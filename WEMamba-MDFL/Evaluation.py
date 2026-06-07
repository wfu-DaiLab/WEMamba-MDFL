import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import lpips
from scipy import linalg
from torchvision.models import inception_v3
import argparse
from tqdm import tqdm
from skimage.metrics import structural_similarity

try:
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["SimHei", "Heiti TC", "WenQuanYi Micro Hei"]
    plt.rcParams["axes.unicode_minus"] = False
    PLT_AVAILABLE = True
except ImportError:
    PLT_AVAILABLE = False
    print("Warning: Matplotlib not found. Visualization will be disabled.")


def calculate_mae(original, reconstructed, max_pixel=255.0):
    absolute_error = np.abs(original - reconstructed)
    mae_percent = np.mean(absolute_error / max_pixel) * 100
    return mae_percent


def calculate_psnr(original, reconstructed, max_pixel=255.0):
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(max_pixel ** 2 / mse)


def calculate_ssim(original, reconstructed, data_range=255.0):
    """
    计算结构相似性指标SSIM。
    使用 scikit-image 的标准实现，并明确指定通道轴。
    """
    if original.ndim != 3 or reconstructed.ndim != 3:
        raise ValueError("Input images must have three dimensions (H, W, C)")

    ssim_value = structural_similarity(
        original,
        reconstructed,
        channel_axis=-1,
        data_range=data_range
    )
    return ssim_value


def calculate_lpips(original, reconstructed, model):
    """计算学习感知图像块相似性LPIPS"""
    device = next(model.parameters()).device
    if not isinstance(original, torch.Tensor):
        original = transforms.ToTensor()(original).unsqueeze(0)
    if not isinstance(reconstructed, torch.Tensor):
        reconstructed = transforms.ToTensor()(reconstructed).unsqueeze(0)

    original = original.to(device)
    reconstructed = reconstructed.to(device)

    original = original * 2 - 1
    reconstructed = reconstructed * 2 - 1

    with torch.no_grad():
        lpips_value = model(original, reconstructed)

    return lpips_value.item()


def calculate_fid(original_images, reconstructed_images, dims=2048, batch_size=8, device='cuda'):
    """计算弗雷歇距离FID（修复cuDNN错误，降低batch_size）"""
    # 修复deprecated警告 + 解决cuDNN错误
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    inception_model = inception_v3(
        weights='IMAGENET1K_V1',  # 替换pretrained=True
        transform_input=False
    ).to(device)
    inception_model.fc = nn.Identity()
    inception_model.eval()

    def get_activations(images, model):
        if not isinstance(images, torch.Tensor):
            images = torch.stack(images)
        images = images.to(device)

        if images.shape[2] != 299 or images.shape[3] != 299:
            images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)

        pred_arr = np.empty((len(images), dims))
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch = batch * 2 - 1

            with torch.no_grad():
                pred = model(batch)

            if pred.shape[1] != dims:
                pred = F.adaptive_avg_pool2d(pred, 1)

            if len(pred.shape) > 2:
                pred = pred.reshape(pred.size(0), -1)

            pred_arr[i:i + batch_size] = pred.cpu().numpy()

            # 清理显存
            del batch, pred
            torch.cuda.empty_cache()

        return pred_arr

    def calculate_activation_statistics(images, model):
        act = get_activations(images, model)
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma

    def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, 'Mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, 'Covariances have different dimensions'

        diff = mu1 - mu2

        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError(f'Imaginary component {m}')
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

    # 清理显存
    torch.cuda.empty_cache()
    mu1, sigma1 = calculate_activation_statistics(original_images, inception_model)
    torch.cuda.empty_cache()
    mu2, sigma2 = calculate_activation_statistics(reconstructed_images, inception_model)
    torch.cuda.empty_cache()

    fid_value = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

    # 恢复默认设置
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    return fid_value


def evaluate_image_repair(original_dir, reconstructed_dir, output_file=None, verbose=True):
    """评估图像修复结果，计算多个指标（按顺序匹配，不依赖文件名）。"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 获取并排序原始/修复文件列表（按文件名排序，保证顺序一致）
    original_files = sorted([f for f in os.listdir(original_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    reconstructed_files = sorted(
        [f for f in os.listdir(reconstructed_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    # 打印基础信息
    print(f"\n=== 匹配信息 ===")
    print(f"原始文件数量: {len(original_files)}")
    print(f"修复文件数量: {len(reconstructed_files)}")

    if not original_files or not reconstructed_files:
        print(
            f"Error: Directory is empty. original_dir: {len(original_files)} files, reconstructed_dir: {len(reconstructed_files)} files.")
        return None

    # 取最小数量，避免索引越界
    min_file_count = min(len(original_files), len(reconstructed_files))
    if len(original_files) != len(reconstructed_files):
        print(f"Warning: 文件数量不匹配！将仅处理前 {min_file_count} 对图片")
        original_files = original_files[:min_file_count]
        reconstructed_files = reconstructed_files[:min_file_count]

    lpips_model = lpips.LPIPS(net='alex').to(device)

    all_mae, all_psnr, all_ssim, all_lpips = [], [], [], []
    original_images, reconstructed_images = [], []

    desc = "Evaluating images"

    # 核心修改：按索引一一匹配，不依赖文件名
    for i in tqdm(range(min_file_count), desc=desc, disable=not verbose):
        orig_file = original_files[i]
        rec_file = reconstructed_files[i]

        try:
            # 加载图片
            orig_img = Image.open(os.path.join(original_dir, orig_file)).convert('RGB')
            rec_img = Image.open(os.path.join(reconstructed_dir, rec_file)).convert('RGB')

            # 统一图片尺寸
            if orig_img.size != rec_img.size:
                rec_img = rec_img.resize(orig_img.size, Image.BICUBIC)

            # 转换为numpy数组计算指标
            orig_np = np.array(orig_img).astype(np.float32)
            rec_np = np.array(rec_img).astype(np.float32)

            # 计算各项指标
            all_mae.append(calculate_mae(orig_np, rec_np))
            all_psnr.append(calculate_psnr(orig_np, rec_np))
            all_ssim.append(calculate_ssim(orig_np, rec_np))
            all_lpips.append(calculate_lpips(orig_img, rec_img, lpips_model))

            # 保存tensor用于FID计算
            original_images.append(transforms.ToTensor()(orig_img))
            reconstructed_images.append(transforms.ToTensor()(rec_img))

        except Exception as e:
            print(f"\nWarning: 处理第 {i} 对图片失败 (原始: {orig_file}, 修复: {rec_file}) - {str(e)}")
            continue

    if not original_images:
        print("Error: No images were processed for evaluation.")
        return None

    # 计算FID
    if verbose: print("计算 FID 中 (这可能需要一些时间)...")
    fid = calculate_fid(original_images, reconstructed_images, device=device)

    # 计算平均指标
    avg_mae = np.mean(all_mae)
    avg_psnr = np.mean(all_psnr)
    avg_ssim = np.mean(all_ssim)
    avg_lpips = np.mean(all_lpips)

    results = {'mae': avg_mae, 'psnr': avg_psnr, 'ssim': avg_ssim, 'lpips': avg_lpips, 'fid': fid}

    # 打印结果
    if verbose:
        print("\n=== 平均指标 ===")
        print(f"实际处理图像数量: {len(original_images)}")
        print(f"MAE (平均绝对误差): {avg_mae:.4f} %")
        print(f"PSNR (峰值信噪比): {avg_psnr:.4f} dB")
        print(f"SSIM (结构相似性): {avg_ssim:.4f}")
        print(f"LPIPS (学习感知图像块相似性): {avg_lpips:.4f}")
        print(f"FID (弗雷歇距离): {fid:.4f}")

    # 保存结果到文件
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=== 图像修复评估结果 ===\n\n")
            f.write(f"原始文件总数: {len(original_files)}\n")
            f.write(f"修复文件总数: {len(reconstructed_files)}\n")
            f.write(f"实际处理图像数量: {len(original_images)}\n\n")
            f.write("\n=== 平均指标 ===\n")
            f.write(f"MAE (平均绝对误差): {avg_mae:.4f} %\n")
            f.write(f"PSNR (峰值信噪比): {avg_psnr:.4f} dB\n")
            f.write(f"SSIM (结构相似性): {avg_ssim:.4f}\n")
            f.write(f"LPIPS (学习感知图像块相似性): {avg_lpips:.4f}\n")
            f.write(f"FID (弗雷歇距离): {fid:.4f}\n")
        if verbose: print(f"\n结果已保存到: {output_file}")

    return results


if __name__ == "__main__":
    DEFAULT_ORIGINAL_DIR = "/home/li/wu/data_set/inpainting/celeba-hq/test_2000_png"
    DEFAULT_RECONSTRUCTED_DIR = "/home/li/li/Runing_codes/TSGDA/demo/nodrop/0.5-0.6"
    DEFAULT_OUTPUT_FILE = "/home/li/wu/fir_work/WalMaFa/comparative_experiment/TSGDAM/Celeba-HQ/0.5-0.6.txt"

    parser = argparse.ArgumentParser(description='Image Inpainting Evaluation Script (按顺序匹配)')
    parser.add_argument('--original_dir', type=str, default=DEFAULT_ORIGINAL_DIR,
                        help=f'Directory of original ground truth images. Default: {DEFAULT_ORIGINAL_DIR}')
    parser.add_argument('--reconstructed_dir', type=str, default=DEFAULT_RECONSTRUCTED_DIR,
                        help=f'Directory of inpainted images. Default: {DEFAULT_RECONSTRUCTED_DIR}')
    parser.add_argument('--output_file', type=str, default=DEFAULT_OUTPUT_FILE,
                        help=f'(Optional) Path to save the evaluation results. Default: {DEFAULT_OUTPUT_FILE}')
    args = parser.parse_args()

    print("开始评估图像修复结果...")
    print(f"原始图像目录: {args.original_dir}")
    print(f"修复图像目录: {args.reconstructed_dir}")

    results = evaluate_image_repair(args.original_dir, args.reconstructed_dir, args.output_file)

    # 生成可视化图表
    if PLT_AVAILABLE and results:
        print("正在生成可视化图表...")
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.bar(['PSNR', 'SSIM'], [results['psnr'], results['ssim']], color=['blue', 'green'])
        plt.title('PSNR and SSIM Metrics')
        plt.ylabel('Value')

        plt.subplot(2, 2, 2)
        plt.bar(['MAE', 'LPIPS'], [results['mae'], results['lpips']], color=['red', 'purple'])
        plt.title('MAE and LPIPS Metrics')
        plt.ylabel('Value')

        plt.subplot(2, 2, 3)
        plt.bar(['FID'], [results['fid']], color='orange')
        plt.title('FID Metric')
        plt.ylabel('Value')

        plt.suptitle('Overall Evaluation Metrics', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        output_fig_path = 'metrics_visualization.png'
        if args.output_file:
            output_fig_path = os.path.splitext(args.output_file)[0] + '.png'

        plt.savefig(output_fig_path)
        print(f"可视化图表已保存到: {output_fig_path}")
        plt.show()