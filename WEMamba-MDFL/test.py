import numpy as np
import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from PIL import Image
import os
import cv2
from skimage import img_as_ubyte
from natsort import natsorted
from glob import glob
import argparse
import time
from collections import OrderedDict
from model.Walmafa import MDFL


def load_checkpoint(model, weights):

    try:
        checkpoint = torch.load(weights, map_location=torch.device('cuda'))
    except Exception as e:
        print(f"Failed to load the weight file: {e}")
        return


    state_dict = None

    if 'generator_state_dict' in checkpoint:
        state_dict = checkpoint['generator_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    if state_dict is None:
        print("Error: No recognizable state_dict found in the checkpoint file!")
        return

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    try:
        model.load_state_dict(new_state_dict, strict=True)
        print("Model weights loaded successfully!")
    except Exception as e:
        print(f"Strict mode failed to load state_dect: {e}")
        print("Attempting to load in non strict mode ..")
        try:
            model.load_state_dict(new_state_dict, strict=False)
            print("Non strict mode loaded successfully!")
        except Exception as e2:
            print(f"Non strict mode loading also failed: {e2}")


def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image Inpainting Demo')
    parser.add_argument('--input_dir', default='', type=str, help='Input images folder')
    parser.add_argument('--mask_dir', default='', type=str, help='Input masks folder')
    parser.add_argument('--result_dir', default='', type=str, help='Directory for results')
    parser.add_argument('--weights', default='', type=str, help='Path to weights')
    args = parser.parse_args()

    inp_dir = args.input_dir
    mask_dir = args.mask_dir
    out_dir = args.result_dir
    os.makedirs(out_dir, exist_ok=True)

    files = natsorted(glob(os.path.join(inp_dir, '*.*'))) 
    masks = natsorted(glob(os.path.join(mask_dir, '*.*')))

    if not files: print(f"No image files were found in '{inpudir}'!")
    if not masks: print(f"No mask files were found in '{mask_dir}'!")

    model = MDFL(inp_channels=4, out_channels=3, dim=32, num_blocks=[3, 4, 5], heads=[8, 8, 8], ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias', skip=False)
    model.cuda()
    model.eval()

    load_checkpoint(model, args.weights)

    print('Start image inpainting...')
    total_time = 0

    if len(files) != len(masks):
        print(f"Warning: The number of images ({len (files)}) does not match the number of masks ({len (masks)})!")

    for i, (file_, mask_file) in enumerate(zip(files, masks)):
        img = Image.open(file_).convert('RGB')
        mask = Image.open(mask_file).convert('L')

        if img.size != mask.size: mask = mask.resize(img.size, Image.NEAREST)

        input_img_tensor = TF.to_tensor(img).unsqueeze(0).cuda()
        input_mask_tensor = TF.to_tensor(mask).unsqueeze(0).cuda()
        input_mask_tensor = (input_mask_tensor > 0.5).float()

        masked_input = input_img_tensor * (1. - input_mask_tensor)

        mul = 16
        h, w = masked_input.shape[2], masked_input.shape[3]
        H, W = ((h + mul) // mul) * mul, ((w + mul) // mul) * mul
        padh = H - h if h % mul != 0 else 0
        padw = W - w if w % mul != 0 else 0
        padded_masked_input = F.pad(masked_input, (0, padw, 0, padh), 'reflect')
        padded_mask = F.pad(input_mask_tensor, (0, padw, 0, padh), 'reflect')
        
        start_time = time.time()
        with torch.no_grad():
            restored_patch = model(padded_masked_input, padded_mask)
        
        end_time = time.time()
        processing_time = end_time - start_time
        total_time += processing_time

        restored_patch = restored_patch[:, :, :h, :w]
        restored_patch = torch.clamp(restored_patch, 0, 1)

        inpainted_result = input_img_tensor * (1. - input_mask_tensor) + restored_patch * input_mask_tensor
        
        inpainted_result = inpainted_result.permute(0, 2, 3, 1).cpu().detach().numpy()
        inpainted_result = img_as_ubyte(inpainted_result[0])

        original_filename = os.path.basename(file_)
        save_img(os.path.join(out_dir, original_filename), inpainted_result)
        
        print(f"Processed {i+1}/{len (files)}: {origina_filename}, Time consumption: {processing_time:. 4f} s")

    avg_time = total_time / len(files) if files else 0
    print("-" * 30)
    print(f"**All image inpainting completed. Results saved to: {out_dir}")
    print(f"Average processing time per image: {avg_time:.4f}s")
