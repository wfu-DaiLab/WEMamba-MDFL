import os
import torch
import yaml
import argparse
import time
import numpy as np
import random

from utils import network_parameters, mkdirs


import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from warmup_scheduler.scheduler import GradualWarmupScheduler
from tqdm import tqdm
from tensorboardX import SummaryWriter

from model.Walmafa import MDFL
from src.discriminator import PatchDiscriminator
from src.losses import PerceptualLoss, StyleLoss, AdversarialLoss
from transform.data_RGB import get_training_data, get_validation_data




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyper-parameters for Inpainting GAN')
    parser.add_argument('-yml_path',
                        default="",
                        type=str)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


    torch.backends.cuda.cufft_plan_cache.max_size = 0
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    with open(args.yml_path, 'r') as config:
        opt = yaml.safe_load(config)
    print("Load repair task GAN training configuration file: %s" % (args.yml_path))

    Train = opt['TRAINING']
    OPT = opt['OPTIM']
    LOSS_WEIGHTS = opt['LOSS_WEIGHTS']

    print('==> Building generators and discriminators')
    generator = MDFL(inp_channels=4, out_channels=3, dim=32, num_blocks=[3, 4, 5], heads=[8, 8, 8],
                        ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias', skip=False)
    discriminator = PatchDiscriminator(in_channels=3)
    generator.cuda()
    discriminator.cuda()
    p_number_g = network_parameters(generator)
    p_number_d = network_parameters(discriminator)

    mode = opt['MODEL']['MODE']
    model_dir = os.path.join(Train['SAVE_DIR'], mode, 'models')
    mkdirs(model_dir)

    optimizer_g = optim.AdamW(generator.parameters(), lr=float(OPT['LR_INITIAL']), betas=(0.9, 0.999),
                              weight_decay=1e-4)
    optimizer_d = optim.AdamW(discriminator.parameters(), lr=float(OPT['LR_INITIAL']), betas=(0.9, 0.999),
                              weight_decay=1e-4)


    warmup_epochs = 10
    milestones = [100, 150]
    gamma = 0.1
    print(f"==> Using the MultiStepLR learning rate strategy, preheat the {warmup_ipochs} round and attenuate it in the {milestones} round ..")
    scheduler_multistep_g = optim.lr_scheduler.MultiStepLR(optimizer_g, milestones=milestones, gamma=gamma)
    scheduler_multistep_d = optim.lr_scheduler.MultiStepLR(optimizer_d, milestones=milestones, gamma=gamma)
    scheduler_g = GradualWarmupScheduler(optimizer_g, multiplier=1, total_epoch=warmup_epochs,
                                         after_scheduler=scheduler_multistep_g)
    scheduler_d = GradualWarmupScheduler(optimizer_d, multiplier=1, total_epoch=warmup_epochs,
                                         after_scheduler=scheduler_multistep_d)

    print('==> Load inpainting dataset**')
    train_dataset = get_training_data(Train['TRAIN_DIR'], Train['TRAIN_MASK_DIR'], {'patch_size': Train['TRAIN_PS']})
    train_loader = DataLoader(dataset=train_dataset, batch_size=OPT['BATCH'], shuffle=True, num_workers=8,
                              drop_last=True)


    print("==> The validation set has been disabled. The model will be saved in rounds.")

    print('==> Initialize all loss functions')
    l1_loss_func = nn.L1Loss()
    perceptual_loss_func = PerceptualLoss()
    style_loss_func = StyleLoss()
    adversarial_loss_func = AdversarialLoss()


    start_epoch = 1
    end_epoch = OPT['EPOCHS']

    if Train['RESUME']:
        latest_checkpoint_path = os.path.join(model_dir, "model_latest.pth")
        if os.path.exists(latest_checkpoint_path):
            print("==> Discovered the latest model file and started restoring training ..")
            checkpoint = torch.load(latest_checkpoint_path)
            start_epoch = checkpoint['epoch'] + 1
            generator.load_state_dict(checkpoint['generator_state_dict'])
            discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
            optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
            scheduler_g.load_state_dict(checkpoint['scheduler_g_state_dict'])
            scheduler_d.load_state_dict(checkpoint['scheduler_d_state_dict'])
            print(f"==> Recovery successful! Training will start from Epoch {start_ epoch}.")
        else:
            print("==> The recovery mode has been enabled, but the model_1atest-pth file cannot be found. Training will start from scratch.")
    else:
        print("==> Recovery mode not enabled, training will start from scratch.")

    print(f'''==> Training details:
    ------------------------------------------------------------------
        Generator parameters:     {p_number_g / (1024 * 1024):.2f}M
        Discriminator parameters:     {p_number_d / (1024 * 1024):.2f}M
        Batch size:     {OPT['BATCH']}
        Starting and ending Epoch:     {start_epoch}~{end_epoch}
        Loss weight:
            - Reconstruction (L1):    {LOSS_WEIGHTS['RECONSTRUCTION']}
            - Perception:        {LOSS_WEIGHTS['PERCEPTUAL']}
            - Style:        {LOSS_WEIGHTS['STYLE']}
            - Adversarial:        {LOSS_WEIGHTS['ADVERSARIAL']}
    ------------------------------------------------------------------''')


    print('==> Start GAN training: ')
    log_dir = os.path.join(Train['SAVE_DIR'], mode, 'log')
    mkdirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir, filename_suffix=f'_{mode}')

    if start_epoch == 1:
        scheduler_g.step()
        scheduler_d.step()

    for epoch in range(start_epoch, end_epoch + 1):
        epoch_start_time = time.time()
        epoch_loss_g_total = 0
        epoch_loss_d_total = 0

        generator.train()
        discriminator.train()

        for i, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{end_epoch}"), 0):
            target, input_, mask = data[0].cuda(), data[1].cuda(), data[2].cuda()
            restored_patch = generator(input_, mask)
            inpainted_result = target * (1 - mask) + restored_patch * mask

            optimizer_d.zero_grad();
            pred_real = discriminator(target);
            loss_d_real = adversarial_loss_func(pred_real, True)
            pred_fake = discriminator(inpainted_result.detach());
            loss_d_fake = adversarial_loss_func(pred_fake, False)
            loss_d = (loss_d_real + loss_d_fake) * 0.5;
            loss_d.backward();
            optimizer_d.step()
            epoch_loss_d_total += loss_d.item()

            optimizer_g.zero_grad();
            pred_fake_for_g = discriminator(inpainted_result);
            loss_g_adv = adversarial_loss_func(pred_fake_for_g, True)
            loss_g_rec = l1_loss_func(restored_patch * mask, target * mask)
            loss_g_perceptual = perceptual_loss_func(inpainted_result, target)
            loss_g_style = style_loss_func(inpainted_result, target)
            loss_g_total = (loss_g_rec * LOSS_WEIGHTS['RECONSTRUCTION'] + loss_g_perceptual * LOSS_WEIGHTS[
                'PERCEPTUAL'] + loss_g_style * LOSS_WEIGHTS['STYLE'] + loss_g_adv * LOSS_WEIGHTS['ADVERSARIAL'])
            loss_g_total.backward();
            optimizer_g.step()
            epoch_loss_g_total += loss_g_total.item()


        scheduler_g.step()
        scheduler_d.step()

        avg_loss_g = epoch_loss_g_total / len(train_loader)
        avg_loss_d = epoch_loss_d_total / len(train_loader)
        print("------------------------------------------------------------------")
        print(f"Epoch: {epoch}\tTime: {time.time() - epoch_start_time:.2f}s")
        print(f"  - Generator Total Loss (Avg): {avg_loss_g:.4f}")
        print(f"  - Discriminator Total Loss (Avg): {avg_loss_d:.4f}")
        print(f"  - Learning Rate (G/D): {scheduler_g.get_last_lr()[0]:.6f} / {scheduler_d.get_last_lr()[0]:.6f}")
        print("------------------------------------------------------------------")
        writer.add_scalar('train/loss_generator_total', avg_loss_g, epoch)
        writer.add_scalar('train/loss_discriminator_total', avg_loss_d, epoch)
        writer.add_scalar('train/lr_generator', scheduler_g.get_last_lr()[0], epoch)

        torch.save({
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_g_state_dict': optimizer_g.state_dict(),
            'optimizer_d_state_dict': optimizer_d.state_dict(),
            'scheduler_g_state_dict': scheduler_g.state_dict(),
            'scheduler_d_state_dict': scheduler_d.state_dict(),
        }, os.path.join(model_dir, "model_latest.pth"))

        torch.save({
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
        }, os.path.join(model_dir, f"model_epoch_{epoch}.pth"))
        print(f"==> Intermediate model saved: model_epoch_{epoch}.pth")

    writer.close()

    try:
        total_finish_time = (time.time() - total_start_time)
        print('Total training time: {:.1f} hours'.format((total_finish_time / 60 / 60)))
    except NameError:
        print("**Training resumed from checkpoint. Total training time cannot be calculated.**")
