

import torch
import torch.nn as nn
import os
from torchvision.utils import save_image
import itertools

def dice_loss(pred, target, smooth=1e-6):
    # 将预测值转换为二进制
    pred = pred.argmax(1)  # 将预测值转换为类索引
    target = target.long()  # 确保目标是长整型以便进行比较

    # 计算每个类的 Dice 损失
    dice = 0
    for c in range(target.max().item() + 1):  # 遍历所有类别
        pred_c = (pred == c).float()
        target_c = (target == c).float()

        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()

        dice += (2. * intersection + smooth) / (union + smooth)

    return 1 - (dice / (target.max().item() + 1)).mean()  # 返回 Dice 损失


def get_output_shape(discriminator, input_shape, device):
    """
    使用伪输入计算 Discriminator 的输出形状
    """
    dummy_input = torch.zeros(1, *input_shape).to(device)  # 确保伪输入在正确的设备上
    with torch.no_grad():
        output = discriminator(dummy_input)
    return output.shape[1:]  # 返回 (C, H, W)

def train_cycle_gan_unet(G_A2B, D_B, G_B2A, D_A, unet, dataloader, num_epochs, lr, device, n_classes, save_path, save_frequency):
    # 确保模型保存路径存在
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 确保图像保存路径存在
    image_save_path = os.path.join(save_path, 'generated_images')
    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path)

    # CycleGAN 损失函数
    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    # UNet 损失函数
    criterion_segmentation = nn.CrossEntropyLoss()

    # optimizer_G = torch.optim.Adam(itertools.chain(G_A2B.parameters(), G_B2A.parameters()), lr=lr, betas=(0.5, 0.999))
    # optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=lr, betas=(0.5, 0.999))
    # optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=lr, betas=(0.5, 0.999))
    # unet_optimizer = torch.optim.Adam(unet.parameters(), lr=lr)

    optimizer_G = torch.optim.Adam(list(G_A2B.parameters()) + list(G_B2A.parameters()), lr=lr, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=lr, betas=(0.5, 0.999))
    unet_optimizer = torch.optim.Adam(unet.parameters(), lr=lr, betas=(0.5, 0.999))


    # 计算 Discriminator 输出形状
    example_input = torch.zeros(1, 1, 256, 256).to(device)  # 假设输入大小是 (1, 256, 256)
    output_shape_A = get_output_shape(D_A, example_input.shape[1:], device)
    output_shape_B = get_output_shape(D_B, example_input.shape[1:], device)


    torch.autograd.set_detect_anomaly(True)
    for epoch in range(num_epochs):
        for i, batch in enumerate(dataloader):
            real_A = batch['A'].to(device)
            real_B = batch['B'].to(device)
            label = batch['label'].to(device)

            # 设置目标
            batch_size = real_A.size(0)
            valid = torch.ones((batch_size, *output_shape_B), requires_grad=False).to(device)
            fake = torch.zeros((batch_size, *output_shape_B), requires_grad=False).to(device)

            # 训练生成器
            optimizer_G.zero_grad()

            # Identity loss
            loss_id_A = criterion_identity(G_B2A(real_A), real_A)
            loss_id_B = criterion_identity(G_A2B(real_B), real_B)
            loss_identity = (loss_id_A + loss_id_B) / 2

            # GAN loss
            fake_B = G_A2B(real_A)
            loss_GAN_A2B = criterion_GAN(D_B(fake_B), valid)
            fake_A = G_B2A(real_B)
            loss_GAN_B2A = criterion_GAN(D_A(fake_A), valid)
            loss_GAN = (loss_GAN_A2B + loss_GAN_B2A) / 2

            # Cycle loss
            recovered_A = G_B2A(fake_B)
            loss_cycle_A = criterion_cycle(recovered_A, real_A)
            recovered_B = G_A2B(fake_A)
            loss_cycle_B = criterion_cycle(recovered_B, real_B)
            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            # 总生成器损失
            loss_G = loss_GAN + 10.0 * loss_cycle + 5.0 * loss_identity
            loss_G.backward(retain_graph=True)
            optimizer_G.step()

            # 训练鉴别器 A
            optimizer_D_A.zero_grad()
            loss_real = criterion_GAN(D_A(real_A), valid)
            loss_fake = criterion_GAN(D_A(fake_A.detach()), fake)
            loss_D_A = (loss_real + loss_fake) / 2
            loss_D_A.backward(retain_graph=True)
            optimizer_D_A.step()

            # 训练鉴别器 B
            optimizer_D_B.zero_grad()
            loss_real = criterion_GAN(D_B(real_B), valid)
            loss_fake = criterion_GAN(D_B(fake_B.detach()), fake)
            loss_D_B = (loss_real + loss_fake) / 2
            loss_D_B.backward(retain_graph=True)
            optimizer_D_B.step()

            # 保存生成的fake_B图片
            save_image(fake_B, os.path.join(image_save_path, f'fake_B_epoch_{epoch}_batch_{i}.png'))


         
            # 训练 UNet
            unet_optimizer.zero_grad()
            output = unet(fake_B)
            loss_segmentation = criterion_segmentation(output, label)
            loss_segmentation.backward()
            unet_optimizer.step()

            # 计算 Dice 损失
            dice_loss_value = dice_loss(output, label)


            # 打印损失
            print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(dataloader)}] \n"
                  f"CycleGAN:"
                  f"[G loss: {loss_G.item():.4f}]"  
                  f"[D_A loss: {loss_D_A.item():.4f}] [D_B loss: {loss_D_B.item():.4f}]\n"
                  f"UNet:"
                  f"[segmentation loss: {loss_segmentation.item():.4f}]"
                  f"[Dice loss: {dice_loss_value.item():.4f}]\n")

            # 保存UNet分割结果
            save_image(output.argmax(1, keepdim=True).float() / (n_classes - 1), os.path.join(image_save_path, f'output_epoch_{epoch}_batch_{i}.png'))
                
        # 根据保存频率保存模型
        if (epoch + 1) % save_frequency == 0:
            torch.save(G_A2B.state_dict(), os.path.join(save_path, f'G_A2B_epoch_{epoch+1}.pth'))
            torch.save(G_B2A.state_dict(), os.path.join(save_path, f'G_B2A_epoch_{epoch+1}.pth'))
            torch.save(unet.state_dict(), os.path.join(save_path, f'unet_epoch_{epoch+1}.pth'))
            print(f"Epoch {epoch+1} 完成，模型已保存。")

    # 最后一个epoch完成后保存模型
    torch.save(G_A2B.state_dict(), os.path.join(save_path, f'G_A2B_final.pth'))
    torch.save(G_B2A.state_dict(), os.path.join(save_path, f'G_B2A_final.pth'))
    torch.save(unet.state_dict(), os.path.join(save_path, f'unet_final.pth'))
    print("训练结束，所有模型已保存。")


