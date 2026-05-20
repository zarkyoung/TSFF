import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchmetrics.functional import structural_similarity_index_measure as SSIM
import os
import numpy as np

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# 感知损失
class VGGPerceptualLoss(nn.Module):
    def __init__(self, layers=None):
        super(VGGPerceptualLoss, self).__init__()

        # 使用新的 'weights' 参数来加载预训练的 VGG19 网络
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features

        # 冻结 VGG19 网络的参数
        for param in vgg.parameters():
            param.requires_grad = False

        # 如果未指定层，默认使用 VGG19 的 'conv1_2', 'conv2_2', 'conv3_2', 'conv4_2', 'conv5_2' 层
        if layers is None:
            layers = ['3', '8', '17', '26', '35']  # 对应于 VGG19 中的卷积层

        self.vgg_layers = nn.ModuleList([vgg[int(layer)] for layer in layers])
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def forward(self, x, y):
        # 将单通道图像复制三次，变成三通道
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if y.shape[1] == 1:
            y = y.repeat(1, 3, 1, 1)

        # 归一化输入
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        y = (y - self.mean.to(y.device)) / self.std.to(y.device)

        loss = 0.0
        for layer in self.vgg_layers:
            x = layer(x).detach()  # 确保在特征提取时没有改变计算图
            y = layer(y).detach()  # 确保在特征提取时没有改变计算图
            # 使用L1损失来计算生成图像和真实图像在特征空间的差异
            loss += F.l1_loss(x, y)

        return loss

# 边缘保留损失：使用 Sobel 算子来提取图像边缘
def edge_preserving_loss(predicted, target):
    # Sobel算子定义
    sobel_x = torch.tensor([[1, 0, -1],
                             [2, 0, -2],
                             [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    sobel_y = torch.tensor([[1, 2, 1],
                             [0, 0, 0],
                             [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    sobel_x = sobel_x.to(predicted.device)
    sobel_y = sobel_y.to(predicted.device)

    # 计算输入和目标的梯度
    input_grad_x = F.conv2d(predicted, sobel_x, padding=1)
    input_grad_y = F.conv2d(predicted, sobel_y, padding=1)
    target_grad_x = F.conv2d(target, sobel_x, padding=1)
    target_grad_y = F.conv2d(target, sobel_y, padding=1)

    # 计算梯度差
    grad_diff = F.l1_loss(input_grad_x + input_grad_y, target_grad_x + target_grad_y)

    # 计算平滑损失
    smooth_input = F.avg_pool2d(predicted, kernel_size=3, stride=1, padding=1)
    smooth_target = F.avg_pool2d(target, kernel_size=3, stride=1, padding=1)

    return grad_diff + F.l1_loss(smooth_input, smooth_target)

# SSIM 损失
def ssim_loss(predicted, target):
    # 使用 torchmetrics 的 SSIM
    ssim_value = SSIM(predicted, target)
    return 1 - ssim_value

# 综合损失
class ReconstructionLoss(nn.Module):
    def __init__(self, config, alpha=0.3, beta=0.4, gamma=0.3):
        super(ReconstructionLoss, self).__init__()
        self.config = config
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.perceptual_loss = VGGPerceptualLoss()

    def forward(self, prediction, target, M1_pre=None, M1=None):
        # 计算边缘保留损失
        edge_loss = edge_preserving_loss(prediction, target)

        # 计算感知损失
        perceptual_loss_value = self.perceptual_loss(prediction, target)

        # 计算SSIM损失
        ssim_loss_value = ssim_loss(prediction, target)

        if self.config.use_twoway:
            loss = (F.mse_loss(prediction, target) + F.mse_loss(M1_pre, M1) +
                    self.alpha * edge_loss + self.beta * perceptual_loss_value + self.gamma * ssim_loss_value)
            return loss
        else:
            loss = (F.mse_loss(prediction, target) +
                    self.alpha * edge_loss + self.beta * perceptual_loss_value + self.gamma * ssim_loss_value)
            return loss

class CAFDBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CAFDBlock, self).__init__()
        residual_fine = [nn.Conv2d(in_channels, out_channels, 3, 1, padding=1),
                         nn.ReLU(),
                         nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)]

        self.residual_fine = nn.Sequential(*residual_fine)
        self.residual_coarse = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)

    def forward(self, coarse, fine):
        coarse_features = self.residual_coarse(coarse)
        fine_features = self.residual_fine(fine) + coarse_features
        return coarse_features, fine_features

class CAFD_extract(nn.Module):
    def __init__(self, Dims):
        super(CAFD_extract, self).__init__()
        self.encoder1 = CAFDBlock(1, Dims//2)
        self.encoder2 = CAFDBlock(Dims//2, Dims//2)

        self.decoder1 = Resblock(Dims, Dims)
        self.decoder2 = Resblock(Dims, Dims)

    def forward(self, coarse, fine):
        coarse, fine = self.encoder1(coarse, fine)
        coarse, fine = self.encoder2(coarse, fine)

        out = self.decoder1(torch.cat((coarse, fine), 1))
        out = self.decoder2(out)
        return out

class Resblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Resblock, self).__init__()
        residual = [nn.Conv2d(in_channels, out_channels, 3, 1, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(out_channels, out_channels, 1, 1)]

        self.residual = nn.Sequential(*residual)

    def forward(self, inputs):
        out = self.residual(inputs)
        out = out * 0.1 + inputs
        return out

class CVFD_extract(nn.Module):
    def __init__(self, Dims):
        super(CVFD_extract, self).__init__()
        self.STF1 = Resblock(1, Dims)
        self.STF2 = Resblock(Dims, Dims)
        self.STF3 = Resblock(Dims, Dims)
        self.STF4 = Resblock(Dims, Dims)

    def forward(self, E0, E1, M1):
        STF_features = self.STF1(M1 + E0 - E1)
        STF_features = self.STF2(STF_features)
        STF_features = self.STF3(STF_features)
        STF_features = self.STF4(STF_features)
        return STF_features


class LocalSpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(LocalSpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, 1, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention_map1 = self.conv1(x)
        attention_map2 = self.conv2(x)
        attention_map3 = self.conv3(x)
        attention_map = self.sigmoid(attention_map1 + attention_map2 + attention_map3)
        return x * attention_map

class GlobalSpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(GlobalSpatialAttention, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        global_context = self.global_pool(x)
        attention_map = self.fc(global_context)
        return x * attention_map

class Fusion(nn.Module):
    def __init__(self, Dims):
        super(Fusion, self).__init__()
        self.LocalSpatialAttention_CAFD = LocalSpatialAttention(Dims)
        self.GlobalSpatialAttention_CAFD = GlobalSpatialAttention(Dims)

        self.LocalSpatialAttention_CVFD = LocalSpatialAttention(Dims)
        self.GlobalSpatialAttention_CVFD = GlobalSpatialAttention(Dims)

        self.CAFD_weight = nn.Parameter(torch.FloatTensor([0.5]))
        self.CVFD_weight = nn.Parameter(torch.FloatTensor([0.5]))

    def forward(self, CAFD, CVFD):
        LocalSpatialAttention_CAFD = self.LocalSpatialAttention_CAFD(CAFD)
        GlobalSpatialAttention_CAFD = self.GlobalSpatialAttention_CAFD(CAFD)
        CAFD_feature = LocalSpatialAttention_CAFD + GlobalSpatialAttention_CAFD

        LocalSpatialAttention_CVFD = self.LocalSpatialAttention_CVFD(CVFD)
        GlobalSpatialAttention_CVFD = self.GlobalSpatialAttention_CVFD(CVFD)
        CVFD_feature = LocalSpatialAttention_CVFD + GlobalSpatialAttention_CVFD

        # 将权重合并到一个张量中
        weights = torch.cat((self.CAFD_weight, self.CVFD_weight))
        # 应用softmax以确保权重和为1
        normalized_weights = torch.softmax(weights, dim=0)
        # 使用归一化后的权重进行加权融合
        out = normalized_weights[0] * CAFD_feature + normalized_weights[1] * CVFD_feature
        return out

class MutiScaleDown(nn.Module):
    def __init__(self, Dims):
        super(MutiScaleDown, self).__init__()
        self.Down1 = nn.Sequential(
            nn.Conv2d(Dims, Dims, 3, 1, padding=1),
            nn.ReLU(),
            nn.Conv2d(Dims, Dims // 2, 1, 1),
            nn.PixelUnshuffle(2),
            nn.ReLU()
        )
        self.Down2 = nn.Sequential(
            nn.Conv2d(2 * Dims, 2 * Dims, 3, 1, padding=1),
            nn.ReLU(),
            nn.Conv2d(2 * Dims, Dims, 1, 1),
            nn.PixelUnshuffle(2),
            nn.ReLU()
        )
        self.Down3 = nn.Sequential(
            nn.Conv2d(4 * Dims, 4 * Dims, 3, 1, padding=1),
            nn.ReLU(),
            nn.Conv2d(4 * Dims, 2 * Dims, 1, 1),
            nn.PixelUnshuffle(2),
            nn.ReLU()
        )

    def forward(self, inputs):
        Down1 = self.Down1(inputs)
        Down2 = self.Down2(Down1)
        Down3 = self.Down3(Down2)
        return Down1, Down2, Down3

class MutiScaleUp(nn.Module):
    def __init__(self, Dims):
        super(MutiScaleUp, self).__init__()
        self.Up2 = nn.Sequential(
            nn.Conv2d(8 * Dims, 16 * Dims, 3, 1, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU()
        )
        self.Conv2 = nn.Sequential(
            nn.Conv2d(8 * Dims, 4 * Dims, 3, 1, padding=1),
            nn.ReLU()
        )
        self.Up1 = nn.Sequential(
            nn.Conv2d(4 * Dims, 8 * Dims, 3, 1, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU()
        )
        self.Conv1 = nn.Sequential(
            nn.Conv2d(4 * Dims, 2 * Dims, 3, 1, padding=1),
            nn.ReLU()
        )
        self.Up0 = nn.Sequential(
            nn.Conv2d(2 * Dims, 4 * Dims, 3, 1, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU()
        )
        self.Conv0 = nn.Sequential(
            nn.Conv2d(2 * Dims, Dims, 3, 1, padding=1),
            nn.ReLU()
        )

    def forward(self, Fusion0, Fusion1, Fusion2, Fusion3):
        Up2 = self.Up2(Fusion3)
        Up2 = torch.cat((Up2, Fusion2), 1)
        Up2 = self.Conv2(Up2)

        Up1 = self.Up1(Up2)
        Up1 = torch.cat((Up1, Fusion1), 1)
        Up1 = self.Conv1(Up1)

        Up0 = self.Up0(Up1)
        Up0 = torch.cat((Up0, Fusion0), 1)
        Up0 = self.Conv0(Up0)
        return Up0

class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config
        
        if self.config.use_cafd:
            self.CAFD_extract = CAFD_extract(self.config.dims)
            self.CAFD_Downs = MutiScaleDown(self.config.dims)

        if self.config.use_cvfd:
            self.CVFD_extract = CVFD_extract(self.config.dims)
            self.CVFD_Downs = MutiScaleDown(self.config.dims)

        self.Up = MutiScaleUp(self.config.dims)
        
        self.Fusion0 = Fusion(self.config.dims)
        self.Fusion1 = Fusion(2 * self.config.dims)
        self.Fusion2 = Fusion(4 * self.config.dims)
        self.Fusion3 = Fusion(8 * self.config.dims)
        
        self.recont = nn.Conv2d(self.config.dims, 1, 3, 1, padding=1)

    def forward(self, E0, E1, M1):
        if self.config.use_cafd and not self.config.use_cvfd:
            CAFD_features = self.CAFD_extract(E0, M1)
            SD_Down1, SD_Down2, SD_Down3 = self.CAFD_Downs(CAFD_features)
            up = self.Up(CAFD_features, SD_Down1, SD_Down2, SD_Down3)
        
        if self.config.use_cvfd and not self.config.use_cafd:
            CVFD_features = self.CVFD_extract(E0, E1, M1)
            STF_Down1, STF_Down2, STF_Down3 = self.CVFD_Downs(CVFD_features)
            up = self.Up(CVFD_features, STF_Down1, STF_Down2, STF_Down3)

        if self.config.use_cvfd and self.config.use_cafd:
            CAFD_features = self.CAFD_extract(E0, M1)
            SD_Down1, SD_Down2, SD_Down3 = self.CAFD_Downs(CAFD_features)
            CVFD_features = self.CVFD_extract(E0, E1, M1)
            STF_Down1, STF_Down2, STF_Down3 = self.CVFD_Downs(CVFD_features)
        
            Fusion0 = self.Fusion0(CAFD_features, CVFD_features)
            Fusion1 = self.Fusion1(SD_Down1, STF_Down1)
            Fusion2 = self.Fusion2(SD_Down2, STF_Down2)
            Fusion3 = self.Fusion3(SD_Down3, STF_Down3)
            up = self.Up(Fusion0, Fusion1, Fusion2, Fusion3)

        out = self.recont(up)
        return out

class Recont(nn.Module):
    def __init__(self, config):
        super(Recont, self).__init__()
        self.config = config

        if self.config.use_twoway:
            self.forward_pre = Generator(self.config)
            self.backward_pre = Generator(self.config)
        else:
            self.forward_pre = Generator(self.config)

        # 初始化权重
        self.__initialize_wights()

    def __initialize_wights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, E0, E1, M1):
        if self.config.use_twoway:
            M0_pre = self.forward_pre(E0, E1, M1)
            M1_pre = self.backward_pre(E1, E0, M0_pre)
            return M0_pre+E0-E1+M1, M1_pre+E1-E0+M0_pre
        else:
            M0_pre = self.forward_pre(E0, E1, M1)
            return M0_pre+E0-E1+M1
        
class Discriminator(nn.Module):
    def __init__(self, input_channels):
        super(Discriminator, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        # 修改最后一层，使其输出一个标量值
        self.conv5 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0)
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.batch_norm3 = nn.BatchNorm2d(256)
        self.batch_norm4 = nn.BatchNorm2d(512)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.batch_norm2(self.conv2(x)))
        x = self.leaky_relu(self.batch_norm3(self.conv3(x)))
        x = self.leaky_relu(self.batch_norm4(self.conv4(x)))
        x = self.conv5(x)
        return x.view(-1)  # 将输出展平为一维张量