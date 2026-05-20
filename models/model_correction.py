import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model import Recont

class SimpleCorrection(nn.Module):
    """简单的修正网络，使用辅助数据对model.py的结果进行修正"""
    
    def __init__(self, config):
        super(SimpleCorrection, self).__init__()
        self.config = config
        
        # 辅助数据编码器（简化版）
        self.dem_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, 1, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.water_vapor_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, 1, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.longwave_encoder = nn.Sequential(
            nn.Conv2d(2, 16, 3, 1, padding=1),  # 2通道：上行+下行
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, 1, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 修正网络：输入为基础预测+辅助特征，输出为修正量
        self.correction_net = nn.Sequential(
            # 输入: 1(基础预测) + 16(DEM) + 16(水汽) + 16(长波) = 49通道
            nn.Conv2d(49, 64, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, 1, padding=1),  # 输出修正量
            nn.Tanh()  # 限制修正幅度在[-1, 1]范围内
        )
        
        # 修正幅度控制参数（可学习）
        self.correction_scale = nn.Parameter(torch.tensor(5.0))  # 最大修正5K
        
    def forward(self, base_prediction, dem, water_vapor, longwave_up, longwave_down):
        """
        Args:base_prediction
            base_prediction: model.py的预测结果 [B, 1, H, W]
            dem: DEM数据 [B, 1, H, W]
            water_vapor: 大气水汽含量 [B, 1, H, W]
            longwave_up: 上行长波辐射 [B, 1, H, W]
            longwave_down: 下行长波辐射 [B, 1, H, W]
        Returns:
            corrected_prediction: 修正后的预测结果 [B, 1, H, W]
        """
        # 编码辅助数据
        dem_features = self.dem_encoder(dem)
        wv_features = self.water_vapor_encoder(water_vapor)
        lw_features = self.longwave_encoder(torch.cat([longwave_up, longwave_down], dim=1))
        
        # 拼接所有特征
        combined_features = torch.cat([
            base_prediction,  # 基础预测
            dem_features,     # DEM特征
            wv_features,      # 水汽特征
            lw_features       # 长波特征
        ], dim=1)
        
        # 计算修正量
        correction = self.correction_net(combined_features)
        correction = correction * self.correction_scale
        
        # 应用修正
        corrected_prediction = base_prediction + correction
        
        return corrected_prediction, correction

class TwoStageModel(nn.Module):
    """两阶段模型：基础预测 + 辅助数据修正"""
    
    def __init__(self, config, base_model_path=None):
        super(TwoStageModel, self).__init__()
        self.config = config

        # 第一阶段：基础模型 (model.py)
        self.base_model = Recont(config)
        
        # 如果提供了预训练模型路径，加载权重
        if base_model_path:
            self.load_base_model(base_model_path)
            # 冻结基础模型参数
            for param in self.base_model.parameters():
                param.requires_grad = False
                
        # 第二阶段：修正网络
        self.correction_model = SimpleCorrection(config)
        
    def load_base_model(self, model_path):
        """加载预训练的基础模型"""
        print(f"Loading base model from: {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if 'generator' in checkpoint:
            self.base_model.load_state_dict(checkpoint['generator'])
        elif 'model_state_dict' in checkpoint:
            self.base_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.base_model.load_state_dict(checkpoint)
            
        print("Base model loaded successfully!")
    
    def freeze_base_model(self):
        """冻结基础模型参数"""
        for param in self.base_model.parameters():
            param.requires_grad = False
            
    def unfreeze_base_model(self):
        """解冻基础模型参数（用于联合微调）"""
        for param in self.base_model.parameters():
            param.requires_grad = True
    
    def forward(self, E0, E1, M1, dem, water_vapor, longwave_up, longwave_down, 
                stage='correction'):
        """
        Args:
            stage: 'base_only', 'correction', 'joint'
        """
        # 第一阶段：基础预测
        if self.config.use_twoway:
            base_prediction, M1_pre = self.base_model(E0, E1, M1)
        else:
            base_prediction = self.base_model(E0, E1, M1)
            M1_pre = None
            
        if stage == 'base_only':
            return base_prediction, M1_pre
        
        # 第二阶段：修正
        corrected_prediction, correction = self.correction_model(
            base_prediction, dem, water_vapor, longwave_up, longwave_down
        )
        
        if stage == 'correction':
            return corrected_prediction, correction, base_prediction
        elif stage == 'joint':
            # 联合训练模式
            return corrected_prediction, M1_pre, base_prediction, correction

class CorrectionLoss(nn.Module):
    """修正网络的损失函数"""
    
    def __init__(self, config):
        super(CorrectionLoss, self).__init__()
        self.config = config
        self.mse_loss = nn.MSELoss()
        
        # 损失权重
        self.correction_weight = 1.0      # 修正结果的权重
        self.regularization_weight = 0.1  # 修正量正则化权重
        
    def forward(self, corrected_pred, target, correction, base_pred=None):
        """
        Args:
            corrected_pred: 修正后的预测
            target: 真实值
            correction: 修正量
            base_pred: 基础预测（可选，用于分析）
        """
        # 主要损失：修正后预测与真实值的差异
        main_loss = self.mse_loss(corrected_pred, target)
        
        # 正则化损失：限制修正量的幅度
        reg_loss = torch.mean(torch.abs(correction))
        
        # 总损失
        total_loss = (self.correction_weight * main_loss + 
                     self.regularization_weight * reg_loss)
        
        # 如果有基础预测，计算改进量
        if base_pred is not None:
            base_loss = self.mse_loss(base_pred, target)
            improvement = base_loss - main_loss
            return total_loss, main_loss, reg_loss, improvement
        
        return total_loss, main_loss, reg_loss

# 使用示例和训练策略
class TwoStageTrainer:
    """两阶段训练器"""
    
    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config
        self.criterion = CorrectionLoss(config)
        
    def train_correction_stage(self, train_loader, val_loader, epochs=50):
        """训练修正阶段"""
        print("=== 开始训练修正网络 ===")
        
        # 只优化修正网络参数
        optimizer = torch.optim.Adam(
            self.model.correction_model.parameters(),
            lr=0.001,  # 较高的学习率，因为只训练小网络
            weight_decay=1e-4
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.8, patience=5
        )
        
        best_loss = float('inf')
        
        for epoch in range(epochs):
            # 训练
            self.model.train()
            train_loss = 0
            train_improvement = 0
            
            for batch_idx, data_dict in enumerate(train_loader):
                # 移动数据到设备
                for key, value in data_dict.items():
                    if isinstance(value, torch.Tensor):
                        data_dict[key] = value.to(self.device)
                
                optimizer.zero_grad()
                
                # 前向传播
                corrected_pred, correction, base_pred = self.model(
                    data_dict['eobj'], data_dict['eref'], data_dict['mref'],
                    data_dict['dem'], data_dict['wv_obj'], 
                    data_dict['lw_up_obj'], data_dict['lw_down_obj'],
                    stage='correction'
                )
                
                # 计算损失
                total_loss, main_loss, reg_loss, improvement = self.criterion(
                    corrected_pred, data_dict['mobj'], correction, base_pred
                )
                
                # 反向传播
                total_loss.backward()
                optimizer.step()
                
                train_loss += main_loss.item()
                train_improvement += improvement.item()
            
            # 验证
            val_loss, val_improvement = self.validate(val_loader)
            scheduler.step(val_loss)
            
            # 日志
            avg_train_loss = train_loss / len(train_loader)
            avg_train_improvement = train_improvement / len(train_loader)
            
            print(f"Epoch [{epoch+1}/{epochs}]")
            print(f"Train Loss: {avg_train_loss:.6f}, Improvement: {avg_train_improvement:.6f}")
            print(f"Val Loss: {val_loss:.6f}, Improvement: {val_improvement:.6f}")
            print(f"LR: {optimizer.param_groups[0]['lr']:.8f}")
            
            # 保存最佳模型
            if val_loss < best_loss:
                best_loss = val_loss
                self.save_model(f'best_correction_model.pth', epoch)
                print(f"New best model saved! Loss: {val_loss:.6f}")
            
            print("-" * 50)
    
    def validate(self, val_loader):
        """验证修正网络"""
        self.model.eval()
        total_loss = 0
        total_improvement = 0
        
        with torch.no_grad():
            for data_dict in val_loader:
                # 移动数据到设备
                for key, value in data_dict.items():
                    if isinstance(value, torch.Tensor):
                        data_dict[key] = value.to(self.device)
                
                # 前向传播
                corrected_pred, correction, base_pred = self.model(
                    data_dict['eobj'], data_dict['eref'], data_dict['mref'],
                    data_dict['dem'], data_dict['wv_obj'], 
                    data_dict['lw_up_obj'], data_dict['lw_down_obj'],
                    stage='correction'
                )
                
                # 计算损失
                _, main_loss, _, improvement = self.criterion(
                    corrected_pred, data_dict['mobj'], correction, base_pred
                )
                
                total_loss += main_loss.item()
                total_improvement += improvement.item()
        
        return total_loss / len(val_loader), total_improvement / len(val_loader)
    
    def save_model(self, filename, epoch):
        """保存模型"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'correction_model_state_dict': self.model.correction_model.state_dict(),
            'config': self.config
        }, filename)

if __name__ == '__main__':
    # 测试代码
    class MockConfig:
        def __init__(self):
            self.use_cafd = True
            self.use_cvfd = True
            self.use_twoway = True
            self.dims = 32
    
    config = MockConfig()
    
    # 创建两阶段模型
    model = TwoStageModel(config)
    
    # 测试前向传播
    batch_size = 2
    H, W = 128, 128
    
    E0 = torch.randn(batch_size, 1, H, W)
    E1 = torch.randn(batch_size, 1, H, W)
    M1 = torch.randn(batch_size, 1, H, W)
    dem = torch.randn(batch_size, 1, H, W)
    wv = torch.randn(batch_size, 1, H, W)
    lw_up = torch.randn(batch_size, 1, H, W)
    lw_down = torch.randn(batch_size, 1, H, W)
    
    with torch.no_grad():
        # 测试基础预测
        base_pred, _ = model(E0, E1, M1, dem, wv, lw_up, lw_down, stage='base_only')
        print(f"Base prediction shape: {base_pred.shape}")
        
        # 测试修正预测
        corrected_pred, correction, _ = model(E0, E1, M1, dem, wv, lw_up, lw_down, stage='correction')
        print(f"Corrected prediction shape: {corrected_pred.shape}")
        print(f"Correction shape: {correction.shape}")
        print(f"Correction range: [{correction.min():.3f}, {correction.max():.3f}]")
    
    print("Two-stage model test passed!")