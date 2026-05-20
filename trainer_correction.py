#!/usr/bin/env python3
"""
两阶段修正网络训练器
专门用于训练基于model.py的修正网络
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
import numpy as np
from timeit import default_timer as timer

from models.model_correction import TwoStageModel, CorrectionLoss
from utils.early_stopping import EarlyStopping

class CorrectionTrainer:
    """修正网络专用训练器"""
    
    def __init__(self, model, train_loader, val_loader, device, config, logger):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.logger = logger

        self.criterion = CorrectionLoss(config)

        self.early_stopping = EarlyStopping(
            patience=getattr(config, 'correction_patience', 15),
            min_delta=getattr(config, 'min_delta', 1e-6)
        )

        self.best_val_loss = float('inf')
        self.best_improvement = 0.0
        self.best_mae_improvement = 0.0
        self.best_correction_mae = float('inf')
        self.best_model_path = None

        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'improvement': [],
            'correction_magnitude': [],
            'base_mae': [],
            'corrected_mae': [],
            'mae_improvement': []
        }
    
    def setup_optimizer(self, stage='correction'):
        """设置不同阶段的优化器"""
        if stage == 'correction':
            # 只优化修正网络
            self.optimizer = optim.Adam(
                self.model.correction_model.parameters(),
                lr=getattr(self.config, 'correction_learning_rate', 0.001),
                weight_decay=getattr(self.config, 'weight_decay', 1e-4)
            )
            
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.8,
                patience=5,
                min_lr=1e-6,
                verbose=True
            )
            
        elif stage == 'joint':
            # 联合优化所有参数
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=getattr(self.config, 'joint_learning_rate', 5e-5),
                weight_decay=getattr(self.config, 'weight_decay', 1e-4)
            )
            
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.9,
                patience=3,
                min_lr=1e-7,
                verbose=True
            )
    
    def train_correction_stage(self, epochs=50):
        """训练修正网络阶段"""
        self.logger.log_info("=" * 60)
        self.logger.log_info("开始训练修正网络")
        self.logger.log_info("=" * 60)
        
        # 设置优化器
        self.setup_optimizer('correction')
        
        # 冻结基础模型
        self.model.freeze_base_model()
        self.logger.log_info("基础模型已冻结")
        
        # 记录修正网络参数数量
        correction_params = sum(p.numel() for p in self.model.correction_model.parameters() if p.requires_grad)
        self.logger.log_info(f"修正网络参数数量: {correction_params:,}")
        
        for epoch in range(epochs):
            start_time = timer()

            train_metrics = self.train_epoch_correction()

            val_metrics = self.validate_correction()

            self.scheduler.step(val_metrics['val_loss'])
            current_lr = self.optimizer.param_groups[0]['lr']

            self.train_history['train_loss'].append(train_metrics['train_loss'])
            self.train_history['val_loss'].append(val_metrics['val_loss'])
            self.train_history['improvement'].append(val_metrics['improvement'])
            self.train_history['correction_magnitude'].append(val_metrics['correction_magnitude'])
            self.train_history['base_mae'].append(val_metrics['base_mae'])
            self.train_history['corrected_mae'].append(val_metrics['corrected_mae'])
            self.train_history['mae_improvement'].append(val_metrics['mae_improvement'])

            epoch_time = timer() - start_time
            self.logger.log_info(f"Epoch [{epoch+1}/{epochs}] ({epoch_time:.1f}s)")
            self.logger.log_info(f"Train Loss: {train_metrics['train_loss']:.6f}")
            self.logger.log_info(f"Val Loss: {val_metrics['val_loss']:.6f}")
            self.logger.log_info(f"Base MAE: {val_metrics['base_mae']:.6f}")
            self.logger.log_info(f"Corrected MAE: {val_metrics['corrected_mae']:.6f}")
            self.logger.log_info(f"MAE Improvement: {val_metrics['mae_improvement']:.6f}")
            self.logger.log_info(f"Correction Mag: {val_metrics['correction_magnitude']:.3f}K")
            self.logger.log_info(f"Learning Rate: {current_lr:.8f}")

            self.logger.log_correction(
                epoch+1, 
                train_metrics['train_loss'], 
                val_metrics['val_loss'],
                val_metrics['base_mae'], 
                val_metrics['corrected_mae'], 
                val_metrics['mae_improvement'],
                val_metrics['correction_magnitude'], 
                current_lr
            )

            should_save_best = False
            save_reason = ""
            
            # 条件1: MAE必须有改进 (修正MAE < 基础MAE)
            if val_metrics['mae_improvement'] > 0:
                # 条件2: 在有MAE改进的前提下，选择验证损失最小的
                if val_metrics['val_loss'] < self.best_val_loss:
                    should_save_best = True
                    save_reason = f"验证损失改进: {self.best_val_loss:.6f} → {val_metrics['val_loss']:.6f}"
                # 或者MAE改进超过了之前的最佳改进
                elif val_metrics['mae_improvement'] > self.best_mae_improvement:
                    should_save_best = True
                    save_reason = f"MAE改进提升: {self.best_mae_improvement:.6f} → {val_metrics['mae_improvement']:.6f}"
            
            if should_save_best:
                self.best_val_loss = val_metrics['val_loss']
                self.best_improvement = val_metrics['improvement']
                self.best_mae_improvement = val_metrics['mae_improvement']
                self.best_correction_mae = val_metrics['corrected_mae']  # 记录修正模型的绝对MAE
                self.best_model_path = self.save_checkpoint(epoch, 'best_correction_model.pth', stage='correction')
                self.logger.log_info(f"   新的最佳修正模型! {save_reason}")
                self.logger.log_info(f"   MAE改进: {val_metrics['mae_improvement']:.6f}")
                self.logger.log_info(f"   修正模型绝对MAE: {val_metrics['corrected_mae']:.6f}")
            elif val_metrics['mae_improvement'] <= 0:
                self.logger.log_info(f"   修正网络未改善基础模型 (MAE改进: {val_metrics['mae_improvement']:.6f})")
            else:
                self.logger.log_info(f"   当前MAE改进: {val_metrics['mae_improvement']:.6f} (最佳: {self.best_mae_improvement:.6f})")

            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, f'correction_checkpoint_{epoch+1}.pth', stage='correction')

            self.early_stopping(val_metrics['val_loss'])
            if self.early_stopping.early_stop:
                self.logger.log_info(f"早停触发于第 {epoch+1} 轮")
                break
            
            self.logger.log_info("-" * 60)
        
        # 保存最终模型
        final_model_path = self.save_checkpoint(epochs-1, 'correction_model_final.pth', stage='correction')
        
        self.logger.log_info("修正网络训练完成!")
        self.logger.log_info(f"最佳修正模型: {self.best_model_path}")
        self.logger.log_info(f"最终修正模型: {final_model_path}")
        
        # 记录最终统计
        self.logger.log_info(f"最佳验证损失: {self.best_val_loss:.6f}")
        self.logger.log_info(f"最佳改进幅度: {self.best_improvement:.6f}")
        
        # 返回最佳模型路径用于第三阶段
        return self.best_model_path if self.best_model_path else final_model_path
    
    def train_joint_stage(self, epochs=20):
        """联合微调阶段"""
        self.logger.log_info("=" * 60)
        self.logger.log_info("开始联合微调")
        self.logger.log_info("=" * 60)

        self.setup_optimizer('joint')

        self.model.unfreeze_base_model()
        self.logger.log_info("基础模型已解冻")

        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.log_info(f"联合训练参数数量: {total_params:,}")
        
        # 重置早停和最佳模型跟踪
        self.early_stopping = EarlyStopping(patience=8, min_delta=1e-6)
        self.best_joint_model_path = None
        self.best_joint_mae_improvement = 0.0  # 跟踪最佳联合MAE改进
        
        # 记录修正阶段的最佳性能作为联合训练的基准
        # 注意：需要记录第二阶段的最佳绝对MAE，而不是相对基础模型的改进值
        self.correction_baseline_mae_improvement = self.best_mae_improvement if hasattr(self, 'best_mae_improvement') else 0.0
        self.correction_best_absolute_mae = getattr(self, 'best_correction_mae', float('inf'))
        self.logger.log_info(f"联合训练基准 - 修正模型最佳MAE改进: {self.correction_baseline_mae_improvement:.6f}")
        self.logger.log_info(f"联合训练基准 - 修正模型最佳绝对MAE: {self.correction_best_absolute_mae:.6f}")
        
        for epoch in range(epochs):
            start_time = timer()

            train_metrics = self.train_epoch_joint()

            val_metrics = self.validate_joint()

            self.scheduler.step(val_metrics['val_loss'])
            current_lr = self.optimizer.param_groups[0]['lr']

            epoch_time = timer() - start_time
            self.logger.log_info(f"Joint Epoch [{epoch+1}/{epochs}] ({epoch_time:.1f}s)")
            self.logger.log_info(f"Train Loss: {train_metrics['train_loss']:.6f}")
            self.logger.log_info(f"Val Loss: {val_metrics['val_loss']:.6f}")
            self.logger.log_info(f"Base MAE: {val_metrics['base_mae']:.6f}")
            self.logger.log_info(f"Joint MAE: {val_metrics['joint_mae']:.6f} (vs 修正基准: {self.correction_best_absolute_mae:.6f})")
            self.logger.log_info(f"MAE Improvement (vs Base): {val_metrics['mae_improvement']:.6f}")
            self.logger.log_info(f"Learning Rate: {current_lr:.8f}")

            self.logger.log_joint(
                epoch+1, 
                train_metrics['train_loss'], 
                val_metrics['val_loss'],
                val_metrics['base_mae'], 
                val_metrics['joint_mae'], 
                val_metrics['mae_improvement'], 
                current_lr
            )
            
            # 保存最佳联合模型 - 新的判断条件（与修正模型比较）
            should_save_best = False
            save_reason = ""
            
            # 修正：第三阶段应该与第二阶段的最佳绝对MAE比较，而不是与基础模型的改进比较
            # 条件1: 联合模型的绝对MAE必须小于修正模型的最佳绝对MAE
            if val_metrics['joint_mae'] < self.correction_best_absolute_mae:
                # 条件2: 在超过修正模型的前提下，选择验证损失最小的或MAE最小的
                if val_metrics['val_loss'] < self.best_val_loss:
                    should_save_best = True
                    save_reason = f"验证损失改进: {self.best_val_loss:.6f} → {val_metrics['val_loss']:.6f}"
                # 或者联合MAE比之前的最佳联合MAE更小
                elif not hasattr(self, 'best_joint_absolute_mae') or val_metrics['joint_mae'] < self.best_joint_absolute_mae:
                    should_save_best = True
                    save_reason = f"联合MAE改进: {getattr(self, 'best_joint_absolute_mae', 'N/A')} → {val_metrics['joint_mae']:.6f}"
            
            if should_save_best:
                self.best_val_loss = val_metrics['val_loss']
                self.best_joint_mae_improvement = val_metrics['mae_improvement']
                self.best_joint_absolute_mae = val_metrics['joint_mae']  # 记录最佳联合模型的绝对MAE
                self.best_joint_model_path = self.save_checkpoint(epoch, 'best_joint_model.pth', stage='joint')
                self.logger.log_info(f"新的最佳联合模型! {save_reason}")
                self.logger.log_info(f"联合MAE: {val_metrics['joint_mae']:.6f} (修正基准: {self.correction_best_absolute_mae:.6f})")
                self.logger.log_info(f"相对基础模型改进: {val_metrics['mae_improvement']:.6f}")
            elif val_metrics['joint_mae'] >= self.correction_best_absolute_mae:
                mae_vs_correction = val_metrics['joint_mae'] - self.correction_best_absolute_mae
                self.logger.log_info(f"   联合模型未超越修正模型 (联合MAE: {val_metrics['joint_mae']:.6f} vs 修正MAE: {self.correction_best_absolute_mae:.6f}, 差距: +{mae_vs_correction:.6f})")
            else:
                current_best_joint = getattr(self, 'best_joint_absolute_mae', float('inf'))
                self.logger.log_info(f"   当前联合MAE: {val_metrics['joint_mae']:.6f} (修正基准: {self.correction_best_absolute_mae:.6f}, 最佳联合: {current_best_joint:.6f})")
            
            # 早停检查
            self.early_stopping(val_metrics['val_loss'])
            if self.early_stopping.early_stop:
                self.logger.log_info(f"早停触发于第 {epoch+1} 轮")
                break
            
            self.logger.log_info("-" * 60)
        
        # 保存最终联合模型
        final_joint_model_path = self.save_checkpoint(epochs-1, 'joint_model_final.pth', stage='joint')
        
        self.logger.log_info("联合微调完成!")
        self.logger.log_info(f"最佳联合模型: {self.best_joint_model_path}")
        self.logger.log_info(f"最终联合模型: {final_joint_model_path}")
        self.logger.log_info(f"修正模型基准绝对MAE: {self.correction_best_absolute_mae:.6f}")
        best_joint_mae = getattr(self, 'best_joint_absolute_mae', float('inf'))
        self.logger.log_info(f"最佳联合绝对MAE: {best_joint_mae:.6f}")
        
        # 修正：评估联合训练效果（与修正模型的绝对MAE比较）
        if best_joint_mae < self.correction_best_absolute_mae:
            mae_improvement = self.correction_best_absolute_mae - best_joint_mae
            self.logger.log_info(f"   联合训练成功! 相对修正模型MAE改进: {mae_improvement:.6f}")
        else:
            mae_degradation = best_joint_mae - self.correction_best_absolute_mae
            self.logger.log_info(f"   联合训练未超越修正模型性能，MAE增加: {mae_degradation:.6f}")
        
        # 返回最佳联合模型路径
        return self.best_joint_model_path if self.best_joint_model_path else final_joint_model_path
    
    def train_epoch_correction(self):
        """修正网络训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_main_loss = 0.0
        total_reg_loss = 0.0
        
        for batch_idx, (mobj, mref, eobj, eref, dem, wv_obj, lw_up_obj, lw_down_obj) in enumerate(self.train_loader):

            mobj = mobj.to(self.device)
            mref = mref.to(self.device)
            eobj = eobj.to(self.device)
            eref = eref.to(self.device)
            dem = dem.to(self.device)
            wv_obj = wv_obj.to(self.device)
            lw_up_obj = lw_up_obj.to(self.device)
            lw_down_obj = lw_down_obj.to(self.device)
            
            self.optimizer.zero_grad()

            corrected_pred, correction, base_pred = self.model(
                eobj, eref, mref,
                dem, wv_obj,
                lw_up_obj, lw_down_obj,
                stage='correction'
            )

            loss, main_loss, reg_loss, _ = self.criterion(
                corrected_pred, mobj, correction, base_pred
            )

            loss.backward()
            
            # 梯度裁剪
            if hasattr(self.config, 'gradient_clip_norm'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.correction_model.parameters(),
                    self.config.gradient_clip_norm
                )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            total_main_loss += main_loss.item()
            total_reg_loss += reg_loss.item()
        
        return {
            'train_loss': total_loss / len(self.train_loader),
            'main_loss': total_main_loss / len(self.train_loader),
            'reg_loss': total_reg_loss / len(self.train_loader)
        }
    
    def train_epoch_joint(self):
        """联合训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (mobj, mref, eobj, eref, dem, wv_obj, lw_up_obj, lw_down_obj) in enumerate(self.train_loader):

            mobj = mobj.to(self.device)
            mref = mref.to(self.device)
            eobj = eobj.to(self.device)
            eref = eref.to(self.device)
            dem = dem.to(self.device)
            wv_obj = wv_obj.to(self.device)
            lw_up_obj = lw_up_obj.to(self.device)
            lw_down_obj = lw_down_obj.to(self.device)
            
            self.optimizer.zero_grad()

            if self.config.use_twoway:
                corrected_pred, M1_pre, base_pred, correction = self.model(
                    eobj, eref, mref,
                    dem, wv_obj,
                    lw_up_obj, lw_down_obj,
                    stage='joint'
                )
                
                # 主损失 + 双向损失
                loss = F.mse_loss(corrected_pred, mobj)
                if M1_pre is not None:
                    loss += F.mse_loss(M1_pre, mref)
            else:
                corrected_pred, _, base_pred, correction = self.model(
                    eobj, eref, mref,
                    dem, wv_obj,
                    lw_up_obj, lw_down_obj,
                    stage='joint'
                )
                
                loss = F.mse_loss(corrected_pred, mobj)
            
            # 添加修正量正则化
            reg_loss = torch.mean(torch.abs(correction))
            total_loss_with_reg = loss + 0.01 * reg_loss

            total_loss_with_reg.backward()

            if hasattr(self.config, 'gradient_clip_norm'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_norm
                )
            
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return {
            'train_loss': total_loss / len(self.train_loader)
        }
    
    def validate_correction(self):
        """验证修正网络"""
        self.model.eval()
        total_loss = 0.0
        total_improvement = 0.0
        total_correction_mag = 0.0
        total_base_mae = 0.0
        total_corrected_mae = 0.0
        
        with torch.no_grad():
            for (mobj, mref, eobj, eref, dem, wv_obj, lw_up_obj, lw_down_obj) in self.val_loader:

                mobj = mobj.to(self.device)
                mref = mref.to(self.device)
                eobj = eobj.to(self.device)
                eref = eref.to(self.device)
                dem = dem.to(self.device)
                wv_obj = wv_obj.to(self.device)
                lw_up_obj = lw_up_obj.to(self.device)
                lw_down_obj = lw_down_obj.to(self.device)
                
                target = mobj
 
                corrected_pred, correction, base_pred = self.model(
                    eobj, eref, mref,
                    dem, wv_obj,
                    lw_up_obj, lw_down_obj,
                    stage='correction'
                )

                _, main_loss, _, improvement = self.criterion(
                    corrected_pred, target, correction, base_pred
                )

                base_mae = torch.abs(base_pred - target).mean()
                corrected_mae = torch.abs(corrected_pred - target).mean()
                
                total_loss += main_loss.item()
                total_improvement += improvement.item()
                total_correction_mag += torch.abs(correction).mean().item()
                total_base_mae += base_mae.item()
                total_corrected_mae += corrected_mae.item()
        
        avg_base_mae = total_base_mae / len(self.val_loader)
        avg_corrected_mae = total_corrected_mae / len(self.val_loader)
        
        return {
            'val_loss': total_loss / len(self.val_loader),
            'improvement': total_improvement / len(self.val_loader),
            'correction_magnitude': total_correction_mag / len(self.val_loader),
            'base_mae': avg_base_mae,
            'corrected_mae': avg_corrected_mae,
            'mae_improvement': avg_base_mae - avg_corrected_mae  # 正值表示改进
        }
    
    def validate_joint(self):
        """验证联合模型"""
        self.model.eval()
        total_loss = 0.0
        total_base_mse = 0.0
        total_corrected_mse = 0.0
        total_base_mae = 0.0
        total_corrected_mae = 0.0
        
        with torch.no_grad():
            for (mobj, mref, eobj, eref, dem, wv_obj, lw_up_obj, lw_down_obj) in self.val_loader:

                mobj = mobj.to(self.device)
                mref = mref.to(self.device)
                eobj = eobj.to(self.device)
                eref = eref.to(self.device)
                dem = dem.to(self.device)
                wv_obj = wv_obj.to(self.device)
                lw_up_obj = lw_up_obj.to(self.device)
                lw_down_obj = lw_down_obj.to(self.device)
                
                target = mobj
                
                # 基础预测
                base_pred, _ = self.model(
                    eobj, eref, mref,
                    dem, wv_obj,
                    lw_up_obj, lw_down_obj,
                    stage='base_only'
                )
                
                # 联合预测 (修正预测)
                joint_pred, correction, _ = self.model(
                    eobj, eref, mref,
                    dem, wv_obj,
                    lw_up_obj, lw_down_obj,
                    stage='correction'
                )
                
                # 计算MSE指标
                loss = F.mse_loss(joint_pred, target)
                base_mse = F.mse_loss(base_pred, target)
                corrected_mse = F.mse_loss(joint_pred, target)
                
                # 计算MAE指标
                base_mae = torch.abs(base_pred - target).mean()
                joint_mae = torch.abs(joint_pred - target).mean()
                
                total_loss += loss.item()
                total_base_mse += base_mse.item()
                total_corrected_mse += corrected_mse.item()
                total_base_mae += base_mae.item()
                total_corrected_mae += joint_mae.item()
        
        avg_base_mae = total_base_mae / len(self.val_loader)
        avg_joint_mae = total_corrected_mae / len(self.val_loader)
        
        return {
            'val_loss': total_loss / len(self.val_loader),
            'base_mse': total_base_mse / len(self.val_loader),
            'corrected_mse': total_corrected_mse / len(self.val_loader),
            'base_mae': avg_base_mae,
            'joint_mae': avg_joint_mae,
            'mae_improvement': avg_base_mae - avg_joint_mae  # 正值表示联合模型比基础模型好
        }
    
    def save_checkpoint(self, epoch, filename, stage='correction'):
        """保存检查点"""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'stage': stage,
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss,
            'train_history': self.train_history
        }
        
        if stage == 'correction':
            checkpoint['correction_model_state_dict'] = self.model.correction_model.state_dict()
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        elif stage == 'joint':
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        
        save_path = checkpoint_dir / filename
        torch.save(checkpoint, save_path)
        
        if 'best' in filename:
            self.logger.log_info(f"最佳修正模型已保存: {save_path}")
        else:
            self.logger.log_info(f"检查点已保存: {save_path}")
        
        return str(save_path)
    
    def load_checkpoint(self, checkpoint_path, stage='correction'):
        """加载检查点"""
        self.logger.log_info(f"加载检查点: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if stage == 'correction' and 'correction_model_state_dict' in checkpoint:
            self.model.correction_model.load_state_dict(checkpoint['correction_model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint and hasattr(self, 'optimizer'):
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'train_history' in checkpoint:
            self.train_history = checkpoint['train_history']
        
        if 'best_val_loss' in checkpoint:
            self.best_val_loss = checkpoint['best_val_loss']
        
        start_epoch = checkpoint.get('epoch', 0) + 1
        self.logger.log_info(f"检查点加载完成，从第 {start_epoch} 轮开始")

        return start_epoch

    def get_training_summary(self):
        """获取训练摘要"""
        if not self.train_history['train_loss']:
            return "没有训练历史数据"
        
        summary = {
            'total_epochs': len(self.train_history['train_loss']),
            'best_val_loss': self.best_val_loss,
            'best_improvement': self.best_improvement,
            'best_mae_improvement': self.best_mae_improvement,
            'final_train_loss': self.train_history['train_loss'][-1],
            'final_val_loss': self.train_history['val_loss'][-1],
            'final_base_mae': self.train_history['base_mae'][-1],
            'final_corrected_mae': self.train_history['corrected_mae'][-1],
            'final_mae_improvement': self.train_history['mae_improvement'][-1],
            'avg_improvement': np.mean(self.train_history['improvement']),
            'avg_correction_magnitude': np.mean(self.train_history['correction_magnitude']),
            'avg_mae_improvement': np.mean(self.train_history['mae_improvement']),
            'mae_improvement_std': np.std(self.train_history['mae_improvement'])
        }
        
        return summary

class BaseModelTrainer:
    """基础模型训练器 (使用model.py)"""
    
    def __init__(self, train_loader, val_loader, device, config, logger):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.logger = logger
        
        # 导入基础模型组件
        from models.model import Recont, Discriminator
        from trainer import Trainer
        
        # 创建模型
        self.generator = Recont(config).to(device)
        self.discriminator = Discriminator(input_channels=1).to(device)
        
        # 优化器
        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=config.learning_rate)
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=config.learning_rate)
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_g, mode='min', factor=config.lr_decay_factor,
            patience=config.lr_patience, min_lr=config.min_lr
        )
        
        # 创建基础训练器
        self.trainer = Trainer(
            self.generator, self.optimizer_g, self.scheduler,
            self.discriminator, self.optimizer_d,
            train_loader, val_loader, device, config, logger
        )
        
        # 最佳模型跟踪
        self.best_val_loss = float('inf')
        self.best_model_path = None
    
    def train(self, epochs):
        """训练基础模型"""
        self.logger.log_info("=" * 60)
        self.logger.log_info("开始训练基础模型 (model.py)")
        self.logger.log_info("=" * 60)
        
        # 记录模型参数
        g_params = sum(p.numel() for p in self.generator.parameters() if p.requires_grad)
        d_params = sum(p.numel() for p in self.discriminator.parameters() if p.requires_grad)
        self.logger.log_info(f"生成器参数: {g_params:,}")
        self.logger.log_info(f"判别器参数: {d_params:,}")
        self.logger.log_info(f"总参数: {g_params + d_params:,}")
        
        # 开始训练 - 使用修改后的训练循环
        self.train_with_best_model_tracking(epochs)
        
        # 保存最终模型
        final_model_path = self.save_base_model(is_final=True)
        
        self.logger.log_info("基础模型训练完成!")
        self.logger.log_info(f"最佳模型: {self.best_model_path}")
        self.logger.log_info(f"最终模型: {final_model_path}")
        
        # 返回最佳模型路径用于第二阶段
        return self.best_model_path if self.best_model_path else final_model_path
    
    def train_with_best_model_tracking(self, epochs):
        """带最佳模型跟踪的训练循环"""
        from utils.early_stopping import EarlyStopping
        
        early_stopping = EarlyStopping(patience=self.config.patience, min_delta=self.config.min_delta)
        
        for epoch in range(epochs):

            train_loss_G, train_loss_D, train_mse, train_mae = self.trainer.train_epoch()

            val_loss, val_mse, val_mae = self.trainer.validate()

            self.scheduler.step(val_loss)
            lr = self.optimizer_g.param_groups[0]['lr']

            self.logger.log_info(f"Epoch: [{epoch+1}/{epochs}]")
            self.logger.log_info(f"Train - Loss G: {train_loss_G:.4f}, Loss D: {train_loss_D:.4f}, MSE: {train_mse:.4f}, MAE: {train_mae:.4f}")
            self.logger.log_info(f"Val - Loss: {val_loss:.4f}, MSE: {val_mse:.4f}, MAE: {val_mae:.4f}")
            self.logger.log_info(f"Learning Rate: {lr:.6f}")

            self.logger.log_base(epoch+1, train_loss_G, train_loss_D, train_mse, train_mae, val_loss, val_mse, val_mae, lr)

            self.save_base_model_checkpoint(epoch, val_loss, is_best=False)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_path = self.save_base_model_checkpoint(epoch, val_loss, is_best=True)
                self.logger.log_info(f"新的最佳基础模型! 损失: {val_loss:.6f}")

            early_stopping(val_loss)
            if early_stopping.early_stop:
                self.logger.log_info(f"早停触发于第 {epoch + 1} 轮")
                break
            
            self.logger.log_info("-" * 60)
    
    def save_base_model_checkpoint(self, epoch, val_loss, is_best=False):
        """保存基础模型检查点"""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 确定保存路径
        if is_best:
            save_path = checkpoint_dir / 'base_model_best.pth'
        else:
            save_path = checkpoint_dir / f'base_model_checkpoint_{epoch+1}.pth'
        
        # 保存检查点
        checkpoint = {
            'epoch': epoch,
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'optimizer_G': self.optimizer_g.state_dict(),
            'optimizer_D': self.optimizer_d.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_val_loss': val_loss if is_best else self.best_val_loss,
            'config': self.config,
            'model_type': 'base_model'
        }
        
        torch.save(checkpoint, save_path)
        
        if is_best:
            self.logger.log_info(f"最佳基础模型已保存: {save_path}")
        
        return str(save_path)
    
    def save_base_model(self, is_final=False):
        """保存基础模型 (最终版本)"""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = checkpoint_dir / 'base_model_final.pth'
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'model_type': 'base_model'
        }, save_path)
        
        self.logger.log_info(f"最终基础模型已保存: {save_path}")
        return str(save_path)