#!/usr/bin/env python3
"""
两阶段BMSTF-GAN主训练脚本
集成基础模型训练和修正网络训练的完整流程
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import yaml
from pathlib import Path
import argparse
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.model_correction import TwoStageModel
from trainer_correction import CorrectionTrainer, BaseModelTrainer
from utils.logger import Logger
from utils.config import Config


def setup_data_loaders(config):
    """设置数据加载器（直接从 .npy 文件加载，类似 main.py 的风格）"""
    print("Setting up data loaders...")

    # ---- 加载训练数据 ----
    train_mobj = np.load(f'{config.dataset_path}/train_mobj.npy')
    train_mref = np.load(f'{config.dataset_path}/train_mref.npy')
    train_eobj = np.load(f'{config.dataset_path}/train_eobj.npy')
    train_eref = np.load(f'{config.dataset_path}/train_eref.npy')
    train_dem = np.load(f'{config.dataset_path}/train_dem.npy')
    train_wv_obj = np.load(f'{config.dataset_path}/train_wv_obj.npy')
    train_lw_up_obj = np.load(f'{config.dataset_path}/train_lw_up_obj.npy')
    train_lw_down_obj = np.load(f'{config.dataset_path}/train_lw_down_obj.npy')

    # ---- 加载验证数据 ----
    val_mobj = np.load(f'{config.dataset_path}/val_mobj.npy')
    val_mref = np.load(f'{config.dataset_path}/val_mref.npy')
    val_eobj = np.load(f'{config.dataset_path}/val_eobj.npy')
    val_eref = np.load(f'{config.dataset_path}/val_eref.npy')
    val_dem = np.load(f'{config.dataset_path}/val_dem.npy')
    val_wv_obj = np.load(f'{config.dataset_path}/val_wv_obj.npy')
    val_lw_up_obj = np.load(f'{config.dataset_path}/val_lw_up_obj.npy')
    val_lw_down_obj = np.load(f'{config.dataset_path}/val_lw_down_obj.npy')

    # ---- 转为 Tensor 并补充通道维 (N, H, W) -> (N, 1, H, W) ----
    def to_tensor(arr):
        return torch.from_numpy(arr.astype(np.float32)).unsqueeze(1)

    train_mobj = to_tensor(train_mobj)
    train_mref = to_tensor(train_mref)
    train_eobj = to_tensor(train_eobj)
    train_eref = to_tensor(train_eref)
    train_dem = to_tensor(train_dem)
    train_wv_obj = to_tensor(train_wv_obj)
    train_lw_up_obj = to_tensor(train_lw_up_obj)
    train_lw_down_obj = to_tensor(train_lw_down_obj)

    val_mobj = to_tensor(val_mobj)
    val_mref = to_tensor(val_mref)
    val_eobj = to_tensor(val_eobj)
    val_eref = to_tensor(val_eref)
    val_dem = to_tensor(val_dem)
    val_wv_obj = to_tensor(val_wv_obj)
    val_lw_up_obj = to_tensor(val_lw_up_obj)
    val_lw_down_obj = to_tensor(val_lw_down_obj)

    # ---- 构建 TensorDataset 与 DataLoader ----
    # 元组顺序须与 trainer_correction.py 中解包顺序一致：
    # (mobj, mref, eobj, eref, dem, wv_obj, lw_up_obj, lw_down_obj)
    train_dataset = TensorDataset(
        train_mobj, train_mref, train_eobj, train_eref,
        train_dem, train_wv_obj, train_lw_up_obj, train_lw_down_obj,
    )
    val_dataset = TensorDataset(
        val_mobj, val_mref, val_eobj, val_eref,
        val_dem, val_wv_obj, val_lw_up_obj, val_lw_down_obj,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        pin_memory=torch.cuda.is_available(),
        num_workers=config.num_workers,
    )

    print(
        f"Train samples: {len(train_dataset)} ({len(train_loader)} batches),"
        f" Val samples: {len(val_dataset)} ({len(val_loader)} batches)"
    )

    return train_loader, val_loader

def train_base_stage(config, device, logger, train_loader, val_loader):
    """第一阶段：训练基础模型"""
    logger.log_info("   开始第一阶段：基础模型训练")
    
    # 创建基础模型训练器
    base_trainer = BaseModelTrainer(train_loader, val_loader, device, config, logger)
    
    # 训练基础模型 - 返回最佳模型路径
    base_epochs = getattr(config, 'base_epochs', 100)
    best_base_model_path = base_trainer.train(base_epochs)
    
    logger.log_info(f"   基础模型训练完成，最佳模型: {best_base_model_path}")
    
    # 返回最佳基础模型路径用于第二阶段
    return best_base_model_path

def train_correction_stage(config, device, logger, train_loader, val_loader, base_model_path):
    """第二阶段：训练修正网络"""
    logger.log_info("   开始第二阶段：修正网络训练")
    
    # 创建两阶段模型
    model = TwoStageModel(config, base_model_path).to(device)
    
    # 创建修正训练器
    correction_trainer = CorrectionTrainer(model, train_loader, val_loader, device, config, logger)
    
    # 训练修正网络 - 返回最佳模型路径
    correction_epochs = getattr(config, 'correction_epochs', 50)
    best_correction_model_path = correction_trainer.train_correction_stage(correction_epochs)

    # 打印训练摘要
    summary = correction_trainer.get_training_summary()
    logger.log_info("修正网络训练摘要:")
    for key, value in summary.items():
        logger.log_info(f"  {key}: {value}")
    
    logger.log_info(f"   修正网络训练完成，最佳模型: {best_correction_model_path}")
    
    # 返回训练器和最佳模型路径
    return correction_trainer, best_correction_model_path

def train_joint_stage(config, device, logger, train_loader, val_loader, best_base_model_path, best_correction_model_path):
    """第三阶段：联合微调"""
    logger.log_info("   开始第三阶段：联合微调")
    logger.log_info(f"使用最佳基础模型: {best_base_model_path}")
    logger.log_info(f"使用最佳修正模型: {best_correction_model_path}")
    
    # 创建新的两阶段模型，加载最佳基础模型
    model = TwoStageModel(config, best_base_model_path).to(device)
    
    # 创建联合训练器
    joint_trainer = CorrectionTrainer(model, train_loader, val_loader, device, config, logger)
    
    # 加载最佳修正模型的状态
    joint_trainer.load_checkpoint(best_correction_model_path, stage='correction')
    
    # 联合微调
    joint_epochs = getattr(config, 'joint_epochs', 20)
    best_joint_model_path = joint_trainer.train_joint_stage(joint_epochs)
    
    logger.log_info("联合微调完成!")
    logger.log_info(f"最佳联合模型: {best_joint_model_path}")
    
    return best_joint_model_path

def evaluate_models(config, device, logger, val_loader):
    """评估不同阶段的模型性能"""
    logger.log_info("   开始模型性能评估")
    
    checkpoint_dir = Path(config.checkpoint_dir)
    
    # 模型路径 - 优先使用最佳模型
    models_to_evaluate = {
        'Base Model (Best)': checkpoint_dir / 'base_model_best.pth',
        'Base Model (Final)': checkpoint_dir / 'base_model_final.pth',
        'Correction Model': checkpoint_dir / 'best_correction_model.pth',
        'Joint Model': checkpoint_dir / 'best_joint_model.pth'
    }
    
    results = {}
    
    for model_name, model_path in models_to_evaluate.items():
        if not model_path.exists():
            logger.log_info(f"   {model_name} 不存在: {model_path}")
            continue
        
        logger.log_info(f"评估 {model_name}...")
        
        try:
            if model_name == 'Base Model':
                # 评估基础模型
                mse = evaluate_base_model(model_path, val_loader, device, config)
                results[model_name] = {'MSE': mse}
            else:
                # 评估两阶段模型
                metrics = evaluate_two_stage_model(model_path, val_loader, device, config)
                results[model_name] = metrics
            
            logger.log_info(f"{model_name} 评估完成")
            
        except Exception as e:
            logger.log_info(f"   {model_name} 评估失败: {str(e)}")
    
    # 打印对比结果
    logger.log_info("=" * 60)
    logger.log_info("模型性能对比")
    logger.log_info("=" * 60)
    
    for model_name, metrics in results.items():
        logger.log_info(f"{model_name}:")
        for metric_name, value in metrics.items():
            logger.log_info(f"  {metric_name}: {value:.6f}")
        logger.log_info("-" * 40)
    
    return results

def evaluate_base_model(model_path, val_loader, device, config):
    """评估基础模型"""
    from models.model import Recont
    
    # 加载基础模型
    model = Recont(config).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['generator_state_dict'])
    model.eval()
    
    total_mse = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for data_dict in val_loader:
            # 移动数据到设备
            for key, value in data_dict.items():
                if isinstance(value, torch.Tensor):
                    data_dict[key] = value.to(device)
            
            # 预测
            if config.use_twoway:
                pred, _ = model(data_dict['eobj'], data_dict['eref'], data_dict['mref'])
            else:
                pred = model(data_dict['eobj'], data_dict['eref'], data_dict['mref'])
            
            # 计算MSE
            mse = torch.nn.functional.mse_loss(pred, data_dict['mobj'])
            total_mse += mse.item() * pred.size(0)
            total_samples += pred.size(0)
    
    return total_mse / total_samples

def evaluate_two_stage_model(model_path, val_loader, device, config):
    """评估两阶段模型"""
    # 加载两阶段模型
    model = TwoStageModel(config).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    total_base_mse = 0.0
    total_corrected_mse = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for data_dict in val_loader:
            # 移动数据到设备
            for key, value in data_dict.items():
                if isinstance(value, torch.Tensor):
                    data_dict[key] = value.to(device)
            
            target = data_dict['mobj']
            
            # 基础预测
            base_pred, _ = model(
                data_dict['eobj'], data_dict['eref'], data_dict['mref'],
                data_dict['dem'], data_dict['wv_obj'],
                data_dict['lw_up_obj'], data_dict['lw_down_obj'],
                stage='base_only'
            )
            
            # 修正预测
            corrected_pred, _, _ = model(
                data_dict['eobj'], data_dict['eref'], data_dict['mref'],
                data_dict['dem'], data_dict['wv_obj'],
                data_dict['lw_up_obj'], data_dict['lw_down_obj'],
                stage='correction'
            )
            
            # 计算MSE
            base_mse = torch.nn.functional.mse_loss(base_pred, target)
            corrected_mse = torch.nn.functional.mse_loss(corrected_pred, target)
            
            total_base_mse += base_mse.item() * target.size(0)
            total_corrected_mse += corrected_mse.item() * target.size(0)
            total_samples += target.size(0)
    
    avg_base_mse = total_base_mse / total_samples
    avg_corrected_mse = total_corrected_mse / total_samples
    improvement = avg_base_mse - avg_corrected_mse
    improvement_rate = (improvement / avg_base_mse) * 100
    
    return {
        'Base MSE': avg_base_mse,
        'Corrected MSE': avg_corrected_mse,
        'Improvement': improvement,
        'Improvement Rate (%)': improvement_rate
    }

def main():
    parser = argparse.ArgumentParser(description='Two-Stage BMSTF-GAN Training')
    parser.add_argument('--config', type=str, default='configs/config_two_stage.yaml',
                       help='Path to config file')
    parser.add_argument('--stage', type=str, 
                       choices=['base', 'correction', 'joint', 'all', 'evaluate'],
                       default='all', help='Training stage to run')
    parser.add_argument('--base_model', type=str, default=None,
                       help='Path to pretrained base model (for correction/joint stages)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--skip_base', action='store_true',
                       help='Skip base model training (use existing base model)')
    
    args = parser.parse_args()
    
    # 设置设备
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        print(f"Using GPU: {device}")
        print(f"GPU Name: {torch.cuda.get_device_name(args.gpu)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(args.gpu).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # 加载配置
    print(f"Loading config from: {args.config}")
    config = Config(args.config)

    # 设置日志
    logger = Logger(config)
    
    logger.log_info("=" * 80)
    logger.log_info("两阶段 BMSTF-GAN 训练开始")
    logger.log_info("=" * 80)
    logger.log_info(f"实验名称: {getattr(config, 'experiment_name', 'two_stage_correction')}")
    logger.log_info(f"训练阶段: {args.stage}")
    logger.log_info(f"设备: {device}")
    logger.log_info(f"配置文件: {args.config}")
    
    # 记录关键配置
    logger.log_info("关键配置:")
    logger.log_info(f"  - Patch size: {config.patch_size}")
    logger.log_info(f"  - Batch size: {config.batch_size}")
    logger.log_info(f"  - Learning rate: {config.learning_rate}")
    logger.log_info(f"  - Base epochs: {getattr(config, 'base_epochs', 100)}")
    logger.log_info(f"  - Correction epochs: {getattr(config, 'correction_epochs', 50)}")
    logger.log_info(f"  - Joint epochs: {getattr(config, 'joint_epochs', 20)}")
    logger.log_info(f"  - Use two-way: {config.use_twoway}")
    
    try:
        # 开始训练计时
        logger.start_training()
        
        # 设置数据加载器
        if args.stage != 'evaluate':
            train_loader, val_loader = setup_data_loaders(config)
        else:
            _, val_loader = setup_data_loaders(config)
            train_loader = None
        
        best_base_model_path = None
        best_correction_model_path = None
        best_joint_model_path = None
        
        # 执行训练阶段
        if args.stage in ['base', 'all']:
            if not args.skip_base:
                # 第一阶段：基础模型训练
                logger.start_stage("基础模型", getattr(config, 'base_epochs', 100))
                best_base_model_path = train_base_stage(config, device, logger, train_loader, val_loader)
                logger.end_stage("基础模型", getattr(config, 'base_epochs', 100))
            else:
                # 优先使用最佳基础模型，否则使用最终模型
                best_path = Path(config.checkpoint_dir) / 'base_model_best.pth'
                final_path = Path(config.checkpoint_dir) / 'base_model_final.pth'
                
                if args.base_model:
                    best_base_model_path = args.base_model
                elif best_path.exists():
                    best_base_model_path = str(best_path)
                else:
                    best_base_model_path = str(final_path)
                    
                logger.log_info(f"跳过基础模型训练，使用现有模型: {best_base_model_path}")
        
        if args.stage in ['correction', 'all']:
            # 确定基础模型路径
            if best_base_model_path is None:
                best_path = Path(config.checkpoint_dir) / 'base_model_best.pth'
                final_path = Path(config.checkpoint_dir) / 'base_model_final.pth'
                
                if args.base_model:
                    best_base_model_path = args.base_model
                elif best_path.exists():
                    best_base_model_path = str(best_path)
                else:
                    best_base_model_path = str(final_path)
            
            if not Path(best_base_model_path).exists():
                raise FileNotFoundError(f"基础模型不存在: {best_base_model_path}")
            
            # 第二阶段：修正网络训练
            logger.start_stage("修正网络", getattr(config, 'correction_epochs', 50))
            correction_trainer, best_correction_model_path = train_correction_stage(
                config, device, logger, train_loader, val_loader, best_base_model_path
            )
            logger.end_stage("修正网络", getattr(config, 'correction_epochs', 50))
        
        if args.stage in ['joint', 'all']:
            # 确定最佳模型路径
            if best_base_model_path is None:
                best_path = Path(config.checkpoint_dir) / 'base_model_best.pth'
                final_path = Path(config.checkpoint_dir) / 'base_model_final.pth'
                
                if best_path.exists():
                    best_base_model_path = str(best_path)
                else:
                    best_base_model_path = str(final_path)
            
            if best_correction_model_path is None:
                best_correction_model_path = str(Path(config.checkpoint_dir) / 'best_correction_model.pth')
            
            if not Path(best_correction_model_path).exists():
                raise FileNotFoundError(f"修正模型不存在: {best_correction_model_path}")
            
            # 第三阶段：联合微调
            logger.start_stage("联合微调", getattr(config, 'joint_epochs', 20))
            best_joint_model_path = train_joint_stage(
                config, device, logger, train_loader, val_loader, 
                best_base_model_path, best_correction_model_path
            )
            logger.end_stage("联合微调", getattr(config, 'joint_epochs', 20))
        
        if args.stage in ['evaluate', 'all']:
            # 模型评估
            evaluate_models(config, device, logger, val_loader)
        
        # 结束训练计时
        logger.end_training()
        
        logger.log_info("所有训练阶段完成!")
        
        # 最终总结
        logger.log_info("=" * 80)
        logger.log_info("训练完成总结")
        logger.log_info("=" * 80)
        
        checkpoint_dir = Path(config.checkpoint_dir)
        available_models = []
        
        if (checkpoint_dir / 'base_model_final.pth').exists():
            available_models.append("基础模型 (base_model_final.pth)")
        if (checkpoint_dir / 'best_correction_model.pth').exists():
            available_models.append("修正模型 (best_correction_model.pth)")
        if (checkpoint_dir / 'best_joint_model.pth').exists():
            available_models.append("联合模型 (best_joint_model.pth)")
        
        logger.log_info("可用模型:")
        for model in available_models:
            logger.log_info(f"     {model}")
        
        logger.log_info(f"检查点目录: {config.checkpoint_dir}")
        logger.log_info(f"日志目录: {config.log_dir}")
        
        # 推荐下一步操作
        logger.log_info("建议下一步操作:")
        logger.log_info("1. 运行分析脚本: python analyze_two_stage.py")
        logger.log_info("2. 在测试集上评估: python main_correction.py --stage evaluate")
        logger.log_info("3. 进行推理预测: python inference_with_best_model.py")
        
    except KeyboardInterrupt:
        logger.log_info("训练被用户中断")
        
    except Exception as e:
        logger.log_info(f"训练失败: {str(e)}")
        import traceback
        logger.log_info("错误详情:")
        logger.log_info(traceback.format_exc())
        raise
        
    finally:
        logger.log_info("=" * 80)
        logger.log_info("训练会话结束")
        logger.log_info("=" * 80)

if __name__ == '__main__':
    main()