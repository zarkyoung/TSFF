import csv
from pathlib import Path
import time
from datetime import datetime

class Logger:
    def __init__(self, config):
        self.config = config
        self.log_dir = config.log_dir
        self.csv_path = Path(config.log_dir) / f'{self.config.experiment_name}_training_log.csv'
        self.best_mse = float('inf')
        self.patience = config.patience
        self.consecutive_no_improvement = 0
        
        # 阶段时间记录
        self.stage_times = {}
        self.training_start_time = None

        if not self.csv_path.exists():
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Epoch', 'Stage', 'Train_loss_G', 'Train_loss_D', 'Train_MSE', 'Train_MAE', 'Val_loss', 'Val_MSE', 'Val_MAE', 'Base_MAE', 'Corrected_MAE', 'MAE_Improvement', 'Correction_Magnitude', 'LR'])

    def log(self, epoch, stage, train_loss_G, train_loss_D, train_mse, train_mae, val_loss, val_mse, val_mae, base_mae=None, corrected_mae=None, mae_improvement=None, correction_magnitude=None, lr=None):
        """统一的CSV记录方法"""
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, stage, 
                train_loss_G or 0, train_loss_D or 0, 
                train_mse or 0, train_mae or 0, 
                val_loss or 0, val_mse or 0, val_mae or 0,
                base_mae or '', corrected_mae or '', 
                mae_improvement or '', correction_magnitude or '',
                lr or 0
            ])
    
    def log_base(self, epoch, train_loss_G, train_loss_D, train_mse, train_mae, val_loss, val_mse, val_mae, lr):
        """记录基础模型训练数据"""
        self.log(epoch, 'base', train_loss_G, train_loss_D, train_mse, train_mae, val_loss, val_mse, val_mae, lr=lr)
    
    def log_correction(self, epoch, train_loss, val_loss, base_mae, corrected_mae, mae_improvement, correction_magnitude, lr):
        """记录修正网络训练数据"""
        self.log(epoch, 'correction', train_loss, 0, 0, 0, val_loss, 0, 0, base_mae, corrected_mae, mae_improvement, correction_magnitude, lr)
    
    def log_joint(self, epoch, train_loss, val_loss, base_mae, joint_mae, mae_improvement, lr):
        """记录联合训练数据"""
        self.log(epoch, 'joint', train_loss, 0, 0, 0, val_loss, 0, 0, base_mae, joint_mae, mae_improvement, 0, lr)

    def log_info(self, message):
        print(message)
        with open(Path(self.log_dir) / f'{self.config.experiment_name}_training_info.log', 'a') as f:
            f.write(f"{message}\n")
    
    def start_training(self):
        """开始整个训练过程的计时"""
        self.training_start_time = time.time()
        start_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.log_info(f"   开始两阶段BMSTF-GAN训练: {start_time_str}")
        self.log_info("=" * 80)
    
    def start_stage(self, stage_name, total_epochs):
        """开始某个训练阶段的计时"""
        start_time = time.time()
        start_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        self.stage_times[stage_name] = {
            'start_time': start_time,
            'total_epochs': total_epochs,
            'start_time_str': start_time_str
        }
        
        self.log_info(f"   开始{stage_name}训练阶段")
        self.log_info(f"   开始时间: {start_time_str}")
        self.log_info(f"   预计轮次: {total_epochs}")
        self.log_info("-" * 60)
    
    def end_stage(self, stage_name, completed_epochs):
        """结束某个训练阶段的计时"""
        if stage_name not in self.stage_times:
            self.log_info(f"   警告: 阶段 {stage_name} 未找到开始时间记录")
            return
        
        end_time = time.time()
        end_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        stage_data = self.stage_times[stage_name]
        duration_seconds = end_time - stage_data['start_time']
        duration_minutes = duration_seconds / 60
        duration_hours = duration_seconds / 3600
        
        # 计算平均每轮时间
        avg_epoch_time = duration_seconds / completed_epochs if completed_epochs > 0 else 0
        avg_epoch_minutes = avg_epoch_time / 60
        
        # 更新阶段数据
        stage_data.update({
            'end_time': end_time,
            'end_time_str': end_time_str,
            'duration_seconds': duration_seconds,
            'duration_minutes': duration_minutes,
            'duration_hours': duration_hours,
            'completed_epochs': completed_epochs,
            'avg_epoch_seconds': avg_epoch_time,
            'avg_epoch_minutes': avg_epoch_minutes
        })
        
        # 记录到日志
        self.log_info("-" * 60)
        self.log_info(f"   {stage_name}训练阶段完成")
        self.log_info(f"   开始时间: {stage_data['start_time_str']}")
        self.log_info(f"   结束时间: {end_time_str}")
        self.log_info(f"   完成轮次: {completed_epochs}/{stage_data['total_epochs']}")
        self.log_info(f"   总用时: {duration_hours:.2f}小时 ({duration_minutes:.1f}分钟)")
        self.log_info(f"   平均每轮: {avg_epoch_minutes:.2f}分钟 ({avg_epoch_time:.1f}秒)")
        self.log_info("=" * 60)
    
    def end_training(self):
        """结束整个训练过程的计时"""
        if self.training_start_time is None:
            self.log_info("   警告: 未找到训练开始时间记录")
            return
        
        end_time = time.time()
        end_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        total_duration = end_time - self.training_start_time
        total_hours = total_duration / 3600
        total_minutes = total_duration / 60
        
        # 计算各阶段统计
        total_epochs = sum(stage.get('completed_epochs', 0) for stage in self.stage_times.values())
        
        self.log_info("   两阶段BMSTF-GAN训练完成!")
        self.log_info(f"   结束时间: {end_time_str}")
        self.log_info(f"   总训练时间: {total_hours:.2f}小时 ({total_minutes:.1f}分钟)")
        self.log_info(f"   总训练轮次: {total_epochs}")
        self.log_info("")
        
        # 详细阶段统计
        self.log_info("   各阶段训练时间统计:")
        self.log_info("-" * 80)
        
        for stage_name, stage_data in self.stage_times.items():
            if 'duration_hours' in stage_data:
                percentage = (stage_data['duration_seconds'] / total_duration) * 100
                self.log_info(f"   {stage_name}:")
                self.log_info(f"     轮次: {stage_data['completed_epochs']}/{stage_data['total_epochs']}")
                self.log_info(f"     用时: {stage_data['duration_hours']:.2f}小时 ({percentage:.1f}%)")
                self.log_info(f"     平均: {stage_data['avg_epoch_minutes']:.2f}分钟/轮")
                self.log_info("")
        
        # 效率分析
        if len(self.stage_times) >= 2:
            self.log_info("⚡ 训练效率分析:")
            stage_names = list(self.stage_times.keys())
            for i, stage_name in enumerate(stage_names):
                stage_data = self.stage_times[stage_name]
                if 'avg_epoch_minutes' in stage_data:
                    self.log_info(f"   {stage_name}: {stage_data['avg_epoch_minutes']:.2f}分钟/轮")
            
            # 比较基础模型和修正网络效率
            if '基础模型' in self.stage_times and '修正网络' in self.stage_times:
                base_avg = self.stage_times['基础模型'].get('avg_epoch_minutes', 0)
                corr_avg = self.stage_times['修正网络'].get('avg_epoch_minutes', 0)
                if base_avg > 0 and corr_avg > 0:
                    speedup = base_avg / corr_avg
                    self.log_info(f"   修正网络比基础模型快 {speedup:.1f}倍")
        
        self.log_info("=" * 80)
    
    def get_stage_summary(self):
        """获取阶段时间摘要"""
        summary = {}
        for stage_name, stage_data in self.stage_times.items():
            if 'duration_hours' in stage_data:
                summary[stage_name] = {
                    'duration_hours': stage_data['duration_hours'],
                    'completed_epochs': stage_data['completed_epochs'],
                    'avg_epoch_minutes': stage_data['avg_epoch_minutes']
                }
        return summary