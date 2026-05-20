import torch
import torch.nn.functional as F
from pathlib import Path
from models.model import ReconstructionLoss
from utils.early_stopping import EarlyStopping
from timeit import default_timer as timer

class Trainer:
    def __init__(self, generator, optimizer_g, scheduler, discriminator, optimizer_d,
                          train_loader, val_loader, device, config, logger):
        self.generator = generator
        self.discriminator = discriminator
        self.optimizer_G = optimizer_g
        self.optimizer_D = optimizer_d
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.mse_loss = torch.nn.MSELoss()
        self.device = device
        self.config = config
        self.logger = logger
        self.best_val_loss = float('inf')
        self.early_stopping = EarlyStopping(patience=config.patience, min_delta=config.min_delta)
        self.criterion = ReconstructionLoss(self.config)

    def train(self, start_epoch):
        
        for epoch in range(start_epoch, self.config.num_epochs):
            train_loss_G, train_loss_D, train_mse, train_mae = self.train_epoch()
            val_loss, val_mse, val_mae = self.validate()
            
            self.scheduler.step(val_loss)

            lr = self.optimizer_G.param_groups[0]['lr']

            self.logger.log(epoch+1, train_loss_G, train_loss_D, train_mse, train_mae, val_loss, val_mse, val_mae, lr)

            self.logger.log_info(f"Epoch: [{epoch+1}/{self.config.num_epochs}]")
            self.logger.log_info(f"Train - Loss G: {train_loss_G:.4f}, Loss D: {train_loss_D:.4f}, MSE: {train_mse:.4f}, MAE: {train_mae:.4f}")
            self.logger.log_info(f"Val - Loss: {val_loss:.4f}, MSE: {val_mse:.4f}, MAE: {val_mae:.4f}")
            self.logger.log_info(f"Learning Rate: {lr:.6f}")

            self.save_checkpoint(epoch, is_best=False)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)

            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                self.logger.log_info(f"Early stopping triggered at epoch {epoch + 1}")
                break

    def train_epoch(self):
        self.generator.train()
        self.discriminator.train()
        total_loss_G, total_loss_D, total_mse, total_mae = 0, 0, 0, 0

        for i, (mobj, mref, eobj, eref, _, _, _, _) in enumerate(self.train_loader):
            t_start = timer()
            mobj = mobj.to(self.device)
            mref = mref.to(self.device)
            eobj = eobj.to(self.device)
            eref = eref.to(self.device)

            # Train Discriminator
            self.optimizer_D.zero_grad()
            
            if self.config.use_twoway:
                prediction, M1_pre = self.generator(eobj, eref, mref)
            else:
                prediction = self.generator(eobj, eref, mref)

            d_real = self.discriminator(mobj)
            d_fake = self.discriminator(prediction.detach())

            loss_D_real = self.mse_loss(d_real, torch.ones_like(d_real))
            loss_D_fake = self.mse_loss(d_fake, torch.zeros_like(d_fake))
            loss_D = (loss_D_real + loss_D_fake) / 2

            loss_D.backward()
            self.optimizer_D.step()

            # Train Generator
            self.optimizer_G.zero_grad()
            
            if self.config.use_twoway:
                prediction, M1_pre = self.generator(eobj, eref, mref)
            else:
                prediction = self.generator(eobj, eref, mref)

            d_fake = self.discriminator(prediction)
            
            loss_G_GAN = self.mse_loss(d_fake, torch.ones_like(d_fake))
            loss_G_L1 = self.criterion(prediction, mobj, M1_pre=M1_pre if self.config.use_twoway else None, M1=mref if self.config.use_twoway else None)
            loss_G = self.config.lambda_gan * loss_G_GAN + loss_G_L1

            loss_G.backward()
            self.optimizer_G.step()

            mse = F.mse_loss(prediction.detach(), mobj)
            mae = F.l1_loss(prediction.detach(), mobj)

            total_loss_G += loss_G.item()
            total_loss_D += loss_D.item()
            total_mse += mse.item()
            total_mae += mae.item()

            t_end = timer()
            print(f"Batch {i}/{len(self.train_loader)} - Loss_G: {loss_G.item():.4f}, Loss_D: {loss_D.item():.4f}, MSE: {mse.item():.4f}, MAE: {mae.item():.4f}, Time: {t_end - t_start}s")

        return (total_loss_G / len(self.train_loader), 
                total_loss_D / len(self.train_loader), 
                total_mse / len(self.train_loader), 
                total_mae / len(self.train_loader))

    def validate(self):
        self.generator.eval()
        total_loss, total_mse, total_mae = 0, 0, 0

        with torch.no_grad():
            for (mobj, mref, eobj, eref, _, _, _, _) in self.val_loader:
                mobj = mobj.to(self.device)
                mref = mref.to(self.device)
                eobj = eobj.to(self.device)
                eref = eref.to(self.device)

                if self.config.use_twoway:
                    prediction, M1_pre = self.generator(eobj, eref, mref)
                    loss = self.criterion(prediction, mobj, M1_pre=M1_pre, M1=mref)
                    mse = F.mse_loss(prediction, mobj).item()
                    mae = F.l1_loss(prediction, mobj)
                    total_loss += loss.item()
                    total_mse += mse
                    total_mae += mae.item()
                else:
                    prediction = self.generator(eobj, eref, mref)
                    loss = self.criterion(prediction, mobj, M1_pre=None, M1=None)
                    mse = F.mse_loss(prediction, mobj).item()
                    mae = F.l1_loss(prediction, mobj)
                    total_loss += loss.item()
                    total_mse += mse
                    total_mae += mae.item()

        return total_loss / len(self.val_loader), total_mse / len(self.val_loader), total_mae / len(self.val_loader)
    
    def test(self, test_loader):
        self.generator.eval()
        total_mse = 0
        
        with torch.no_grad():
            for batch in test_loader:
                mobj, mref, eobj, eref = [item.to(self.device) for item in batch]
                if self.config.use_twoway:
                    fake_mobj, _ = self.generator(eobj, eref, mref)
                    mse = F.mse_loss(fake_mobj, mobj)
                    total_mse += mse.item()
                else:
                    fake_mobj = self.generator(eobj, eref, mref)
                    mse = F.mse_loss(fake_mobj, mobj)
                    total_mse += mse.item()
    
        avg_mse = total_mse / len(test_loader)
        print(f"Test Results - MSE: {avg_mse:.4f}")

    def save_checkpoint(self, epoch, is_best=False):
        state = {
            'epoch': epoch,
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict(),
            'optimizer_D': self.optimizer_D.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        
        if is_best:
            best_model_path = Path(self.config.checkpoint_dir) / f'{self.config.experiment_name}_best_model.pth'
            torch.save(state, best_model_path)
            self.logger.log_info(f"Saved best model to {best_model_path}")
        else:
            current_model_path = Path(self.config.checkpoint_dir) / f'{self.config.experiment_name}_model_epoch.pth'
            torch.save(state, current_model_path)
            self.logger.log_info(f"Saved model checkpoint to {current_model_path}")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.best_val_loss = checkpoint['best_val_loss']
        return checkpoint['epoch']