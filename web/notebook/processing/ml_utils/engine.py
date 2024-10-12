import numpy as np
import torch

from tqdm import tqdm
from metric_utils import eval_metric


class Engine:
    def __init__(self, model, 
                 optimizer, scheduler, criterion,
                 train_data_loader, valid_data_loader,
                 num_classes, epochs, device):

        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader

        self.model = model
        self.num_classes = num_classes
        self.epochs = epochs
        self.device = device

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
    
        self.train_iters = 0
        self.valid_iters = 0
        self.trained_epochs = 0

    def get_num_epochs(self):
        return self.trained_epochs
    
    def update_num_epoch(self, epoch):
        self.epochs = epoch

    def load_from_ckpt(self, ckpt_path):
        checkpoint = torch.load(ckpt_path)
        
        self.trained_epochs = checkpoint['epoch']
        self.train_iters = checkpoint['train_iters']
        self.valid_iters = checkpoint['valid_iters']
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def fit(self):
        self.model.train()
        train_running_loss = 0.0
        train_running_inter, train_running_union = 0, 0
        train_running_correct, train_running_label = 0, 0
    
        # calculate the number of batches
        num_batches = int(self.train_data_loader.len / self.train_data_loader.batch_size)
        prog_bar = tqdm(self.train_data_loader, total=num_batches)

        for data, target in prog_bar:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(data)

            # don't need this for unet
            outputs = outputs['out']
            
            _, predictions = torch.max(target, dim=1)
            loss = self.criterion(outputs, predictions)
            train_running_loss += loss.item()

            correct, labeled, inter, union = eval_metric(outputs, predictions, self.num_classes)
            # for IoU
            train_running_inter += inter
            train_running_union += union
            # for pixel accuracy
            train_running_correct += correct
            train_running_label += labeled

            loss.backward()
            self.optimizer.step()
            
            train_running_IoU = 1.0 * inter / (np.spacing(1) + union)
            train_running_mIoU = train_running_IoU.mean()
            train_running_pixacc = 1.0 * correct / (np.spacing(1) + labeled)

            prog_bar.set_description(desc=f"Loss: {loss:.4f} | mIoU: {train_running_mIoU:.4f} | PixAcc: {train_running_pixacc:.4f}")

            self.train_iters += 1
        
        self.scheduler.step()
        # metrics per epoch
        # IoU and mIoU
        IoU = 1.0 * train_running_inter / (np.spacing(1) + train_running_union)
        mIoU = IoU.mean()
        # pixel accuracy
        pixel_acc = 1.0 * train_running_correct / (np.spacing(1) + train_running_label)    
        train_loss = train_running_loss / num_batches
        self.trained_epochs += 1

        return train_loss, mIoU, pixel_acc
    
    def validate(self):
        self.model.eval()
        valid_running_loss = 0.0
        valid_running_inter, valid_running_union = 0, 0
        valid_running_correct, valid_running_label = 0, 0
        # calculate the number of batches
        num_batches = int(self.valid_data_loader.len / self.valid_data_loader.batch_size)
        with torch.no_grad():
            prog_bar = tqdm(self.valid_data_loader, total=num_batches)
            for data, target in prog_bar:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                outputs = outputs['out']
                
                _, predictions = torch.max(target, dim=1)
                loss = self.criterion(outputs, predictions)
                valid_running_loss += loss.item()

                correct, labeled, inter, union = eval_metric(outputs, predictions, self.num_classes)
                valid_running_inter += inter
                valid_running_union += union
                
                valid_running_correct += correct
                valid_running_label += labeled
                
                valid_running_IoU = 1.0 * inter / (np.spacing(1) + union)
                valid_running_mIoU = valid_running_IoU.mean()
                valid_running_pixacc = 1.0 * correct / (np.spacing(1) + labeled)
                
                prog_bar.set_description(desc=f"Loss: {loss:.4f} | mIoU: {valid_running_mIoU:.4f} | PixAcc: {valid_running_pixacc:.4f}")

                self.valid_iters += 1
            
        valid_loss = valid_running_loss / num_batches

        # metrics per epoch
        # IoU and mIoU
        IoU = 1.0 * valid_running_inter / (np.spacing(1) + valid_running_union)
        mIoU = IoU.mean()
        # pixel accuracy
        pixel_acc = 1.0 * valid_running_correct / (np.spacing(1) + valid_running_label)
        ##############################
        return valid_loss, mIoU, pixel_acc
    
    def save_model(self, path):
        torch.save({
            'epoch': self.trained_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.criterion,
            'valid_iters': self.valid_iters, 
            'train_iters': self.train_iters}, 
            f"{path}.pth")
