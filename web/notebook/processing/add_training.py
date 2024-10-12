import numpy as np
import torch
import torch.nn as nn

from ml_utils.dataloader import DefectDataset, getDefectDatasetLoaders
from ml_utils.engine import Engine
from ml_utils.logger import Logger, create_base_logger
from ml_utils.model import get_model_deeplabv3_resnet50
from torch.optim import lr_scheduler


def get_engine():
    dataset = DefectDataset("dataset_addon/")
    loaders_dict = getDefectDatasetLoaders(dataset, batch_size=4)

    model = get_model_deeplabv3_resnet50()

    device = "cpu"
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_classes = 8
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    # следующий эксперимент
    weight = np.concatenate((np.ones(num_classes-1), np.array([0.001])))
    weight = torch.from_numpy(weight).float().to(device)

    criterion = nn.CrossEntropyLoss(weight=weight) 

    model = model.to(device)
    engine = Engine(model, 
                    optimizer=optimizer, 
                    scheduler=scheduler,
                    criterion=criterion,
                    train_data_loader=loaders_dict["train"],
                    valid_data_loader=loaders_dict["test"],
                    num_classes=num_classes, epochs=20, device=device)
    
    return engine


def launch_training(training_dir):

    stats_logger = Logger(f"{training_dir}/stats.csv")
    base_logger = create_base_logger(training_dir)
    engine = get_engine()

    prev_best_value = 0
    engine.load_from_ckpt(f"ml_utils/checkpoints/best.pth")
    engine.epochs = engine.trained_epochs + 20 # дополнительные эпохи для дообучения
    for e in range(engine.trained_epochs, engine.epochs):
        base_logger.debug(f"Epoch {e+1}: TRAIN")
        train_loss, train_mIoU, train_pixel_acc = engine.fit()
        base_logger.debug(f"Loss: {train_loss:.6f} | Metrics: mIoU={train_mIoU:.4f}, pixel_acc={train_pixel_acc:.4f}")
        stats_logger.update_training_data(train_loss, train_mIoU, train_pixel_acc)

        base_logger.debug(f"Epoch {e+1}: VALID")
        val_loss, val_mIoU, val_pixel_acc = engine.validate()
        base_logger.debug(f"Loss: {val_loss:.6f} | Metrics: mIoU={val_mIoU:.4f}, pixel_acc={val_pixel_acc:.4f}\n")
        stats_logger.update_val_data(val_loss, val_mIoU, val_pixel_acc)

        stats_logger.write_line(e+1)

        engine.save_model(f"{training_dir}/last")
        if val_mIoU > prev_best_value:
            engine.save_model(f"{training_dir}/best")
            prev_best_value = val_mIoU
            base_logger.info("Updated Best weights")