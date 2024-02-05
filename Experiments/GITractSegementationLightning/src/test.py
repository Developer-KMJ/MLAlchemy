import os
import sys
from lightning_fabric import seed_everything

import torch

# Included due to the common dir being a sibling of the src dir
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

from data.data_module import MriDataModule
from model import UNet
from common import config
from common.consts import MriPlane

def main():
    seed_everything(42)
    
    torch.set_float32_matmul_precision('medium')
    
    mri_data = MriDataModule(data_dir = config.DATA_DIR, zip_file = config.ZIP_FILE, mri_plane=MriPlane.AXIAL)

    logger_dir = os.path.join(config.OUT_DIR, 'logs')
    csv_logger = CSVLogger(logger_dir, name='unet_axial-test')
    
    # Load the model from a checkpoint
    model = UNet.load_from_checkpoint(checkpoint_path="//home//kevin//Documents//MRIProject-Working//out//SelectedModels//Axial-unet-epoch=04-val_loss=0.00.ckpt", num_classes=3)
    model.write_intermediate = True

    trainer = pl.Trainer(accelerator='gpu',
                         logger=csv_logger)
    trainer.test(model, 
                datamodule=mri_data)
            
if __name__ == '__main__':
    main()