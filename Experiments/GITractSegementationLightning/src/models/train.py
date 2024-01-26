import os

# Related third party imports
import pytorch_lightning as pl
import torch
from lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from torchvision.transforms import functional as F

# Local application/library specific imports
from common import config, LossMetric
from data import MriDataModule
from model import UNet


def main():
    seed_everything(42)
    
    torch.set_float32_matmul_precision('medium')
    
    model = UNet(num_classes=3)
    
    model.set_write_intermediate(True)
    
    mri_data = MriDataModule(data_dir = config.DATA_DIR, zip_file = config.ZIP_FILE)

    logger_dir = os.path.join(config.OUT_DIR, 'logs')
    csv_logger = CSVLogger(logger_dir, name='unet_axial')
    
    checkpoint_dir = os.path.join(config.OUT_DIR, 'checkpoints')
    trainer = pl.Trainer(num_sanity_val_steps=0,
                        accelerator='gpu',
                        logger=csv_logger,
                        max_epochs=10,
                        callbacks=[ModelCheckpoint(monitor=LossMetric.VAL_LOSS,
                            dirpath=checkpoint_dir,
                            filename='unet-{epoch:02d}-{val_loss:.2f}',
                            save_top_k=10,
                            mode='min',
                            enable_version_counter=True)])
    
    trainer.fit(model,
                datamodule=mri_data) 
    
if __name__ == '__main__':
    main()