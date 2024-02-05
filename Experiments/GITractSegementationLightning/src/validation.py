import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from common.consts import LossMetric

from data.data_module import MriDataModule
from model import UNet
from common import config

def main():
    # torch.set_float32_matmul_precision('medium')
    
    model = UNet(num_classes=3)
    model.set_write_intermediate(True)
    mri_data = MriDataModule(data_dir = config.DATA_DIR, zip_file = config.ZIP_FILE)

    logger_dir = os.path.join(config.OUT_DIR, 'logs')
    csv_logger = CSVLogger(logger_dir, name='unet_axial')
    
    checkpoint_dir = os.path.join(config.OUT_DIR, 'checkpoints')
    trainer = pl.Trainer(
        accelerator='gpu', 
        precision='16-mixed',
        logger=csv_logger,
        callbacks=[ModelCheckpoint(monitor=LossMetric.VAL_LOSS,
                                      dirpath=checkpoint_dir,
                                      filename='unet-{epoch:02d}-{val_loss:.2f}',
                                      save_top_k=6,
                                      mode='min',
                                      enable_version_counter=True),
                   EarlyStopping(monitor=LossMetric.VAL_LOSS, 
                                 min_delta=0.00001,
                                 patience=3, 
                                 mode='min', 
                                 verbose=True)])
    
    trainer.validate(model, 
                val_dataloaders=mri_data.val_dataloader())
    
if __name__ == '__main__':
    main()