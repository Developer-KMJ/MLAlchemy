
import pytorch_lightning
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from dataset import PetsDataModule

# from model import ResnetTest
from model_mixup import ResnetTest
import config as Config

data_module = PetsDataModule(
    Config.DATA_DIR_BASE, 
    Config.BATCH_SIZE, 
    Config.NUM_WORKERS)

model = ResnetTest(
    num_classes=data_module.get_num_classes())

checkpoint_callback = ModelCheckpoint(dirpath="./checkpoints", save_top_k=5, monitor="val_accuracy", mode="max")
early_stop_callback = EarlyStopping(monitor="val_accuracy", min_delta=0.001, patience=25, verbose=True, mode="max")

trainer = Trainer(
    accelerator="gpu", 
    devices=1, 
    enable_checkpointing=True, 
    # precision=16,
    callbacks=[checkpoint_callback, early_stop_callback])

# Pick point based on suggestion
# tuner = Tuner(trainer)
# lr_finder = tuner.lr_find(model, datamodule=data_module, min_lr=1e-08, max_lr=1e-3, num_training=250)
# new_lr = lr_finder.suggestion()
# print("new lr: ", new_lr)
# model.learning_rate = new_lr

trainer.fit(
    model,
    datamodule=data_module,
    ckpt_path="./checkpoints/epoch=126-step=93853-v1.ckpt")

trainer.test(
    model, 
    datamodule=data_module)