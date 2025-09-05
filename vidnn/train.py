import argparse
import platform
import os

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping

# from pytorch_lightning.plugins import DDPPlugin
from torchinfo import summary

from vidnn.utils.yaml_helper import get_configs
from vidnn.data.datamodule import YoloDataModule
from vidnn.models.yolo.detect.mobilenetv3_yolov8 import YOLOv8MobileNet
from vidnn.module.yolo_detector import YoloDetector

# from utils.utility import make_model_names

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def train(cfg):
    data_module = YoloDataModule(cfg)

    model = YOLOv8MobileNet(num_classes=len(cfg["names"]))

    summary(model, input_size=(1, cfg["channels"], cfg["imgsz"], cfg["imgsz"]), device="cpu")

    model_module = YoloDetector(model=model, cfg=cfg)

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(monitor="val_loss", save_last=True, every_n_epochs=cfg["save_freq"]),
        EarlyStopping(monitor="val_loss", patience=30, verbose=True),
    ]

    trainer = pl.Trainer(
        max_epochs=cfg["epochs"],
        # logger=TensorBoardLogger(cfg["save_dir"], make_model_name(cfg), default_hp_metric=False),
        logger=TensorBoardLogger(cfg["save_dir"], default_hp_metric=False),
        accelerator=cfg["accelerator"],
        devices=cfg["devices"],
        # plugins=DDPPlugin(find_unused_parameters=False) if platform.system() != "Windows" else None,
        callbacks=callbacks,
        **cfg["trainer_options"],
    )

    trainer.fit(model_module, data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, type=str, help="config file")
    args = parser.parse_args()
    cfg = get_configs(args.cfg)

    train(cfg)
