import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping

from torchinfo import summary

from vidnn.utils.yaml_helper import get_configs
from vidnn.data.datamodule import YoloDataModule
from vidnn.models.yolo.detect.mobilenetv3_yolov8 import YOLOv8MobileNet
from vidnn.module.yolo_detector import YoloDetector
from vidnn.utils.torch_utils import model_info


def train(cfg):
    # Dataloaders
    data_module = YoloDataModule(cfg)
    data_module.setup(stage="fit")
    train_dataloaders = data_module.train_dataloader()
    val_dataloaders = data_module.val_dataloader()

    # Model
    model = YOLOv8MobileNet(num_classes=len(cfg["names"]))
    # model_info(model, imgsz=cfg["imgsz"])

    # summary(model, input_size=(1, cfg["channels"], cfg["imgsz"], cfg["imgsz"]), device="cpu")
    model_module = YoloDetector(model=model, cfg=cfg, steps_per_epoch=len(train_dataloaders))
    model_info(model_module, imgsz=cfg["imgsz"])
    model_module = torch.compile(model_module)

    # # Setup train
    # callbacks = [
    #     # LearningRateMonitor(logging_interval="step"),
    #     LearningRateMonitor(),
    #     ModelCheckpoint(monitor="fitness", save_last=True, every_n_epochs=cfg["trainer_options"]["check_val_every_n_epoch"]),
    #     EarlyStopping(monitor="fitness", patience=cfg["patience"], verbose=True, mode="max"),
    # ]

    # trainer = pl.Trainer(
    #     max_epochs=cfg["epochs"],
    #     # logger=TensorBoardLogger(cfg["save_dir"], make_model_name(cfg), default_hp_metric=False),
    #     logger=TensorBoardLogger(cfg["save_dir"], default_hp_metric=False),
    #     # accelerator=cfg["accelerator"],
    #     # devices=cfg["devices"],
    #     # plugins=DDPPlugin(find_unused_parameters=False) if platform.system() != "Windows" else None,
    #     callbacks=callbacks,
    #     **cfg["trainer_options"],
    # )

    # trainer.fit(
    #     model=model_module,
    #     train_dataloaders=train_dataloaders,
    #     val_dataloaders=val_dataloaders,
    # )


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--cfg", required=True, type=str, help="config file")
    # args = parser.parse_args()
    # cfg = get_configs(args.cfg)

    cfg = get_configs("/mnt/michael/vidnn/vidnn/configs/yolo.yaml")
    train(cfg)
