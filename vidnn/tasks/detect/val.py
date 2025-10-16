import os
import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping

from vidnn.utils import check_configs
from vidnn.utils.yaml_helper import get_configs
from vidnn.utils.module_select import get_data_module, get_model, get_model_module
from vidnn.utils import LOGGER
from vidnn.data.datamodule import YoloDataModule
from vidnn.models.detect.mobilenetv3_yolov8 import YOLOv8MobileNet
from vidnn.module.yolo_detector import YoloDetector


def val(cfg):
    # Check cfg
    cfg = check_configs(cfg)

    # Dataloaders
    # data_module = YoloDataModule(cfg)
    data_module = get_data_module(cfg)
    data_module.setup(stage="fit")
    train_dataloaders = data_module.train_dataloader()
    val_dataloaders = data_module.val_dataloader()

    # Model
    model = YOLOv8MobileNet(num_classes=len(cfg["names"]))
    model_module = YoloDetector(
        model=model,
        cfg=cfg,
        steps_per_epoch=len(train_dataloaders),
    )
    # model_module = YoloDetector.load_from_checkpoint(
    #     checkpoint_path=ckpt,
    #     model=model,
    #     cfg=cfg,
    #     steps_per_epoch=len(train_dataloaders),
    # )
    # model_module = torch.compile(model_module)

    # model = get_model(cfg)
    # model_module = get_model_module(model=model, cfg=cfg, steps_per_epoch=len(train_dataloaders))
    ckpt_path = cfg.get("ckpt")
    if ckpt_path and os.path.exists(ckpt_path):
        LOGGER.info(f"Loading weights from checkpoint: {ckpt_path}")

        # 체크포인트 파일을 불러옵니다.
        # map_location='cpu'는 GPU 메모리 관련 오류를 방지하는 안전한 방법입니다.
        # 모델은 이후 Trainer에 의해 자동으로 올바른 장치(device)로 이동됩니다.
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # PyTorch Lightning 체크포인트는 'state_dict' 키 아래에 모델 가중치를 저장합니다.
        state_dict = checkpoint["state_dict"]

        # state_dict를 모델에 적용합니다.
        # strict=False는 전이 학습(transfer learning)에 유용합니다.
        # 체크포인트와 현재 모델 간에 일부 레이어 이름이 다르거나 없을 경우 오류 없이
        # 일치하는 레이어의 가중치만 불러옵니다.
        model_module.load_state_dict(state_dict, strict=False)

    trainer = pl.Trainer(
        logger=False,
        accelerator=cfg["trainer_options"]["accelerator"],
        devices=cfg["trainer_options"]["devices"],
    )

    trainer.validate(
        model=model_module,
        dataloaders=val_dataloaders,
    )


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--cfg", required=True, type=str, help="config file")
    # parser.add_argument('--ckpt', required=True, type=str, help='checkpoints file')
    # args = parser.parse_args()
    # cfg = get_configs(args.cfg)
    # val(cfg, args.ckpt)

    cfg = get_configs("/mnt/michael/vidnn/vidnn/configs/yolo.yaml")
    # ckpt = "/mnt/michael/vidnn/runs/lightning_logs/version_1/checkpoints/last.ckpt"
    val(cfg)
