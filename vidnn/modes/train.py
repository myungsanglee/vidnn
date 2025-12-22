import os
import torch

torch.set_float32_matmul_precision("high")

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from vidnn.utils import check_configs
from vidnn.utils.yaml_helper import get_configs
from vidnn.utils.module_select import get_data_module, get_model, get_model_module
from vidnn.utils.callbacks import Float32LrMonitor


def train(cfg):
    # Check cfg
    cfg = check_configs(cfg)

    # Resume logic
    ckpt_path = None
    version = None
    if cfg.get("resume"):  # Use .get for safety
        version = cfg.get("version")
        if version is not None:
            ckpt_path = os.path.join(
                cfg["save_dir"],
                cfg["model"],
                f"version_{version}",
                "weights",
                "last.ckpt",
            )
            if not os.path.exists(ckpt_path):
                print(f"Warning: Checkpoint path {ckpt_path} not found. Starting new training.")
                ckpt_path = None
                version = None  # Start a new version if checkpoint is not found
                return
        else:
            print("Warning: 'resume' is true but 'version' is not specified. Starting new training.")
            return

    # Dataloaders
    data_module = get_data_module(cfg)
    data_module.setup(stage="fit")
    train_dataloaders = data_module.train_dataloader()
    val_dataloaders = data_module.val_dataloader()

    # Model
    model = get_model(cfg)
    model_module = get_model_module(model=model, cfg=cfg, steps_per_epoch=len(train_dataloaders))

    # Setup train
    logger = TensorBoardLogger(
        save_dir=os.path.join(cfg["save_dir"], cfg["model"]),
        name=None,
        version=version,
        default_hp_metric=False,
    )
    run_version = logger.version  # When resuming, this will be the specified version. Otherwise, it will be a new version string.

    # The ModelCheckpoint dirpath should be consistent.
    checkpoint_dir = os.path.join(
        cfg["save_dir"],
        cfg["model"],
        f"version_{run_version}",
        "weights",
    )

    callbacks = [
        Float32LrMonitor() if torch.backends.mps.is_available() else LearningRateMonitor(),
        EarlyStopping(
            monitor="fitness",
            patience=cfg["patience"],
            verbose=True,
            mode="max",
        ),
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            monitor="fitness",
            save_last=True,
            every_n_epochs=cfg["trainer_options"]["check_val_every_n_epoch"],
            mode="max",
        ),
    ]

    trainer = pl.Trainer(
        max_epochs=cfg["epochs"],
        logger=logger,
        callbacks=callbacks,
        **cfg["trainer_options"],
    )

    trainer.fit(
        model=model_module,
        train_dataloaders=train_dataloaders,
        val_dataloaders=val_dataloaders,
        ckpt_path=ckpt_path,
    )


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--cfg", required=True, type=str, help="config file")
    # args = parser.parse_args()
    # cfg = get_configs(args.cfg)

    # cfg = get_configs("vidnn/configs/yolo.yaml")
    # cfg = get_configs("vidnn/configs/yolo-obb.yaml")
    cfg = get_configs("vidnn/configs/yolo-ladybug.yaml")
    train(cfg)
