import os
import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping

from vidnn.utils import check_configs
from vidnn.utils.yaml_helper import get_configs
from vidnn.utils.module_select import get_data_module, get_model, get_model_module


def train(cfg):
    # Check cfg
    cfg = check_configs(cfg)

    # Dataloaders
    data_module = get_data_module(cfg)
    data_module.setup(stage="fit")
    train_dataloaders = data_module.train_dataloader()
    val_dataloaders = data_module.val_dataloader()

    # Model
    model = get_model(cfg)
    model_module = get_model_module(model=model, cfg=cfg, steps_per_epoch=len(train_dataloaders))
    # model_module = torch.compile(model_module)

    # Setup train
    logger = TensorBoardLogger(
        save_dir=os.path.join(cfg["save_dir"], cfg["model"]),
    )
    run_version = logger.version

    callbacks = [
        LearningRateMonitor(),
        EarlyStopping(
            monitor="fitness",
            patience=cfg["patience"],
            verbose=True,
            mode="max",
        ),
        ModelCheckpoint(
            dirpath=os.path.join(cfg["save_dir"], cfg["model"], "weights", f"version_{run_version}"),
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
    )


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--cfg", required=True, type=str, help="config file")
    # args = parser.parse_args()
    # cfg = get_configs(args.cfg)

    cfg = get_configs("/mnt/michael/vidnn/vidnn/configs/yolo.yaml")
    train(cfg)
