import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping, Callback

from vidnn.utils.yaml_helper import get_configs
from vidnn.data.datamodule import YoloDataModule
from vidnn.module.yolo_detector import YoloDetector
from vidnn.models.detect.mobilenetv3_large_100_yolo import MobileNetV3Yolo

import optuna
from optuna.integration import PyTorchLightningPruningCallback as OptunaPruningCallback


class PatchedPruningCallback(Callback):
    def __init__(self, trial, monitor):
        super().__init__()
        self._pruning_callback = OptunaPruningCallback(trial, monitor)

    def on_validation_end(self, trainer, pl_module):
        self._pruning_callback.on_validation_end(trainer, pl_module)


def objective(trial: optuna.trial.Trial):
    # === 1. 하이퍼파라미터 제안 ===
    lr0 = trial.suggest_float("lr0", 1e-3, 1e-2, log=True)
    momentum = trial.suggest_float("momentum", 9e-1, 99e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-3, log=True)
    optimizer = trial.suggest_categorical("optimizer", ["SGD", "RMSProp", "Adam", "Adamax", "AdamW", "NAdam", "RAdam"])
    cos_lr = trial.suggest_categorical("cos_lr", [True, False])

    # Get config
    cfg = get_configs("/mnt/michael/vidnn/vidnn/configs/yolo.yaml")
    cfg["lr0"] = lr0
    cfg["momentum"] = momentum
    cfg["weight_decay"] = weight_decay
    cfg["optimizer"] = optimizer
    cfg["cos_lr"] = cos_lr
    # set for optuna
    cfg["epochs"] = 100
    cfg["patience"] = 10
    cfg["save_dir"] = "/mnt/michael/vidnn/optuna"

    # Dataloaders
    data_module = YoloDataModule(cfg)
    data_module.setup(stage="fit")
    train_dataloaders = data_module.train_dataloader()
    val_dataloaders = data_module.val_dataloader()

    # Model
    model = MobileNetV3Yolo(num_classes=len(cfg["names"]))
    model_module = YoloDetector(model=model, cfg=cfg, steps_per_epoch=len(train_dataloaders))
    model_module = torch.compile(model_module)

    # Setup train
    callbacks = [
        # LearningRateMonitor(logging_interval="step"),
        LearningRateMonitor(),
        ModelCheckpoint(monitor="fitness", save_last=True, every_n_epochs=cfg["trainer_options"]["check_val_every_n_epoch"], mode="max"),
        EarlyStopping(monitor="fitness", patience=cfg["patience"], verbose=True, mode="max"),
        PatchedPruningCallback(trial, monitor="fitness"),
    ]

    trainer = pl.Trainer(
        max_epochs=cfg["epochs"],
        # logger=TensorBoardLogger(cfg["save_dir"], make_model_name(cfg), default_hp_metric=False),
        logger=TensorBoardLogger(cfg["save_dir"], default_hp_metric=False),
        # accelerator=cfg["accelerator"],
        # devices=cfg["devices"],
        # plugins=DDPPlugin(find_unused_parameters=False) if platform.system() != "Windows" else None,
        callbacks=callbacks,
        **cfg["trainer_options"],
    )

    trainer.fit(
        model=model_module,
        train_dataloaders=train_dataloaders,
        val_dataloaders=val_dataloaders,
    )


if __name__ == "__main__":
    # "가지치기"를 수행할 Pruner 설정
    pruner = optuna.pruners.MedianPruner()

    # 1. Study 생성
    # direction="minimize": objective 함수가 반환하는 값을 '최소화'하는 것이 목표
    # 만약 정확도(accuracy)를 기준으로 한다면 "maximize"로 설정
    study = optuna.create_study(direction="maximize", pruner=pruner)

    # 2. 최적화 실행
    # n_trials=20: 총 20번의 다른 하이퍼파라미터 조합으로 시도
    study.optimize(objective, n_trials=20)

    # 3. 결과 출력
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)  # 가장 좋았던 fitness
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
