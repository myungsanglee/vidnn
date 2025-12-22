import os
import sys
import torch

torch.set_float32_matmul_precision("high")

import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping, Callback

from vidnn.utils.yaml_helper import get_configs
from vidnn.utils import check_configs
from vidnn.utils.yaml_helper import get_configs
from vidnn.utils.module_select import get_data_module, get_model, get_model_module
from vidnn.utils.callbacks import Float32LrMonitor

import optuna
from optuna.integration import PyTorchLightningPruningCallback as OptunaPruningCallback


class PatchedPruningCallback(Callback):
    def __init__(self, trial, monitor):
        super().__init__()
        self._pruning_callback = OptunaPruningCallback(trial, monitor)

    def on_validation_end(self, trainer, pl_module):
        self._pruning_callback.on_validation_end(trainer, pl_module)


def objective(trial: optuna.trial.Trial, cfg: dict):
    # === 1. 하이퍼파라미터 제안 ===
    lr0 = trial.suggest_float("lr0", 1e-3, 1e-2, log=True)
    momentum = trial.suggest_float("momentum", 9e-1, 99e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-3, log=True)
    optimizer = trial.suggest_categorical("optimizer", ["SGD", "RMSProp", "Adam", "Adamax", "AdamW", "NAdam", "RAdam"])
    cos_lr = trial.suggest_categorical("cos_lr", [True, False])

    # Get config
    # cfg = get_configs(cfg_path)
    # cfg = check_configs(cfg)
    cfg["lr0"] = lr0
    cfg["momentum"] = momentum
    cfg["weight_decay"] = weight_decay
    cfg["optimizer"] = optimizer
    cfg["cos_lr"] = cos_lr
    # set for optuna
    cfg["epochs"] = 100
    cfg["patience"] = 10
    cfg["save_dir"] = "optuna"

    # --- Resume Logic ---
    resumed_trial_number = trial.user_attrs.get("resumes_trial")
    if resumed_trial_number is not None:
        path_trial_number = resumed_trial_number
        print(f"INFO: Trial #{trial.number} is resuming from specified trial #{resumed_trial_number}.")
    else:
        path_trial_number = trial.number

    study_name = trial.study.study_name
    trial_version = f"trial_{path_trial_number}"

    logger = TensorBoardLogger(
        save_dir=os.path.join(cfg["save_dir"], study_name),
        name=cfg["model"],
        version=trial_version,
        default_hp_metric=False,
    )

    checkpoint_dir = os.path.join(logger.log_dir, "weights")
    ckpt_path = os.path.join(checkpoint_dir, "last.ckpt")
    if not os.path.exists(ckpt_path):
        ckpt_path = None
    else:
        print(f"INFO: Found checkpoint for trial #{path_trial_number} at: {ckpt_path}")

    # Dataloaders, Model, Callbacks
    data_module = get_data_module(cfg)
    data_module.setup(stage="fit")
    train_dataloaders = data_module.train_dataloader()
    val_dataloaders = data_module.val_dataloader()
    model = get_model(cfg)
    model_module = get_model_module(model=model, cfg=cfg, steps_per_epoch=len(train_dataloaders))
    callbacks = [
        Float32LrMonitor() if torch.backends.mps.is_available() else LearningRateMonitor(),
        EarlyStopping(monitor="fitness", patience=cfg["patience"], verbose=True, mode="max"),
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            monitor="fitness",
            save_last=True,
            every_n_epochs=cfg["trainer_options"]["check_val_every_n_epoch"],
            mode="max",
        ),
        PatchedPruningCallback(trial, monitor="fitness"),
    ]

    trainer = pl.Trainer(
        max_epochs=cfg["epochs"],
        logger=logger,
        callbacks=callbacks,
        **cfg["trainer_options"],
    )

    # --- Execute Training ---
    trainer.fit(
        model=model_module,
        train_dataloaders=train_dataloaders,
        val_dataloaders=val_dataloaders,
        ckpt_path=ckpt_path,
    )

    # --- Handle Interruption and Completion ---
    if trainer.interrupted:
        print(f"WARN: Trial #{trial.number} was interrupted by the user. Marking as FAIL.")
        raise RuntimeError(f"Trial #{trial.number} interrupted.")

    if "fitness" not in trainer.callback_metrics:
        print(f"WARN: Trial #{trial.number} finished without a 'fitness' metric. Marking as FAIL.")
        raise RuntimeError(f"Trial #{trial.number} did not produce a 'fitness' metric.")

    # 성공적으로 완료되면 최종 메트릭 반환
    return trainer.callback_metrics["fitness"].item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=20, help="Number of trials to run.")
    parser.add_argument("--study-name", type=str, default="obb", help="Name of the study.")
    parser.add_argument("--resume-trial", type=int, help="Specify a trial number to resume.")
    # parser.add_argument("--config", type=str, default="vidnn/configs/yolo.yaml", help="Path to the config file.")
    parser.add_argument("--config", type=str, default="vidnn/configs/yolo-obb.yaml", help="Path to the config file.")
    args = parser.parse_args()

    # get cfg
    cfg = get_configs(args.config)
    cfg = check_configs(cfg)
    print(type(cfg))

    # DB 저장 경로 설정
    save_dir = "optuna"  # objective 함수와 일관성 유지
    study_name = args.study_name
    db_dir = os.path.join(save_dir, study_name, cfg["model"])
    os.makedirs(db_dir, exist_ok=True)
    storage_name = f"sqlite:///{os.path.join(db_dir, 'optuna.db')}"

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=10,
        interval_steps=1,
    )

    # 1. Study 생성 또는 로드
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_name,
        direction="maximize",
        pruner=pruner,
        load_if_exists=True,
    )

    # 2. 특정 trial 재개 로직
    if args.resume_trial is not None:
        trial_to_resume = None
        # get_trials()는 모든 상태의 trial을 포함합니다.
        for t in study.get_trials(deepcopy=False):
            if t.number == args.resume_trial:
                trial_to_resume = t
                break

        if trial_to_resume is None:
            print(f"ERROR: Trial #{args.resume_trial} not found in study '{args.study_name}'.", file=sys.stderr)
            sys.exit(1)

        # 재개를 위해 지정된 trial의 파라미터로 새로운 trial을 대기열에 추가
        print(f"INFO: Enqueuing parameters from trial #{args.resume_trial} for a new run.")
        study.enqueue_trial(trial_to_resume.params, user_attrs={"resumes_trial": trial_to_resume.number})

    # 3. 최적화 실행
    study.optimize(lambda trial: objective(trial, cfg), n_trials=args.n_trials)

    # 4. 결과 출력
    print("Number of finished trials: ", len(study.trials))
    trial = study.best_trial

    print(f"  Best trial: {trial.number}")
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    main()
