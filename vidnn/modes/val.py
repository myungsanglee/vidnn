import pytorch_lightning as pl

from vidnn.utils import check_configs
from vidnn.utils.yaml_helper import get_configs
from vidnn.utils.module_select import get_data_module, get_model, get_model_module


def val(cfg):
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

    # Setup validation
    trainer = pl.Trainer(
        logger=False,
        **cfg["trainer_options"],
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

    # cfg = get_configs("vidnn/configs/yolo.yaml")
    cfg = get_configs("vidnn/configs/yolo-obb.yaml")
    val(cfg)
