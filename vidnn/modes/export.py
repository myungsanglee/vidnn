import os
import argparse
import torch

from vidnn.utils.yaml_helper import get_configs
from vidnn.utils.module_select import get_model, get_model_module
from vidnn.utils.torch_utils import model_info


def export(cfg_path, ckpt_path, save_path):
    """
    Exports a trained LightningModule from a checkpoint, ready for inference.
    The saved file will contain the full module, including the .predict() method.

    Args:
        cfg_path (str): Path to the configuration YAML file used for training.
        ckpt_path (str): Path to the PyTorch Lightning checkpoint (.ckpt) file.
        save_path (str): Path to save the exported module file (.pt).
    """
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint file not found at {ckpt_path}")
        return

    print(f"Loading configuration from {cfg_path}")
    cfg = get_configs(cfg_path)
    
    # Manually set the checkpoint path in the config.
    # The get_model_module function will use this to load the weights.
    cfg['ckpt'] = ckpt_path

    print("Creating model structure...")
    # The base model structure is needed to initialize the LightningModule
    model = get_model(cfg)

    print("Creating and loading LightningModule from checkpoint...")
    # The get_model_module function already contains the logic to load the checkpoint.
    # 'steps_per_epoch' is not needed for inference, so we pass 0.
    model_module = get_model_module(model=model, cfg=cfg, steps_per_epoch=0)
    
    # The weights are already loaded inside get_model_module.
    model_module.eval() # Set the module to evaluation mode

    model_info(model_module.model, imgsz=cfg["imgsz"])

    print(f"Saving exported LightningModule to {save_path}")
    # Save the entire LightningModule
    torch.save(model_module, save_path)
    print("\nExport complete!")
    print(f"You can now load the module and predict directly:")
    print(f"  model = torch.load('{save_path}')")
    print(f"  prediction, _ = model.predict('path/to/image.jpg')")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export a full LightningModule from a checkpoint for easy inference."
    )
    parser.add_argument(
        "--cfg",
        required=True,
        type=str,
        help="Path to the training configuration YAML file.",
    )
    parser.add_argument(
        "--ckpt", required=True, type=str, help="Path to the checkpoint (.ckpt) file."
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="exported_module.pt",
        help="Path to save the exported module (.pt) file.",
    )
    args = parser.parse_args()

    export(args.cfg, args.ckpt, args.save_path)

    # Example usage from command line:
    # python vidnn/modes/export.py --cfg vidnn/configs/yolo-ladybug.yaml --ckpt path/to/your/last.ckpt --save_path ladybug_detector_module.pt
