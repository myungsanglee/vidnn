import cv2
import numpy as np
import torch

from vidnn.data.datamodule import YoloDataModule
from vidnn.utils.yaml_helper import get_configs


def check_dataloader():
    yaml_path = "/mnt/michael/vidnn/vidnn/configs/yolo.yaml"
    cfg = get_configs(yaml_path)

    print("Initializing YoloDataModule with v8 augmentations...")
    data_module = YoloDataModule(cfg=cfg)

    print("Setting up data...")
    data_module.setup(stage="fit")
    # dataloader = data_module.train_dataloader()
    dataloader = data_module.val_dataloader()
    batch = next(iter(dataloader))

    targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)

    batch_img = batch["img"]
    batch_size = batch_img.shape[0]
    print(f"Images tensor shape: {batch_img.size()}")
    print(f"Targets tensor shape: {targets.size()}")

    for i in range(batch_size):
        img = batch_img[i].permute(1, 2, 0).numpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        height, width = img.shape[:2]

        indices = torch.where(targets[:, 0] == i)[0]
        target = targets[indices].numpy()[..., 1:]
        cls = target[..., :1]
        bboxes = target[..., 1:]
        for cls_id, bbox in zip(cls, bboxes):
            class_id = int(cls_id[0])
            cx, cy, w, h = bbox
            x1 = int((cx - w / 2) * width)
            y1 = int((cy - h / 2) * height)
            x2 = int((cx + w / 2) * width)
            y2 = int((cy + h / 2) * height)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img,
                f"cls: {class_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

        cv2.imshow("test", img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    check_dataloader()
