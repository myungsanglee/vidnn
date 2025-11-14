import os
import torch
import cv2
import numpy as np

from vidnn.utils import check_configs
from vidnn.utils.yaml_helper import get_configs
from vidnn.utils.module_select import get_data_module
from vidnn.utils.ops import xywhr2xyxyxyxy, xywh2xyxy
from vidnn.utils.plotting import Annotator, colors
from vidnn.data.utils import IMG_FORMATS


def main(cfg, mode="train"):
    # Check cfg
    cfg = check_configs(cfg)

    # Check task
    task = cfg["task"]
    assert task in ["detect", "obb"], "Now only support detect, obb tasks"

    # Dataloaders
    data_module = get_data_module(cfg)
    data_module.setup(stage="fit")
    if mode == "train":
        dataloader = data_module.train_dataloader()
    elif mode == "val":
        dataloader = data_module.val_dataloader()
    else:
        print("mode only support train, val mode")
        return

    for batch in dataloader:
        images, bboxes, batch_idx, cls, img_file = batch["img"], batch["bboxes"], batch["batch_idx"], batch["cls"], batch["im_file"]
        for i, image in enumerate(images):
            # Convert tensor to numpy array for visualization
            img = image.permute(1, 2, 0).numpy().copy()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Get indices
            indices = torch.where(batch_idx == i)[0]
            tmp_bboxes, tmp_cls, tmp_img_file = bboxes[indices], cls[indices], img_file[i]

            # Denormalize bboxes
            h, w, _ = img.shape
            tmp_bboxes[..., [0, 2]] *= w
            tmp_bboxes[..., [1, 3]] *= h

            # Convert bbox format
            if task == "detect":
                tmp_bboxes = xywh2xyxy(tmp_bboxes)
            elif task == "obb":
                tmp_bboxes = xywhr2xyxyxyxy(tmp_bboxes)

            # Annotator
            annotator = Annotator(img)

            # annotate bbox
            for bbox, cls_idx in zip(tmp_bboxes, tmp_cls):
                cls_idx = int(cls_idx[0])
                color = colors(cls_idx, True)
                annotator.box_label(bbox, label=str(cls_idx), color=color)

            # get result
            img = annotator.result()

            # Show image
            cv2.imshow("Image", img)
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q") or key == 27:
                cv2.destroyAllWindows()
                return
            elif key == ord("c"):
                # save image
                img_num = 1
                save_dir = cfg["save_dir"]
                img_list = [x for x in os.listdir(save_dir) if x.rpartition(".")[-1].lower() in IMG_FORMATS]
                if len(img_list):
                    img_list = sorted(img_list)
                    filename = os.path.basename(img_list[-1])
                    filename = os.path.splitext(filename)[0]
                    number_str = filename.split("_")[-1]
                    img_num = int(number_str) + 1
                save_img_filename = os.path.join(save_dir, f"img_{img_num:05d}.jpg")
                cv2.imwrite(save_img_filename, img)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # cfg = get_configs("vidnn/configs/yolo.yaml")
    cfg = get_configs("vidnn/configs/yolo-obb.yaml")
    main(cfg)
    # main(cfg, "val")
