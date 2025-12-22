import os
import torch
import cv2
import numpy as np

from vidnn.utils import check_configs
from vidnn.utils.yaml_helper import get_configs
from vidnn.utils.module_select import get_data_module
from vidnn.utils.ops import xywhr2xyxyxyxy, xywh2xyxy
from vidnn.utils.plotting import Annotator, colors
from vidnn.data.utils import IMG_FORMATS, letterbox


def main(cfg, mode="train"):
    # Check cfg
    cfg = check_configs(cfg)

    # Check task
    task = cfg["task"]
    assert task in ["detect", "obb", "segment", "pose", "classify"], "Now only support detect, obb tasks"

    # Dataloaders
    data_module = get_data_module(cfg)
    data_module.setup(stage="fit")
    if mode == "train":
        dataloader = data_module.train_dataloader()
    elif mode == "val":
        dataloader = data_module.val_dataloader()
    else:
        print("only support train, val mode")
        return

    for batch in dataloader:
        for k in {"cls", "bboxes", "conf", "masks", "keypoints", "batch_idx", "img"}:
            if k not in batch:
                continue
            if k == "cls" and batch[k].ndim == 2:
                batch[k] = batch[k].squeeze(1)  # squeeze if shape is (n, 1)
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].cpu().numpy()

        cls = batch.get("cls", np.zeros(0, dtype=np.int64))
        batch_idx = batch.get("batch_idx", np.zeros(cls.shape, dtype=np.int64))
        bboxes = batch.get("bboxes", np.zeros(0, dtype=np.float32))
        confs = batch.get("conf", None)
        masks = batch.get("masks", np.zeros(0, dtype=np.uint8))
        kpts = batch.get("keypoints", np.zeros(0, dtype=np.float32))
        images = batch.get("img", np.zeros((0, 3, 640, 640), dtype=np.float32))  # default to input images

        if len(images) and isinstance(images, torch.Tensor):
            images = images.cpu().float().numpy()

        _, _, h, w = images.shape  # height, width
        # bs = min(bs, 16)  # limit plot images
        # ns = np.ceil(bs**0.5)  # number of subplots (square)
        if np.max(images[0]) <= 1:
            images *= 255  # de-normalise (optional)

        # fs = int((h + w) * ns * 0.01)  # font size
        # fs = max(fs, 18)  # ensure that the font size is large enough to be easily readable.
        for i, image in enumerate(images):
            # Convert tensor to numpy array for visualization
            img = image.transpose(1, 2, 0).copy()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Annotator
            # annotator = Annotator(img, line_width=round(fs / 10), font_size=fs)
            annotator = Annotator(img)

            if len(cls) > 0:
                idx = batch_idx == i
                tmp_bboxes = bboxes[idx]
                tmp_cls = cls[idx]

                if len(tmp_bboxes):
                    if tmp_bboxes[:, :4].max() <= 1.1:
                        # Denormalize bboxes
                        tmp_bboxes[..., [0, 2]] *= w
                        tmp_bboxes[..., [1, 3]] *= h
                    is_obb = tmp_bboxes.shape[-1] == 5  # xywhr
                    tmp_bboxes = xywhr2xyxyxyxy(tmp_bboxes) if is_obb else xywh2xyxy(tmp_bboxes)
                    # annotate bbox
                    for j, bbox in enumerate(tmp_bboxes.astype(np.int64).tolist()):
                        cls_idx = int(tmp_cls[j])
                        color = colors(cls_idx, True)
                        annotator.box_label(bbox, label=str(cls_idx), color=color)

                    # for bbox, cls_idx in zip(tmp_bboxes, tmp_cls):
                    #     color = colors(int(cls_idx), True)
                    #     annotator.box_label(bbox, label=str(cls_idx), color=color)

            # # Get indices
            # indices = torch.where(batch_idx == i)[0]
            # tmp_bboxes, tmp_cls, tmp_img_file = bboxes[indices], cls[indices], img_file[i]

            # # Denormalize bboxes
            # h, w, _ = img.shape
            # tmp_bboxes[..., [0, 2]] *= w
            # tmp_bboxes[..., [1, 3]] *= h

            # # Convert bbox format
            # if task == "obb":
            #     tmp_bboxes = xywhr2xyxyxyxy(tmp_bboxes)
            # else:
            #     tmp_bboxes = xywh2xyxy(tmp_bboxes)

            # annotate masks
            # if masks is not None:
            #     im_gpu = letterbox(annotator.result(), masks.shape[1:])
            #     im_gpu = torch.as_tensor(img, dtype=torch.float16, device=masks.device).permute(2, 0, 1).flip(0).contiguous() / 255
            #     annotator.masks(masks, colors=[colors(int(x), True) for x in tmp_cls], im_gpu=im_gpu)

            # # annotate bbox
            # for bbox, cls_idx in zip(tmp_bboxes, tmp_cls):
            #     cls_idx = int(cls_idx[0])
            #     color = colors(cls_idx, True)
            #     annotator.box_label(bbox, label=str(cls_idx), color=color)

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
    cfg = get_configs("vidnn/configs/yolo.yaml")
    # cfg = get_configs("vidnn/configs/yolo-obb.yaml")
    # cfg = get_configs("vidnn/configs/yolo-seg.yaml")
    main(cfg)
    # main(cfg, "val")
