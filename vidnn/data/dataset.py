import cv2
import numpy as np
import torch
import random
import os
import math
from torch.utils.data import Dataset
import albumentations as A

from .utils import xywhn2xyxy, xyxy2xywhn

IMG_FORMATS = {"bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm", "heic"}  # image suffixes


class YoloDataset(Dataset):
    def __init__(self, img_path, imgsz=640, augment=False, cfg=None, bgr=0.0):
        self.img_path = img_path
        self.imgsz = imgsz
        self.augment = augment
        self.cfg = cfg if cfg is not None else {}
        self.img_files = self.get_img_files(img_path)
        self.label_files = [x.replace("images", "labels").rsplit(".", 1)[0] + ".txt" for x in self.img_files]
        self.indices = range(len(self.img_files))
        self.bgr = bgr
        self.albumentations = A.Compose(
            [
                A.Blur(p=0.01),
                A.MedianBlur(p=0.01),
                A.ToGray(p=0.01),
                A.CLAHE(p=0.01),
                A.RandomBrightnessContrast(p=0.0),
                A.RandomGamma(p=0.0),
                A.ImageCompression(quality_range=(75, 100), p=0.0),
            ]
        )

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        label = self.get_image_and_label(index)

        if self.augment:
            # Mosaic
            if random.random() < self.cfg.get("mosaic", 0.0):
                label = self._mosaic4(label)
                # print(label)

            # Random Perspective
            label = self._random_perspective(
                label,
                degrees=self.cfg.get("degrees", 0.0),
                translate=self.cfg.get("translate", 0.1),
                scale=self.cfg.get("scale", 0.5),
                shear=self.cfg.get("shear", 0.0),
                perspective=self.cfg.get("perspective", 0.0),
            )

            # Albumentations
            if label["img"].shape[2] == 3:  # Only apply Albumentation on 3-channel images
                label["img"] = self.albumentations(image=label["img"])["image"]  # transformed

            # Random HSV
            label = self._random_hsv(
                label,
                hgain=self.cfg.get("hsv_h", 0.5),
                sgain=self.cfg.get("hsv_s", 0.5),
                vgain=self.cfg.get("hsv_v", 0.5),
            )

            # Random Horizontal Flip
            if random.random() < self.cfg.get("fliplr", 0.0):
                label = self._random_horizontal_flip(label)

        else:
            label = self._letterbox(label, new_shape=self.imgsz)

        # Final formatting
        img = label.pop("img")
        h, w = img.shape[:2]
        cls = label.pop("cls")
        bboxes = label.pop("bboxes")
        bboxes = xyxy2xywhn(bboxes, w=w, h=h, clip=True) if label["bbox_format"] == "xyxy" else bboxes
        nl = len(bboxes)

        if len(img.shape) < 3:
            img = np.expand_dims(img, -1)
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img[::-1] if random.uniform(0, 1) > self.bgr and img.shape[0] == 3 else img)
        img = torch.from_numpy(img)
        label["img"] = img
        label["cls"] = torch.from_numpy(cls) if nl else torch.zeros(nl)
        label["bboxes"] = torch.from_numpy(bboxes) if nl else torch.zeros((nl, 4))
        label["batch_idx"] = torch.zeros(nl)

        return label

    def get_img_files(self, img_path):
        try:
            img_files = sorted([os.path.join(img_path, x) for x in os.listdir(img_path) if x.rpartition(".")[-1].lower() in IMG_FORMATS])
            assert img_files, f"No images found in {img_path}."
        except Exception as e:
            raise FileNotFoundError(f"{e}")
        return img_files

    def get_image_and_label(self, index):
        """
        Get and return label information from the dataset.

        Args:
            index (int): Index of the image to retrieve.

        Returns:
            (Dict[str, Any]): Label dictionary with image and metadata.
        """
        label = {}

        # get image
        label["img"], label["ori_shape"], label["resized_shape"] = self.load_image(index)
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )  # for evaluation

        # get label
        lb = self.load_label(index)
        label["cls"] = lb[:, 0:1]  # n, 1
        label["bboxes"] = lb[:, 1:]  # n, 4
        label["bbox_format"] = "xywh"

        return label

    def load_image(self, i):
        filename = self.img_files[i]
        im = self.imread(filename)
        if im is None:
            raise FileNotFoundError(f"Image Not Found {filename}")
        h0, w0 = im.shape[:2]  # orig hw
        r = self.imgsz / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
            im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        if im.ndim == 2:
            im = im[..., None]

        return im, (h0, w0), im.shape[:2]

    def imread(self, filename, flags=cv2.IMREAD_COLOR):
        """
        Read an image from a file with multilanguage filename support.

        Args:
            filename (str): Path to the file to read.
            flags (int, optional): Flag that can take values of cv2.IMREAD_*. Controls how the image is read.

        Returns:
            (np.ndarray | None): The read image array, or None if reading fails.
        """
        file_bytes = np.fromfile(filename, np.uint8)
        if filename.endswith((".tiff", ".tif")):
            success, frames = cv2.imdecodemulti(file_bytes, cv2.IMREAD_UNCHANGED)
            if success:
                # Handle RGB images in tif/tiff format
                return frames[0] if len(frames) == 1 and frames[0].ndim == 3 else np.stack(frames, axis=2)
            return None
        else:
            im = cv2.imdecode(file_bytes, flags)
            return im[..., None] if im is not None and im.ndim == 2 else im  # Always ensure 3 dimensions

    def load_label(self, i):
        lb_file = self.label_files[i]
        if os.path.isfile(lb_file):
            with open(lb_file, encoding="utf-8") as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                lb = np.array(lb, dtype=np.float32)
            if len(lb):
                assert lb.shape[1] == 5, f"labels require 5 columns, {lb.shape[1]} columns detected"
                points = lb[:, 1:]
                # Coordinate points check with 1% tolerance
                assert points.max() <= 1.01, f"non-normalized or out of bounds coordinates {points[points > 1.01]}"
                assert lb.min() >= -0.01, f"negative class labels {lb[lb < -0.01]}"
            else:
                lb = np.zeros((0, 5), dtype=np.float32)
        else:
            lb = np.zeros((0, 5), dtype=np.float32)
        return lb

    def _mosaic4(self, label):
        cls4 = []
        bboxes4 = []
        s = self.imgsz
        border = (-self.imgsz // 2, -self.imgsz // 2)  # width, height
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in border)  # mosaic center x, y
        indices = [-1] + random.choices(self.indices, k=3)
        for i, idx in enumerate(indices):
            labels_patch = label if idx == -1 else self.get_image_and_label(idx)
            # Load image
            img = labels_patch["img"]
            h, w = labels_patch.pop("resized_shape")
            cls = labels_patch["cls"]
            bboxes = labels_patch["bboxes"]

            # if len(labels):
            #     # Convert normalized xywh to absolute xyxy
            #     labels[:, 1:5] = xywhn2xyxy(labels[:, 1:5], w=w, h=h)

            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            if len(bboxes) and labels_patch["bbox_format"] == "xywh":
                bboxes = xywhn2xyxy(bboxes, w, h, padw, padh)
                # labels[:, 1:5] += (padw, padh, padw, padh)
            bboxes4.append(bboxes)
            cls4.append(cls)

        cls4 = np.concatenate(cls4, 0)
        bboxes4 = np.concatenate(bboxes4, 0)
        np.clip(bboxes4, 0, 2 * s, out=bboxes4)

        label["img"] = img4
        label["cls"] = cls4
        label["bboxes"] = bboxes4
        label["bbox_format"] = "xyxy"
        label["resized_shape"] = (self.imgsz * 2, self.imgsz * 2)
        label["mosaic_border"] = border

        return label

    def _letterbox(self, label, new_shape=(640, 640), auto=False, scale_fill=False, scaleup=True, center=True, stride=32):
        img = label.get("img")
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = label.pop("rect_shape", new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scale_fill:
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        if center:
            dw /= 2  # divide padding into 2 sides
            dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
            if img.ndim == 2:
                img = img[..., None]

        top, bottom = int(round(dh - 0.1)) if center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if center else 0, int(round(dw + 0.1))
        h, w, c = img.shape
        if c == 3:
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        else:  # multispectral
            pad_img = np.full((h + top + bottom, w + left + right, c), fill_value=114, dtype=img.dtype)
            pad_img[top : top + h, left : left + w] = img
            img = pad_img

        if label.get("ratio_pad"):
            label["ratio_pad"] = (label["ratio_pad"], (left, top))  # for evaluation

        label["img"] = img
        label["resized_shape"] = new_shape
        bboxes = label["bboxes"]
        if len(bboxes) and label["bbox_format"] == "xywh":
            bboxes = xywhn2xyxy(bboxes, w, h, left, top)
            label["bboxes"] = bboxes
            label["bbox_format"] = "xyxy"
        return label

    def _random_perspective(self, label, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0):
        img = label["img"]
        cls = label["cls"]
        bboxes = label["bboxes"]
        if label["bbox_format"] == "xywh":
            bboxes = xywhn2xyxy(bboxes, img.shape[1], img.shape[0])
            label["bbox_format"] == "xyxy"
        border = label.pop("mosaic_border", (0, 0))
        size = img.shape[1] + border[1] * 2, img.shape[0] + border[0] * 2  # w, h

        # Center
        C = np.eye(3, dtype=np.float32)

        C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
        C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

        # Perspective
        P = np.eye(3, dtype=np.float32)
        P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
        P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

        # Rotation and Scale
        R = np.eye(3, dtype=np.float32)
        a = random.uniform(-degrees, degrees)
        s = random.uniform(1 - scale, 1 + scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear
        S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear

        # Translation
        T = np.eye(3, dtype=np.float32)
        T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * size[0]  # x translation
        T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * size[1]  # y translation

        # Combined rotation matrix
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
        # Affine image
        if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
            if perspective:
                img = cv2.warpPerspective(img, M, dsize=size, borderValue=(114, 114, 114))
            else:  # affine
                img = cv2.warpAffine(img, M[:2], dsize=size, borderValue=(114, 114, 114))
            if img.ndim == 2:
                img = img[..., None]

        # Apply bboxes
        n = len(bboxes)
        if n == 0:
            new_bboxes = bboxes
        else:
            xy = np.ones((n * 4, 3), dtype=bboxes.dtype)
            xy[:, :2] = bboxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

            # Create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new_bboxes = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1)), dtype=bboxes.dtype).reshape(4, n).T
        # clip
        new_bboxes[:, [0, 2]] = new_bboxes[:, [0, 2]].clip(0, size[0])
        new_bboxes[:, [1, 3]] = new_bboxes[:, [1, 3]].clip(0, size[1])
        # Filter bboxes
        bboxes[:, 0] *= s
        bboxes[:, 1] *= s
        bboxes[:, 2] *= s
        bboxes[:, 3] *= s
        # Make the bboxes have the same scale with new_bboxes
        box1 = bboxes.T
        box2 = new_bboxes.T
        wh_thr = 2
        ar_thr = 100
        area_thr = 0.1
        eps = 1e-16
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
        i = (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates

        label["img"] = img
        label["cls"] = cls[i]
        label["bboxes"] = new_bboxes[i]
        label["resize_shape"] = img.shape[:2]
        return label

    def _random_hsv(self, label, hgain=0.5, sgain=0.5, vgain=0.5):
        img = label["img"]
        if img.shape[-1] != 3:  # only apply to RGB images
            return label
        if hgain or sgain or vgain:
            dtype = img.dtype  # uint8

            r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain]  # random gains
            x = np.arange(0, 256, dtype=r.dtype)
            # lut_hue = ((x * (r[0] + 1)) % 180).astype(dtype)   # original hue implementation from ultralytics<=8.3.78
            lut_hue = ((x + r[0] * 180) % 180).astype(dtype)
            lut_sat = np.clip(x * (r[1] + 1), 0, 255).astype(dtype)
            lut_val = np.clip(x * (r[2] + 1), 0, 255).astype(dtype)
            lut_sat[0] = 0  # prevent pure white changing color, introduced in 8.3.79

            hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
            im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed
        return label

    def _random_horizontal_flip(self, label):
        img = label["img"]
        bboxes = label["bboxes"]
        if label["bbox_format"] == "xywh":
            bboxes = xywhn2xyxy(bboxes, img.shape[1], img.shape[0])
        w = img.shape[1]

        img = np.fliplr(img)
        x1 = bboxes[:, 0].copy()
        x2 = bboxes[:, 2].copy()
        bboxes[:, 0] = w - x2
        bboxes[:, 2] = w - x1

        label["img"] = np.ascontiguousarray(img)
        label["bboxes"] = bboxes
        return label

    @staticmethod
    def collate_fn(batch):
        new_batch = {}
        batch = [dict(sorted(b.items())) for b in batch]  # make sure the keys are in the same order
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k in {"img"}:
                value = torch.stack(value, 0)
            if k in {"bboxes", "cls"}:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i  # add target image index for build_targets()
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch
