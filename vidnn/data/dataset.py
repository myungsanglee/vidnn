import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import os

from .augmentations import (
    letterbox,
    xywhn2xyxy,
    xyxy2xywhn,
    albumentations_transforms,
    random_perspective,
    augment_hsv,
)


class YoloDataset(Dataset):
    def __init__(self, img_path, imgsz=640, augment=False, hyp=None):
        self.img_path = img_path
        self.imgsz = imgsz
        self.augment = augment
        self.hyp = hyp if hyp is not None else {}

        self.img_files = sorted(
            [
                os.path.join(img_path, f)
                for f in os.listdir(img_path)
                if f.endswith((".jpg", ".png"))
            ]
        )
        self.label_files = [
            f.replace("images", "labels")
            .replace(".png", ".txt")
            .replace(".jpg", ".txt")
            for f in self.img_files
        ]
        self.indices = range(len(self.img_files))

        if self.augment:
            self.transform = albumentations_transforms(self.hyp)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        # 1. Load image and labels
        if self.augment and random.random() < self.hyp.get("mosaic", 0.0):
            # Load mosaic: returns image and labels in absolute xyxy format
            img, labels = self.load_mosaic(index)
        else:
            # Load single image
            img, (h0, w0), (h, w) = self.load_image(index)
            # Letterbox
            img, ratio, pad = letterbox(
                img, self.imgsz, auto=False, scaleup=self.augment
            )
            labels = self.load_label(index)
            if len(labels):
                # Convert normalized xywh to absolute xyxy
                labels[:, 1:5] = xywhn2xyxy(
                    labels[:, 1:5], w=w, h=h, padw=pad[0], padh=pad[1]
                )

        # 2. Random Perspective
        if self.augment:
            img, labels = random_perspective(
                img,
                labels,
                degrees=self.hyp.get("degrees", 0.0),
                translate=self.hyp.get("translate", 0.1),
                scale=self.hyp.get("scale", 0.5),
                shear=self.hyp.get("shear", 0.0),
                perspective=self.hyp.get("perspective", 0.0),
            )

        # 3. MixUp
        if self.augment and random.random() < self.hyp.get("mixup", 0.0):
            # Load second image for mixup
            img2, labels2 = self.__load_mixup_image()
            r = np.random.beta(8.0, 8.0)
            img = (img * r + img2 * (1 - r)).astype(np.uint8)
            labels = np.concatenate((labels, labels2), 0)

        # 4. Convert labels from absolute xyxy to normalized yolo format for albumentations
        h, w = img.shape[:2]
        if len(labels):
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=w, h=h, clip=True)

        # 5. Apply Albumentations
        if self.augment:
            transformed = self.transform(
                image=img, bboxes=labels[:, 1:], class_labels=labels[:, 0]
            )
            img = transformed["image"]
            labels = np.array(
                [
                    [c] + list(b)
                    for c, b in zip(transformed["class_labels"], transformed["bboxes"])
                ],
                dtype=np.float32,
            )

        # 6. Random HSV
        if self.augment:
            img = augment_hsv(
                img,
                hgain=self.hyp.get("hsv_h", 0.5),
                sgain=self.hyp.get("hsv_s", 0.5),
                vgain=self.hyp.get("hsv_v", 0.5),
            )

        # 7. Final formatting
        if len(labels) == 0:
            labels = np.zeros((0, 5), dtype=np.float32)

        labels_out = torch.zeros((len(labels), 6))
        if len(labels):
            labels_out[:, 1:] = torch.from_numpy(labels)

        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).transpose(0, 2).transpose(1, 2).float() / 255.0

        return img, labels_out

    def __load_mixup_image(self):
        # Simplified loader for mixup that mimics the main path
        idx = random.randint(0, len(self) - 1)
        if self.hyp.get("mosaic", 0.0) > 0 and random.random() < self.hyp.get(
            "mosaic", 0.0
        ):
            return self.load_mosaic(idx)
        else:
            img, (h0, w0), (h, w) = self.load_image(idx)
            img, ratio, pad = letterbox(
                img, self.imgsz, auto=False, scaleup=self.augment
            )
            labels = self.load_label(idx)
            if len(labels):
                labels[:, 1:5] = xywhn2xyxy(
                    labels[:, 1:5], w=w, h=h, padw=pad[0], padh=pad[1]
                )
            return img, labels

    def load_image(self, i):
        path = self.img_files[i]
        im = cv2.imread(path)
        assert im is not None, f"Image Not Found {path}"
        h0, w0 = im.shape[:2]
        r = self.imgsz / max(h0, w0)
        if r != 1:
            im = cv2.resize(
                im, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_LINEAR
            )
        return im, (h0, w0), im.shape[:2]

    def load_label(self, i):
        path = self.label_files[i]
        if os.path.exists(path):
            with open(path) as f:
                labels = np.array(
                    [x.split() for x in f.read().strip().splitlines()], dtype=np.float32
                )
        else:
            labels = np.zeros((0, 5), dtype=np.float32)
        return labels

    def load_mosaic(self, index):
        labels4 = []
        s = self.imgsz
        yc, xc = (int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2))
        indices = [index] + random.choices(self.indices, k=3)

        for i, idx in enumerate(indices):
            img, _, (h, w) = self.load_image(idx)
            labels = self.load_label(idx)
            if len(labels):
                # Convert normalized xywh to absolute xyxy
                labels[:, 1:5] = xywhn2xyxy(labels[:, 1:5], w=w, h=h)

            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            if len(labels):
                labels[:, 1:5] += (padw, padh, padw, padh)
            labels4.append(labels)

        labels4 = np.concatenate(labels4, 0)
        np.clip(labels4[:, 1:5], 0, 2 * s, out=labels4[:, 1:5])

        # Final crop
        x_crop, y_crop = xc - s // 2, yc - s // 2
        img4 = img4[y_crop : y_crop + s, x_crop : x_crop + s]

        # Adjust labels to final crop
        if len(labels4):
            labels4[:, 1:5] -= (x_crop, y_crop, x_crop, y_crop)
            np.clip(labels4[:, 1:5], 0, s, out=labels4[:, 1:5])

        # Filter out small boxes
        if len(labels4):
            i = ((labels4[:, 3] - labels4[:, 1]) > 2) & (
                (labels4[:, 4] - labels4[:, 2]) > 2
            )
            labels4 = labels4[i]

        return img4, labels4

    @staticmethod
    def collate_fn(batch):
        im, label = zip(*batch)
        for i, l in enumerate(label):
            l[:, 0] = i
        return torch.stack(im, 0), torch.cat(label, 0)
