import math
import random
import cv2
import numpy as np
import albumentations as A


def letterbox(
    im,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scaleFill=False,
    scaleup=True,
    stride=32,
):
    """Resize and pad image to new_shape with stride-multiple constraints."""
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def random_perspective(im, targets=(), degrees=10, translate=0.1, scale=0.1, shear=10, perspective=0.0):
    """Applies random perspective, rotation, scaling, and shear transformations."""
    height, width, _ = im.shape

    # Center
    C = np.eye(3)
    C[0, 2] = -im.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -im.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation

    # Combined rotation matrix
    M = T @ S @ R @ P @ C
    if height > 0 and width > 0:
        im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(114, 114, 114))

    # Transform label coordinates
    n = len(targets)
    if n:
        xy = np.ones((n * 4, 3))
        # targets are [class, x1, y1, x2, y2]
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform

        # Filter out points behind the camera
        z = xy[:, 2]
        z_positive = z > 1e-6  # Threshold to avoid division by zero

        # Reshape z_positive to match the structure of xy
        z_positive = z_positive.reshape(n, 4).all(axis=1)

        targets = targets[z_positive]
        xy = xy[z_positive.repeat(4)]

        if not len(targets):
            return im, targets

        xy = (xy[:, :2] / xy[:, 2:3]).reshape(-1, 8)  # perspective rescale or divide by z

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, -1).T

        # clip
        new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
        new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # filter small boxes
        w = new[:, 2] - new[:, 0]
        h = new[:, 3] - new[:, 1]
        area = w * h

        # You can adjust this area threshold
        area_threshold = 1000.0
        i = area > area_threshold

        targets = targets[i]
        new = new[i]

        targets[:, 1:5] = new

    return im, targets


def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
    """HSV color-space augmentation."""
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)
    return im


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    """Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right."""
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = np.empty_like(x, dtype=np.float32)  # faster than clone/copy
    y[..., 0] = w * (x[..., 0] - x[..., 2] / 2) + padw  # top left x
    y[..., 1] = h * (x[..., 1] - x[..., 3] / 2) + padh  # top left y
    y[..., 2] = w * (x[..., 0] + x[..., 2] / 2) + padw  # bottom right x
    y[..., 3] = h * (x[..., 1] + x[..., 3] / 2) + padh  # bottom right y
    return y


def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    """Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right."""
    if clip:
        # x = np.clip(x, 0, (w - 1, h - 1, w - 1, h - 1))
        x[..., [0, 2]] = x[..., [0, 2]].clip(0, w - eps)  # x1, x2
        x[..., [1, 3]] = x[..., [1, 3]].clip(0, h - eps)  # y1, y2
    y = np.empty_like(x, dtype=np.float32)  # faster than clone/copy
    y[..., 0] = ((x[..., 0] + x[..., 2]) / 2) / w  # x center
    y[..., 1] = ((x[..., 1] + x[..., 3]) / 2) / h  # y center
    y[..., 2] = (x[..., 2] - x[..., 0]) / w  # width
    y[..., 3] = (x[..., 3] - x[..., 1]) / h  # height
    return y


def albumentations_transforms(hyp):
    """YOLOv8 style augmentations using albumentations."""
    return A.Compose(
        [
            # # --- Geometric Transformations ---
            # A.Affine(
            #     rotate=hyp.get("degrees", 0.0),  # FIX: Changed 'degrees' to 'rotate'
            #     translate_percent=hyp.get("translate", 0.1),
            #     scale=(1 - hyp.get("scale", 0.5), 1 + hyp.get("scale", 0.5)),
            #     shear=hyp.get("shear", 0.0),
            #     p=0.5,
            # ),
            # A.Perspective(scale=(0.0, hyp.get("perspective", 0.0)), p=0.5),
            # # --- Color Transformations ---
            # A.ColorJitter(
            #     saturation=hyp.get("hsv_s", 0.7), hue=hyp.get("hsv_h", 0.015), p=0.5
            # ),
            # A.Blur(p=0.01),
            # A.MedianBlur(p=0.01),
            A.Blur(p=0.01),
            A.MedianBlur(p=0.01),
            A.ToGray(p=0.01),
            A.CLAHE(p=0.01),
            A.RandomBrightnessContrast(p=0.0),
            A.RandomGamma(p=0.0),
            A.ImageCompression(quality_range=(75, 100), p=0.0),
            A.HorizontalFlip(p=hyp.get("fliplr", 0.5)),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
    )
