import math
import random
import cv2
import albumentations as A
import os
import numpy as np
from PIL import Image, ImageOps

from vidnn.utils import LOGGER
from vidnn.utils.ops import segments2boxes


IMG_FORMATS = {"bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm", "heic"}  # image suffixes
VID_FORMATS = {"asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv", "webm"}  # video suffixes
FORMATS_HELP_MSG = f"Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}"


def get_hash(paths):
    """Return a single hash value of a list of paths (files or dirs)."""
    size = 0
    for p in paths:
        try:
            size += os.stat(p).st_size
        except OSError:
            continue
    h = __import__("hashlib").sha256(str(size).encode())  # hash sizes
    h.update("".join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


def exif_size(img):
    """Return exif-corrected PIL size."""
    s = img.size  # (width, height)
    if img.format == "JPEG":  # only support JPEG images
        try:
            if exif := img.getexif():
                rotation = exif.get(274, None)  # the EXIF key for the orientation tag is 274
                if rotation in {6, 8}:  # rotation 270 or 90
                    s = s[1], s[0]
        except Exception:
            pass
    return s


def verify_image_label(args):
    """Verify one image-label pair."""
    im_file, lb_file, prefix, keypoint, num_cls, nkpt, ndim, single_cls = args
    # Number (missing, found, empty, corrupt), message, segments, keypoints
    nm, nf, ne, nc, msg, segments, keypoints = 0, 0, 0, 0, "", [], None
    try:
        # Verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        shape = (shape[1], shape[0])  # hw
        assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
        assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}. {FORMATS_HELP_MSG}"
        if im.format.lower() in {"jpg", "jpeg"}:
            with open(im_file, "rb") as f:
                f.seek(-2, 2)
                if f.read() != b"\xff\xd9":  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)
                    msg = f"{prefix}{im_file}: corrupt JPEG restored and saved"

        # Verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found
            with open(lb_file, encoding="utf-8") as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any(len(x) > 6 for x in lb) and (not keypoint):  # is segment
                    classes = np.array([x[0] for x in lb], dtype=np.float32)
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
                    lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                lb = np.array(lb, dtype=np.float32)
            if nl := len(lb):
                if keypoint:
                    assert lb.shape[1] == (5 + nkpt * ndim), f"labels require {(5 + nkpt * ndim)} columns each"
                    points = lb[:, 5:].reshape(-1, ndim)[:, :2]
                else:
                    assert lb.shape[1] == 5, f"labels require 5 columns, {lb.shape[1]} columns detected"
                    points = lb[:, 1:]
                # Coordinate points check with 1% tolerance
                assert points.max() <= 1.01, f"non-normalized or out of bounds coordinates {points[points > 1.01]}"
                assert lb.min() >= -0.01, f"negative class labels {lb[lb < -0.01]}"

                # All labels
                if single_cls:
                    lb[:, 0] = 0
                max_cls = lb[:, 0].max()  # max label count
                assert max_cls < num_cls, (
                    f"Label class {int(max_cls)} exceeds dataset class count {num_cls}. " f"Possible class labels are 0-{num_cls - 1}"
                )
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:  # duplicate row check
                    lb = lb[i]  # remove duplicates
                    if segments:
                        segments = [segments[x] for x in i]
                    msg = f"{prefix}{im_file}: {nl - len(i)} duplicate labels removed"
            else:
                ne = 1  # label empty
                lb = np.zeros((0, (5 + nkpt * ndim) if keypoint else 5), dtype=np.float32)
        else:
            nm = 1  # label missing
            lb = np.zeros((0, (5 + nkpt * ndim) if keypoints else 5), dtype=np.float32)
        if keypoint:
            keypoints = lb[:, 5:].reshape(-1, nkpt, ndim)
            if ndim == 2:
                kpt_mask = np.where((keypoints[..., 0] < 0) | (keypoints[..., 1] < 0), 0.0, 1.0).astype(np.float32)
                keypoints = np.concatenate([keypoints, kpt_mask[..., None]], axis=-1)  # (nl, nkpt, 3)
        lb = lb[:, :5]
        return im_file, lb, shape, segments, keypoints, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f"{prefix}{im_file}: ignoring corrupt image/label: {e}"
        return [None, None, None, None, None, nm, nf, ne, nc, msg]


def load_dataset_cache_file(path):
    """Load *.cache dictionary from path."""
    import gc

    gc.disable()  # reduce pickle load time
    cache = np.load(str(path), allow_pickle=True).item()  # load dict
    gc.enable()
    return cache


def save_dataset_cache_file(prefix, path, x, version):
    """Save an Ultralytics dataset *.cache dictionary x to path."""
    x["version"] = version  # add cache version
    if is_dir_writeable(path.parent):
        if path.exists():
            path.unlink()  # remove *.cache file if exists
        with open(str(path), "wb") as file:  # context manager here fixes windows async np.save bug
            np.save(file, x)
        LOGGER.info(f"{prefix}New cache created: {path}")
    else:
        LOGGER.warning(f"{prefix}Cache directory {path.parent} is not writeable, cache not saved.")


def is_dir_writeable(dir_path):
    """
    Check if a directory is writeable.

    Args:
        dir_path (str | Path): The path to the directory.

    Returns:
        (bool): True if the directory is writeable, False otherwise.
    """
    return os.access(str(dir_path), os.W_OK)


def load_image_from_source(source, imgsz=640):
    if isinstance(source, str):
        im = imread(source)
        if im is None:
            raise FileNotFoundError(f"Image Not Found {source}")
        h0, w0 = im.shape[:2]  # orig hw
        orig_img = im
        r = imgsz / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            w, h = (min(math.ceil(w0 * r), imgsz), min(math.ceil(h0 * r), imgsz))
            im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        if im.ndim == 2:
            im = im[..., None]

        return im, orig_img[..., None] if orig_img.ndim == 2 else orig_img

    elif isinstance(source, np.array):
        h0, w0 = source.shape[:2]  # orig hw
        orig_img = im
        r = imgsz / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            w, h = (min(math.ceil(w0 * r), imgsz), min(math.ceil(h0 * r), imgsz))
            source = cv2.resize(source, (w, h), interpolation=cv2.INTER_LINEAR)
        if source.ndim == 2:
            source = source[..., None]

        return source, orig_img[..., None] if orig_img.ndim == 2 else orig_img

    else:
        raise TypeError(f"Source only support str or numpy array")


def imread(filename, flags=cv2.IMREAD_COLOR):
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


def letterbox(
    img,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=False,
    scaleFill=False,
    scaleup=True,
    stride=32,
):
    """Resize and pad image to new_shape with stride-multiple constraints."""
    shape = img.shape[:2]  # current shape [height, width]
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
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        if img.ndim == 2:
            img = img[..., None]

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    h, w, c = img.shape
    if c == 3:
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    else:  # multispectral
        pad_img = np.full((h + top + bottom, w + left + right, c), fill_value=114, dtype=img.dtype)
        pad_img[top : top + h, left : left + w] = img
        img = pad_img

    # return img, ratio, (dw, dh)
    return img


def polygon2mask(imgsz: tuple[int, int], polygons: list[np.ndarray], color: int = 1, downsample_ratio: int = 1) -> np.ndarray:
    """
    Convert a list of polygons to a binary mask of the specified image size.

    Args:
        imgsz (tuple[int, int]): The size of the image as (height, width).
        polygons (list[np.ndarray]): A list of polygons. Each polygon is an array with shape (N, M), where
                                     N is the number of polygons, and M is the number of points such that M % 2 = 0.
        color (int, optional): The color value to fill in the polygons on the mask.
        downsample_ratio (int, optional): Factor by which to downsample the mask.

    Returns:
        (np.ndarray): A binary mask of the specified image size with the polygons filled in.
    """
    mask = np.zeros(imgsz, dtype=np.uint8)
    polygons = np.asarray(polygons, dtype=np.int32)
    polygons = polygons.reshape((polygons.shape[0], -1, 2))
    cv2.fillPoly(mask, polygons, color=color)
    nh, nw = (imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio)
    # Note: fillPoly first then resize is trying to keep the same loss calculation method when mask-ratio=1
    return cv2.resize(mask, (nw, nh))


def polygons2masks(imgsz: tuple[int, int], polygons: list[np.ndarray], color: int, downsample_ratio: int = 1) -> np.ndarray:
    """
    Convert a list of polygons to a set of binary masks of the specified image size.

    Args:
        imgsz (tuple[int, int]): The size of the image as (height, width).
        polygons (list[np.ndarray]): A list of polygons. Each polygon is an array with shape (N, M), where
                                     N is the number of polygons, and M is the number of points such that M % 2 = 0.
        color (int): The color value to fill in the polygons on the masks.
        downsample_ratio (int, optional): Factor by which to downsample each mask.

    Returns:
        (np.ndarray): A set of binary masks of the specified image size with the polygons filled in.
    """
    return np.array([polygon2mask(imgsz, [x.reshape(-1)], color, downsample_ratio) for x in polygons])


def polygons2masks_overlap(imgsz: tuple[int, int], segments: list[np.ndarray], downsample_ratio: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Return a (640, 640) overlap mask."""
    masks = np.zeros(
        (imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio),
        dtype=np.int32 if len(segments) > 255 else np.uint8,
    )
    areas = []
    ms = []
    for segment in segments:
        mask = polygon2mask(
            imgsz,
            [segment.reshape(-1)],
            downsample_ratio=downsample_ratio,
            color=1,
        )
        ms.append(mask.astype(masks.dtype))
        areas.append(mask.sum())
    areas = np.asarray(areas)
    index = np.argsort(-areas)
    ms = np.array(ms)[index]
    for i in range(len(segments)):
        mask = ms[i] * (i + 1)
        masks = masks + mask
        masks = np.clip(masks, a_min=0, a_max=i + 1)
    return masks, index


# def random_perspective(im, targets=(), degrees=10, translate=0.1, scale=0.1, shear=10, perspective=0.0):
#     """Applies random perspective, rotation, scaling, and shear transformations."""
#     height, width, _ = im.shape

#     # Center
#     C = np.eye(3)
#     C[0, 2] = -im.shape[1] / 2  # x translation (pixels)
#     C[1, 2] = -im.shape[0] / 2  # y translation (pixels)

#     # Perspective
#     P = np.eye(3)
#     P[2, 0] = random.uniform(-perspective, perspective)  # x perspective
#     P[2, 1] = random.uniform(-perspective, perspective)  # y perspective

#     # Rotation and Scale
#     R = np.eye(3)
#     a = random.uniform(-degrees, degrees)
#     s = random.uniform(1 - scale, 1 + scale)
#     R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

#     # Shear
#     S = np.eye(3)
#     S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear
#     S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear

#     # Translation
#     T = np.eye(3)
#     T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation
#     T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation

#     # Combined rotation matrix
#     M = T @ S @ R @ P @ C
#     if height > 0 and width > 0:
#         im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(114, 114, 114))

#     # Transform label coordinates
#     n = len(targets)
#     if n:
#         xy = np.ones((n * 4, 3))
#         # targets are [class, x1, y1, x2, y2]
#         xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
#         xy = xy @ M.T  # transform

#         # Filter out points behind the camera
#         z = xy[:, 2]
#         z_positive = z > 1e-6  # Threshold to avoid division by zero

#         # Reshape z_positive to match the structure of xy
#         z_positive = z_positive.reshape(n, 4).all(axis=1)

#         targets = targets[z_positive]
#         xy = xy[z_positive.repeat(4)]

#         if not len(targets):
#             return im, targets

#         xy = (xy[:, :2] / xy[:, 2:3]).reshape(-1, 8)  # perspective rescale or divide by z

#         # create new boxes
#         x = xy[:, [0, 2, 4, 6]]
#         y = xy[:, [1, 3, 5, 7]]
#         new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, -1).T

#         # clip
#         new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
#         new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

#         # filter small boxes
#         w = new[:, 2] - new[:, 0]
#         h = new[:, 3] - new[:, 1]
#         area = w * h

#         # You can adjust this area threshold
#         area_threshold = 1000.0
#         i = area > area_threshold

#         targets = targets[i]
#         new = new[i]

#         targets[:, 1:5] = new

#     return im, targets


# def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
#     """HSV color-space augmentation."""
#     if hgain or sgain or vgain:
#         r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
#         hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
#         dtype = im.dtype  # uint8

#         x = np.arange(0, 256, dtype=r.dtype)
#         lut_hue = ((x * r[0]) % 180).astype(dtype)
#         lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
#         lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

#         im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
#         cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)
#     return im


# def albumentations_transforms(hyp):
#     """YOLOv8 style augmentations using albumentations."""
#     return A.Compose(
#         [
#             # # --- Geometric Transformations ---
#             # A.Affine(
#             #     rotate=hyp.get("degrees", 0.0),  # FIX: Changed 'degrees' to 'rotate'
#             #     translate_percent=hyp.get("translate", 0.1),
#             #     scale=(1 - hyp.get("scale", 0.5), 1 + hyp.get("scale", 0.5)),
#             #     shear=hyp.get("shear", 0.0),
#             #     p=0.5,
#             # ),
#             # A.Perspective(scale=(0.0, hyp.get("perspective", 0.0)), p=0.5),
#             # # --- Color Transformations ---
#             # A.ColorJitter(
#             #     saturation=hyp.get("hsv_s", 0.7), hue=hyp.get("hsv_h", 0.015), p=0.5
#             # ),
#             # A.Blur(p=0.01),
#             # A.MedianBlur(p=0.01),
#             A.Blur(p=0.01),
#             A.MedianBlur(p=0.01),
#             A.ToGray(p=0.01),
#             A.CLAHE(p=0.01),
#             A.RandomBrightnessContrast(p=0.0),
#             A.RandomGamma(p=0.0),
#             A.ImageCompression(quality_range=(75, 100), p=0.0),
#             A.HorizontalFlip(p=hyp.get("fliplr", 0.5)),
#         ],
#         bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
#     )
