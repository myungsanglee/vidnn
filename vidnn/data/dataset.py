import glob
import cv2
import numpy as np
import torch
import random
import os
import math
from copy import deepcopy
from tqdm import tqdm
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from torch.utils.data import Dataset
import albumentations as A

from vidnn.utils import LOGGER, NUM_THREADS, LOCAL_RANK
from vidnn.data.utils import (
    IMG_FORMATS,
    verify_image_label,
    get_hash,
    save_dataset_cache_file,
    load_dataset_cache_file,
    polygons2masks_overlap,
    polygons2masks,
)
from vidnn.utils.ops import resample_segments, xywh2xyxy, xyxy2xywh, segment2box, xyxyxyxy2xywhr

DATASET_CACHE_VERSION = "1.0.0"


class YoloDataset(Dataset):
    def __init__(self, img_path, imgsz=640, augment=False, cfg={}, bgr=0.0, prefix=""):
        self.img_path = img_path
        self.imgsz = imgsz
        self.augment = augment
        self.cfg = cfg
        self.bgr = bgr
        self.prefix = prefix
        task = cfg["task"]
        self.use_segments = task == "segment"
        self.use_keypoints = task == "pose"
        self.use_obb = task == "obb"
        assert not (self.use_segments and self.use_keypoints), "Can not use both segments and keypoints."
        self.single_cls = self.cfg.get("single_cls", False)
        self.img_files = self.get_img_files(img_path)
        self.label_files = [x.replace("images", "labels").rsplit(".", 1)[0] + ".txt" for x in self.img_files]
        self.labels = self.get_labels()
        self.indices = range(len(self.labels))
        self.ni = len(self.labels)  # number of images
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
        self.flip_idx = self.cfg.get("flip_idx", [])
        if self.use_keypoints:
            kpt_shape = self.cfg.get("kpt_shape", None)
            if len(self.flip_idx) == 0 and (self.cfg["fliplr"] > 0.0 or self.cfg["flipud"] > 0.0):
                self.cfg["fliplr"] = self.cfg["flipud"] = 0.0  # both fliplr and flipud require flip_idx
                LOGGER.warning("No 'flip_idx' array defined in yaml file, disabling 'fliplr' and 'flipud' augmentations.")
            elif self.flip_idx and (len(self.flip_idx) != kpt_shape[0]):
                raise ValueError(f"data flip_idx={self.flip_idx} length must be equal to kpt_shape[0]={kpt_shape[0]}")

        self.mask_ratio = self.cfg["mask_ratio"]
        self.overlap_mask = self.cfg["overlap_mask"]

        # Buffer thread for mosaic images
        self.buffer = []  # buffer size = batch size
        self.max_buffer_length = min((self.ni, cfg["batch_size"] * 8, 1000)) if self.augment else 0
        self.ims, self.im_hw0, self.im_hw = [None] * self.ni, [None] * self.ni, [None] * self.ni

    def __len__(self):
        return len(self.labels)

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
        segments = label.pop("segments")
        keypoints = label.pop("keypoints")
        bbox_format = label.pop("bbox_format")
        normalized = label.pop("normalized")
        if bbox_format == "xyxy":
            # bboxes = xyxy2xywhn(bboxes, w=w, h=h, clip=True) if label["bbox_format"] == "xyxy" else bboxes
            # convert bbox
            bboxes = xyxy2xywh(bboxes)
            normalized = False
        elif bbox_format == "xywh":
            # denormalize
            bboxes[:, 0] *= w
            bboxes[:, 1] *= h
            bboxes[:, 2] *= w
            bboxes[:, 3] *= h
            segments[..., 0] *= w
            segments[..., 1] *= h
            if keypoints is not None:
                keypoints[..., 0] *= w
                keypoints[..., 1] *= h
            normalized = False
        nl = len(bboxes)

        if self.use_segments:
            if nl:
                if self.overlap_mask:
                    masks, sorted_idx = polygons2masks_overlap((h, w), segments, downsample_ratio=self.mask_ratio)
                    masks = masks[None]  # (640, 640) -> (1, 640, 640)
                    segments = segments[sorted_idx] if len(segments) else segments
                    keypoints = keypoints[sorted_idx] if keypoints is not None else None
                    bboxes = bboxes[sorted_idx]
                    cls = cls[sorted_idx]
                else:
                    masks = polygons2masks((h, w), segments, color=1, downsample_ratio=self.mask_ratio)
                masks = torch.from_numpy(masks)
            else:
                masks = torch.zeros(1 if self.overlap_mask else nl, img.shape[0] // self.mask_ratio, img.shape[1] // self.mask_ratio)
            label["masks"] = masks

        if len(img.shape) < 3:
            img = np.expand_dims(img, -1)
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img[::-1] if random.uniform(0, 1) > self.bgr and img.shape[0] == 3 else img)
        img = torch.from_numpy(img)
        label["img"] = img
        label["cls"] = torch.from_numpy(cls) if nl else torch.zeros(nl)
        label["bboxes"] = torch.from_numpy(bboxes) if nl else torch.zeros((nl, 4))

        if self.use_keypoints:
            label["keypoints"] = torch.empty(0, 3) if keypoints is None else torch.from_numpy(keypoints)
            if not normalized:
                label["keypoints"][..., 0] /= w
                label["keypoints"][..., 1] /= h

        if self.use_obb:
            label["bboxes"] = xyxyxyxy2xywhr(torch.from_numpy(segments)) if len(segments) else torch.zeros((0, 5))

        if not normalized:
            label["bboxes"][:, [0, 2]] /= w
            label["bboxes"][:, [1, 3]] /= h

        label["batch_idx"] = torch.zeros(nl)

        return label

    def cache_labels(self, path: Path = Path("./labels.cache")):
        """
        Cache dataset labels, check images and read shapes.

        Args:
            path (Path): Path where to save the cache file.

        Returns:
            (dict): Dictionary containing cached labels and related information.
        """
        x = {"labels": []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        total = len(self.img_files)
        nkpt, ndim = self.cfg.get("kpt_shape", (0, 0))
        if self.use_keypoints and (nkpt <= 0 or ndim not in {2, 3}):
            raise ValueError(
                "'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'"
            )
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                func=verify_image_label,
                iterable=zip(
                    self.img_files,
                    self.label_files,
                    repeat(self.prefix),
                    repeat(self.use_keypoints),
                    repeat(len(self.cfg["names"])),
                    repeat(nkpt),
                    repeat(ndim),
                    repeat(self.single_cls),
                ),
            )
            pbar = tqdm(results, desc=desc, total=total)
            for im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x["labels"].append(
                        {
                            "im_file": im_file,
                            "shape": shape,
                            "cls": lb[:, 0:1],  # n, 1
                            "bboxes": lb[:, 1:],  # n, 4
                            "segments": segments,
                            "keypoints": keypoint,
                            "normalized": True,
                            "bbox_format": "xywh",
                        }
                    )
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"{self.prefix}No labels found in {path}.")
        x["hash"] = get_hash(self.label_files + self.img_files)
        x["results"] = nf, nm, ne, nc, len(self.img_files)
        x["msgs"] = msgs  # warnings
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    def get_labels(self):
        """
        Return dictionary of labels for YOLO training.

        This method loads labels from disk or cache, verifies their integrity, and prepares them for training.

        Returns:
            (List[dict]): List of label dictionaries, each containing information about an image and its annotations.
        """
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")
        try:
            cache, exists = load_dataset_cache_file(cache_path), True  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash(self.label_files + self.img_files)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            tqdm(None, desc=self.prefix + d, total=n, initial=n)  # display results
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))  # display warnings

        # Read cache
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        labels = cache["labels"]
        if not labels:
            raise RuntimeError(f"No valid images found in {cache_path}. Images with incorrectly formatted labels are ignored.")
        self.im_files = [lb["im_file"] for lb in labels]  # update im_files

        # Check if the dataset is all boxes or all segments
        lengths = ((len(lb["cls"]), len(lb["bboxes"]), len(lb["segments"])) for lb in labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f"Box and segment counts should be equal, but got len(segments) = {len_segments}, "
                f"len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. "
                "To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset."
            )
            for lb in labels:
                lb["segments"] = []
        if len_cls == 0:
            LOGGER.warning(f"Labels are missing or empty in {cache_path}, training may not work correctly.")
        return labels

    def get_img_files(self, img_path):
        try:
            f = []  # image files
            for p in img_path if isinstance(img_path, list) else [img_path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / "**" / "*.*"), recursive=True)
                    # F = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p, encoding="utf-8") as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace("./", parent) if x.startswith("./") else x for x in t]  # local to global path
                        # F += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise FileNotFoundError(f"{self.prefix}{p} does not exist")
            img_files = sorted(x.replace("/", os.sep) for x in f if x.rpartition(".")[-1].lower() in IMG_FORMATS)
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert img_files, f"{self.prefix}No images found in {img_path}."
        except Exception as e:
            raise FileNotFoundError(f"{self.prefix}Error loading data from {img_path}") from e
        return img_files

    def get_image_and_label(self, index):
        """
        Get and return label information from the dataset.

        Args:
            index (int): Index of the image to retrieve.

        Returns:
            (Dict[str, Any]): Label dictionary with image and metadata.
        """
        label = deepcopy(self.labels[index])
        label.pop("shape", None)  # shape is for rect, remove it
        # get image
        label["img"], label["ori_shape"], label["resized_shape"] = self.load_image(index)
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )  # for evaluation

        # update labels info
        segments = label.pop("segments", [])
        keypoints = label.pop("keypoints", None)

        # NOTE: do NOT resample oriented boxes
        segment_resamples = 100 if self.use_obb else 1000
        if len(segments) > 0:
            # make sure segments interpolate correctly if original length is greater than segment_resamples
            max_len = max(len(s) for s in segments)
            segment_resamples = (max_len + 1) if segment_resamples < max_len else segment_resamples
            # list[np.array(segment_resamples, 2)] * num_samples
            segments = np.stack(resample_segments(segments, n=segment_resamples), axis=0)
        else:
            segments = np.zeros((0, segment_resamples, 2), dtype=np.float32)
        label["segments"] = segments
        label["keypoints"] = keypoints

        return label

    def load_image(self, i):
        im, filename = self.ims[i], self.img_files[i]
        if im is None:
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

            # Add to buffer if training with augmentations
            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
                self.buffer.append(i)
                if 1 < len(self.buffer) >= self.max_buffer_length:  # prevent empty buffer
                    j = self.buffer.pop(0)
                    self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

            return im, (h0, w0), im.shape[:2]

        return self.ims[i], self.im_hw0[i], self.im_hw[i]

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

    def close_mosaic(self):
        self.cfg["mosaic"] = 0.0

    def _mosaic4(self, label):
        cls4 = []
        bboxes4 = []
        segments4 = []
        keypoints4 = []
        s = self.imgsz
        border = (-self.imgsz // 2, -self.imgsz // 2)  # width, height
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in border)  # mosaic center x, y
        # indices = [-1] + random.choices(self.indices, k=3)
        indices = [-1] + random.choices(list(self.buffer), k=3)
        for i, idx in enumerate(indices):
            labels_patch = label if idx == -1 else self.get_image_and_label(idx)
            # Load image
            img = labels_patch["img"]
            h, w = labels_patch.pop("resized_shape")
            cls = labels_patch["cls"]
            bboxes = labels_patch["bboxes"]
            segments = labels_patch["segments"]
            keypoints = labels_patch["keypoints"]

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
                # convert bbox
                bboxes = xywh2xyxy(bboxes)
                # denormalize
                # denormalize
                bboxes[:, 0] *= w
                bboxes[:, 1] *= h
                bboxes[:, 2] *= w
                bboxes[:, 3] *= h
                segments[..., 0] *= w
                segments[..., 1] *= h
                if keypoints is not None:
                    keypoints[..., 0] *= w
                    keypoints[..., 1] *= h
                # add padding
                bboxes[:, 0] += padw
                bboxes[:, 1] += padh
                bboxes[:, 2] += padw
                bboxes[:, 3] += padh
                segments[..., 0] += padw
                segments[..., 1] += padh
                if keypoints is not None:
                    keypoints[..., 0] += padw
                    keypoints[..., 1] += padh

                # bboxes = xywhn2xyxy(bboxes, w, h, padw, padh)
                # labels[:, 1:5] += (padw, padh, padw, padh)
            bboxes4.append(bboxes)
            cls4.append(cls)
            segments4.append(segments)
            if keypoints is not None:
                keypoints4.append(keypoints)

        # concatenate
        cls4 = np.concatenate(cls4, 0)
        bboxes4 = np.concatenate(bboxes4, 0)
        seg_len = [seg.shape[1] for seg in segments4]
        if len(frozenset(seg_len)) > 1:  # resample segments if there's different length
            max_len = max(seg_len)
            segments4 = np.concatenate(
                [
                    (
                        resample_segments(list(seg), max_len) if len(seg) else np.zeros((0, max_len, 2), dtype=np.float32)
                    )  # re-generating empty segments
                    for seg in segments4
                ],
                axis=0,
            )
        else:
            segments4 = np.concatenate(segments4, axis=0)
        keypoints4 = np.concatenate(keypoints, axis=0) if len(keypoints4) > 0 else None

        # clip
        np.clip(bboxes4, 0, 2 * s, out=bboxes4)
        np.clip(segments4, 0, 2 * s, out=segments4)
        if keypoints4 is not None:
            # Set out of bounds visibility to zero
            keypoints4[..., 2][
                (keypoints4[..., 0] < 0) | (keypoints4[..., 0] > 2 * s) | (keypoints4[..., 1] < 0) | (keypoints4[..., 1] > 2 * s)
            ] = 0.0
            np.clip(keypoints4, 0, 2 * s, out=keypoints4)

        # update label
        label["img"] = img4
        label["cls"] = cls4
        label["bboxes"] = bboxes4
        label["segments"] = segments4
        label["keypoints"] = keypoints4
        label["bbox_format"] = "xyxy"
        label["resized_shape"] = (self.imgsz * 2, self.imgsz * 2)
        label["mosaic_border"] = border
        label["normalized"] = False

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

        # update label
        label["img"] = img
        label["resized_shape"] = new_shape
        bboxes = label["bboxes"]
        segments = label["segments"]
        keypoints = label["keypoints"]
        if len(bboxes) and label["bbox_format"] == "xywh":
            # convert bbox
            bboxes = xywh2xyxy(bboxes)
            # denormalize
            bboxes[:, 0] *= w
            bboxes[:, 1] *= h
            bboxes[:, 2] *= w
            bboxes[:, 3] *= h
            segments[..., 0] *= w
            segments[..., 1] *= h
            if keypoints is not None:
                keypoints[..., 0] *= w
                keypoints[..., 1] *= h
            # scale
            scale_w, scale_h = ratio
            bboxes[:, 0] *= scale_w
            bboxes[:, 1] *= scale_h
            bboxes[:, 2] *= scale_w
            bboxes[:, 3] *= scale_h
            segments[..., 0] *= scale_w
            segments[..., 1] *= scale_h
            if keypoints is not None:
                keypoints[..., 0] *= scale_w
                keypoints[..., 1] *= scale_h
            # add padding
            bboxes[:, 0] += left
            bboxes[:, 1] += top
            bboxes[:, 2] += left
            bboxes[:, 3] += top
            segments[..., 0] += left
            segments[..., 1] += top
            if keypoints is not None:
                keypoints[..., 0] += left
                keypoints[..., 1] += top

            # bboxes = xywhn2xyxy(bboxes, w, h, left, top)
            label["bboxes"] = bboxes
            label["bbox_format"] = "xyxy"
            label["normalized"] = False
        return label

    def _random_perspective(self, label, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0):
        if "mosaic_border" not in label:
            label = self._letterbox(label, new_shape=self.imgsz)
        img = label["img"]
        cls = label["cls"]
        bboxes = label["bboxes"]
        segments = label["segments"]
        keypoints = label["keypoints"]
        if label["bbox_format"] == "xywh":
            # convert bbox
            bboxes = xywh2xyxy(bboxes)
            # denormalize
            w, h = img.shape[:2][::-1]
            bboxes[:, 0] *= w
            bboxes[:, 1] *= h
            bboxes[:, 2] *= w
            bboxes[:, 3] *= h
            segments[..., 0] *= w
            segments[..., 1] *= h
            if keypoints is not None:
                keypoints[..., 0] *= w
                keypoints[..., 1] *= h
            # bboxes = xywhn2xyxy(bboxes, img.shape[1], img.shape[0])
            label["bbox_format"] == "xyxy"
            label["normalized"] = False
        border = label.pop("mosaic_border", (0, 0))
        size = img.shape[1] + border[1] * 2, img.shape[0] + border[0] * 2  # w, h

        # affine transform
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
        S = np.eye(3, dtype=np.float32)
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

        # apply segments
        # Update bboxes if there are segments.
        if len(segments):
            n, num = segments.shape[:2]
            if n == 0:
                new_bboxes = []
                segments = segments
            else:
                xy = np.ones((n * num, 3), dtype=segments.dtype)
                segments = segments.reshape(-1, 2)
                xy[:, :2] = segments
                xy = xy @ M.T  # transform
                xy = xy[:, :2] / xy[:, 2:3]
                segments = xy.reshape(n, -1, 2)
                new_bboxes = np.stack([segment2box(xy, size[0], size[1]) for xy in segments], 0)
                segments[..., 0] = segments[..., 0].clip(new_bboxes[:, 0:1], new_bboxes[:, 2:3])
                segments[..., 1] = segments[..., 1].clip(new_bboxes[:, 1:2], new_bboxes[:, 3:4])

        # apply keypoints
        if keypoints is not None:
            n, nkpt = keypoints.shape[:2]
            if n == 0:
                keypoints = keypoints
            else:
                xy = np.ones((n * nkpt, 3), dtype=keypoints.dtype)
                visible = keypoints[..., 2].reshape(n * nkpt, 1)
                xy[:, :2] = keypoints[..., :2].reshape(n * nkpt, 2)
                xy = xy @ M.T  # transform
                xy = xy[:, :2] / xy[:, 2:3]  # perspective rescale or affine
                out_mask = (xy[:, 0] < 0) | (xy[:, 1] < 0) | (xy[:, 0] > size[0]) | (xy[:, 1] > size[1])
                visible[out_mask] = 0
                keypoints = np.concatenate([xy, visible], axis=-1).reshape(n, nkpt, 3)

        # clip
        new_bboxes[:, [0, 2]] = new_bboxes[:, [0, 2]].clip(0, size[0])
        new_bboxes[:, [1, 3]] = new_bboxes[:, [1, 3]].clip(0, size[1])
        segments[..., 0] = segments[..., 0].clip(0, size[0])
        segments[..., 1] = segments[..., 1].clip(0, size[1])
        if keypoints is not None:
            # Set out of bounds visibility to zero
            keypoints[..., 2][(keypoints[..., 0] < 0) | (keypoints[..., 0] > size[0]) | (keypoints[..., 1] < 0) | (keypoints[..., 1] > size[1])] = 0.0
            keypoints[..., 0] = keypoints[..., 0].clip(0, size[0])
            keypoints[..., 1] = keypoints[..., 1].clip(0, size[1])

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
        area_thr = 0.01 if len(segments) else 0.10
        eps = 1e-16
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
        i = (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates

        label["img"] = img
        label["cls"] = cls[i]
        label["bboxes"] = new_bboxes[i]
        label["segments"] = segments[i] if len(segments) else segments
        label["keypoints"] = keypoints[i] if keypoints is not None else None
        label["resized_shape"] = img.shape[:2]
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
        h, w = img.shape[:2]
        bboxes = label["bboxes"]
        segments = label["segments"]
        keypoints = label["keypoints"]
        if label["bbox_format"] == "xywh":
            # convert bbox
            bboxes = xywh2xyxy(bboxes)
            # denormalize
            bboxes[:, 0] *= w
            bboxes[:, 1] *= h
            bboxes[:, 2] *= w
            bboxes[:, 3] *= h
            segments[..., 0] *= w
            segments[..., 1] *= h
            if keypoints is not None:
                keypoints[..., 0] *= w
                keypoints[..., 1] *= h
            label["bbox_format"] == "xyxy"
            label["normalized"] = False
            # bboxes = xywhn2xyxy(bboxes, img.shape[1], img.shape[0])

        # flip coordinates horizontally
        img = np.fliplr(img)
        x1 = bboxes[:, 0].copy()
        x2 = bboxes[:, 2].copy()
        bboxes[:, 0] = w - x2
        bboxes[:, 2] = w - x1
        segments[..., 0] = w - segments[..., 0]
        if keypoints is not None:
            keypoints[..., 0] = w - keypoints[..., 0]
            if self.flip_idx is not None:
                keypoints = np.ascontiguousarray(keypoints[:, self.flip_idx, :])

        label["img"] = np.ascontiguousarray(img)
        label["bboxes"] = bboxes
        label["segments"] = segments
        label["keypoints"] = keypoints
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
            if k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"}:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i  # add target image index for build_targets()
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch
