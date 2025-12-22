import math
import warnings
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch

from vidnn.utils import ops


class Colors:
    """
    color palette for visualization and plotting.

    This class provides methods to work with the color palette, including converting hex color codes to
    RGB values and accessing predefined color schemes for object detection and pose estimation.

    Attributes:
        palette (list[tuple]): List of RGB color tuples for general use.
        n (int): The number of colors in the palette.
        pose_palette (np.ndarray): A specific color palette array for pose estimation with dtype np.uint8.

    ## Color Palette

    | Index | Color                                                             | HEX       | RGB               |
    |-------|-------------------------------------------------------------------|-----------|-------------------|
    | 0     | <i class="fa-solid fa-square fa-2xl" style="color: #042aff;"></i> | `#042aff` | (4, 42, 255)      |
    | 1     | <i class="fa-solid fa-square fa-2xl" style="color: #0bdbeb;"></i> | `#0bdbeb` | (11, 219, 235)    |
    | 2     | <i class="fa-solid fa-square fa-2xl" style="color: #f3f3f3;"></i> | `#f3f3f3` | (243, 243, 243)   |
    | 3     | <i class="fa-solid fa-square fa-2xl" style="color: #00dfb7;"></i> | `#00dfb7` | (0, 223, 183)     |
    | 4     | <i class="fa-solid fa-square fa-2xl" style="color: #111f68;"></i> | `#111f68` | (17, 31, 104)     |
    | 5     | <i class="fa-solid fa-square fa-2xl" style="color: #ff6fdd;"></i> | `#ff6fdd` | (255, 111, 221)   |
    | 6     | <i class="fa-solid fa-square fa-2xl" style="color: #ff444f;"></i> | `#ff444f` | (255, 68, 79)     |
    | 7     | <i class="fa-solid fa-square fa-2xl" style="color: #cced00;"></i> | `#cced00` | (204, 237, 0)     |
    | 8     | <i class="fa-solid fa-square fa-2xl" style="color: #00f344;"></i> | `#00f344` | (0, 243, 68)      |
    | 9     | <i class="fa-solid fa-square fa-2xl" style="color: #bd00ff;"></i> | `#bd00ff` | (189, 0, 255)     |
    | 10    | <i class="fa-solid fa-square fa-2xl" style="color: #00b4ff;"></i> | `#00b4ff` | (0, 180, 255)     |
    | 11    | <i class="fa-solid fa-square fa-2xl" style="color: #dd00ba;"></i> | `#dd00ba` | (221, 0, 186)     |
    | 12    | <i class="fa-solid fa-square fa-2xl" style="color: #00ffff;"></i> | `#00ffff` | (0, 255, 255)     |
    | 13    | <i class="fa-solid fa-square fa-2xl" style="color: #26c000;"></i> | `#26c000` | (38, 192, 0)      |
    | 14    | <i class="fa-solid fa-square fa-2xl" style="color: #01ffb3;"></i> | `#01ffb3` | (1, 255, 179)     |
    | 15    | <i class="fa-solid fa-square fa-2xl" style="color: #7d24ff;"></i> | `#7d24ff` | (125, 36, 255)    |
    | 16    | <i class="fa-solid fa-square fa-2xl" style="color: #7b0068;"></i> | `#7b0068` | (123, 0, 104)     |
    | 17    | <i class="fa-solid fa-square fa-2xl" style="color: #ff1b6c;"></i> | `#ff1b6c` | (255, 27, 108)    |
    | 18    | <i class="fa-solid fa-square fa-2xl" style="color: #fc6d2f;"></i> | `#fc6d2f` | (252, 109, 47)    |
    | 19    | <i class="fa-solid fa-square fa-2xl" style="color: #a2ff0b;"></i> | `#a2ff0b` | (162, 255, 11)    |

    ## Pose Color Palette

    | Index | Color                                                             | HEX       | RGB               |
    |-------|-------------------------------------------------------------------|-----------|-------------------|
    | 0     | <i class="fa-solid fa-square fa-2xl" style="color: #ff8000;"></i> | `#ff8000` | (255, 128, 0)     |
    | 1     | <i class="fa-solid fa-square fa-2xl" style="color: #ff9933;"></i> | `#ff9933` | (255, 153, 51)    |
    | 2     | <i class="fa-solid fa-square fa-2xl" style="color: #ffb266;"></i> | `#ffb266` | (255, 178, 102)   |
    | 3     | <i class="fa-solid fa-square fa-2xl" style="color: #e6e600;"></i> | `#e6e600` | (230, 230, 0)     |
    | 4     | <i class="fa-solid fa-square fa-2xl" style="color: #ff99ff;"></i> | `#ff99ff` | (255, 153, 255)   |
    | 5     | <i class="fa-solid fa-square fa-2xl" style="color: #99ccff;"></i> | `#99ccff` | (153, 204, 255)   |
    | 6     | <i class="fa-solid fa-square fa-2xl" style="color: #ff66ff;"></i> | `#ff66ff` | (255, 102, 255)   |
    | 7     | <i class="fa-solid fa-square fa-2xl" style="color: #ff33ff;"></i> | `#ff33ff` | (255, 51, 255)    |
    | 8     | <i class="fa-solid fa-square fa-2xl" style="color: #66b2ff;"></i> | `#66b2ff` | (102, 178, 255)   |
    | 9     | <i class="fa-solid fa-square fa-2xl" style="color: #3399ff;"></i> | `#3399ff` | (51, 153, 255)    |
    | 10    | <i class="fa-solid fa-square fa-2xl" style="color: #ff9999;"></i> | `#ff9999` | (255, 153, 153)   |
    | 11    | <i class="fa-solid fa-square fa-2xl" style="color: #ff6666;"></i> | `#ff6666` | (255, 102, 102)   |
    | 12    | <i class="fa-solid fa-square fa-2xl" style="color: #ff3333;"></i> | `#ff3333` | (255, 51, 51)     |
    | 13    | <i class="fa-solid fa-square fa-2xl" style="color: #99ff99;"></i> | `#99ff99` | (153, 255, 153)   |
    | 14    | <i class="fa-solid fa-square fa-2xl" style="color: #66ff66;"></i> | `#66ff66` | (102, 255, 102)   |
    | 15    | <i class="fa-solid fa-square fa-2xl" style="color: #33ff33;"></i> | `#33ff33` | (51, 255, 51)     |
    | 16    | <i class="fa-solid fa-square fa-2xl" style="color: #00ff00;"></i> | `#00ff00` | (0, 255, 0)       |
    | 17    | <i class="fa-solid fa-square fa-2xl" style="color: #0000ff;"></i> | `#0000ff` | (0, 0, 255)       |
    | 18    | <i class="fa-solid fa-square fa-2xl" style="color: #ff0000;"></i> | `#ff0000` | (255, 0, 0)       |
    | 19    | <i class="fa-solid fa-square fa-2xl" style="color: #ffffff;"></i> | `#ffffff` | (255, 255, 255)   |
    """

    def __init__(self):
        """Initialize colors as hex = matplotlib.colors.TABLEAU_COLORS.values()."""
        hexs = (
            "042AFF",
            "0BDBEB",
            "F3F3F3",
            "00DFB7",
            "111F68",
            "FF6FDD",
            "FF444F",
            "CCED00",
            "00F344",
            "BD00FF",
            "00B4FF",
            "DD00BA",
            "00FFFF",
            "26C000",
            "01FFB3",
            "7D24FF",
            "7B0068",
            "FF1B6C",
            "FC6D2F",
            "A2FF0B",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)
        self.pose_palette = np.array(
            [
                [255, 128, 0],
                [255, 153, 51],
                [255, 178, 102],
                [230, 230, 0],
                [255, 153, 255],
                [153, 204, 255],
                [255, 102, 255],
                [255, 51, 255],
                [102, 178, 255],
                [51, 153, 255],
                [255, 153, 153],
                [255, 102, 102],
                [255, 51, 51],
                [153, 255, 153],
                [102, 255, 102],
                [51, 255, 51],
                [0, 255, 0],
                [0, 0, 255],
                [255, 0, 0],
                [255, 255, 255],
            ],
            dtype=np.uint8,
        )

    def __call__(self, i: int | torch.Tensor, bgr: bool = False) -> tuple:
        """
        Convert hex color codes to RGB values.

        Args:
            i (int | torch.Tensor): Color index.
            bgr (bool, optional): Whether to return BGR format instead of RGB.

        Returns:
            (tuple): RGB or BGR color tuple.
        """
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h: str) -> tuple:
        """Convert hex color codes to RGB values (i.e. default PIL order)."""
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'


class Annotator:
    """
    Annotator for train/val mosaics and JPGs and predictions annotations.

    Attributes:
        im (np.ndarray): The image to annotate.
        font (ImageFont.truetype | ImageFont.load_default): Font used for text annotations.
        lw (float): Line width for drawing.
        skeleton (list[list[int]]): Skeleton structure for keypoints.
        limb_color (list[int]): Color palette for limbs.
        kpt_color (list[int]): Color palette for keypoints.
        dark_colors (set): Set of colors considered dark for text contrast.
        light_colors (set): Set of colors considered light for text contrast.
    """

    def __init__(
        self,
        im,
        line_width: int | None = None,
        font_size: int | None = None,
        font: str = "Arial.ttf",
    ):
        assert isinstance(im, np.ndarray), "Image not numpy array. Use OpenCV images."
        assert im.data.contiguous, "Image not contiguous. Apply np.ascontiguousarray(im) to Annotator input images."

        self.lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)

        if im.shape[2] == 1:  # handle grayscale
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        elif im.shape[2] > 3:  # multispectral
            im = np.ascontiguousarray(im[..., :3])

        self.im = im if im.flags.writeable else im.copy()
        self.tf = max(self.lw - 1, 1)  # font thickness
        self.sf = self.lw / 3  # font scale

        # Pose
        self.skeleton = [
            [16, 14],
            [14, 12],
            [17, 15],
            [15, 13],
            [12, 13],
            [6, 12],
            [7, 13],
            [6, 7],
            [6, 8],
            [7, 9],
            [8, 10],
            [9, 11],
            [2, 3],
            [1, 2],
            [1, 3],
            [2, 4],
            [3, 5],
            [4, 6],
            [5, 7],
        ]

        self.limb_color = colors.pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
        self.kpt_color = colors.pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
        self.dark_colors = {
            (235, 219, 11),
            (243, 243, 243),
            (183, 223, 0),
            (221, 111, 255),
            (0, 237, 204),
            (68, 243, 0),
            (255, 255, 0),
            (179, 255, 1),
            (11, 255, 162),
        }
        self.light_colors = {
            (255, 42, 4),
            (79, 68, 255),
            (255, 0, 189),
            (255, 180, 0),
            (186, 0, 221),
            (0, 192, 38),
            (255, 36, 125),
            (104, 0, 123),
            (108, 27, 255),
            (47, 109, 252),
            (104, 31, 17),
        }

    def get_txt_color(self, color: tuple = (128, 128, 128), txt_color: tuple = (255, 255, 255)) -> tuple:
        """
        Assign text color based on background color.

        Args:
            color (tuple, optional): The background color of the rectangle for text (B, G, R).
            txt_color (tuple, optional): The color of the text (R, G, B).

        Returns:
            (tuple): Text color for label.

        Examples:
            >>> from vidnn.utils.plotting import Annotator
            >>> im0 = cv2.imread("test.png")
            >>> annotator = Annotator(im0, line_width=10)
            >>> annotator.get_txt_color(color=(104, 31, 17))  # return (255, 255, 255)
        """
        if color in self.dark_colors:
            return 104, 31, 17
        elif color in self.light_colors:
            return 255, 255, 255
        else:
            return txt_color

    def box_label(self, box, label: str = "", color: tuple = (128, 128, 128), txt_color: tuple = (255, 255, 255)):
        """
        Draw a bounding box on an image with a given label.

        Args:
            box (tuple): The bounding box coordinates (x1, y1, x2, y2).
            label (str, optional): The text label to be displayed.
            color (tuple, optional): The background color of the rectangle (B, G, R).
            txt_color (tuple, optional): The color of the text (R, G, B).

        Examples:
            >>> from ultralytics.utils.plotting import Annotator
            >>> im0 = cv2.imread("test.png")
            >>> annotator = Annotator(im0, line_width=10)
            >>> annotator.box_label(box=[10, 20, 30, 40], label="person")
        """
        txt_color = self.get_txt_color(color, txt_color)
        if isinstance(box, torch.Tensor):
            box = box.tolist()

        multi_points = isinstance(box[0], list)  # multiple points with shape (n, 2)
        p1 = [int(b) for b in box[0]] if multi_points else (int(box[0]), int(box[1]))

        (
            cv2.polylines(self.im, [np.asarray(box, dtype=int)], True, color, self.lw)
            if multi_points
            else cv2.rectangle(self.im, p1, (int(box[2]), int(box[3])), color, thickness=self.lw, lineType=cv2.LINE_AA)
        )
        if label:
            w, h = cv2.getTextSize(label, 0, fontScale=self.sf, thickness=self.tf)[0]  # text width, height
            h += 3  # add pixels to pad text
            outside = p1[1] >= h  # label fits outside box
            if p1[0] > self.im.shape[1] - w:  # shape is (h, w), check if label extend beyond right side of image
                p1 = self.im.shape[1] - w, p1[1]
            p2 = p1[0] + w, p1[1] - h if outside else p1[1] + h
            cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(
                self.im,
                label,
                (p1[0], p1[1] - 2 if outside else p1[1] + h - 1),
                0,
                self.sf,
                txt_color,
                thickness=self.tf,
                lineType=cv2.LINE_AA,
            )

    def masks(self, masks, colors, im_gpu: torch.Tensor = None, alpha: float = 0.5, retina_masks: bool = False):
        """Plot masks on image.

        Args:
            masks (torch.Tensor | np.ndarray): Predicted masks with shape: [n, h, w]
            colors (list[list[int]]): Colors for predicted masks, [[r, g, b] * n]
            im_gpu (torch.Tensor | None): Image is in cuda, shape: [3, h, w], range: [0, 1]
            alpha (float, optional): Mask transparency: 0.0 fully transparent, 1.0 opaque.
            retina_masks (bool, optional): Whether to use high resolution masks or not.
        """
        if im_gpu is None:
            assert isinstance(masks, np.ndarray), "`masks` must be a np.ndarray if `im_gpu` is not provided."
            overlay = self.im.copy()
            for i, mask in enumerate(masks):
                overlay[mask.astype(bool)] = colors[i]
            self.im = cv2.addWeighted(self.im, 1 - alpha, overlay, alpha, 0)
        else:
            assert isinstance(masks, torch.Tensor), "'masks' must be a torch.Tensor if 'im_gpu' is provided."
            if len(masks) == 0:
                self.im[:] = im_gpu.permute(1, 2, 0).contiguous().cpu().numpy() * 255
                return
            if im_gpu.device != masks.device:
                im_gpu = im_gpu.to(masks.device)

            ih, iw = self.im.shape[:2]
            if not retina_masks:
                # Use scale_masks to properly remove padding and upsample, convert bool to float first
                masks = ops.scale_masks(masks[None].float(), (ih, iw))[0] > 0.5
                # Convert original BGR image to RGB tensor
                im_gpu = torch.from_numpy(self.im).to(masks.device).permute(2, 0, 1).flip(0).contiguous().float() / 255.0

            colors = torch.tensor(colors, device=masks.device, dtype=torch.float32) / 255.0  # shape(n,3)
            colors = colors[:, None, None]  # shape(n,1,1,3)
            masks = masks.unsqueeze(3)  # shape(n,h,w,1)
            masks_color = masks * (colors * alpha)  # shape(n,h,w,3)
            inv_alpha_masks = (1 - masks * alpha).cumprod(0)  # shape(n,h,w,1)
            mcs = masks_color.max(dim=0).values  # shape(n,h,w,3)

            im_gpu = im_gpu.flip(dims=[0]).permute(1, 2, 0).contiguous()  # shape(h,w,3)
            im_gpu = im_gpu * inv_alpha_masks[-1] + mcs
            self.im[:] = (im_gpu * 255).byte().cpu().numpy()

    def kpts(
        self,
        kpts,
        shape: tuple = (640, 640),
        radius: int | None = None,
        kpt_line: bool = True,
        conf_thres: float = 0.25,
        kpt_color: tuple | None = None,
    ):
        """Plot keypoints on the image.

        Args:
            kpts (torch.Tensor): Keypoints, shape [17, 3] (x, y, confidence).
            shape (tuple, optional): Image shape (h, w).
            radius (int, optional): Keypoint radius.
            kpt_line (bool, optional): Draw lines between keypoints.
            conf_thres (float, optional): Confidence threshold.
            kpt_color (tuple, optional): Keypoint color (B, G, R).

        Notes:
            - `kpt_line=True` currently only supports human pose plotting.
            - Modifies self.im in-place.
            - If self.pil is True, converts image to numpy array and back to PIL.
        """
        radius = radius if radius is not None else self.lw
        nkpt, ndim = kpts.shape
        is_pose = nkpt == 17 and ndim in {2, 3}
        kpt_line &= is_pose  # `kpt_line=True` for now only supports human pose plotting
        for i, k in enumerate(kpts):
            color_k = kpt_color or (self.kpt_color[i].tolist() if is_pose else colors(i))
            x_coord, y_coord = k[0], k[1]
            if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
                if len(k) == 3:
                    conf = k[2]
                    if conf < conf_thres:
                        continue
                cv2.circle(self.im, (int(x_coord), int(y_coord)), radius, color_k, -1, lineType=cv2.LINE_AA)

        if kpt_line:
            ndim = kpts.shape[-1]
            for i, sk in enumerate(self.skeleton):
                pos1 = (int(kpts[(sk[0] - 1), 0]), int(kpts[(sk[0] - 1), 1]))
                pos2 = (int(kpts[(sk[1] - 1), 0]), int(kpts[(sk[1] - 1), 1]))
                if ndim == 3:
                    conf1 = kpts[(sk[0] - 1), 2]
                    conf2 = kpts[(sk[1] - 1), 2]
                    if conf1 < conf_thres or conf2 < conf_thres:
                        continue
                if pos1[0] % shape[1] == 0 or pos1[1] % shape[0] == 0 or pos1[0] < 0 or pos1[1] < 0:
                    continue
                if pos2[0] % shape[1] == 0 or pos2[1] % shape[0] == 0 or pos2[0] < 0 or pos2[1] < 0:
                    continue
                cv2.line(
                    self.im,
                    pos1,
                    pos2,
                    kpt_color or self.limb_color[i].tolist(),
                    thickness=int(np.ceil(self.lw / 2)),
                    lineType=cv2.LINE_AA,
                )

    def result(self):
        """Return annotated image as array."""
        return np.asarray(self.im)
