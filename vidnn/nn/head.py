import math
import torch
import torch.nn as nn

from .conv import Conv
from .block import DFL
from vidnn.utils.tal import dist2bbox, dist2rbox, make_anchors


class DetectHeadV1(nn.Module):
    """
    Detect head for object detection models.

    This class implements the detection head used in YOLO models for predicting bounding boxes and class probabilities.
    It supports both training and inference modes

    Attributes:
        max_det (int): Maximum detections per image.
        shape (tuple): Input shape.
        anchors (torch.Tensor): Anchor points.
        strides (torch.Tensor): Feature map strides.
        num_classes (int): Number of classes.
        num_heads (int): Number of detection layers.
        reg_max (int): DFL channels.
        num_outputs (int): Number of outputs per anchor.
        stride (torch.Tensor): Strides computed during build.
        cv2 (nn.ModuleList): Convolution layers for box regression.
        cv3 (nn.ModuleList): Convolution layers for classification.
        dfl (nn.Module): Distribution Focal Loss layer.

    Methods:
        forward: Perform forward pass and return predictions.
        bias_init: Initialize detection head biases.
        decode_bboxes: Decode bounding boxes from predictions.
    """

    max_det = 300  # max_det
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, num_classes, in_channels, reg_max=16):  # in_channels: [P3_넥_채널, P4_넥_채널, P5_넥_채널]
        """
        Initialize the detection layer with specified number of classes and channels.

        Args:
            num_classes (int): Number of classes.
            in_channels (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_heads = len(in_channels)  # number of detection layers 3개의 출력 스케일 (P3, P4, P5)
        self.reg_max = reg_max  # DFL 분포의 최대 범위 (보통 16)
        self.num_outputs = num_classes + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.num_heads)  # strides computed during build
        c2, c3 = max((16, in_channels[0] // 4, self.reg_max * 4)), max(in_channels[0], min(self.num_classes, 100))  # channels
        self.cv2 = nn.ModuleList(nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in in_channels)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.num_classes, 1)) for x in in_channels)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):  # x: [P3_넥_출력, P4_넥_출력, P5_넥_출력]
        for i in range(self.num_heads):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return x
        y = self._inference(x)
        return (y, x)

    def _inference(self, x):
        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.num_outputs, -1) for xi in x], 2)
        if self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        box, cls = x_cat.split((self.reg_max * 4, self.num_classes), 1)

        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        return torch.cat((dbox, cls.sigmoid()), 1)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.num_classes] = math.log(5 / m.num_classes / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes, anchors, xywh=True):
        """Decode bounding boxes from predictions."""
        return dist2bbox(bboxes, anchors, xywh=xywh, dim=1)


class OBB(DetectHeadV1):
    """
    OBB detection head for detection with rotation models.

    This class extends the Detect head to include oriented bounding box prediction with rotation angles.

    Attributes:
        num_extra (int): Number of extra parameters.
        cv4 (nn.ModuleList): Convolution layers for angle prediction.
        angle (torch.Tensor): Predicted rotation angles.

    Methods:
        forward: Concatenate and return predicted bounding boxes and class probabilities.
        decode_bboxes: Decode rotated bounding boxes.
    """

    def __init__(self, num_classes=80, num_extra=1, in_channels=()):
        """
        Initialize OBB with number of classes `nc` and layer channels `ch`.

        Args:
            num_classes (int): Number of classes.
            num_extra (int): Number of extra parameters.
            in_channels (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(num_classes, in_channels)
        self.num_extra = num_extra  # number of extra parameters

        c4 = max(in_channels[0] // 4, self.num_extra)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.num_extra, 1)) for x in in_channels)

    def forward(self, x):
        """Concatenate and return predicted bounding boxes and class probabilities."""
        bs = x[0].shape[0]  # batch size
        angle = torch.cat([self.cv4[i](x[i]).view(bs, self.num_extra, -1) for i in range(self.num_heads)], 2)  # OBB theta logits
        # NOTE: set `angle` as an attribute so that `decode_bboxes` could use it.
        angle = (angle.sigmoid() - 0.25) * math.pi  # [-pi/4, 3pi/4]
        # angle = angle.sigmoid() * math.pi / 2  # [0, pi/2]
        if not self.training:
            self.angle = angle
        x = DetectHeadV1.forward(self, x)
        if self.training:
            return x, angle
        return (torch.cat([x[0], angle], 1), (x[1], angle))

    def decode_bboxes(self, bboxes, anchors):
        """Decode rotated bounding boxes."""
        return dist2rbox(bboxes, self.angle, anchors, dim=1)
