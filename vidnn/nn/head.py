import math
import torch
import torch.nn as nn

from .conv import Conv
from .block import DFL
from vidnn.utils.tal import dist2bbox, dist2rbox, make_anchors


class DetectHeadV1(nn.Module):

    max_det = 300  # max_det
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, num_classes, in_channels, reg_max=16):  # in_channels: [P3_넥_채널, P4_넥_채널, P5_넥_채널]
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
