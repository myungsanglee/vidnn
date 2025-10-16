import torch
import torch.nn as nn

from .conv import Conv, Concat
from .block import C2f, C3k2
from vidnn.utils.ops import make_divisible


class YoloNeckV1(nn.Module):
    """
    YOLOv8의 PAN(Path Aggregation Network) 스타일 넥 구조를 구현합니다.
    하향식(top-down, FPN) 및 상향식(bottom-up, PAN) 경로를 통해
    다양한 스케일의 특징 맵을 효과적으로 융합합니다.
    """

    def __init__(self, in_channels, scale=None):  # in_channels: [P3_채널, P4_채널, P5_채널]
        super().__init__()
        self.in_channels = in_channels
        self.num_scales = len(in_channels)  # 3 (P3, P4, P5)
        self.upsample = nn.Upsample(None, 2, "nearest")
        self.concat = Concat(1)
        p3c, p4c, p5c = in_channels
        c2f_channels = [512, 256, 512, 1024]
        n = 3
        if scale is not None:
            assert len(scale) == 3, "scale 값은 리스트로 [depth, width, max_channels] 3개의 값이 필요합니다."
            depth, width, max_channels = scale
            n = max(round(n * depth), 1) if n > 1 else n  # depth gain
            c2f_channels = [make_divisible(min(x, max_channels) * width, 8) for x in c2f_channels]

        # FPN (Top-down pathway): P5 -> P4, P4 -> P3 방향으로 특징 융합
        # P5 특징을 P4 채널 수로 줄이고, P4 특징과 함께 융합
        # self.up_p5_to_p4 = nn.Upsample(None, 2, "nearest")
        # self.concat_p5_p4 = Concat(1)
        self.fuse_p4 = C2f(p4c + p5c, c2f_channels[0], n)

        # P4 특징을 P3 채널 수로 줄이고, P3 특징과 함께 융합
        # self.up_p4_to_p3 = nn.Upsample(None, 2, "nearest")
        # self.concat_p4_p3 = Concat(1)
        self.fuse_p3 = C2f(p3c + c2f_channels[0], c2f_channels[1], n)

        # PAN (Bottom-up pathway): P3 -> P4, P4 -> P5 방향으로 특징 융합
        # P3 융합 특징을 다운샘플링하여 P4의 공간 해상도와 맞춤
        self.down_conv_p3_to_p4 = Conv(c2f_channels[1], c2f_channels[1], 3, 2)  # k=3, s=2 -> p=1 자동
        self.fuse_pan_p4 = C2f(c2f_channels[0] + c2f_channels[1], c2f_channels[2], n)

        # P4 융합 특징을 다운샘플링하여 P5의 공간 해상도와 맞춤
        self.down_conv_p4_to_p5 = Conv(c2f_channels[2], c2f_channels[2], 3, 2)  # k=3, s=2 -> p=1 자동
        self.fuse_pan_p5 = C2f(p5c + c2f_channels[2], c2f_channels[3], n)

        # 넥의 최종 출력 채널 (헤드로 전달될 채널)
        self.neck_out_channels = c2f_channels[1:]

    def forward(self, features):  # features: [P3_입력, P4_입력, P5_입력] (백본으로부터)
        assert len(features) == self.num_scales, "백본으로부터 3개의 특징 맵이 필요합니다."
        p3_in, p4_in, p5_in = features

        # Top-down pathway (FPN)
        # P5 특징을 업샘플링하여 P4의 공간 해상도와 맞춤
        p5_up = self.upsample(p5_in)
        p4_fused = self.fuse_p4(self.concat([p5_up, p4_in]))  # P4와 업샘플링된 P5를 융합

        # P4 융합 특징을 업샘플링하여 P3의 공간 해상도와 맞춤
        p4_up = self.upsample(p4_fused)
        p3_fused = self.fuse_p3(self.concat([p4_up, p3_in]))  # P3와 업샘플링된 P4를 융합

        # Bottom-up pathway (PAN)
        # P3 융합 특징을 다운샘플링하여 P4의 공간 해상도와 맞춤
        p3_down = self.down_conv_p3_to_p4(p3_fused)
        # P4 융합 특징(FPN에서 온)과 다운샘플링된 P3를 융합
        p4_pan_fused = self.fuse_pan_p4(self.concat([p3_down, p4_fused]))

        # P4 PAN 융합 특징을 다운샘플링하여 P5의 공간 해상도와 맞춤
        p4_down = self.down_conv_p4_to_p5(p4_pan_fused)
        # P5 입력 특징(백본에서 온)과 다운샘플링된 P4 PAN 융합 특징을 융합
        p5_pan_fused = self.fuse_pan_p5(self.concat([p4_down, p5_in]))

        # 최종 넥 출력: P3, P4, P5 스케일의 융합된 특징 맵
        return [p3_fused, p4_pan_fused, p5_pan_fused]
