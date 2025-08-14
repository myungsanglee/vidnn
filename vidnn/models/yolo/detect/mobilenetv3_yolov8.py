import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# -----------------------------------------------------------------------------
# 1. Helper Modules (기본적인 Conv, Bottleneck, C2f-like 블록 및 DFL 모듈)
#    YOLOv8 모델의 각 구성 요소를 만드는 데 사용되는 기본적인 빌딩 블록입니다.
# -----------------------------------------------------------------------------


class Conv(nn.Module):
    """
    표준 Convolution 레이어: Conv2d + BatchNorm2d + SiLU 활성화 함수.
    이 모듈은 YOLOv8 모델 전체에서 기본 구성 요소로 사용됩니다.
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        # k: 커널 크기, s: 스트라이드, p: 패딩 (None이면 자동 계산), g: 그룹 수
        if (
            p is None
        ):  # 패딩이 명시되지 않으면 자동으로 계산 (커널 크기 기반, stride 무관)
            p = k // 2  # 'same' 패딩 효과를 위해 항상 k // 2로 설정
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        # 활성화 함수: 기본적으로 SiLU를 사용하며, 다른 활성화 함수를 지정할 수 있습니다.
        self.act = (
            nn.SiLU()
            if act is True
            else (act if isinstance(act, nn.Module) else nn.Identity())
        )

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        # 훈련 후 추론 시 BatchNorm을 Convolution에 통합하기 위한 메서드 (여기서는 단순화)
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    """
    C2f_like 모듈 내에서 사용되는 Bottleneck 블록입니다.
    잔차 연결(residual connection)을 포함하여 깊은 네트워크의 학습을 용이하게 합니다.
    """

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        # c1: 입력 채널, c2: 출력 채널, shortcut: 잔차 연결 사용 여부
        # g: 그룹 수, e: 확장 비율 (중간 채널을 c2 * e로 설정)
        self.conv1 = Conv(c1, int(c2 * e), 1, 1)  # 1x1 Conv로 채널 축소
        self.conv2 = Conv(int(c2 * e), c2, 3, 1, 1, g)  # 3x3 Conv
        self.add = shortcut and c1 == c2  # 입력/출력 채널이 같을 때만 잔차 연결

    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))


class C2f_like(nn.Module):
    """
    YOLOv8의 C2f 모듈과 유사하게 구현한 것입니다.
    입력 특징 맵을 두 부분으로 나누고, 한 부분은 Bottleneck 블록들을 통과시킨 후
    모든 중간 출력과 다른 부분(원본의 절반)을 함께 연결하여 특징 융합을 수행합니다.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        # c1: 입력 채널, c2: 출력 채널, n: Bottleneck 블록 반복 횟수
        # e: hidden 채널 비율 (c2 * e)
        self.c = int(c2 * e)  # hidden channels
        self.conv1 = Conv(
            c1, 2 * self.c, 1, 1
        )  # 입력을 두 배의 채널로 확장 후 분할 준비
        # 최종 Concatenate될 모든 부분의 채널을 합산하여 conv2의 입력 채널을 정의
        # 2*self.c는 conv1의 출력 (두 부분으로 나e기 전)을 의미하며,
        # 이후 n개의 Bottleneck 블록이 각 self.c 채널을 가지므로 (2 + n) * self.c
        self.conv2 = Conv((2 + n) * self.c, c2, 1, 1)  # 최종 출력 채널을 맞추는 Conv
        # n개의 Bottleneck 블록들을 모듈 리스트로 생성
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )

    def forward(self, x):
        y = list(
            self.conv1(x).chunk(2, 1)
        )  # 입력을 2개로 분할: [원본의 절반, 원본의 절반]
        # 분할된 한 부분(y[-1], 즉 두 번째 절반)을 Bottleneck 블록들을 통과시킴
        # 각 Bottleneck 블록의 출력은 y 리스트에 추가됨
        y.extend(m(y[-1]) for m in self.m)
        # 모든 부분(원본의 첫 번째 절반 + 처리된 두 번째 절반 및 모든 중간 Bottleneck 출력)을 다시 연결하고
        # 최종 Conv2를 통과시켜 특징 융합을 완료
        return self.conv2(torch.cat(y, 1))


class DFL(nn.Module):
    """
    Distribution Focal Loss (DFL)의 Integral 모듈입니다.
    바운딩 박스 예측에서 각 오프셋(x, y, w, h)을 분포로 예측한 값을
    실제 좌표 값으로 변환하는 데 사용됩니다.
    """

    def __init__(self, reg_max=16):  # reg_max: 분포의 최대 범위 (기본 16)
        super().__init__()
        self.reg_max = reg_max
        # 가중치 텐서: 0, 1, ..., reg_max-1
        # 이 텐서는 학습되지 않습니다 (requires_grad_(False))
        self.register_buffer("proj_weights", torch.arange(reg_max, dtype=torch.float32))

    def forward(self, x):
        # x: (B, 4, reg_max, H, W) 형태로 Softmax가 적용된 확률 분포
        # 이 x는 YOLOv8Head에서 reg_dist.view(B, 4, self.reg_max, H, W) 후 Softmax를 거친 결과여야 합니다.

        # proj_weights는 (reg_max,) 형태. x와 곱하기 위해 (1, 1, reg_max, 1, 1)로 브로드캐스트.
        # x의 reg_max 차원(dim=2)과 proj_weights를 곱하고, 그 차원을 따라 합산합니다.
        # 결과 shape: (B, 4, H, W)
        return (x * self.proj_weights.view(1, 1, self.reg_max, 1, 1)).sum(dim=2)


# -----------------------------------------------------------------------------
# 2. YOLOv8-like Neck (PAN-like 구조)
#    백본에서 추출된 다양한 스케일의 특징 맵을 통합하고 풍부하게 만듭니다.
# -----------------------------------------------------------------------------


class YOLOv8Neck(nn.Module):
    """
    YOLOv8의 PAN(Path Aggregation Network) 스타일 넥 구조를 구현합니다.
    하향식(top-down, FPN) 및 상향식(bottom-up, PAN) 경로를 통해
    다양한 스케일의 특징 맵을 효과적으로 융합합니다.
    """

    def __init__(self, in_channels):  # in_channels: [P3_채널, P4_채널, P5_채널]
        super().__init__()
        self.in_channels = in_channels
        self.num_scales = len(in_channels)  # 3 (P3, P4, P5)

        # FPN (Top-down pathway): P5 -> P4, P4 -> P3 방향으로 특징 융합
        # P5 특징을 P4 채널 수로 줄이고, P4 특징과 함께 융합
        self.up_conv_p5_to_p4 = Conv(in_channels[2], in_channels[1], 1, 1)
        self.fuse_p4 = C2f_like(in_channels[1] * 2, in_channels[1])

        # P4 특징을 P3 채널 수로 줄이고, P3 특징과 함께 융합
        self.up_conv_p4_to_p3 = Conv(in_channels[1], in_channels[0], 1, 1)
        self.fuse_p3 = C2f_like(in_channels[0] * 2, in_channels[0])

        # PAN (Bottom-up pathway): P3 -> P4, P4 -> P5 방향으로 특징 융합
        # P3 융합 특징을 다운샘플링하여 P4의 공간 해상도와 맞춤
        self.down_conv_p3_to_p4 = Conv(
            in_channels[0], in_channels[1], 3, 2
        )  # k=3, s=2 -> p=1 자동
        self.fuse_pan_p4 = C2f_like(in_channels[1] * 2, in_channels[1])

        # P4 융합 특징을 다운샘플링하여 P5의 공간 해상도와 맞춤
        self.down_conv_p4_to_p5 = Conv(
            in_channels[1], in_channels[2], 3, 2
        )  # k=3, s=2 -> p=1 자동
        self.fuse_pan_p5 = C2f_like(in_channels[2] * 2, in_channels[2])

        # 넥의 최종 출력 채널 (헤드로 전달될 채널)
        self.neck_out_channels = in_channels

    def forward(self, features):  # features: [P3_입력, P4_입력, P5_입력] (백본으로부터)
        assert (
            len(features) == self.num_scales
        ), "백본으로부터 3개의 특징 맵이 필요합니다."
        p3_in, p4_in, p5_in = features

        # Top-down pathway (FPN)
        # P5 특징을 업샘플링하여 P4의 공간 해상도와 맞춤
        p5_up = F.interpolate(
            self.up_conv_p5_to_p4(p5_in), size=p4_in.shape[2:], mode="nearest"
        )
        p4_fused = self.fuse_p4(
            torch.cat([p4_in, p5_up], 1)
        )  # P4와 업샘플링된 P5를 융합

        # P4 융합 특징을 업샘플링하여 P3의 공간 해상도와 맞춤
        p4_up = F.interpolate(
            self.up_conv_p4_to_p3(p4_fused), size=p3_in.shape[2:], mode="nearest"
        )
        p3_fused = self.fuse_p3(
            torch.cat([p3_in, p4_up], 1)
        )  # P3와 업샘플링된 P4를 융합

        # Bottom-up pathway (PAN)
        # P3 융합 특징을 다운샘플링하여 P4의 공간 해상도와 맞춤
        p3_down = self.down_conv_p3_to_p4(p3_fused)
        # P4 융합 특징(FPN에서 온)과 다운샘플링된 P3를 융합
        p4_pan_fused = self.fuse_pan_p4(torch.cat([p4_fused, p3_down], 1))

        # P4 PAN 융합 특징을 다운샘플링하여 P5의 공간 해상도와 맞춤
        self.down_conv_p4_to_p5 = Conv(
            self.in_channels[1], self.in_channels[2], 3, 2
        )  # k=3, s=2 -> p=1 자동
        p4_down = self.down_conv_p4_to_p5(p4_pan_fused)
        # P5 입력 특징(백본에서 온)과 다운샘플링된 P4 PAN 융합 특징을 융합
        p5_pan_fused = self.fuse_pan_p5(torch.cat([p5_in, p4_down], 1))

        # 최종 넥 출력: P3, P4, P5 스케일의 융합된 특징 맵
        return [p3_fused, p4_pan_fused, p5_pan_fused]


# -----------------------------------------------------------------------------
# 3. YOLOv8-like Head (Decoupled Head with DFL)
#    넥에서 처리된 특징 맵을 바탕으로 바운딩 박스와 클래스 예측을 수행합니다.
# -----------------------------------------------------------------------------


class YOLOv8Head(nn.Module):
    """
    YOLOv8의 디커플드 헤드(Decoupled Head)를 구현합니다.
    바운딩 박스 회귀(regression)와 클래스 분류(classification)를 위한
    별도의 분리된 경로를 가집니다. DFL(Distribution Focal Loss)을 포함합니다.
    """

    def __init__(
        self, num_classes, in_channels, reg_max=16
    ):  # in_channels: [P3_넥_채널, P4_넥_채널, P5_넥_채널]
        super().__init__()
        self.num_classes = num_classes
        self.num_heads = len(in_channels)  # 3개의 출력 스케일 (P3, P4, P5)
        self.reg_max = reg_max  # DFL 분포의 최대 범위 (보통 16)

        self.reg_convs = nn.ModuleList()  # 바운딩 박스 회귀를 위한 Conv 레이어들
        self.cls_convs = nn.ModuleList()  # 클래스 분류를 위한 Conv 레이어들
        self.reg_preds = nn.ModuleList()  # 최종 바운딩 박스 분포 예측 레이어
        self.cls_preds = nn.ModuleList()  # 최종 클래스 예측 레이어

        # DFL 변환을 위한 모듈 (모든 스케일에 공통으로 사용)
        self.dfl = DFL(self.reg_max)

        # 각 스케일(P3, P4, P5)에 대해 별도의 회귀 및 분류 헤드 생성
        for i in range(self.num_heads):
            # 회귀 브랜치 (예: 2개의 Conv 레이어 + 최종 예측 레이어)
            self.reg_convs.append(
                nn.Sequential(
                    Conv(in_channels[i], in_channels[i], 3, 1),
                    Conv(in_channels[i], in_channels[i], 3, 1),
                )
            )
            # 바운딩 박스 예측: 4 (left, right, top, bottom) * reg_max (분포)
            self.reg_preds.append(
                nn.Conv2d(in_channels[i], 4 * self.reg_max, 1)
            )  # 1x1 Conv

            # 분류 브랜치 (예: 2개의 Conv 레이어 + 최종 예측 레이어)
            self.cls_convs.append(
                nn.Sequential(
                    Conv(in_channels[i], in_channels[i], 3, 1),
                    Conv(in_channels[i], in_channels[i], 3, 1),
                )
            )
            # 클래스 예측: num_classes (각 클래스에 대한 확률)
            self.cls_preds.append(nn.Conv2d(in_channels[i], num_classes, 1))  # 1x1 Conv

    def forward(self, features):  # features: [P3_넥_출력, P4_넥_출력, P5_넥_출력]
        reg_outputs = []  # 바운딩 박스 예측 결과들을 저장할 리스트
        cls_outputs = []  # 클래스 예측 결과들을 저장할 리스트

        # 각 특징 맵 스케일(P3, P4, P5)에 대해 예측 수행
        for i, x in enumerate(features):
            # 회귀 브랜치
            reg_feat = self.reg_convs[i](x)
            # (B, 4 * reg_max, H, W) 형태의 분포 예측
            reg_dist = self.reg_preds[i](reg_feat)

            # DFL을 사용하여 분포 예측을 실제 바운딩 박스 좌표로 변환
            B, _, H, W = reg_dist.shape
            # reg_dist를 (B, 4, reg_max, H, W) 형태로 재구성
            reg_dist_reshaped = reg_dist.view(B, 4, self.reg_max, H, W)
            # reg_max 차원(dim=2)에 대해 Softmax를 적용하여 확률 분포로 만듦
            reg_softmax = F.softmax(reg_dist_reshaped, dim=2)
            # DFL 모듈을 사용하여 최종 바운딩 박스 오프셋 예측 (B, 4, H, W)
            reg_pred = self.dfl(
                reg_softmax
            )  # DFL 모듈 내부에서 sum을 수행하여 (B, 4, H, W) 반환

            reg_outputs.append(reg_pred)

            # 분류 브랜치
            cls_feat = self.cls_convs[i](x)
            cls_outputs.append(self.cls_preds[i](cls_feat))

        # [bbox_preds_P3, bbox_preds_P4, bbox_preds_P5], [cls_preds_P3, cls_preds_P4, cls_preds_P5] 형태로 반환
        return reg_outputs, cls_outputs


# -----------------------------------------------------------------------------
# 4. 전체 YOLOv8-like MobileNet 모델
# -----------------------------------------------------------------------------


class YOLOv8MobileNet(nn.Module):
    """
    timm의 MobileNetV3 Large 백본, 사용자 정의 Neck, 그리고 Decoupled Head를
    결합한 YOLOv8-like 객체 감지 모델입니다.
    """

    def __init__(self, num_classes=80):  # COCO 데이터셋 기본 80개 클래스
        super().__init__()
        self.num_classes = num_classes

        # 백본: timm 라이브러리에서 ImageNet으로 사전 학습된 MobileNetV3 Large 1.0 로드
        # features_only=True는 중간 레이어의 특징 맵 리스트를 반환하도록 합니다.
        self.backbone = timm.create_model(
            "mobilenetv3_large_100", pretrained=True, features_only=True
        )

        # MobileNetV3 Large 1.0의 P3, P4, P5에 해당하는 특징 맵의 채널 수 정의
        # timm의 MobileNetV3는 일반적으로 5개의 특징 맵을 반환하며,
        # 인덱스 2, 3, 4가 각각 stride 8 (P3), 16 (P4), 32 (P5)에 해당합니다.
        # 해당 채널 수는 [40, 112, 960] 입니다.
        backbone_out_channels = [40, 112, 960]

        # Neck: PAN-like FPN/PAN 구조
        self.neck = YOLOv8Neck(in_channels=backbone_out_channels)

        # Head: Decoupled 예측 헤드 (DFL 포함)
        # 넥의 출력 채널을 헤드의 입력 채널로 사용합니다.
        neck_out_channels = self.neck.neck_out_channels
        self.head = YOLOv8Head(num_classes=num_classes, in_channels=neck_out_channels)

    def forward(self, x):
        # 1. 백본 (Backbone)
        # MobileNetV3는 특징 맵 리스트를 반환합니다.
        # 여기서 인덱스 2부터 마지막까지 (P3, P4, P5에 해당)를 선택합니다.
        backbone_features = self.backbone(x)[2:]

        # 2. 넥 (Neck)
        neck_outputs = self.neck(backbone_features)

        # 3. 헤드 (Head)
        reg_preds, cls_preds = self.head(neck_outputs)

        # 최종 예측 결과: [bbox_preds_P3, bbox_preds_P4, bbox_preds_P5],
        #               [cls_preds_P3, cls_preds_P4, cls_preds_P5]
        return reg_preds, cls_preds


# -----------------------------------------------------------------------------
# 5. 모델 사용 예시
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # 객체 클래스 수 (예: COCO의 80개 클래스)
    num_classes = 80
    # YOLOv8MobileNet 모델 인스턴스 생성
    model = YOLOv8MobileNet(num_classes=num_classes)
    model.eval()  # 추론 모드로 설정

    # 더미 입력 이미지 (Batch Size, Channels, Height, Width)
    # YOLOv8은 보통 640x640 이미지 크기를 사용합니다.
    dummy_input = torch.randn(1, 3, 640, 640)

    print("모델 구조:")
    print(model)

    print("\n더미 입력으로 모델 순전파 테스트 중...")
    with torch.no_grad():  # 역전파 계산 비활성화 (추론 시)
        reg_preds, cls_preds = model(dummy_input)

    print("\n예측 결과 형태:")
    for i, (reg_p, cls_p) in enumerate(zip(reg_preds, cls_preds)):
        print(f"  스케일 {i} (P{i+3} 해당):")
        print(f"    바운딩 박스 예측 형태: {reg_p.shape}")  # [B, 4, H, W] (DFL 적용 후)
        print(f"    클래스 예측 형태: {cls_p.shape}")  # [B, num_classes, H, W]

    # 예상 출력 해상도 (640x640 입력 기준):
    # P3: 80x80 (640/8)
    # P4: 40x40 (640/16)
    # P5: 20x20 (640/32)
