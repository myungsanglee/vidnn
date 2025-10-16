import torch
import torch.nn as nn
import timm

from vidnn.nn.neck import YoloNeckV1
from vidnn.nn.head import DetectHeadV1
from vidnn.utils.torch_utils import model_info


class MobileNetV3Yolo(nn.Module):
    """
    timm의 MobileNetV3 Large 백본, 사용자 정의 Neck, 그리고 Decoupled Head를
    결합한 객체 감지 모델입니다.
    """

    def __init__(self, num_classes=80):  # COCO 데이터셋 기본 80개 클래스
        super().__init__()
        self.num_classes = num_classes

        # 백본: timm 라이브러리에서 ImageNet으로 사전 학습된 MobileNetV3 Large 1.0 로드
        # features_only=True는 중간 레이어의 특징 맵 리스트를 반환하도록 합니다.
        self.backbone = timm.create_model("mobilenetv3_large_100", pretrained=True, features_only=True)

        # MobileNetV3 Large 1.0의 P3, P4, P5에 해당하는 특징 맵의 채널 수 정의
        # timm의 MobileNetV3는 일반적으로 5개의 특징 맵을 반환하며,
        # 인덱스 2, 3, 4가 각각 stride 8 (P3), 16 (P4), 32 (P5)에 해당합니다.
        # 해당 채널 수는 [40, 112, 960] 입니다.
        backbone_out_channels = [40, 112, 960]

        # Neck: PAN-like FPN/PAN 구조
        self.neck = YoloNeckV1(in_channels=backbone_out_channels, scale=[0.33, 0.50, 1024])

        # Head: Decoupled 예측 헤드 (DFL 포함)
        # 넥의 출력 채널을 헤드의 입력 채널로 사용합니다.
        neck_out_channels = self.neck.neck_out_channels
        self.head = DetectHeadV1(num_classes=num_classes, in_channels=neck_out_channels)

    def forward(self, x):
        # 1. 백본 (Backbone)
        # MobileNetV3는 특징 맵 리스트를 반환합니다.
        # 여기서 인덱스 2부터 마지막까지 (P3, P4, P5에 해당)를 선택합니다.
        backbone_features = self.backbone(x)[2:]

        # 2. 넥 (Neck)
        neck_outputs = self.neck(backbone_features)

        # 3. 헤드 (Head)
        preds = self.head(neck_outputs)

        # 최종 예측 결과
        return preds


if __name__ == "__main__":
    # 객체 클래스 수 (예: COCO의 80개 클래스)
    num_classes = 80

    # YOLOv8MobileNet 모델 인스턴스 생성
    model = MobileNetV3Yolo(num_classes=num_classes)

    # Build strides
    m = model.head
    if isinstance(m, DetectHeadV1):
        s = 256
        m.inplace = True

        model.eval()  # Avoid changing batch statistics until training begins
        m.training = True  # Setting it to True to properly return strides
        m.stride = torch.tensor([s / x.shape[-2] for x in model(torch.zeros(1, 3, s, s))])  # forward
        model.train()  # Set model back to training(default) mode
        m.bias_init()  # only run once

    # Init weights, biases
    # initialize_weights
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in {nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU}:
            m.inplace = True

    # 더미 입력 이미지 (Batch Size, Channels, Height, Width)
    dummy_input = torch.randn(1, 3, 640, 640)

    # Model Info
    model_info(model, True)

    print("\n더미 입력으로 모델 순전파 테스트 중...")
    with torch.no_grad():  # 역전파 계산 비활성화 (추론 시)
        preds = model(dummy_input)

    print("\n예측 결과 형태:")
    for i in range(len(preds)):
        print(f"  스케일 {i} (P{i+3} 해당):")
        print(f"    예측 형태: {preds[i].shape}")

    # 예상 출력 해상도 (640x640 입력 기준):
    # P3: 80x80 (640/8)
    # P4: 40x40 (640/16)
    # P5: 20x20 (640/32)
