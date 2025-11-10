import time
import torch
import cv2
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
)

from vidnn.utils.yaml_helper import get_configs
from vidnn.data.datamodule import YoloDataModule
from vidnn.models.detect.mobilenetv3_yolov8 import YOLOv8MobileNet
from vidnn.module.tasks import DetectorModule


def predict(cfg, ckpt):
    # Model
    model = YOLOv8MobileNet(num_classes=len(cfg["names"]))
    model_module = DetectorModule.load_from_checkpoint(
        checkpoint_path=ckpt,
        model=model,
        cfg=cfg,
        steps_per_epoch=0,
    )
    model_module.eval()
    if torch.cuda.is_available:
        model_module = model_module.cuda()

    # Image
    img_path = "/mnt/michael/vidnn/vidnn/bus.jpg"
    # img_path = "/mnt/michael/vidnn/vidnn/zidane.jpg"

    # Inference
    with torch.no_grad():
        preds, orig_img = model_module.predict(img_path, imgsz=cfg["imgsz"])
    pred = preds[0].cpu().numpy()
    boxes = pred[:, :4]
    confs = pred[:, 4]
    classes = pred[:, 5]
    names = cfg["names"]

    # draw bboxes
    for box, cls, conf in zip(boxes, classes, confs):
        # get bbox
        x1, y1, x2, y2 = [int(round(x)) for x in box]

        # draw bbox and info
        orig_img = cv2.rectangle(
            orig_img,
            (x1, y1),
            (x2, y2),
            color=(0, 255, 0),
            thickness=2,
        )
        orig_img = cv2.putText(
            orig_img,
            (f"{names[int(cls)]} ({conf:.2f})"),
            (x1, y1 - 5),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=1.5,
            color=(0, 255, 0),
            thickness=2,
        )

    cv2.imshow("Predict", orig_img)
    key = cv2.waitKey(0)

    # Check inference time
    # time_list = []
    # for i in range(100):
    #     start_time = time.time()
    #     with torch.no_grad():
    #         preds, orig_img = model_module.predict(img_path, imgsz=cfg["imgsz"])
    #     end_time = time.time()
    #     time_list.append(end_time - start_time)
    # avg = (sum(time_list) / len(time_list)) * 1000
    # print(f"Inference time: {avg:.2f} ms")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--cfg", required=True, type=str, help="config file")
    # parser.add_argument('--ckpt', required=True, type=str, help='checkpoints file')
    # args = parser.parse_args()
    # cfg = get_configs(args.cfg)
    # val(cfg, args.ckpt)

    cfg = get_configs("/mnt/michael/vidnn/vidnn/configs/yolo.yaml")
    ckpt = "/mnt/michael/vidnn/runs/lightning_logs/version_1/checkpoints/last.ckpt"
    predict(cfg, ckpt)
