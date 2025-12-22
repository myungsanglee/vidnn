from copy import deepcopy
import torch

from vidnn.utils.plotting import Annotator, colors
from vidnn.utils.ops import xywhr2xyxyxyxy


class Predictor:
    def __init__(self, model_path, task="detect", conf=0.25, iou=0.6, imgsz=640, max_det=300):
        self.model = torch.load(model_path, weights_only=False)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.to(torch.device("cuda"))
        elif torch.backends.mps.is_available():
            self.model = self.model.to(torch.device("mps"))
        self.names = self.model.names
        self.task = task
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.max_det = max_det
        self.pred = None
        self.orig_img = None
        self.annotated_img = None

    def __call__(self, source):
        return self.predict(source)

    @torch.no_grad()
    def predict(self, source):
        # inference
        self.pred, self.orig_img = self.model.predict(
            source,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            max_det=self.max_det,
        )

        # annotator
        annotator = Annotator(deepcopy(self.orig_img))
        boxes = xywhr2xyxyxyxy(self.pred[:, :5]) if self.task == "obb" else self.pred[:, :4]
        confs = self.pred[:, 5] if self.task == "obb" else self.pred[:, 4]
        classes = self.pred[:, 6] if self.task == "obb" else self.pred[:, 5]

        # draw bboxes
        for box, cls, conf in zip(boxes, classes, confs):
            color = colors(int(cls), True)
            annotator.box_label(box, label=self.names[int(cls)], color=color)

        # get result
        self.annotated_img = annotator.result()

        return self.pred, self.orig_img

    def get_annotated_image(self):
        return self.annotated_img
