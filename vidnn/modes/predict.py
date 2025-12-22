import time
import torch
import cv2
from copy import deepcopy

from vidnn.utils import check_configs
from vidnn.utils.yaml_helper import get_configs
from vidnn.utils.module_select import get_data_module, get_model, get_model_module
from vidnn.utils.plotting import Annotator, colors
from vidnn.utils.ops import xywhr2xyxyxyxy


def predict(cfg):
    # Check cfg
    cfg = check_configs(cfg)
    task = cfg["task"]

    # Model
    # model = get_model(cfg)
    # model_module = get_model_module(model=model, cfg=cfg, steps_per_epoch=1)
    model_module = torch.load("runs/ladybug.pt", weights_only=False)
    model_module.eval()
    if torch.cuda.is_available():
        model_module = model_module.to(torch.device("cuda"))
    elif torch.backends.mps.is_available():
        model_module = model_module.to(torch.device("mps"))

    # Image
    # img_path = "../datasets/images/bus.jpg"
    # img_path = "../datasets/images/zidane.jpg"
    # img_path = "../datasets/images/airplane.jpg"
    # img_path = "../datasets/images/vehicle.jpg"
    # img_path = "../datasets/images/tennis_court.jpg"
    # img_path = "../datasets/images/P0352.jpg"
    # img_path = "../datasets/images/zivid_image_00001.jpg"
    img_path = "../datasets/images/zivid_image_00037.jpg"

    # Inference
    # # with torch.no_grad():
    pred, orig_img = model_module.predict(img_path, imgsz=cfg["imgsz"])
    # pred = pred.cpu().numpy()
    # print(pred)
    annotator = Annotator(deepcopy(orig_img))
    boxes = xywhr2xyxyxyxy(pred[:, :5]) if task == "obb" else pred[:, :4]
    confs = pred[:, 5] if task == "obb" else pred[:, 4]
    classes = pred[:, 6] if task == "obb" else pred[:, 5]
    names = cfg["names"]

    # draw bboxes
    for box, cls, conf in zip(boxes, classes, confs):
        color = colors(int(cls), True)
        annotator.box_label(box, label=names[int(cls)], color=color)

    # get result
    orig_img = annotator.result()

    cv2.imshow("Predict", orig_img)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Check inference time
    # time_list = []
    # for i in range(100):
    #     start_time = time.time()
    #     # with torch.no_grad():
    #     preds, orig_img = model_module.predict(img_path, imgsz=cfg["imgsz"])
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

    # cfg = get_configs("vidnn/configs/yolo.yaml")
    # cfg = get_configs("vidnn/configs/yolo-obb.yaml")
    cfg = get_configs("vidnn/configs/yolo-ladybug.yaml")
    predict(cfg)
