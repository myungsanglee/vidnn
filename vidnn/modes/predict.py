import time
import torch
import cv2

from vidnn.utils import check_configs
from vidnn.utils.yaml_helper import get_configs
from vidnn.utils.module_select import get_data_module, get_model, get_model_module


def predict(cfg):
    # Check cfg
    cfg = check_configs(cfg)

    # Model
    model = get_model(cfg)
    model_module = get_model_module(model=model, cfg=cfg, steps_per_epoch=1)
    model_module.eval()
    if torch.cuda.is_available():
        model_module = model_module.to(torch.device("cuda"))
    elif torch.backends.mps.is_available():
        model_module = model_module.to(torch.device("mps"))

    # Image
    img_path = "/Users/michael/Project/datasets/images/bus.jpg"
    # img_path = "/Users/michael/Project/datasets/images/zidane.jpg"
    # img_path = "/Users/michael/Project/datasets/images/test.jpg"

    # Inference
    with torch.no_grad():
        pred, orig_img = model_module.predict(img_path, imgsz=cfg["imgsz"])
    pred = pred.cpu().numpy()
    print(pred)
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
    cv2.destroyAllWindows()

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

    cfg = get_configs("vidnn/configs/yolo.yaml")
    # cfg = get_configs("vidnn/configs/yolo-obb.yaml")
    predict(cfg)
