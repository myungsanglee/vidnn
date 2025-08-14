import cv2
import numpy as np
from data.datamodule import YoloDataModule
from utils.yaml_helper import get_configs


def check_dataloader():
    """Initializes the YoloDataModule, and visualizes the result."""

    yaml_path = "configs/yolo.yaml"
    cfg = get_configs(yaml_path)

    print("Initializing YoloDataModule with v8 augmentations...")
    data_module = YoloDataModule(cfg=cfg)

    print("Setting up data...")
    data_module.setup(stage="fit")
    dataloader = data_module.train_dataloader()
    # dataloader = data_module.val_dataloader()
    images, labels = next(iter(dataloader))

    print("\n--- Batch Information ---")
    print(f"Images tensor shape: {images.shape}")
    print(f"Labels tensor shape: {labels.shape}")
    print("-------------------------\n")

    # --- 시각화 ---
    for i in range(len(images)):
        img_tensor = images[i]
        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = (img_np * 255).astype(np.uint8)
        img_np = np.ascontiguousarray(img_np)

        first_img_labels = labels[labels[:, 0] == i]
        print(f"Found {len(first_img_labels)} labels for the first image.")

        for label in first_img_labels:
            class_id = int(label[1])
            x_center, y_center, width, height = label[2:]

            h, w, _ = img_np.shape
            x1 = int((x_center - width / 2) * w)
            y1 = int((y_center - height / 2) * h)
            x2 = int((x_center + width / 2) * w)
            y2 = int((y_center + height / 2) * h)

            cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img_np,
                f"cls: {class_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

        # output_path = "augmented_sample_v8.jpg"
        cv2.imshow("test", img_np)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"\nSuccessfully visualized the first sample of the batch.")
    # print(f"Check the output image at: {output_path}")


if __name__ == "__main__":
    check_dataloader()
