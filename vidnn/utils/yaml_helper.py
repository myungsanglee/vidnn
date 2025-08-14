import re
import yaml


def get_configs(file, hard=True):
    with open(file, errors="ignore") as f:
        try:
            configs = yaml.safe_load(f)
        except yaml.YAMLError as e:
            if hard:
                raise e
            raise Exception(f"WARNING ⚠️ YAML file {file} is invalid: {e}")
    return configs


if __name__ == "__main__":
    cfg_path = "/mnt/michael/vidnn/vidnn/configs/yolo.yaml"
    cfg = get_configs(cfg_path)
    print(cfg)
    print(type(cfg))
    print(cfg["test"])
    print(type(cfg["test"]))
