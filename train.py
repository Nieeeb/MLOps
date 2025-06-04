from nets.net import Net
import torch
import yaml
from utils import load_model






def main():
    with open(r"configs\params.yaml") as config:
        try:
            params = yaml.safe_load(config)
        except yaml.YAMLError as exc:
            print(exc)

    model = load_model()
    





# Vi skal også huske at gemme trænings konfigurationen
