import yaml
from datetime import datetime
import torch
import os


class Config:
    def __init__(self, yml_path):
        with open(yml_path, "r", encoding="utf-8") as f:
            self.content = yaml.load(f.read())
        self.content["time_stamp"] = "{0:%Y-%m-%dT%H-%M-%S/}".format(
            datetime.now()
        )
        self.content["folder"] = os.path.dirname(yml_path)
        self.content["device"] = (
            torch.device("cuda:{}".format(self.content["device_id"]))
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

    def __getattr__(self, key):
        return self.content[str(key)]

    def __str__(self):
        string = ""
        for k, v in self.content.items():
            string += "{}: {}   ".format(k, v)
        return string



p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
conf = Config(os.path.join(p, "configure.yml"))
infer_conf = Config(os.path.join(p, "inference_configure.yml"))
