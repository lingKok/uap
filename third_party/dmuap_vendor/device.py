"""设备解析：唯一改动点，让官方代码可在 CPU 上做冒烟测试。

官方实现把 "cuda:0" 硬编码在 4 处。为了让接入代码能在无 GPU 环境先跑通
（避免把形状/label 错误带上集群），这里统一改为读环境变量 UAP_DEV：
    UAP_DEV=cpu  -> CPU（本地冒烟测试用，极慢，只验证数值流正确）
    未设置       -> cuda:0（集群正式跑，与官方默认完全一致）

除设备放置外，算法逻辑一行未改（见 vendor 目录 diff 说明）。
"""
import os
import torch


def get_device():
    dev = os.environ.get("UAP_DEV", "").strip()
    if dev:
        return torch.device(dev)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
