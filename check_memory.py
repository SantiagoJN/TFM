from pynvml import *
import torch

nvmlInit()
h = nvmlDeviceGetHandleByIndex(0)
info = nvmlDeviceGetMemoryInfo(h)
print(f'~~~~ {torch.cuda.get_device_name(0)} INFO ~~~~')
print(f'total    : {info.total/1000000} MB')
print(f'free     : {info.free/1000000} MB')
print(f'used     : {info.used/1000000} MB')
