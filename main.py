import os
import cv2
import torch
import logging
import numpy as np
from scipy.io import loadmat

from utils import utils_logger
from utils import utils_image as util
from utils import utils_deblur
from utils import utils_sisr as sr

class inference(object):
    def __init__(self, model_name:str, 
                 model_path:str,
                 kernels:str,
                 scale_factor:int, 
                 task_current:str='sr',
                 need_degradation:bool=True,
                 show_img:bool=False,
                 save_LR:bool=True,
                 save_SR:bool=True) -> None:
        super(inference, self).__init__()
        pass

def demo1():
    inf = inference("", "", "", 4)
    print("test")

if __name__ == '__main__':
    import fire
    fire.Fire() 