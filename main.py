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

from models.usrnet import USRNet as net

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
        self.model_name = model_name
        self.model_path = model_path
        self.kernels = kernels
        self.scale_factor = scale_factor
        self.task_current = task_current
        self.need_degradation = need_degradation
        self.show_img = show_img
        self.save_LR = save_LR
        self.save_SR = save_SR

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("device:", self.device)

        self.loadmat()
        self.makeModel()

    def loadmat(self):
        kernel = loadmat(os.path.join(self.kernels, 'kernels_bicubicx234.mat'))['kernels']
        kernel = kernel[0, self.scale_factor-2].astype(np.float64)
        self.kernel = util.single2tensor4(kernel[..., np.newaxis])

    def makeModel(self):
        if 'tiny' in self.model_name:
            self.model = net(n_iter=6, h_nc=32, in_nc=4, out_nc=3, nc=[16, 32, 64, 64],
                    nb=2, act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose")
        else:
            self.model = net(n_iter=8, h_nc=64, in_nc=4, out_nc=3, nc=[64, 128, 256, 512],
                        nb=2, act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose")

    def loadModel(self):
        self.model.load_state_dict(torch.load(self.model_path), strict=True)
        self.model.eval()
        for key, v in self.model.named_parameters():
            v.requires_grad = False

def demo1():
    model_name = 'usrnet'
    model_path = "/media/dyf-ai/d3/weights/super-resolution/model_zoo-20210628T133354Z-001/model_zoo/usrnet.pth"
    kernels = "./kernels"

    inf = inference(model_name, model_path, kernels, 4)
    print("test")

if __name__ == '__main__':
    import fire
    fire.Fire() 