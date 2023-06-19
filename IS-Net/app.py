import gradio as gr
import numpy as np
import torch
# import cv2
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from models import *
import time

input_size = [640,640]
dataset_path = "../demo_datasets/your_dataset"  # Your dataset path
result_path = "../demo_datasets/your_dataset_result"  # The folder path that you want to save the results

class Predictor:
    def __init__(self):
        self.model = model_init()
        self.preprocess = preprocess

    def __call__(self, img):
        time_begin = time.time()
        im_shp = img.shape[0:2]
        img_ = self.preprocess(img)
        time_end_pre = time.time()
        print('the prepro ',time_end_pre - time_begin)
        result = self.model(img_)
        time_begin_postpro = time.time()
        im_result = self.postprocess(result, im_shp)
        time_end = time.time()
        print('the time cost of postpro',time_end-time_begin_postpro)
        print("time cost is ",time_end-time_begin)
        return im_result

    def postprocess(self,im_tensor,im_shp):
        result = im_tensor
        result = torch.squeeze(F.upsample(result[0][0], im_shp, mode='bilinear'), 0)
        ma = torch.max(result)
        mi = torch.min(result)
        result = (result - mi) / (ma - mi)
        im_result = (result * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)
        print("the max value ", im_result.max(), im_result.dtype)
        im_result = im_result.astype("float") / 255
        if im_result.shape[-1] ==1 :
            im_result = im_result[...,0]
        return im_result

def model_init():
    model_path = "../saved_models/IS-Net/isnet-general-use.pth"  # the model path
    net = ISNetDIS()

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_path))
        net = net.cuda()
    else:
        net.load_state_dict(torch.load(model_path, map_location="cpu"))
    net.eval()
    return net

def preprocess(im):
    print("the image max()", im.max(), im.dtype)
    im = im[..., :3]
    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]
    im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
    im_tensor = F.upsample(torch.unsqueeze(im_tensor, 0), input_size, mode="bilinear").type(torch.uint8)
    image = torch.divide(im_tensor, 255.0)
    try:
        image = normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
    except Exception as e:
        print(2)
    if torch.cuda.is_available():
        image = image.cuda()
    return image


if __name__ == "__main__":
    predictor = Predictor()
    demo = gr.Interface(fn=predictor.__call__, inputs="image", outputs="image")

    demo.launch(share = True)
