import torch
from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn
import torch.nn as nn
import cv2
import numpy as np
from visualize import display_instances
import skimage.io

coco_names = {'0': 'background', '1': 'person', '2': 'bicycle', '3': 'car', '4': 'motorcycle', '5': 'airplane',
              '6': 'bus', '7': 'train', '8': 'truck', '9': 'boat', '10': 'traffic light', '11': 'fire hydrant',
              '13': 'stop sign', '14': 'parking meter', '15': 'bench', '16': 'bird', '17': 'cat', '18': 'dog',
              '19': 'horse', '20': 'sheep', '21': 'cow', '22': 'elephant', '23': 'bear', '24': 'zebra', '25': 'giraffe',
              '27': 'backpack', '28': 'umbrella', '31': 'handbag', '32': 'tie', '33': 'suitcase', '34': 'frisbee',
              '35': 'skis', '36': 'snowboard', '37': 'sports ball', '38': 'kite', '39': 'baseball bat',
              '40': 'baseball glove', '41': 'skateboard', '42': 'surfboard', '43': 'tennis racket', '44': 'bottle',
              '46': 'wine glass', '47': 'cup', '48': 'fork', '49': 'knife', '50': 'spoon', '51': 'bowl', '52': 'banana',
              '53': 'apple', '54': 'sandwich', '55': 'orange', '56': 'broccoli', '57': 'carrot', '58': 'hot dog',
              '59': 'pizza', '60': 'donut', '61': 'cake', '62': 'chair', '63': 'couch', '64': 'potted plant',
              '65': 'bed', '67': 'dining table', '70': 'toilet', '72': 'tv', '73': 'laptop', '74': 'mouse',
              '75': 'remote', '76': 'keyboard', '77': 'cell phone', '78': 'microwave', '79': 'oven', '80': 'toaster',
              '81': 'sink', '82': 'refrigerator', '84': 'book', '85': 'clock', '86': 'vase', '87': 'scissors',
              '88': 'teddybear', '89': 'hair drier', '90': 'toothbrush'}

SCORE_THRESHOLD = 0.5

model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()


img1 = skimage.io.imread("img2.jpeg")
img2 = skimage.io.imread("img1.jpeg")

x = [
    torch.from_numpy(np.transpose(img1 / 255.0, axes=[2, 0, 1])).to(torch.float32),
    torch.from_numpy(np.transpose(img2 / 255.0, axes=[2, 0, 1])).to(torch.float32)
]

out = model(x)
filtered_idx = (out[0]['scores'] >= SCORE_THRESHOLD)

boxes = out[0]['boxes'][filtered_idx].detach()
labels = out[0]['labels'][filtered_idx].detach()
scores = out[0]['scores'][filtered_idx].detach()
masks = out[0]['masks'][filtered_idx].detach()

display_instances(img1, boxes, masks, labels, coco_names, scores,
                  title="My desk", fig_save_path="img2_mask_rcnn_th0.5.jpeg")
