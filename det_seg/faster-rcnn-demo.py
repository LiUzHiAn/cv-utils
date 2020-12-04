from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
import torch
import cv2
import numpy as np
import random


def random_color():
    b = random.randint(0, 255)
    g = random.randint(0, 255)
    r = random.randint(0, 255)

    return (b, g, r)


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

SCORE_THRESHOLD = 0.6

model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
img1 = cv2.imread("img1.jpeg", cv2.COLOR_BGR2RGB) / 255.0
img2 = cv2.imread("img2.jpeg", cv2.COLOR_BGR2RGB) / 255.0
x = [
    torch.from_numpy(np.transpose(img1, axes=[2, 0, 1])).to(torch.float32),
    torch.from_numpy(np.transpose(img1, axes=[2, 0, 1])).to(torch.float32)
]

out = model(x)

boxes = out[0]['boxes']
labels = out[0]['labels']
scores = out[0]['scores']
for idx in range(boxes.shape[0]):
    if scores[idx] >= SCORE_THRESHOLD:
        x1, y1, x2, y2 = boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3]
        name = coco_names.get(str(labels[idx].item()))
        # cv2.rectangle(img,(x1,y1),(x2,y2),colors[labels[idx].item()],thickness=2)
        cv2.rectangle(img1, (x1, y1), (x2, y2), random_color(), thickness=2)
        cv2.putText(img1, text=name, org=(x1, y1 + 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(0, 0, 255))

cv2.imshow('img1', img1)
cv2.waitKey(0)
cv2.imwrite('img_out',img1)
