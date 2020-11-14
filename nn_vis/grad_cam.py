import torch.nn.functional as F
import matplotlib.cm as cm
import numpy as np
import cv2
import torch
from torchvision import models, transforms
from PIL import Image
import pydevd
pydevd.settrace(suspend=False, trace_only_current_thread=True)

def denormalize(img_tensor):
    """
    denormalize an img tensor (i.e. multiply variance then add mean)
    """
    means = np.array([0.485, 0.456, 0.406])
    means = np.reshape(means, newshape=(1, 1, 3))
    stds = np.array([0.229, 0.224, 0.225])
    stds = np.reshape(stds, newshape=(1, 1, 3))

    img = np.transpose(img_tensor.numpy(), axes=[1, 2, 0])
    img *= stds
    img += means
    return img


def visualize_gradcam_with_img(gcam, raw_image, save_fig_name=None):
    """
    visualize grad cam with original image.

    raw_image: within range [0,1]
    """

    gcam = gcam - np.min(gcam)
    gcam = gcam / np.max(gcam)

    heatmap = cv2.applyColorMap(np.uint8(255 * gcam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    gcam = heatmap + np.float32(raw_image[:, :, ::-1])
    gcam = gcam / np.max(gcam)
    if save_fig_name is not None:
        cv2.imwrite(save_fig_name, np.uint8(255 * gcam))
    else:
        cv2.imshow("gram_cam", gcam)
        cv2.waitKey(0)


# https://github.com/kazuto1011/grad-cam-pytorch/blob/
# fd10ff7fc85ae064938531235a5dd3889ca46fed/grad_cam.py#L35#
class Grad_CAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # which layer to compute CAM grad
        self.target_layer = target_layer

        self.feature_maps_pool = {}
        self.grads_pool = {}

        self.register_hooks()

    def register_hooks(self):

        def save_feature_maps(name):
            def forward_hook_fn(module, input, output):
                self.feature_maps_pool[name] = output

            return forward_hook_fn

        def save_grads(name):
            def backward_hook_fn(module, grad_input, grad_output):
                # grad_input is a tuple w.r.t. each weight and input,
                # for the activated feature maps (i.e., a tuple contains only one tensor),
                # we extract that explicitly.
                self.grads_pool[name] = grad_output[0]

            return backward_hook_fn

        for name, module in self.model.named_modules():
            # if name == self.candidate_layer:
            module.register_forward_hook(save_feature_maps(name))
            module.register_backward_hook(save_grads(name))

    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    def visualize(self, input_image, target_class):
        input_image_size = input_image.size(2), input_image.size(3)
        # only use feature extractor part
        logits = self.model(input_image)
        probs = F.softmax(logits, dim=1)
        pred_class = probs.argmax().item()

        # generate one-hot code, as the starting point of back propagation
        one_hot = torch.zeros(probs.shape, dtype=torch.float)

        if target_class is not None:
            one_hot[0][target_class] = 1
        else:
            one_hot[0][pred_class] = 1

        self.model.zero_grad()
        logits.backward(one_hot)

        fmaps = self._find(self.feature_maps_pool, self.target_layer)
        grads = self._find(self.grads_pool, self.target_layer)

        weights = F.adaptive_avg_pool2d(grads, 1)
        gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
        gcam = F.relu(gcam)
        gcam = F.interpolate(
            gcam, input_image_size, mode="bilinear", align_corners=False
        )

        B, C, H, W = gcam.shape
        gcam = gcam.view(B, -1)
        gcam -= gcam.min(dim=1, keepdim=True)[0]
        gcam /= gcam.max(dim=1, keepdim=True)[0]
        gcam = gcam.view(B, C, H, W)

        return gcam


def grad_cam_resnet50(img_path, save_path):
    I = Image.open(img_path).convert('RGB')
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    size = 224

    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
    ])

    tensor = transform(I).unsqueeze(0).requires_grad_()
    model = models.resnet50(pretrained=True)

    grad_cam = Grad_CAM(model, target_layer="layer4.2")
    gcam = grad_cam.visualize(tensor, None)
    visualize_gradcam_with_img(gcam.detach().numpy()[0][0],
                               denormalize(tensor[0].detach()),
                               save_fig_name=save_path)


if __name__ == '__main__':
    grad_cam_resnet50("./plane.jpeg", None)
