# code reference: https://zhuanlan.zhihu.com/p/75054200

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt


class Guided_backprop:
    def __init__(self, model):
        self.model = model
        self.image_recon = None
        self.activation_maps = []
        self.model.eval()
        self.register_hooks()

    def register_hooks(self):
        def first_layer_hook_fn(module, grad_input, grad_output):
            self.image_recon = grad_input[0]

        def forward_hook_fn(module, input, output):
            # save the output of each ReLU layer,which is used for guided backpropagation then.
            self.activation_maps.append(output)

        def backward_hook_fn(module, grad_input, grad_output):
            # use activation maps in forward pass as guide
            guide = self.activation_maps.pop()
            # only use the "activated" positions
            guide[guide > 0] = 1

            pos_grad_output = torch.clamp(grad_output[0], min=0.0)
            new_grad_input = pos_grad_output * guide

            # ReLU's gradient
            return (new_grad_input,)

        modules = list(self.model.modules())
        for module in modules:
            if isinstance(module, nn.ReLU):
                module.register_forward_hook(forward_hook_fn)
                module.register_backward_hook(backward_hook_fn)

        first_layer = modules[1]
        first_layer.register_backward_hook(first_layer_hook_fn)

    def visualize(self, input_image, target_class):
        model_output = self.model(input_image)
        self.model.zero_grad()
        pred_class = model_output.argmax().item()

        # generate one-hot code, as the starting point of back propagation
        grad_target_map = torch.zeros(model_output.shape, dtype=torch.float)

        if target_class is not None:
            grad_target_map[0][target_class] = 1
        else:
            grad_target_map[0][pred_class] = 1

        model_output.backward(grad_target_map)
        result = self.image_recon.data[0].permute(1, 2, 0)
        return result.numpy()


def normalize(img):
    img_norm = (img - img.mean()) / img.std()
    # re-normalize with mean=0.5, std=0.1
    img_norm *= 0.1
    img_norm += 0.5
    img_norm = img_norm.clip(0, 1)
    return img_norm


def guided_propagation_resnet50(img_path, save_path):
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

    guided_bp = Guided_backprop(model)
    result = guided_bp.visualize(tensor, None)

    result = normalize(result)
    plt.imsave(save_path, result)
    # plt.show()


if __name__ == '__main__':
    guided_propagation_resnet50("./dog.jpeg", "./dog-vis.jpeg")
