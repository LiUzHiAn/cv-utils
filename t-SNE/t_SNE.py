import cv2
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from torch.utils.data import Dataset
from torchvision.utils import make_grid
import torch
import numpy as np
from sklearn import manifold

# plt.switch_backend('agg')
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.patheffects as path_eff


class ClusterDataset(Dataset):
    def __init__(self, root, dataset_type, img_type, training=True):
        """
        Specific dataset to be clustered.

        Args:
            root: root path to store raw data
            dataset_type:  specified dataset type, must be one of ['MNIST', 'FashionMNIST', 'CIFAR10']
            img_type: type of image in the dataset, must be one of ['rgb', 'grayscale']
            training: whether to use the training set
        """
        assert img_type in ['rgb', 'grayscale']
        assert dataset_type in ['MNIST', 'FashionMNIST', 'CIFAR10']

        self.training = training
        self.dataset_type = dataset_type
        self.img_type = img_type

        if dataset_type == 'MNIST':
            self.dataset = MNIST(root, train=self.training, download=True)
        elif dataset_type == 'FashionMNIST':
            self.dataset = FashionMNIST(root, train=self.training, download=True)
        elif dataset_type == 'CIFAR10':
            self.dataset = CIFAR10(root, train=self.training, download=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_raw, label = self.dataset[idx]
        return img_raw, label


def plot_image(arr, show_ticks=False, show_image=True, save_pth=''):
    """
    Plot an 2d image, to be shown in window or saved to file.

    Args:
        arr (3D ndarray): with shape [H,W,C], C==1 for grayscale, C==3 for rgb
        show_ticks: whether to show ticks in figure
        show_image: show in a window
        save_pth: save into file, if specified, the `show_image` must be False

    Returns:

    """
    if not show_image:
        assert save_pth != ''

    fig = plt.figure()

    if not show_ticks:
        plt.xticks([])
        plt.yticks([])

    plt.imshow(arr)
    if show_image:
        plt.show()
    else:
        plt.savefig(save_pth, dpi=1000, bbox_inches='tight')

    plt.close(fig)


def plot_2d_tsne(proj, y=None, imgs=None, label_dict=None, vis_style="image_scatter",
                 size=20, marker='o', show_ticks=False,
                 show_legend=False, save_path='', show_fig=True):
    """
    Plot 2D projected low-dimension data of t-SNE result.

    Args:
        proj: projected data with shape [num_samples, num_feats_dim]
        y: corresponding label index with shape [num_samples,]
        imgs: corresponding image data with shape [num_samples,C,H,W]
        vis_style: t-SNE visualization style
            if set to "image_scatter", then use AnnotationBox to show original images at projected 2D position;
            if set to "point_scatter", then treat every sample as a point, and scattered in colors for each label.
        label_dict: label index to class name mapping.
        size: size of points
        marker: marker of points
        show_ticks: whether to show ticks in figure, if False, we put text to each cluster center
        show_legend: whether to show legend
        save_path: path to be saved
        show_fig: show figure in a window, if False, the `save_path` must be given.

    Returns:

    """

    x = np.asarray(proj)

    x_min, x_max = np.min(x, 0), np.max(x, 0)
    X = (x - x_min) / (x_max - x_min)

    assert len(x.shape) == 2
    if y is not None:
        y = np.asarray(y)
        assert len(y.shape) == 1

    fig = plt.figure(figsize=(16, 10))
    ax = plt.subplot(111)

    if not show_ticks:
        plt.xticks([])
        plt.yticks([])

    if y is None:
        plt.scatter(x[:, 0], x[:, 1], s=size, c='k', marker=marker)
    else:
        # Visualization style
        if vis_style == "image_scatter":
            # just something big
            shown_images = np.array([[1., 1.]])
            for i in range(len(x)):
                dist = np.sum((X[i] - shown_images) ** 2, 1)
                if np.min(dist) < 4e-5:
                    # don't show points that are too close
                    continue
                shown_images = np.r_[shown_images, [X[i]]]
                # convert Tensor [C,H,W] to [H,W,C]
                img_np = np.transpose(imgs[i].numpy(), axes=[1, 2, 0])
                # downscale for better visualization
                img_np = cv2.resize(img_np, dsize=(14, 14))
                if len(img_np.shape) > 2 and img_np.shape[2] == 1:
                    img_np = np.squeeze(img_np, axis=2)
                imagebox = AnnotationBbox(OffsetImage(img_np, cmap=plt.cm.gray_r), X[i], frameon=False, )
                ax.add_artist(imagebox)

        elif vis_style == "point_scatter":
            jet = plt.get_cmap('jet')
            y_mapping = {i: label_idx for i, label_idx in enumerate(set(y))}
            c_norm = colors.Normalize(vmin=0, vmax=len(y_mapping) - 1)
            scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=jet)
            for i, label_idx in y_mapping.items():
                color_val = scalar_map.to_rgba(i)
                selected = x[y == label_idx]
                label = label_idx if label_dict is None else label_dict[label_idx]
                plt.scatter(selected[:, 0], selected[:, 1], s=size, c=[color_val], marker=marker, label=label)

        if show_legend:
            plt.legend(loc='upper right')
        else:
            for label_idx in range(len(label_dict)):
                selected = x[y == label_idx]
                # centering position of a specific cluster
                center_x, center_y = np.median(selected, axis=0)
                label = label_idx if label_dict is None else label_dict[label_idx]
                txt = plt.text(center_x, center_y, str(label), fontsize=15)
                txt.set_path_effects([path_eff.Stroke(linewidth=3, foreground='w'), path_eff.Normal()])

    if show_fig:
        plt.show()
    else:
        plt.savefig(save_path)

    plt.close(fig)


def load_image_data(data_root='./data', dataset_type="FashionMNIST", img_type="grayscale", training=True,
                    num_data_used=5000, verbose=False):
    """
    Load specified number of image data for given dataset.

    Args:
        data_root: root path to download dataset
        dataset_type:  specified dataset type, must be one of ['MNIST', 'FashionMNIST', 'CIFAR10']
        img_type: type of image in the dataset, must be one of ['rgb', 'grayscale']
        training: whether to use the training set
        num_data_used: how many samples to be used

    Returns:
        imgs (Tensor): with shape [num_samples,C,H,W]
        labels: corresponding label for each image with shape [num_samples,]
        labels_dict: label index to class name mapping

    """
    ds = ClusterDataset(data_root, dataset_type, img_type, training=training)
    imgs, labels = ds.dataset.data, ds.dataset.targets

    if len(labels) <= num_data_used:
        num_data_used = len(labels)
    data_used_idx = np.random.choice(np.arange(len(labels)), num_data_used)
    imgs, labels = imgs[data_used_idx], labels[data_used_idx]

    # wrap to [bs,C,H,W] format and scale into [0,1]
    imgs = torch.unsqueeze(imgs, dim=1) if ds.img_type == 'grayscale' else imgs
    imgs = imgs / 255.0

    if verbose:
        NUM_VIS = 100
        random_choice = np.random.choice(np.arange(len(labels)), NUM_VIS)
        grid_vis = make_grid(imgs[random_choice], nrow=10, padding=0, normalize=True, range=(0, 1))
        # show grid sampled images
        plot_image(np.transpose(grid_vis.numpy(), axes=[1, 2, 0]))

    labels_dict = {}
    for idx, cls in enumerate(ds.dataset.classes):
        labels_dict[idx] = cls

    return imgs, labels, labels_dict


def tsne_demo():
    # 1. load img data
    imgs, labels, labels_dict = load_image_data(dataset_type="FashionMNIST", training=False, num_data_used=1000)

    # 2. apply t-SNE to project the data into low-dimension space, 2 here.
    # For image data, we simply flatten the spatial dimensions (i.e., H and W),
    # TODO: applying a NN like VGG to extract semantic features.
    imgs_flatten = torch.flatten(imgs, start_dim=1)
    proj = manifold.TSNE(n_components=2, random_state=2021).fit_transform(imgs_flatten)

    # 3. visualize the projected data for each cluster
    plot_2d_tsne(proj, labels, imgs, labels_dict, vis_style="image_scatter", show_legend=True,
                 show_fig=False, save_path="t-SNE-FashionMNIST_pointscatter.png")


if __name__ == '__main__':
    tsne_demo()
