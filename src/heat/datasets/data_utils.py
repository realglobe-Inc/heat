import numpy as np
from PIL import ImageFilter
from torch.utils.data.dataloader import default_collate
from torchvision import transforms

from heat.utils.nn_utils import positional_encoding_2d


def random_blur(radius=2.0):
    blur = GaussianBlur(radius=radius)
    full_transform = transforms.RandomApply([blur], p=0.3)
    return full_transform


class ImageFilterTransform:
    def __init__(self):
        raise NotImplementedError

    def __call__(self, img):
        return img.filter(self.filter)


class GaussianBlur(ImageFilterTransform):
    def __init__(self, radius=2.0):
        self.filter = ImageFilter.GaussianBlur(radius=radius)


def collate_fn(data):
    batched_data = {}
    for field in data[0]:
        if field in ["annot", "rec_mat"]:
            batch_values = [item[field] for item in data]
        else:
            batch_values = default_collate([d[field] for d in data])
        if field in ["pixel_features", "pixel_labels", "gauss_labels"]:
            batch_values = batch_values.float()
        batched_data[field] = batch_values

    return batched_data


def get_pixel_features(image_size, d_pe=128):
    all_pe = positional_encoding_2d(d_pe, image_size, image_size)
    pixels_x = np.arange(0, image_size)
    pixels_y = np.arange(0, image_size)

    xv, yv = np.meshgrid(pixels_x, pixels_y)
    all_pixels = []
    for i in range(xv.shape[0]):
        ps = np.stack([xv[i], yv[i]], axis=-1)
        all_pixels.append(ps)
    pixels = np.stack(all_pixels, axis=0)

    pixel_features = all_pe[:, pixels[:, :, 1], pixels[:, :, 0]]
    pixel_features = pixel_features.permute(1, 2, 0)
    return pixels, pixel_features
