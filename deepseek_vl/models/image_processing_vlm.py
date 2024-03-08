from PIL import Image
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional
from typing import List, Union, Tuple
from transformers import PretrainedConfig, AutoImageProcessor
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_utils import to_numpy_array
from transformers.utils import logging

logger = logging.get_logger(__name__)

ImageType = Union[np.ndarray, torch.Tensor, Image.Image]
IMAGENET_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGENET_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


class VLMImageProcessorConfig(PretrainedConfig):

    model_type = "deepseek_vlm"
    image_size: int
    min_size: int
    image_mean: Union[Tuple[float, float, float], List[float]]
    image_std: Union[Tuple[float, float, float], List[float]]
    rescale_factor: float
    do_normalize: bool

    def __init__(
            self,
            image_size: int,
            min_size: int = 14,
            image_mean: Union[Tuple[float, float, float], List[float]] = (0.48145466, 0.4578275, 0.40821073),
            image_std: Union[Tuple[float, float, float], List[float]] = (0.26862954, 0.26130258, 0.27577711),
            rescale_factor: float = 1.0 / 255.0,
            do_normalize: bool = True, **kwargs
    ):

        self.image_size = image_size
        self.min_size = min_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize

        super().__init__(**kwargs)


class VLMImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values"]

    def __init__(
            self,
            image_size: int,
            min_size: int = 14,
            image_mean: Union[Tuple[float, float, float], List[float]] = (0.48145466, 0.4578275, 0.40821073),
            image_std: Union[Tuple[float, float, float], List[float]] = (0.26862954, 0.26130258, 0.27577711),
            rescale_factor: float = 1.0 / 255.0,
            do_normalize: bool = True, **kwargs
    ):
        super().__init__(**kwargs)

        self.image_size = image_size
        self.rescale_factor = rescale_factor
        self.image_mean = image_mean
        self.image_std = image_std
        self.min_size = min_size
        self.do_normalize = do_normalize

        if image_mean is None:
            self.background_color = (127, 127, 127)
        else:
            self.background_color = tuple([int(x * 255) for x in image_mean])

    def resize(self, pil_img: Image) -> np.ndarray:
        """

        Args:
            pil_img (PIL.Image): [H, W, 3] in PIL.Image in RGB

        Returns:
            x (np.ndarray): [3, self.image_size, self.image_size]
        """

        width, height = pil_img.size
        max_size = max(width, height)

        size = [
            max(int(height / max_size * self.image_size), self.min_size),
            max(int(width / max_size * self.image_size), self.min_size)
        ]

        if width <= 0 or height <= 0 or size[0] <= 0 or size[1] <= 0:
            print(f"orig size = {pil_img.size}, new size = {size}")
            raise ValueError("Invalid size!")

        pil_img = torchvision.transforms.functional.resize(
            pil_img, size, interpolation=torchvision.transforms.functional.InterpolationMode.BICUBIC, antialias=True
        )

        pil_img = expand2square(pil_img, self.background_color)
        x = to_numpy_array(pil_img)

        # [H, W, 3] -> [3, H, W]
        x = np.transpose(x, (2, 0, 1))

        return x

    def preprocess(self, images, return_tensors: str = "pt", **kwargs) -> BatchFeature:

        # resize and pad to [self.image_size, self.image_size]
        # then convert from [H, W, 3] to [3, H, W]
        images: List[np.ndarray] = [self.resize(image) for image in images]

        # resacle from [0, 255] -> [0, 1]
        images = [
            self.rescale(image=image, scale=self.rescale_factor, input_data_format="channels_first")
            for image in images
        ]

        # normalize
        if self.do_normalize:
            images = [
                self.normalize(image=image, mean=self.image_mean, std=self.image_std,
                               input_data_format="channels_first")
                for image in images
            ]

        data = {"pixel_values": images}
        return BatchFeature(data=data, tensor_type=return_tensors)

    @property
    def default_shape(self):
        return [3, self.image_size, self.image_size]


# AutoConfig.register("deepseek_vlm", VLMImageProcessorConfig)
AutoImageProcessor.register(VLMImageProcessorConfig, VLMImageProcessor)


if __name__ == "__main__":
    image_processor = VLMImageProcessor(
        image_size=1024,
        image_mean=IMAGENET_INCEPTION_MEAN,
        image_std=IMAGENET_INCEPTION_STD,
        do_normalize=True
    )
