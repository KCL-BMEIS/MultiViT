
from typing import Callable, Dict, List, Optional, Tuple, Union

import os

import logging

import io

from collections import defaultdict

import time

import numpy as np

from PIL import Image


import torch
from torch import nn

from datasets import Dataset

import batchgenerators as bgs

from transformers.image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from transformers.image_transforms import resize, to_channel_dimension_format
from transformers.image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
)
from transformers.utils import TensorType, logging as tlogging


tlogger = tlogging.get_logger(__name__)



class CachedLoader:

    def __init__(
            self,
            data_path,
    ):
        self.data_path = data_path
        self.images = dict()


    def __getitem__(self, item):
        with torch.no_grad():
            if item not in self.images:
                with open(os.path.join(self.data_path, item), 'rb') as f:
                    self.images[item] = f.read()

                logger = logging.getLogger(f"CachedLoader")
                logger.info(f"caching '{item}'")

            image = np.array(Image.open(io.BytesIO(self.images[item])))
            image = torch.tensor(image).requires_grad_(False).permute(2, 0, 1)
            image = image * (1.0 / 255.0)

            return image



class BasicLoader:

    recip = 1.0 / 255.0

    def __init__(self, data_path):
        self.data_path = data_path


    def __getitem__(self, item):
        n = np.array(Image.open(os.path.join(self.data_path, item)))
        t = torch.tensor(n).permute(2, 0, 1) * BasicLoader.recip
        return t.requires_grad_(False)



class MultiViTSimpleDataset(Dataset):

    def __init__(
            self,
            image_loader,
            dataframe,
            processor=None,
            rng=None,
            **kwargs
    ):
        logger = logging.getLogger(f"{__class__.__name__}.__init__")

        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng

        self.image_loader = image_loader
        self.dataframe = dataframe
        self.processor = processor


    def __len__(self):
        return len(self.dataframe.index)


    def __getitem__(self, index):

        rdict = dict()
        patient_fields = list()
        image_fps = [self.dataframe['image_name'].iloc[i] for i in index]
        patient_fields = [self.dataframe.iloc[i].to_dict() for i in index]
        pixel_values = [self.image_loader[i] for i in image_fps]
        pixel_values = self.processor({'pixel_values': pixel_values}, return_tensors='pt')

        rdict['pixel_values'] = pixel_values['pixel_values']
        for k in patient_fields[0].keys():
            rdict[k] = [patient_fields[r][k] for r in range(len(index))]
        return rdict



class MultiViTDataset(Dataset):

    def __init__(
            self,
            image_loader,
            dataframe,
            images_per_sample,
            processor=None,
            rng=None,
            **kwargs
        ):
        logger = logging.getLogger(f"{__class__.__name__}.__init__")
        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng

        self.image_loader = image_loader
        self.dataframe = dataframe
        self.images_per_sample = images_per_sample
        logger.info(f"dataset fields: {self.dataframe.columns}")
        self.subjects = self.dataframe['patient_id'].unique()
        logger.info(f"subject_ids: len({len(self.subjects)}: {self.subjects}")

        id_to_sample = defaultdict(list)
        for i, _id in enumerate(self.dataframe['patient_id']):
            id_to_sample[_id].append(i)
        self.id_to_sample = id_to_sample

        logger.info(f"id to sample mapping: {id_to_sample.items()}")

        self.processor = processor


    def __len__(self):
        return len(self.subjects)


    def __getitem__(self, index):
        batch_size = len(index)
        rdict = dict()
        processor_inputs = list()
        patient_fields = list()

        # construct the initial collections as (batch, sample, ...)
        for _ind in index:
            pid = self.subjects[_ind]
            image_indices = self.id_to_sample[pid]
            rows_to_use = self.rng.choice(image_indices, self.images_per_sample, replace=False)
            processor_inputs.append([self.dataframe['image_name'].iloc[r] for r in rows_to_use])
            patient_fields.append([self.dataframe.iloc[r].to_dict() for r in rows_to_use])

        # transpose the collections to be (sample, batch, ...)
        processor_inputs = list(zip(*processor_inputs))
        patient_fields = list(zip(*patient_fields))

        pixel_values = [[self.image_loader[pi] for pi in s] for s in processor_inputs]
        processor_outputs = [self.processor({'pixel_values': pv}, return_tensors='pt') for pv in pixel_values]
        rdict['pixel_values'] = [po['pixel_values'] for po in processor_outputs]
        if 'pixel_values_hr' in processor_outputs[0]:
            rdict['pixel_values_hr'] = [po['pixel_values_hr'] for po in processor_outputs]
        for k in patient_fields[0][0].keys():
            rdict[k] = [[patient_fields[i][j][k] for j in range(len(index))] for i in range(self.images_per_sample)]
        return rdict



class MultiViTImageProcessor(BaseImageProcessor):

    def __init__(
            self,
            size: Tuple[int, int],
            size_hr: Tuple[int, int] | None = None,
            augment: Callable | None = None,
            image_mean: Tuple[float, float, float] | None = None,
            image_std: Tuple[float, float, float] | None = None,
    ):
        if not isinstance(size, tuple) or len(size) != 2 or not all(isinstance(i, int) for i in size):
            raise ValueError(f"size {size} must be a tuple of two integers")
        if size_hr is not None:
            if not isinstance(size_hr, tuple) or len(size_hr) != 2 or not all(isinstance(i, int) for i in size_hr):
                raise ValueError(f"size_hr {size_hr} must be a tuple of two integers")
        if augment is not None and not callable(augment):
            raise ValueError(f"augment {augment} must be a callable")

        self.size = size
        self.size_hr = size_hr
        self.image_mean = image_mean
        self.image_std = image_std
        self.augment = augment


    def preprocess(
        self,
        images,
        return_tensors: str = 'pt',
        **kwargs
    ):
        with torch.no_grad():
            t0 = time.time()

            if self.augment is None:
                raise ValueError("self.augment must be set")
            images_lr, images_hr, augments = self.augment(images['pixel_values'], self.size, self.size_hr)

            if self.image_mean != None and self.image_std != None:
                images_lr = [normalize(image, self.image_mean, self.image_std) for image in images_lr]
                if self.size_hr is not None:
                    images_hr = [normalize(image, self.image_mean, self.image_std) for image in images_hr]

            data = {"pixel_values": torch.cat(images_lr)}
            if images_hr is not None:
                data["pixel_values_hr"] = torch.cat(images_hr)

            print(f"Preprocessing time: {time.time() - t0}")

        return data



def normalize(image, mean, std):
    """
    Normalise the data with the given mean and standard deviation.
    The data is assumed to have the following shape:
      - (batch, channels(RGB), height, width)
    'mean' and 'std' should be a sequence of values for each channel.
    """
    if not isinstance(mean, tuple) and len(mean) != image.shape[1]:
        raise ValueError(f"mean {mean} must be a tuple of length 3")
    if not isinstance(std, tuple) and len(std) != image.shape[1]:
        raise ValueError(f"std {std} must be a tuple of length 3")

    mean = torch.as_tensor(mean, dtype=image.dtype, device=image.device)
    mean = mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    std = torch.as_tensor(std, dtype=image.dtype, device=image.device)
    std = std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    image_out = (image - mean) / std

    return image_out



def preprocess_nnunet():

    # SpatialTransform(p=0.2, rot=(-30, 30), scale=(0.7, 1.4))
    # GaussianNoiseTransform(p=0.1, variance=(0, 0.1))
    # GaussianBlurTransform(p=0.2, sigma=(0.5, 1.0))
    # BrightnessMultiplicativeTransform(p=0.15, multiplier=(0.75, 1.25))
    # ConstrastAugmentationTransform(p=0.15, factor=(0.75, 1.25))
    # SimulateLowResolutionTransform(p=0.25, zoom=(0.5, 1.0))
    # GammaTransform(p=0.1, gamma_range=(0.7, 1.5))
    # GammaTransform(p=0.3, gamma_range=(0.7, 1.5))
    # MirrorTransform(p=1.0)
    # ~MaskTransform()
    # ~RemoveLabelTransform()
    # ~RenameTransform()
    # ~DownsampleSegForDSTransform2()
    # ~NumpyToTensor()


    transforms = [
        bgs.SpatialTransform(p)
    ]

    def _inner(data):
        pass

    return _inner