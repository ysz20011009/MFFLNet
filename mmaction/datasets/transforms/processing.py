# Copyright (c) OpenMMLab. All rights reserved.
import random
import warnings
from numbers import Number
from typing import Sequence
import torch
import cv2
import mmcv
import mmengine
import numpy as np
from mmcv.transforms import BaseTransform
from mmcv.transforms.utils import cache_randomness
from torch.nn.modules.utils import _pair
from mmcv import imresize
import copy
from mmaction.registry import TRANSFORMS


def _combine_quadruple(a, b):
    return a[0] + a[2] * b[0], a[1] + a[3] * b[1], a[2] * b[2], a[3] * b[3]


def _flip_quadruple(a):
    return 1 - a[0] - a[2], a[1], a[2], a[3]


def _init_lazy_if_proper(results, lazy):
    """Initialize lazy operation properly.

    Make sure that a lazy operation is properly initialized,
    and avoid a non-lazy operation accidentally getting mixed in.

    Required keys in results are "imgs" if "img_shape" not in results,
    otherwise, Required keys in results are "img_shape", add or modified keys
    are "img_shape", "lazy".
    Add or modified keys in "lazy" are "original_shape", "crop_bbox", "flip",
    "flip_direction", "interpolation".

    Args:
        results (dict): A dict stores data pipeline result.
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    if 'img_shape' not in results:
        results['img_shape'] = results['imgs'][0].shape[:2]
    if lazy:
        if 'lazy' not in results:
            img_h, img_w = results['img_shape']
            lazyop = dict()
            lazyop['original_shape'] = results['img_shape']
            lazyop['crop_bbox'] = np.array([0, 0, img_w, img_h],
                                           dtype=np.float32)
            lazyop['flip'] = False
            lazyop['flip_direction'] = None
            lazyop['interpolation'] = None
            results['lazy'] = lazyop
    else:
        assert 'lazy' not in results, 'Use Fuse after lazy operations'


@TRANSFORMS.register_module()
class Fuse(BaseTransform):
    """Fuse lazy operations.

    Fusion order:
        crop -> resize -> flip

    Required keys are "imgs", "img_shape" and "lazy", added or modified keys
    are "imgs", "lazy".
    Required keys in "lazy" are "crop_bbox", "interpolation", "flip_direction".
    """

    def transform(self, results):
        """Fuse lazy operations.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if 'lazy' not in results:
            raise ValueError('No lazy operation detected')
        lazyop = results['lazy']
        imgs = results['imgs']

        # crop
        left, top, right, bottom = lazyop['crop_bbox'].round().astype(int)
        imgs = [img[top:bottom, left:right] for img in imgs]

        # resize
        img_h, img_w = results['img_shape']
        if lazyop['interpolation'] is None:
            interpolation = 'bilinear'
        else:
            interpolation = lazyop['interpolation']
        imgs = [
            mmcv.imresize(img, (img_w, img_h), interpolation=interpolation)
            for img in imgs
        ]

        # flip
        if lazyop['flip']:
            for img in imgs:
                mmcv.imflip_(img, lazyop['flip_direction'])

        results['imgs'] = imgs
        del results['lazy']

        return results


@TRANSFORMS.register_module()
class RandomCrop(BaseTransform):
    """Vanilla square random crop that specifics the output size.

    Required keys in results are "img_shape", "keypoint" (optional), "imgs"
    (optional), added or modified keys are "keypoint", "imgs", "lazy"; Required
    keys in "lazy" are "flip", "crop_bbox", added or modified key is
    "crop_bbox".

    Args:
        size (int): The output size of the images.
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self, size, lazy=False):
        if not isinstance(size, int):
            raise TypeError(f'Size must be an int, but got {type(size)}')
        self.size = size
        self.lazy = lazy

    @staticmethod
    def _crop_kps(kps, crop_bbox):
        """Static method for cropping keypoint."""
        return kps - crop_bbox[:2]

    @staticmethod
    def _crop_imgs(imgs, crop_bbox):
        """Static method for cropping images."""
        x1, y1, x2, y2 = crop_bbox
        return [img[y1:y2, x1:x2] for img in imgs]

    @staticmethod
    def _box_crop(box, crop_bbox):
        """Crop the bounding boxes according to the crop_bbox.

        Args:
            box (np.ndarray): The bounding boxes.
            crop_bbox(np.ndarray): The bbox used to crop the original image.
        """

        x1, y1, x2, y2 = crop_bbox
        img_w, img_h = x2 - x1, y2 - y1

        box_ = box.copy()
        box_[..., 0::2] = np.clip(box[..., 0::2] - x1, 0, img_w - 1)
        box_[..., 1::2] = np.clip(box[..., 1::2] - y1, 0, img_h - 1)
        return box_

    def _all_box_crop(self, results, crop_bbox):
        """Crop the gt_bboxes and proposals in results according to crop_bbox.

        Args:
            results (dict): All information about the sample, which contain
                'gt_bboxes' and 'proposals' (optional).
            crop_bbox(np.ndarray): The bbox used to crop the original image.
        """
        results['gt_bboxes'] = self._box_crop(results['gt_bboxes'], crop_bbox)
        if 'proposals' in results and results['proposals'] is not None:
            assert results['proposals'].shape[1] == 4
            results['proposals'] = self._box_crop(results['proposals'],
                                                  crop_bbox)
        return results

    def transform(self, results):
        """Performs the RandomCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, self.lazy)
        if 'keypoint' in results:
            assert not self.lazy, ('Keypoint Augmentations are not compatible '
                                   'with lazy == True')

        img_h, img_w = results['img_shape']
        assert self.size <= img_h and self.size <= img_w

        y_offset = 0
        x_offset = 0
        if img_h > self.size:
            y_offset = int(np.random.randint(0, img_h - self.size))
        if img_w > self.size:
            x_offset = int(np.random.randint(0, img_w - self.size))

        if 'crop_quadruple' not in results:
            results['crop_quadruple'] = np.array(
                [0, 0, 1, 1],  # x, y, w, h
                dtype=np.float32)

        x_ratio, y_ratio = x_offset / img_w, y_offset / img_h
        w_ratio, h_ratio = self.size / img_w, self.size / img_h

        old_crop_quadruple = results['crop_quadruple']
        old_x_ratio, old_y_ratio = old_crop_quadruple[0], old_crop_quadruple[1]
        old_w_ratio, old_h_ratio = old_crop_quadruple[2], old_crop_quadruple[3]
        new_crop_quadruple = [
            old_x_ratio + x_ratio * old_w_ratio,
            old_y_ratio + y_ratio * old_h_ratio, w_ratio * old_w_ratio,
            h_ratio * old_h_ratio
        ]
        results['crop_quadruple'] = np.array(
            new_crop_quadruple, dtype=np.float32)

        new_h, new_w = self.size, self.size

        crop_bbox = np.array(
            [x_offset, y_offset, x_offset + new_w, y_offset + new_h])
        results['crop_bbox'] = crop_bbox

        results['img_shape'] = (new_h, new_w)

        if not self.lazy:
            if 'keypoint' in results:
                results['keypoint'] = self._crop_kps(results['keypoint'],
                                                     crop_bbox)
            if 'imgs' in results:
                results['imgs'] = self._crop_imgs(results['imgs'], crop_bbox)
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')

            # record crop_bbox in lazyop dict to ensure only crop once in Fuse
            lazy_left, lazy_top, lazy_right, lazy_bottom = lazyop['crop_bbox']
            left = x_offset * (lazy_right - lazy_left) / img_w
            right = (x_offset + new_w) * (lazy_right - lazy_left) / img_w
            top = y_offset * (lazy_bottom - lazy_top) / img_h
            bottom = (y_offset + new_h) * (lazy_bottom - lazy_top) / img_h
            lazyop['crop_bbox'] = np.array([(lazy_left + left),
                                            (lazy_top + top),
                                            (lazy_left + right),
                                            (lazy_top + bottom)],
                                           dtype=np.float32)

        # Process entity boxes
        if 'gt_bboxes' in results:
            assert not self.lazy
            results = self._all_box_crop(results, results['crop_bbox'])

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}(size={self.size}, '
                    f'lazy={self.lazy})')
        return repr_str


@TRANSFORMS.register_module()
class RandomResizedCrop(RandomCrop):
    """Random crop that specifics the area and height-weight ratio range.

    Required keys in results are "img_shape", "crop_bbox", "imgs" (optional),
    "keypoint" (optional), added or modified keys are "imgs", "keypoint",
    "crop_bbox" and "lazy"; Required keys in "lazy" are "flip", "crop_bbox",
    added or modified key is "crop_bbox".

    Args:
        area_range (Tuple[float]): The candidate area scales range of
            output cropped images. Default: (0.08, 1.0).
        aspect_ratio_range (Tuple[float]): The candidate aspect ratio range of
            output cropped images. Default: (3 / 4, 4 / 3).
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self,
                 area_range=(0.08, 1.0),
                 aspect_ratio_range=(3 / 4, 4 / 3),
                 lazy=False):
        self.area_range = area_range
        self.aspect_ratio_range = aspect_ratio_range
        self.lazy = lazy
        if not mmengine.is_tuple_of(self.area_range, float):
            raise TypeError(f'Area_range must be a tuple of float, '
                            f'but got {type(area_range)}')
        if not mmengine.is_tuple_of(self.aspect_ratio_range, float):
            raise TypeError(f'Aspect_ratio_range must be a tuple of float, '
                            f'but got {type(aspect_ratio_range)}')

    @staticmethod
    def get_crop_bbox(img_shape,
                      area_range,
                      aspect_ratio_range,
                      max_attempts=10):
        """Get a crop bbox given the area range and aspect ratio range.

        Args:
            img_shape (Tuple[int]): Image shape
            area_range (Tuple[float]): The candidate area scales range of
                output cropped images. Default: (0.08, 1.0).
            aspect_ratio_range (Tuple[float]): The candidate aspect
                ratio range of output cropped images. Default: (3 / 4, 4 / 3).
                max_attempts (int): The maximum of attempts. Default: 10.
            max_attempts (int): Max attempts times to generate random candidate
                bounding box. If it doesn't qualified one, the center bounding
                box will be used.
        Returns:
            (list[int]) A random crop bbox within the area range and aspect
            ratio range.
        """
        assert 0 < area_range[0] <= area_range[1] <= 1
        assert 0 < aspect_ratio_range[0] <= aspect_ratio_range[1]

        img_h, img_w = img_shape
        area = img_h * img_w

        min_ar, max_ar = aspect_ratio_range
        aspect_ratios = np.exp(
            np.random.uniform(
                np.log(min_ar), np.log(max_ar), size=max_attempts))
        target_areas = np.random.uniform(*area_range, size=max_attempts) * area
        candidate_crop_w = np.round(np.sqrt(target_areas *
                                            aspect_ratios)).astype(np.int32)
        candidate_crop_h = np.round(np.sqrt(target_areas /
                                            aspect_ratios)).astype(np.int32)

        for i in range(max_attempts):
            crop_w = candidate_crop_w[i]
            crop_h = candidate_crop_h[i]
            if crop_h <= img_h and crop_w <= img_w:
                x_offset = random.randint(0, img_w - crop_w)
                y_offset = random.randint(0, img_h - crop_h)
                return x_offset, y_offset, x_offset + crop_w, y_offset + crop_h

        # Fallback
        crop_size = min(img_h, img_w)
        x_offset = (img_w - crop_size) // 2
        y_offset = (img_h - crop_size) // 2
        return x_offset, y_offset, x_offset + crop_size, y_offset + crop_size

    def transform(self, results):
        """Performs the RandomResizeCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, self.lazy)
        if 'keypoint' in results:
            assert not self.lazy, ('Keypoint Augmentations are not compatible '
                                   'with lazy == True')

        img_h, img_w = results['img_shape']

        left, top, right, bottom = self.get_crop_bbox(
            (img_h, img_w), self.area_range, self.aspect_ratio_range)
        new_h, new_w = bottom - top, right - left

        if 'crop_quadruple' not in results:
            results['crop_quadruple'] = np.array(
                [0, 0, 1, 1],  # x, y, w, h
                dtype=np.float32)

        x_ratio, y_ratio = left / img_w, top / img_h
        w_ratio, h_ratio = new_w / img_w, new_h / img_h

        old_crop_quadruple = results['crop_quadruple']
        old_x_ratio, old_y_ratio = old_crop_quadruple[0], old_crop_quadruple[1]
        old_w_ratio, old_h_ratio = old_crop_quadruple[2], old_crop_quadruple[3]
        new_crop_quadruple = [
            old_x_ratio + x_ratio * old_w_ratio,
            old_y_ratio + y_ratio * old_h_ratio, w_ratio * old_w_ratio,
            h_ratio * old_h_ratio
        ]
        results['crop_quadruple'] = np.array(
            new_crop_quadruple, dtype=np.float32)

        crop_bbox = np.array([left, top, right, bottom])
        results['crop_bbox'] = crop_bbox
        results['img_shape'] = (new_h, new_w)

        if not self.lazy:
            if 'keypoint' in results:
                results['keypoint'] = self._crop_kps(results['keypoint'],
                                                     crop_bbox)
            if 'imgs' in results:
                results['imgs'] = self._crop_imgs(results['imgs'], crop_bbox)
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')

            # record crop_bbox in lazyop dict to ensure only crop once in Fuse
            lazy_left, lazy_top, lazy_right, lazy_bottom = lazyop['crop_bbox']
            left = left * (lazy_right - lazy_left) / img_w
            right = right * (lazy_right - lazy_left) / img_w
            top = top * (lazy_bottom - lazy_top) / img_h
            bottom = bottom * (lazy_bottom - lazy_top) / img_h
            lazyop['crop_bbox'] = np.array([(lazy_left + left),
                                            (lazy_top + top),
                                            (lazy_left + right),
                                            (lazy_top + bottom)],
                                           dtype=np.float32)

        if 'gt_bboxes' in results:
            assert not self.lazy
            results = self._all_box_crop(results, results['crop_bbox'])

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'area_range={self.area_range}, '
                    f'aspect_ratio_range={self.aspect_ratio_range}, '
                    f'lazy={self.lazy})')
        return repr_str


@TRANSFORMS.register_module()
class MultiScaleCrop(RandomCrop):
    """Crop images with a list of randomly selected scales.

    Randomly select the w and h scales from a list of scales. Scale of 1 means
    the base size, which is the minimal of image width and height. The scale
    level of w and h is controlled to be smaller than a certain value to
    prevent too large or small aspect ratio.

    Required keys are "img_shape", "imgs" (optional), "keypoint" (optional),
    added or modified keys are "imgs", "crop_bbox", "img_shape", "lazy" and
    "scales". Required keys in "lazy" are "crop_bbox", added or modified key is
    "crop_bbox".

    Args:
        input_size (int | tuple[int]): (w, h) of network input.
        scales (tuple[float]): width and height scales to be selected.
        max_wh_scale_gap (int): Maximum gap of w and h scale levels.
            Default: 1.
        random_crop (bool): If set to True, the cropping bbox will be randomly
            sampled, otherwise it will be sampler from fixed regions.
            Default: False.
        num_fixed_crops (int): If set to 5, the cropping bbox will keep 5
            basic fixed regions: "upper left", "upper right", "lower left",
            "lower right", "center". If set to 13, the cropping bbox will
            append another 8 fix regions: "center left", "center right",
            "lower center", "upper center", "upper left quarter",
            "upper right quarter", "lower left quarter", "lower right quarter".
            Default: 5.
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self,
                 input_size = (224,224),
                 scales=(0.6,0.7,0.8,0.9,1),
                 max_wh_scale_gap=2,
                 random_crop=False,
                 num_fixed_crops=13,
                 lazy=False):
        self.input_size = _pair(input_size)
        if not mmengine.is_tuple_of(self.input_size, int):
            raise TypeError(f'Input_size must be int or tuple of int, '
                            f'but got {type(input_size)}')

        if not isinstance(scales, tuple):
            raise TypeError(f'Scales must be tuple, but got {type(scales)}')

        if num_fixed_crops not in [5, 13]:
            raise ValueError(f'Num_fix_crops must be in {[5, 13]}, '
                             f'but got {num_fixed_crops}')

        self.scales = scales
        self.max_wh_scale_gap = max_wh_scale_gap
        self.random_crop = random_crop
        self.num_fixed_crops = num_fixed_crops
        self.lazy = lazy

    def transform(self, results):
        """Performs the MultiScaleCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, self.lazy)
        if 'keypoint' in results:
            assert not self.lazy, ('Keypoint Augmentations are not compatible '
                                   'with lazy == True')

        img_h, img_w = results['img_shape']
        base_size = min(img_h, img_w)
        crop_sizes = [int(base_size * s) for s in self.scales]

        candidate_sizes = []
        for i, h in enumerate(crop_sizes):
            for j, w in enumerate(crop_sizes):
                if abs(i - j) <= self.max_wh_scale_gap:
                    candidate_sizes.append([w, h])

        crop_size = random.choice(candidate_sizes)
        for i in range(2):
            if abs(crop_size[i] - self.input_size[i]) < 3:
                crop_size[i] = self.input_size[i]

        crop_w, crop_h = crop_size

        if self.random_crop:
            x_offset = random.randint(0, img_w - crop_w)
            y_offset = random.randint(0, img_h - crop_h)
        else:
            w_step = (img_w - crop_w) // 4
            h_step = (img_h - crop_h) // 4
            candidate_offsets = [
                (0, 0),  # upper left
                (4 * w_step, 0),  # upper right
                (0, 4 * h_step),  # lower left
                (4 * w_step, 4 * h_step),  # lower right
                (2 * w_step, 2 * h_step),  # center
            ]
            if self.num_fixed_crops == 13:
                extra_candidate_offsets = [
                    (0, 2 * h_step),  # center left
                    (4 * w_step, 2 * h_step),  # center right
                    (2 * w_step, 4 * h_step),  # lower center
                    (2 * w_step, 0 * h_step),  # upper center
                    (1 * w_step, 1 * h_step),  # upper left quarter
                    (3 * w_step, 1 * h_step),  # upper right quarter
                    (1 * w_step, 3 * h_step),  # lower left quarter
                    (3 * w_step, 3 * h_step)  # lower right quarter
                ]
                candidate_offsets.extend(extra_candidate_offsets)
            x_offset, y_offset = random.choice(candidate_offsets)

        new_h, new_w = crop_h, crop_w

        crop_bbox = np.array(
            [x_offset, y_offset, x_offset + new_w, y_offset + new_h])
        results['crop_bbox'] = crop_bbox
        results['img_shape'] = (new_h, new_w)
        results['scales'] = self.scales

        if 'crop_quadruple' not in results:
            results['crop_quadruple'] = np.array(
                [0, 0, 1, 1],  # x, y, w, h
                dtype=np.float32)

        x_ratio, y_ratio = x_offset / img_w, y_offset / img_h
        w_ratio, h_ratio = new_w / img_w, new_h / img_h

        old_crop_quadruple = results['crop_quadruple']
        old_x_ratio, old_y_ratio = old_crop_quadruple[0], old_crop_quadruple[1]
        old_w_ratio, old_h_ratio = old_crop_quadruple[2], old_crop_quadruple[3]
        new_crop_quadruple = [
            old_x_ratio + x_ratio * old_w_ratio,
            old_y_ratio + y_ratio * old_h_ratio, w_ratio * old_w_ratio,
            h_ratio * old_h_ratio
        ]
        results['crop_quadruple'] = np.array(
            new_crop_quadruple, dtype=np.float32)

        if not self.lazy:
            if 'keypoint' in results:
                results['keypoint'] = self._crop_kps(results['keypoint'],
                                                     crop_bbox)
            if 'imgs' in results:
                results['imgs'] = self._crop_imgs(results['imgs'], crop_bbox)
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')

            # record crop_bbox in lazyop dict to ensure only crop once in Fuse
            lazy_left, lazy_top, lazy_right, lazy_bottom = lazyop['crop_bbox']
            left = x_offset * (lazy_right - lazy_left) / img_w
            right = (x_offset + new_w) * (lazy_right - lazy_left) / img_w
            top = y_offset * (lazy_bottom - lazy_top) / img_h
            bottom = (y_offset + new_h) * (lazy_bottom - lazy_top) / img_h
            lazyop['crop_bbox'] = np.array([(lazy_left + left),
                                            (lazy_top + top),
                                            (lazy_left + right),
                                            (lazy_top + bottom)],
                                           dtype=np.float32)

        if 'gt_bboxes' in results:
            assert not self.lazy
            results = self._all_box_crop(results, results['crop_bbox'])

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'input_size={self.input_size}, scales={self.scales}, '
                    f'max_wh_scale_gap={self.max_wh_scale_gap}, '
                    f'random_crop={self.random_crop}, '
                    f'num_fixed_crops={self.num_fixed_crops}, '
                    f'lazy={self.lazy})')
        return repr_str


@TRANSFORMS.register_module()
class Resize(BaseTransform):
    """Resize images to a specific size.

    Required keys are "img_shape", "modality", "imgs" (optional), "keypoint"
    (optional), added or modified keys are "imgs", "img_shape", "keep_ratio",
    "scale_factor", "lazy", "resize_size". Required keys in "lazy" is None,
    added or modified key is "interpolation".

    Args:
        scale (float | Tuple[int]): If keep_ratio is True, it serves as scaling
            factor or maximum size:
            If it is a float number, the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, the image will
            be rescaled as large as possible within the scale.
            Otherwise, it serves as (w, h) of output size.
        keep_ratio (bool): If set to True, Images will be resized without
            changing the aspect ratio. Otherwise, it will resize images to a
            given size. Default: True.
        interpolation (str): Algorithm used for interpolation,
            accepted values are "nearest", "bilinear", "bicubic", "area",
            "lanczos". Default: "bilinear".
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self,
                 scale,
                 keep_ratio=True,
                 interpolation='bilinear',
                 lazy=False):
        if isinstance(scale, float):
            if scale <= 0:
                raise ValueError(f'Invalid scale {scale}, must be positive.')
        elif isinstance(scale, tuple):
            max_long_edge = max(scale)
            max_short_edge = min(scale)
            if max_short_edge == -1:
                # assign np.inf to long edge for rescaling short edge later.
                scale = (np.inf, max_long_edge)
        else:
            raise TypeError(
                f'Scale must be float or tuple of int, but got {type(scale)}')
        self.scale = scale
        self.keep_ratio = keep_ratio
        self.interpolation = interpolation
        self.lazy = lazy

    def _resize_imgs(self, imgs, new_w, new_h):
        """Static method for resizing keypoint."""
        return [
            mmcv.imresize(
                img, (new_w, new_h), interpolation=self.interpolation)
            for img in imgs
        ]

    @staticmethod
    def _resize_kps(kps, scale_factor):
        """Static method for resizing keypoint."""
        return kps * scale_factor

    @staticmethod
    def _box_resize(box, scale_factor):
        """Rescale the bounding boxes according to the scale_factor.

        Args:
            box (np.ndarray): The bounding boxes.
            scale_factor (np.ndarray): The scale factor used for rescaling.
        """
        assert len(scale_factor) == 2
        scale_factor = np.concatenate([scale_factor, scale_factor])
        return box * scale_factor

    def transform(self, results):
        """Performs the Resize augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """

        _init_lazy_if_proper(results, self.lazy)
        if 'keypoint' in results:
            assert not self.lazy, ('Keypoint Augmentations are not compatible '
                                   'with lazy == True')

        if 'scale_factor' not in results:
            results['scale_factor'] = np.array([1, 1], dtype=np.float32)
        img_h, img_w = results['img_shape']

        if self.keep_ratio:
            new_w, new_h = mmcv.rescale_size((img_w, img_h), self.scale)
        else:
            new_w, new_h = self.scale

        self.scale_factor = np.array([new_w / img_w, new_h / img_h],
                                     dtype=np.float32)

        results['img_shape'] = (new_h, new_w)
        results['keep_ratio'] = self.keep_ratio
        results['scale_factor'] = results['scale_factor'] * self.scale_factor

        if not self.lazy:
            if 'imgs' in results:
                results['imgs'] = self._resize_imgs(results['imgs'], new_w,
                                                    new_h)
            if 'keypoint' in results:
                results['keypoint'] = self._resize_kps(results['keypoint'],
                                                       self.scale_factor)
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')
            lazyop['interpolation'] = self.interpolation

        if 'gt_bboxes' in results:
            assert not self.lazy
            results['gt_bboxes'] = self._box_resize(results['gt_bboxes'],
                                                    self.scale_factor)
            if 'proposals' in results and results['proposals'] is not None:
                assert results['proposals'].shape[1] == 4
                results['proposals'] = self._box_resize(
                    results['proposals'], self.scale_factor)

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'scale={self.scale}, keep_ratio={self.keep_ratio}, '
                    f'interpolation={self.interpolation}, '
                    f'lazy={self.lazy})')
        return repr_str


@TRANSFORMS.register_module()
class RandomRescale(BaseTransform):
    """Randomly resize images so that the short_edge is resized to a specific
    size in a given range. The scale ratio is unchanged after resizing.

    Required keys are "imgs", "img_shape", "modality", added or modified
    keys are "imgs", "img_shape", "keep_ratio", "scale_factor", "resize_size",
    "short_edge".

    Args:
        scale_range (tuple[int]): The range of short edge length. A closed
            interval.
        interpolation (str): Algorithm used for interpolation:
            "nearest" | "bilinear". Default: "bilinear".
    """

    def __init__(self, scale_range, interpolation='bilinear'):
        self.scale_range = scale_range
        # make sure scale_range is legal, first make sure the type is OK
        assert mmengine.is_tuple_of(scale_range, int)
        assert len(scale_range) == 2
        assert scale_range[0] < scale_range[1]
        assert np.all([x > 0 for x in scale_range])

        self.keep_ratio = True
        self.interpolation = interpolation

    def transform(self, results):
        """Performs the Resize augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        short_edge = np.random.randint(self.scale_range[0],
                                       self.scale_range[1] + 1)
        resize = Resize((-1, short_edge),
                        keep_ratio=True,
                        interpolation=self.interpolation,
                        lazy=False)
        results = resize(results)

        results['short_edge'] = short_edge
        return results

    def __repr__(self):
        scale_range = self.scale_range
        repr_str = (f'{self.__class__.__name__}('
                    f'scale_range=({scale_range[0]}, {scale_range[1]}), '
                    f'interpolation={self.interpolation})')
        return repr_str


@TRANSFORMS.register_module()
class Flip(BaseTransform):
    """Flip the input images with a probability.

    Reverse the order of elements in the given imgs with a specific direction.
    The shape of the imgs is preserved, but the elements are reordered.

    Required keys are "img_shape", "modality", "imgs" (optional), "keypoint"
    (optional), added or modified keys are "imgs", "keypoint", "lazy" and
    "flip_direction". Required keys in "lazy" is None, added or modified key
    are "flip" and "flip_direction". The Flip augmentation should be placed
    after any cropping / reshaping augmentations, to make sure crop_quadruple
    is calculated properly.

    Args:
        flip_ratio (float): Probability of implementing flip. Default: 0.5.
        direction (str): Flip imgs horizontally or vertically. Options are
            "horizontal" | "vertical". Default: "horizontal".
        flip_label_map (Dict[int, int] | None): Transform the label of the
            flipped image with the specific label. Default: None.
        left_kp (list[int]): Indexes of left keypoints, used to flip keypoints.
            Default: None.
        right_kp (list[ind]): Indexes of right keypoints, used to flip
            keypoints. Default: None.
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """
    _directions = ['horizontal', 'vertical']

    def __init__(self,
                 flip_ratio=0.5,
                 direction='horizontal',
                 flip_label_map=None,
                 left_kp=None,
                 right_kp=None,
                 lazy=False):
        if direction not in self._directions:
            raise ValueError(f'Direction {direction} is not supported. '
                             f'Currently support ones are {self._directions}')
        self.flip_ratio = flip_ratio
        self.direction = direction
        self.flip_label_map = flip_label_map
        self.left_kp = left_kp
        self.right_kp = right_kp
        self.lazy = lazy

    def _flip_imgs(self, imgs, modality):
        """Utility function for flipping images."""
        _ = [mmcv.imflip_(img, self.direction) for img in imgs]
        lt = len(imgs)
        if modality == 'Flow':
            # The 1st frame of each 2 frames is flow-x
            for i in range(0, lt, 2):
                imgs[i] = mmcv.iminvert(imgs[i])
        return imgs

    def _flip_kps(self, kps, kpscores, img_width):
        """Utility function for flipping keypoint."""
        kp_x = kps[..., 0]
        kp_x[kp_x != 0] = img_width - kp_x[kp_x != 0]
        new_order = list(range(kps.shape[2]))
        if self.left_kp is not None and self.right_kp is not None:
            for left, right in zip(self.left_kp, self.right_kp):
                new_order[left] = right
                new_order[right] = left
        kps = kps[:, :, new_order]
        if kpscores is not None:
            kpscores = kpscores[:, :, new_order]
        return kps, kpscores

    @staticmethod
    def _box_flip(box, img_width):
        """Flip the bounding boxes given the width of the image.

        Args:
            box (np.ndarray): The bounding boxes.
            img_width (int): The img width.
        """
        box_ = box.copy()
        box_[..., 0::4] = img_width - box[..., 2::4]
        box_[..., 2::4] = img_width - box[..., 0::4]
        return box_

    def transform(self, results):
        """Performs the Flip augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, self.lazy)
        if 'keypoint' in results:
            assert not self.lazy, ('Keypoint Augmentations are not compatible '
                                   'with lazy == True')
            assert self.direction == 'horizontal', (
                'Only horizontal flips are'
                'supported for human keypoints')

        modality = results['modality']
        if modality == 'Flow':
            assert self.direction == 'horizontal'

        flip = np.random.rand() < self.flip_ratio

        results['flip'] = flip
        results['flip_direction'] = self.direction
        img_width = results['img_shape'][1]

        if self.flip_label_map is not None and flip:
            results['label'] = self.flip_label_map.get(results['label'],
                                                       results['label'])

        if not self.lazy:
            if flip:
                if 'imgs' in results:
                    results['imgs'] = self._flip_imgs(results['imgs'],
                                                      modality)
                if 'keypoint' in results:
                    kp = results['keypoint']
                    kpscore = results.get('keypoint_score', None)
                    kp, kpscore = self._flip_kps(kp, kpscore, img_width)
                    results['keypoint'] = kp
                    if 'keypoint_score' in results:
                        results['keypoint_score'] = kpscore
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Use one Flip please')
            lazyop['flip'] = flip
            lazyop['flip_direction'] = self.direction

        if 'gt_bboxes' in results and flip:
            assert not self.lazy and self.direction == 'horizontal'
            width = results['img_shape'][1]
            results['gt_bboxes'] = self._box_flip(results['gt_bboxes'], width)
            if 'proposals' in results and results['proposals'] is not None:
                assert results['proposals'].shape[1] == 4
                results['proposals'] = self._box_flip(results['proposals'],
                                                      width)

        return results

    def __repr__(self):
        repr_str = (
            f'{self.__class__.__name__}('
            f'flip_ratio={self.flip_ratio}, direction={self.direction}, '
            f'flip_label_map={self.flip_label_map}, lazy={self.lazy})')
        return repr_str


@TRANSFORMS.register_module()
class ColorJitter(BaseTransform):
    """Perform ColorJitter to each img.

    Required keys are "imgs", added or modified keys are "imgs".

    Args:
        brightness (float | tuple[float]): The jitter range for brightness, if
            set as a float, the range will be (1 - brightness, 1 + brightness).
            Default: 0.5.
        contrast (float | tuple[float]): The jitter range for contrast, if set
            as a float, the range will be (1 - contrast, 1 + contrast).
            Default: 0.5.
        saturation (float | tuple[float]): The jitter range for saturation, if
            set as a float, the range will be (1 - saturation, 1 + saturation).
            Default: 0.5.
        hue (float | tuple[float]): The jitter range for hue, if set as a
            float, the range will be (-hue, hue). Default: 0.1.
    """

    @staticmethod
    def check_input(val, max, base):
        if isinstance(val, tuple):
            assert base - max <= val[0] <= val[1] <= base + max
            return val
        assert val <= max
        return (base - val, base + val)

    @staticmethod
    def rgb_to_grayscale(img):
        return 0.2989 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]

    @staticmethod
    def adjust_contrast(img, factor):
        val = np.mean(ColorJitter.rgb_to_grayscale(img))
        return factor * img + (1 - factor) * val

    @staticmethod
    def adjust_saturation(img, factor):
        gray = np.stack([ColorJitter.rgb_to_grayscale(img)] * 3, axis=-1)
        return factor * img + (1 - factor) * gray

    @staticmethod
    def adjust_hue(img, factor):
        img = np.clip(img, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        offset = int(factor * 255)
        hsv[..., 0] = (hsv[..., 0] + offset) % 180
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return img.astype(np.float32)

    def __init__(self, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1):
        self.brightness = self.check_input(brightness, 1, 1)
        self.contrast = self.check_input(contrast, 1, 1)
        self.saturation = self.check_input(saturation, 1, 1)
        self.hue = self.check_input(hue, 0.5, 0)
        self.fn_idx = np.random.permutation(4)

    def transform(self, results):
        """Perform ColorJitter.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        imgs = results['imgs']
        num_clips, clip_len = 1, len(imgs)

        new_imgs = []
        for i in range(num_clips):
            b = np.random.uniform(
                low=self.brightness[0], high=self.brightness[1])
            c = np.random.uniform(low=self.contrast[0], high=self.contrast[1])
            s = np.random.uniform(
                low=self.saturation[0], high=self.saturation[1])
            h = np.random.uniform(low=self.hue[0], high=self.hue[1])
            start, end = i * clip_len, (i + 1) * clip_len

            for img in imgs[start:end]:
                img = img.astype(np.float32)
                for fn_id in self.fn_idx:
                    if fn_id == 0 and b != 1:
                        img *= b
                    if fn_id == 1 and c != 1:
                        img = self.adjust_contrast(img, c)
                    if fn_id == 2 and s != 1:
                        img = self.adjust_saturation(img, s)
                    if fn_id == 3 and h != 0:
                        img = self.adjust_hue(img, h)
                img = np.clip(img, 0, 255).astype(np.uint8)
                new_imgs.append(img)
        results['imgs'] = new_imgs
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'brightness={self.brightness}, '
                    f'contrast={self.contrast}, '
                    f'saturation={self.saturation}, '
                    f'hue={self.hue})')
        return repr_str


@TRANSFORMS.register_module()
class CenterCrop(RandomCrop):
    """Crop the center area from images.

    Required keys are "img_shape", "imgs" (optional), "keypoint" (optional),
    added or modified keys are "imgs", "keypoint", "crop_bbox", "lazy" and
    "img_shape". Required keys in "lazy" is "crop_bbox", added or modified key
    is "crop_bbox".

    Args:
        crop_size (int | tuple[int]): (w, h) of crop size.
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self, crop_size, lazy=False):
        self.crop_size = _pair(crop_size)
        self.lazy = lazy
        if not mmengine.is_tuple_of(self.crop_size, int):
            raise TypeError(f'Crop_size must be int or tuple of int, '
                            f'but got {type(crop_size)}')

    def transform(self, results):
        """Performs the CenterCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, self.lazy)
        if 'keypoint' in results:
            assert not self.lazy, ('Keypoint Augmentations are not compatible '
                                   'with lazy == True')

        img_h, img_w = results['img_shape']
        crop_w, crop_h = self.crop_size

        left = (img_w - crop_w) // 2
        top = (img_h - crop_h) // 2
        right = left + crop_w
        bottom = top + crop_h
        new_h, new_w = bottom - top, right - left

        crop_bbox = np.array([left, top, right, bottom])
        results['crop_bbox'] = crop_bbox
        results['img_shape'] = (new_h, new_w)

        if 'crop_quadruple' not in results:
            results['crop_quadruple'] = np.array(
                [0, 0, 1, 1],  # x, y, w, h
                dtype=np.float32)

        x_ratio, y_ratio = left / img_w, top / img_h
        w_ratio, h_ratio = new_w / img_w, new_h / img_h

        old_crop_quadruple = results['crop_quadruple']
        old_x_ratio, old_y_ratio = old_crop_quadruple[0], old_crop_quadruple[1]
        old_w_ratio, old_h_ratio = old_crop_quadruple[2], old_crop_quadruple[3]
        new_crop_quadruple = [
            old_x_ratio + x_ratio * old_w_ratio,
            old_y_ratio + y_ratio * old_h_ratio, w_ratio * old_w_ratio,
            h_ratio * old_h_ratio
        ]
        results['crop_quadruple'] = np.array(
            new_crop_quadruple, dtype=np.float32)

        if not self.lazy:
            if 'keypoint' in results:
                results['keypoint'] = self._crop_kps(results['keypoint'],
                                                     crop_bbox)
            if 'imgs' in results:
                results['imgs'] = self._crop_imgs(results['imgs'], crop_bbox)
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')

            # record crop_bbox in lazyop dict to ensure only crop once in Fuse
            lazy_left, lazy_top, lazy_right, lazy_bottom = lazyop['crop_bbox']
            left = left * (lazy_right - lazy_left) / img_w
            right = right * (lazy_right - lazy_left) / img_w
            top = top * (lazy_bottom - lazy_top) / img_h
            bottom = bottom * (lazy_bottom - lazy_top) / img_h
            lazyop['crop_bbox'] = np.array([(lazy_left + left),
                                            (lazy_top + top),
                                            (lazy_left + right),
                                            (lazy_top + bottom)],
                                           dtype=np.float32)

        if 'gt_bboxes' in results:
            assert not self.lazy
            results = self._all_box_crop(results, results['crop_bbox'])

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}(crop_size={self.crop_size}, '
                    f'lazy={self.lazy})')
        return repr_str


@TRANSFORMS.register_module()
class ThreeCrop(BaseTransform):
    """Crop images into three crops.

    Crop the images equally into three crops with equal intervals along the
    shorter side.
    Required keys are "imgs", "img_shape", added or modified keys are "imgs",
    "crop_bbox" and "img_shape".

    Args:
        crop_size(int | tuple[int]): (w, h) of crop size.
    """

    def __init__(self, crop_size):
        self.crop_size = _pair(crop_size)
        if not mmengine.is_tuple_of(self.crop_size, int):
            raise TypeError(f'Crop_size must be int or tuple of int, '
                            f'but got {type(crop_size)}')

    def transform(self, results):
        """Performs the ThreeCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, False)
        if 'gt_bboxes' in results or 'proposals' in results:
            warnings.warn('ThreeCrop cannot process bounding boxes')

        imgs = results['imgs']
        img_h, img_w = results['imgs'][0].shape[:2]
        crop_w, crop_h = self.crop_size
        assert crop_h == img_h or crop_w == img_w

        if crop_h == img_h:
            w_step = (img_w - crop_w) // 2
            offsets = [
                (0, 0),  # left
                (2 * w_step, 0),  # right
                (w_step, 0),  # middle
            ]
        elif crop_w == img_w:
            h_step = (img_h - crop_h) // 2
            offsets = [
                (0, 0),  # top
                (0, 2 * h_step),  # down
                (0, h_step),  # middle
            ]

        cropped = []
        crop_bboxes = []
        for x_offset, y_offset in offsets:
            bbox = [x_offset, y_offset, x_offset + crop_w, y_offset + crop_h]
            crop = [
                img[y_offset:y_offset + crop_h, x_offset:x_offset + crop_w]
                for img in imgs
            ]
            cropped.extend(crop)
            crop_bboxes.extend([bbox for _ in range(len(imgs))])

        crop_bboxes = np.array(crop_bboxes)
        results['imgs'] = cropped
        results['crop_bbox'] = crop_bboxes
        results['img_shape'] = results['imgs'][0].shape[:2]

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}(crop_size={self.crop_size})'
        return repr_str


@TRANSFORMS.register_module()
class TenCrop(BaseTransform):
    """Crop the images into 10 crops (corner + center + flip).

    Crop the four corners and the center part of the image with the same
    given crop_size, and flip it horizontally.
    Required keys are "imgs", "img_shape", added or modified keys are "imgs",
    "crop_bbox" and "img_shape".

    Args:
        crop_size(int | tuple[int]): (w, h) of crop size.
    """

    def __init__(self, crop_size):
        self.crop_size = _pair(crop_size)
        if not mmengine.is_tuple_of(self.crop_size, int):
            raise TypeError(f'Crop_size must be int or tuple of int, '
                            f'but got {type(crop_size)}')

    def transform(self, results):
        """Performs the TenCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, False)

        if 'gt_bboxes' in results or 'proposals' in results:
            warnings.warn('TenCrop cannot process bounding boxes')

        imgs = results['imgs']

        img_h, img_w = results['imgs'][0].shape[:2]
        crop_w, crop_h = self.crop_size

        w_step = (img_w - crop_w) // 4
        h_step = (img_h - crop_h) // 4

        offsets = [
            (0, 0),  # upper left
            (4 * w_step, 0),  # upper right
            (0, 4 * h_step),  # lower left
            (4 * w_step, 4 * h_step),  # lower right
            (2 * w_step, 2 * h_step),  # center
        ]

        img_crops = list()
        crop_bboxes = list()
        for x_offset, y_offsets in offsets:
            crop = [
                img[y_offsets:y_offsets + crop_h, x_offset:x_offset + crop_w]
                for img in imgs
            ]
            flip_crop = [np.flip(c, axis=1).copy() for c in crop]
            bbox = [x_offset, y_offsets, x_offset + crop_w, y_offsets + crop_h]
            img_crops.extend(crop)
            img_crops.extend(flip_crop)
            crop_bboxes.extend([bbox for _ in range(len(imgs) * 2)])

        crop_bboxes = np.array(crop_bboxes)
        results['imgs'] = img_crops
        results['crop_bbox'] = crop_bboxes
        results['img_shape'] = results['imgs'][0].shape[:2]

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}(crop_size={self.crop_size})'
        return repr_str


@TRANSFORMS.register_module()
class RandomErasing(BaseTransform):
    """Randomly selects a rectangle region in an image and erase pixels.
    basically refer mmcls.

    **Required Keys:**

    - img

    **Modified Keys:**

    - img

    Args:
        erase_prob (float): Probability that image will be randomly erased.
            Default: 0.5
        min_area_ratio (float): Minimum erased area / input image area
            Default: 0.02
        max_area_ratio (float): Maximum erased area / input image area
            Default: 1/3
        aspect_range (sequence | float): Aspect ratio range of erased area.
            if float, it will be converted to (aspect_ratio, 1/aspect_ratio)
            Default: (3/10, 10/3)
        mode (str): Fill method in erased area, can be:

            - const (default): All pixels are assign with the same value.
            - rand: each pixel is assigned with a random value in [0, 255]

        fill_color (sequence | Number): Base color filled in erased area.
            Defaults to (128, 128, 128).
        fill_std (sequence | Number, optional): If set and ``mode`` is 'rand',
            fill erased area with random color from normal distribution
            (mean=fill_color, std=fill_std); If not set, fill erased area with
            random color from uniform distribution (0~255). Defaults to None.

    Note:
        See `Random Erasing Data Augmentation
        <https://arxiv.org/pdf/1708.04896.pdf>`_

        This paper provided 4 modes: RE-R, RE-M, RE-0, RE-255, and use RE-M as
        default. The config of these 4 modes are:

        - RE-R: RandomErasing(mode='rand')
        - RE-M: RandomErasing(mode='const', fill_color=(123.67, 116.3, 103.5))
        - RE-0: RandomErasing(mode='const', fill_color=0)
        - RE-255: RandomErasing(mode='const', fill_color=255)
    """

    def __init__(self,
                 erase_prob=0.5,
                 min_area_ratio=0.02,
                 max_area_ratio=1 / 3,
                 aspect_range=(3 / 10, 10 / 3),
                 mode='const',
                 fill_color=(128, 128, 128),
                 fill_std=None):
        assert isinstance(erase_prob, float) and 0. <= erase_prob <= 1.
        assert isinstance(min_area_ratio, float) and 0. <= min_area_ratio <= 1.
        assert isinstance(max_area_ratio, float) and 0. <= max_area_ratio <= 1.
        assert min_area_ratio <= max_area_ratio, \
            'min_area_ratio should be smaller than max_area_ratio'
        if isinstance(aspect_range, float):
            aspect_range = min(aspect_range, 1 / aspect_range)
            aspect_range = (aspect_range, 1 / aspect_range)
        assert isinstance(aspect_range, Sequence) and len(aspect_range) == 2 \
            and all(isinstance(x, float) for x in aspect_range), \
            'aspect_range should be a float or Sequence with two float.'
        assert all(x > 0 for x in aspect_range), \
            'aspect_range should be positive.'
        assert aspect_range[0] <= aspect_range[1], \
            'In aspect_range (min, max), min should be smaller than max.'
        assert mode in ['const', 'rand'], \
            'Please select `mode` from ["const", "rand"].'
        if isinstance(fill_color, Number):
            fill_color = [fill_color] * 3
        assert isinstance(fill_color, Sequence) and len(fill_color) == 3 \
            and all(isinstance(x, Number) for x in fill_color), \
            'fill_color should be a float or Sequence with three int.'
        if fill_std is not None:
            if isinstance(fill_std, Number):
                fill_std = [fill_std] * 3
            assert isinstance(fill_std, Sequence) and len(fill_std) == 3 \
                and all(isinstance(x, Number) for x in fill_std), \
                'fill_std should be a float or Sequence with three int.'

        self.erase_prob = erase_prob
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.aspect_range = aspect_range
        self.mode = mode
        self.fill_color = fill_color
        self.fill_std = fill_std

    def _img_fill_pixels(self, img, top, left, h, w):
        """Fill pixels to the patch of image."""
        if self.mode == 'const':
            patch = np.empty((h, w, 3), dtype=np.uint8)
            patch[:, :] = np.array(self.fill_color, dtype=np.uint8)
        elif self.fill_std is None:
            # Uniform distribution
            patch = np.random.uniform(0, 256, (h, w, 3)).astype(np.uint8)
        else:
            # Normal distribution
            patch = np.random.normal(self.fill_color, self.fill_std, (h, w, 3))
            patch = np.clip(patch.astype(np.int32), 0, 255).astype(np.uint8)

        img[top:top + h, left:left + w] = patch
        return img

    def _fill_pixels(self, imgs, top, left, h, w):
        """Fill pixels to the patch of each image in frame clip."""
        return [self._img_fill_pixels(img, top, left, h, w) for img in imgs]

    @cache_randomness
    def random_disable(self):
        """Randomly disable the transform."""
        return np.random.rand() > self.erase_prob

    @cache_randomness
    def random_patch(self, img_h, img_w):
        """Randomly generate patch the erase."""
        # convert the aspect ratio to log space to equally handle width and
        # height.
        log_aspect_range = np.log(
            np.array(self.aspect_range, dtype=np.float32))
        aspect_ratio = np.exp(np.random.uniform(*log_aspect_range))
        area = img_h * img_w
        area *= np.random.uniform(self.min_area_ratio, self.max_area_ratio)

        h = min(int(round(np.sqrt(area * aspect_ratio))), img_h)
        w = min(int(round(np.sqrt(area / aspect_ratio))), img_w)
        top = np.random.randint(0, img_h - h) if img_h > h else 0
        left = np.random.randint(0, img_w - w) if img_w > w else 0
        return top, left, h, w

    def transform(self, results):
        """
        Args:
            results (dict): Results dict from pipeline

        Returns:
            dict: Results after the transformation.
        """
        if self.random_disable():
            return results

        imgs = results['imgs']
        img_h, img_w = imgs[0].shape[:2]

        imgs = self._fill_pixels(imgs, *self.random_patch(img_h, img_w))

        results['imgs'] = imgs

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(erase_prob={self.erase_prob}, '
        repr_str += f'min_area_ratio={self.min_area_ratio}, '
        repr_str += f'max_area_ratio={self.max_area_ratio}, '
        repr_str += f'aspect_range={self.aspect_range}, '
        repr_str += f'mode={self.mode}, '
        repr_str += f'fill_color={self.fill_color}, '
        repr_str += f'fill_std={self.fill_std})'
        return repr_str


@TRANSFORMS.register_module()
class LRSL(BaseTransform):
    """Crop images along the longer side with specified number of crops.

    Required keys are "imgs", "img_shape", added or modified keys are "imgs",
    "crop_bbox" and "img_shape".

    Args:
        crop_size(int | tuple[int]): (w, h) of crop size.
        num_crops (int): Number of crops to take along the longer side.
    """

    def __init__(self, crop_size=224, num_crops=5):
        self.crop_size = _pair(crop_size)
        self.num_crops = num_crops
        if not mmengine.is_tuple_of(self.crop_size, int):
            raise TypeError(f'Crop_size must be int or tuple of int, '
                            f'but got {type(crop_size)}')
        if num_crops < 2:
            raise ValueError(f'num_crops must be at least 2, got {num_crops}')

    def transform(self, results):
        """Performs the LongSideCrop augmentation."""
        _init_lazy_if_proper(results, False)

        if 'gt_bboxes' in results or 'proposals' in results:
            warnings.warn('LongSideCrop cannot process bounding boxes')

        imgs = results['imgs']
        img_h, img_w = results['imgs'][0].shape[:2]
        crop_w, crop_h = self.crop_size

        # Determine longer side
        if img_w >= img_h:  # Width is longer
            long_size = img_w
            short_size = img_h
            crop_long = crop_w
            crop_short = crop_h
            is_width_longer = True
        else:  # Height is longer
            long_size = img_h
            short_size = img_w
            crop_long = crop_h
            crop_short = crop_w
            is_width_longer = False

        # Calculate step size between crops (including start and end)
        step = (long_size - crop_long) / (self.num_crops - 1)

        img_crops = list()
        crop_bboxes = list()

        for i in range(self.num_crops):
            # Calculate offset along the longer side
            long_offset = round(i * step)

            if is_width_longer:
                # Crop horizontally (width is longer)
                x_offset = long_offset
                y_offset = (short_size - crop_short) // 2  # Center vertically
                bbox = [x_offset, y_offset, x_offset + crop_w, y_offset + crop_h]
            else:
                # Crop vertically (height is longer)
                x_offset = (short_size - crop_short) // 2  # Center horizontally
                y_offset = long_offset
                bbox = [x_offset, y_offset, x_offset + crop_w, y_offset + crop_h]

            # Crop the images
            crop = []
            for img in imgs:
                if is_width_longer:
                    cropped_img = img[y_offset:y_offset + crop_h,
                                      x_offset:x_offset + crop_w]
                else:
                    cropped_img = img[y_offset:y_offset + crop_h,
                                      x_offset:x_offset + crop_w]
                crop.append(cropped_img)

            # Add original crop
            img_crops.extend(crop)
            crop_bboxes.extend([bbox for _ in range(len(imgs))])

            # Add flipped crop
            flip_crop = [np.flip(c, axis=1).copy() for c in crop]
            img_crops.extend(flip_crop)
            crop_bboxes.extend([bbox for _ in range(len(imgs))])

        crop_bboxes = np.array(crop_bboxes)
        results['imgs'] = img_crops
        results['crop_bbox'] = crop_bboxes
        results['img_shape'] = results['imgs'][0].shape[:2]

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}(crop_size={self.crop_size}, '
                    f'num_crops={self.num_crops})')
        return repr_str

@TRANSFORMS.register_module()
class shiziCrop(BaseTransform):
    """Crop the images into 10 crops (top, bottom, left, right, center + flip).

    Crop the top, bottom, left, right and center parts of the image with the
    same given crop_size, and flip it horizontally.
    Required keys are "imgs", "img_shape", added or modified keys are "imgs",
    "crop_bbox" and "img_shape".

    Args:
        crop_size(int | tuple[int]): (w, h) of crop size.
    """

    def __init__(self, crop_size):
        self.crop_size = _pair(crop_size)
        if not mmengine.is_tuple_of(self.crop_size, int):
            raise TypeError(f'Crop_size must be int or tuple of int, '
                            f'but got {type(crop_size)}')

    def transform(self, results):
        """Performs the TenCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, False)

        if 'gt_bboxes' in results or 'proposals' in results:
            warnings.warn('TenCrop cannot process bounding boxes')

        imgs = results['imgs']
        img_h, img_w = results['imgs'][0].shape[:2]
        crop_w, crop_h = self.crop_size

        # Calculate center position
        center_x = (img_w - crop_w) // 2
        center_y = (img_h - crop_h) // 2

        # Define five regions: top, bottom, left, right, center
        offsets = [
            (center_x, 0),  # top (centered horizontally)
            (center_x, img_h - crop_h),  # bottom (centered horizontally)
            (0, center_y),  # left (centered vertically)
            (img_w - crop_w, center_y),  # right (centered vertically)
            (center_x, center_y),  # center
        ]

        img_crops = list()
        crop_bboxes = list()
        for x_offset, y_offset in offsets:
            # Crop the region
            crop = [
                img[y_offset:y_offset + crop_h, x_offset:x_offset + crop_w]
                for img in imgs
            ]
            # Generate horizontal flip
            flip_crop = [np.flip(c, axis=1).copy() for c in crop]
            bbox = [x_offset, y_offset, x_offset + crop_w, y_offset + crop_h]
            # Add original and flipped crops
            img_crops.extend(crop)
            img_crops.extend(flip_crop)
            crop_bboxes.extend([bbox for _ in range(len(imgs) * 2)])

        crop_bboxes = np.array(crop_bboxes)
        results['imgs'] = img_crops
        results['crop_bbox'] = crop_bboxes
        results['img_shape'] = results['imgs'][0].shape[:2]

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}(crop_size={self.crop_size})'
        return repr_str


@TRANSFORMS.register_module()
class suiji2Crop(RandomCrop):
    """Random crop that specifics the area and height-weight ratio range.

    Modified to support conditional cropping:
    If label==1, crop fixed size at random position.
    Otherwise, use original random scaling strategy.

    Required keys in results are "img_shape", "crop_bbox", "imgs" (optional),
    "keypoint" (optional), added or modified keys are "imgs", "keypoint",
    "crop_bbox" and "lazy"; Required keys in "lazy" are "flip", "crop_bbox",
    added or modified key is "crop_bbox".

    Args:
        area_range (Tuple[float]): The candidate area scales range of
            output cropped images. Default: (0.08, 1.0).
        aspect_ratio_range (Tuple[float]): The candidate aspect ratio range of
            output cropped images. Default: (3 / 4, 4 / 3).
        fixed_crop_size (int | tuple[int]): Fixed crop size (w, h) used when
            label == 1. Default: None.
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self,
                 area_range=(0.08, 1.0),
                 aspect_ratio_range=(3 / 4, 4 / 3),
                 fixed_crop_size=224,
                 lazy=False):
        self.area_range = area_range
        self.aspect_ratio_range = aspect_ratio_range
        self.fixed_crop_size = _pair(fixed_crop_size) if fixed_crop_size else None
        self.lazy = lazy
        if not mmengine.is_tuple_of(self.area_range, float):
            raise TypeError(f'Area_range must be a tuple of float, '
                            f'but got {type(area_range)}')
        if not mmengine.is_tuple_of(self.aspect_ratio_range, float):
            raise TypeError(f'Aspect_ratio_range must be a tuple of float, '
                            f'but got {type(aspect_ratio_range)}')
        if self.fixed_crop_size and not mmengine.is_tuple_of(self.fixed_crop_size, int):
            raise TypeError(f'fixed_crop_size must be int or tuple of int, '
                            f'but got {type(fixed_crop_size)}')

    @staticmethod
    def get_crop_bbox(img_shape,
                      area_range,
                      aspect_ratio_range,
                      max_attempts=10):
        """Get a crop bbox given the area range and aspect ratio range.

        Args:
            img_shape (Tuple[int]): Image shape
            area_range (Tuple[float]): The candidate area scales range of
                output cropped images. Default: (0.08, 1.0).
            aspect_ratio_range (Tuple[float]): The candidate aspect
                ratio range of output cropped images. Default: (3 / 4, 4 / 3).
                max_attempts (int): The maximum of attempts. Default: 10.
            max_attempts (int): Max attempts times to generate random candidate
                bounding box. If it doesn't qualified one, the center bounding
                box will be used.
        Returns:
            (list[int]) A random crop bbox within the area range and aspect
            ratio range.
        """
        assert 0 < area_range[0] <= area_range[1] <= 1
        assert 0 < aspect_ratio_range[0] <= aspect_ratio_range[1]

        img_h, img_w = img_shape
        area = img_h * img_w

        min_ar, max_ar = aspect_ratio_range
        aspect_ratios = np.exp(
            np.random.uniform(
                np.log(min_ar), np.log(max_ar), size=max_attempts))
        target_areas = np.random.uniform(*area_range, size=max_attempts) * area
        candidate_crop_w = np.round(np.sqrt(target_areas *
                                            aspect_ratios)).astype(np.int32)
        candidate_crop_h = np.round(np.sqrt(target_areas /
                                            aspect_ratios)).astype(np.int32)

        for i in range(max_attempts):
            crop_w = candidate_crop_w[i]
            crop_h = candidate_crop_h[i]
            if crop_h <= img_h and crop_w <= img_w:
                x_offset = random.randint(0, img_w - crop_w)
                y_offset = random.randint(0, img_h - crop_h)
                return x_offset, y_offset, x_offset + crop_w, y_offset + crop_h

        # Fallback
        crop_size = min(img_h, img_w)
        x_offset = (img_w - crop_size) // 2
        y_offset = (img_h - crop_size) // 2
        return x_offset, y_offset, x_offset + crop_size, y_offset + crop_size

    def transform(self, results):
        """Performs the RandomResizeCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, self.lazy)
        if 'keypoint' in results:
            assert not self.lazy, ('Keypoint Augmentations are not compatible '
                                   'with lazy == True')

        img_h, img_w = results['img_shape']

        # Conditional cropping for label == 1
        if self.fixed_crop_size and 'label' in results and results['label'] == 1:
            crop_w, crop_h = self.fixed_crop_size
            # Ensure crop size is valid
            if crop_w > img_w or crop_h > img_h:
                raise ValueError(
                    f"Crop size {self.fixed_crop_size} larger than image size {(img_w, img_h)}"
                )
            left = random.randint(0, img_w - crop_w)
            top = random.randint(0, img_h - crop_h)
            right = left + crop_w
            bottom = top + crop_h
        else:
            # Original random scaled cropping
            left, top, right, bottom = self.get_crop_bbox(
                (img_h, img_w), self.area_range, self.aspect_ratio_range)

        new_h, new_w = bottom - top, right - left

        if 'crop_quadruple' not in results:
            results['crop_quadruple'] = np.array(
                [0, 0, 1, 1],  # x, y, w, h
                dtype=np.float32)

        x_ratio, y_ratio = left / img_w, top / img_h
        w_ratio, h_ratio = new_w / img_w, new_h / img_h

        old_crop_quadruple = results['crop_quadruple']
        old_x_ratio, old_y_ratio = old_crop_quadruple[0], old_crop_quadruple[1]
        old_w_ratio, old_h_ratio = old_crop_quadruple[2], old_crop_quadruple[3]
        new_crop_quadruple = [
            old_x_ratio + x_ratio * old_w_ratio,
            old_y_ratio + y_ratio * old_h_ratio, w_ratio * old_w_ratio,
            h_ratio * old_h_ratio
        ]
        results['crop_quadruple'] = np.array(
            new_crop_quadruple, dtype=np.float32)

        crop_bbox = np.array([left, top, right, bottom])
        results['crop_bbox'] = crop_bbox
        results['img_shape'] = (new_h, new_w)

        if not self.lazy:
            if 'keypoint' in results:
                results['keypoint'] = self._crop_kps(results['keypoint'],
                                                     crop_bbox)
            if 'imgs' in results:
                results['imgs'] = self._crop_imgs(results['imgs'], crop_bbox)
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')

            # record crop_bbox in lazyop dict to ensure only crop once in Fuse
            lazy_left, lazy_top, lazy_right, lazy_bottom = lazyop['crop_bbox']
            left = left * (lazy_right - lazy_left) / img_w
            right = right * (lazy_right - lazy_left) / img_w
            top = top * (lazy_bottom - lazy_top) / img_h
            bottom = bottom * (lazy_bottom - lazy_top) / img_h
            lazyop['crop_bbox'] = np.array([(lazy_left + left),
                                            (lazy_top + top),
                                            (lazy_left + right),
                                            (lazy_top + bottom)],
                                           dtype=np.float32)

        if 'gt_bboxes' in results:
            assert not self.lazy
            results = self._all_box_crop(results, results['crop_bbox'])

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'area_range={self.area_range}, '
                    f'aspect_ratio_range={self.aspect_ratio_range}, '
                    f'fixed_crop_size={self.fixed_crop_size}, '
                    f'lazy={self.lazy})')
        return repr_str

@TRANSFORMS.register_module()
class DualCenterCrop(RandomCrop):
    """Random crop that specifics the area and height-weight ratio range.

    Required keys in results are "img_shape", "crop_bbox", "imgs" (optional),
    "keypoint" (optional), added or modified keys are "imgs", "keypoint",
    "crop_bbox" and "lazy"; Required keys in "lazy" are "flip", "crop_bbox",
    added or modified key is "crop_bbox".

    Args:
        area_range (Tuple[float]): The candidate area scales range of
            output cropped images. Default: (0.08, 1.0).
        aspect_ratio_range (Tuple[float]): The candidate aspect ratio range of
            output cropped images. Default: (3 / 4, 4 / 3).
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self,
                 area_range=(0.08, 1.0),
                 aspect_ratio_range=(3 / 4, 4 / 3),
                 lazy=False):
        self.area_range = area_range
        self.aspect_ratio_range = aspect_ratio_range
        self.lazy = lazy
        if not mmengine.is_tuple_of(self.area_range, float):
            raise TypeError(f'Area_range must be a tuple of float, '
                            f'but got {type(area_range)}')
        if not mmengine.is_tuple_of(self.aspect_ratio_range, float):
            raise TypeError(f'Aspect_ratio_range must be a tuple of float, '
                            f'but got {type(aspect_ratio_range)}')

    @staticmethod
    def get_crop_bbox(img_shape,
                      area_range,
                      aspect_ratio_range,
                      max_attempts=10):
        """Get a crop bbox given the area range and aspect ratio range.

        Modified to implement the dual-center crop strategy when applicable.

        Args:
            img_shape (Tuple[int]): Image shape
            area_range (Tuple[float]): The candidate area scales range of
                output cropped images. Default: (0.08, 1.0).
            aspect_ratio_range (Tuple[float]): The candidate aspect
                ratio range of output cropped images. Default: (3 / 4, 4 / 3).
                max_attempts (int): The maximum of attempts. Default: 10.
            max_attempts (int): Max attempts times to generate random candidate
                bounding box. If it doesn't qualified one, the center bounding
                box will be used.
        Returns:
            (list[int]) A random crop bbox within the area range and aspect
            ratio range.
        """
        assert 0 < area_range[0] <= area_range[1] <= 1
        assert 0 < aspect_ratio_range[0] <= aspect_ratio_range[1]

        img_h, img_w = img_shape
        area = img_h * img_w

        min_ar, max_ar = aspect_ratio_range
        aspect_ratios = np.exp(
            np.random.uniform(
                np.log(min_ar), np.log(max_ar), size=max_attempts))
        target_areas = np.random.uniform(*area_range, size=max_attempts) * area
        candidate_crop_w = np.round(np.sqrt(target_areas *
                                            aspect_ratios)).astype(np.int32)
        candidate_crop_h = np.round(np.sqrt(target_areas /
                                            aspect_ratios)).astype(np.int32)

        for i in range(max_attempts):
            crop_w = candidate_crop_w[i]
            crop_h = candidate_crop_h[i]
            if crop_h <= img_h and crop_w <= img_w:
                # Determine long side strategy
                if img_w >= img_h:  # Width is longer
                    if img_w > 2 * crop_w:
                        # Divide width into two parts and center crop each
                        half_width = img_w // 2
                        part = random.choice([0, 1])
                        if part == 0:  # Left part
                            x_offset = max(0, (half_width - crop_w) // 2)
                        else:  # Right part
                            x_offset = min(img_w - crop_w,
                                          half_width + (half_width - crop_w) // 2)
                        y_offset = random.randint(0, img_h - crop_h)
                    else:
                        # Allow overlapping crops
                        x_offset = random.randint(0, img_w - crop_w)
                        y_offset = random.randint(0, img_h - crop_h)
                else:  # Height is longer
                    if img_h > 2 * crop_h:
                        # Divide height into two parts and center crop each
                        half_height = img_h // 2
                        part = random.choice([0, 1])
                        if part == 0:  # Top part
                            y_offset = max(0, (half_height - crop_h) // 2)
                        else:  # Bottom part
                            y_offset = min(img_h - crop_h,
                                          half_height + (half_height - crop_h) // 2)
                        x_offset = random.randint(0, img_w - crop_w)
                    else:
                        # Allow overlapping crops
                        x_offset = random.randint(0, img_w - crop_w)
                        y_offset = random.randint(0, img_h - crop_h)

                return x_offset, y_offset, x_offset + crop_w, y_offset + crop_h

        # Fallback
        crop_size = min(img_h, img_w)
        x_offset = (img_w - crop_size) // 2
        y_offset = (img_h - crop_size) // 2
        return x_offset, y_offset, x_offset + crop_size, y_offset + crop_size

    def transform(self, results):
        """Performs the RandomResizeCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, self.lazy)
        if 'keypoint' in results:
            assert not self.lazy, ('Keypoint Augmentations are not compatible '
                                   'with lazy == True')

        img_h, img_w = results['img_shape']

        left, top, right, bottom = self.get_crop_bbox(
            (img_h, img_w), self.area_range, self.aspect_ratio_range)
        new_h, new_w = bottom - top, right - left

        if 'crop_quadruple' not in results:
            results['crop_quadruple'] = np.array(
                [0, 0, 1, 1],  # x, y, w, h
                dtype=np.float32)

        x_ratio, y_ratio = left / img_w, top / img_h
        w_ratio, h_ratio = new_w / img_w, new_h / img_h

        old_crop_quadruple = results['crop_quadruple']
        old_x_ratio, old_y_ratio = old_crop_quadruple[0], old_crop_quadruple[1]
        old_w_ratio, old_h_ratio = old_crop_quadruple[2], old_crop_quadruple[3]
        new_crop_quadruple = [
            old_x_ratio + x_ratio * old_w_ratio,
            old_y_ratio + y_ratio * old_h_ratio, w_ratio * old_w_ratio,
            h_ratio * old_h_ratio
        ]
        results['crop_quadruple'] = np.array(
            new_crop_quadruple, dtype=np.float32)

        crop_bbox = np.array([left, top, right, bottom])
        results['crop_bbox'] = crop_bbox
        results['img_shape'] = (new_h, new_w)

        if not self.lazy:
            if 'keypoint' in results:
                results['keypoint'] = self._crop_kps(results['keypoint'],
                                                     crop_bbox)
            if 'imgs' in results:
                results['imgs'] = self._crop_imgs(results['imgs'], crop_bbox)
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')

            # record crop_bbox in lazyop dict to ensure only crop once in Fuse
            lazy_left, lazy_top, lazy_right, lazy_bottom = lazyop['crop_bbox']
            left = left * (lazy_right - lazy_left) / img_w
            right = right * (lazy_right - lazy_left) / img_w
            top = top * (lazy_bottom - lazy_top) / img_h
            bottom = bottom * (lazy_bottom - lazy_top) / img_h
            lazyop['crop_bbox'] = np.array([(lazy_left + left),
                                            (lazy_top + top),
                                            (lazy_left + right),
                                            (lazy_top + bottom)],
                                           dtype=np.float32)

        if 'gt_bboxes' in results:
            assert not self.lazy
            results = self._all_box_crop(results, results['crop_bbox'])

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'area_range={self.area_range}, '
                    f'aspect_ratio_range={self.aspect_ratio_range}, '
                    f'lazy={self.lazy})')
        return repr_str

@TRANSFORMS.register_module()
class FusionCrop(RandomCrop):
    """Random crop that specifics the area and height-weight ratio range.

    Modified to perform both random crop and center crop with the same size,
    then randomly select one of them.

    Required keys in results are "img_shape", "crop_bbox", "imgs" (optional),
    "keypoint" (optional), added or modified keys are "imgs", "keypoint",
    "crop_bbox" and "lazy"; Required keys in "lazy" are "flip", "crop_bbox",
    added or modified key is "crop_bbox".

    Args:
        area_range (Tuple[float]): The candidate area scales range of
            output cropped images. Default: (0.08, 1.0).
        aspect_ratio_range (Tuple[float]): The candidate aspect ratio range of
            output cropped images. Default: (3 / 4, 4 / 3).
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self,
                 area_range=(0.08, 1.0),
                 aspect_ratio_range=(3 / 4, 4 / 3),
                 lazy=False):
        self.area_range = area_range
        self.aspect_ratio_range = aspect_ratio_range
        self.lazy = lazy
        if not mmengine.is_tuple_of(self.area_range, float):
            raise TypeError(f'Area_range must be a tuple of float, '
                            f'but got {type(area_range)}')
        if not mmengine.is_tuple_of(self.aspect_ratio_range, float):
            raise TypeError(f'Aspect_ratio_range must be a tuple of float, '
                            f'but got {type(aspect_ratio_range)}')

    @staticmethod
    def get_crop_bbox(img_shape,
                      area_range,
                      aspect_ratio_range,
                      max_attempts=10):
        """Get a crop bbox given the area range and aspect ratio range.

        Args:
            img_shape (Tuple[int]): Image shape
            area_range (Tuple[float]): The candidate area scales range of
                output cropped images. Default: (0.08, 1.0).
            aspect_ratio_range (Tuple[float]): The candidate aspect
                ratio range of output cropped images. Default: (3 / 4, 4 / 3).
                max_attempts (int): The maximum of attempts. Default: 10.
            max_attempts (int): Max attempts times to generate random candidate
                bounding box. If it doesn't qualified one, the center bounding
                box will be used.
        Returns:
            (list[int]) A random crop bbox within the area range and aspect
            ratio range.
        """
        assert 0 < area_range[0] <= area_range[1] <= 1
        assert 0 < aspect_ratio_range[0] <= aspect_ratio_range[1]

        img_h, img_w = img_shape
        area = img_h * img_w

        min_ar, max_ar = aspect_ratio_range
        aspect_ratios = np.exp(
            np.random.uniform(
                np.log(min_ar), np.log(max_ar), size=max_attempts))
        target_areas = np.random.uniform(*area_range, size=max_attempts) * area
        candidate_crop_w = np.round(np.sqrt(target_areas *
                                            aspect_ratios)).astype(np.int32)
        candidate_crop_h = np.round(np.sqrt(target_areas /
                                            aspect_ratios)).astype(np.int32)

        for i in range(max_attempts):
            crop_w = candidate_crop_w[i]
            crop_h = candidate_crop_h[i]
            if crop_h <= img_h and crop_w <= img_w:
                x_offset = random.randint(0, img_w - crop_w)
                y_offset = random.randint(0, img_h - crop_h)
                return x_offset, y_offset, x_offset + crop_w, y_offset + crop_h

        # Fallback
        crop_size = min(img_h, img_w)
        x_offset = (img_w - crop_size) // 2
        y_offset = (img_h - crop_size) // 2
        return x_offset, y_offset, x_offset + crop_size, y_offset + crop_size

    def transform(self, results):
        """Performs the RandomResizeCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, self.lazy)
        if 'keypoint' in results:
            assert not self.lazy, ('Keypoint Augmentations are not compatible '
                                   'with lazy == True')

        img_h, img_w = results['img_shape']

        # Get random crop bbox
        left, top, right, bottom = self.get_crop_bbox(
            (img_h, img_w), self.area_range, self.aspect_ratio_range)
        new_h, new_w = bottom - top, right - left

        # Calculate center crop bbox with same size as random crop
        center_left = (img_w - new_w) // 2
        center_top = (img_h - new_h) // 2
        center_right = center_left + new_w
        center_bottom = center_top + new_h

        # Randomly choose between random crop and center crop
        if random.random() < 0.5:
            # Use random crop
            final_bbox = [left, top, right, bottom]
        else:
            # Use center crop
            final_bbox = [center_left, center_top, center_right, center_bottom]
            left, top, right, bottom = final_bbox

        if 'crop_quadruple' not in results:
            results['crop_quadruple'] = np.array(
                [0, 0, 1, 1],  # x, y, w, h
                dtype=np.float32)

        x_ratio, y_ratio = left / img_w, top / img_h
        w_ratio, h_ratio = new_w / img_w, new_h / img_h

        old_crop_quadruple = results['crop_quadruple']
        old_x_ratio, old_y_ratio = old_crop_quadruple[0], old_crop_quadruple[1]
        old_w_ratio, old_h_ratio = old_crop_quadruple[2], old_crop_quadruple[3]
        new_crop_quadruple = [
            old_x_ratio + x_ratio * old_w_ratio,
            old_y_ratio + y_ratio * old_h_ratio, w_ratio * old_w_ratio,
            h_ratio * old_h_ratio
        ]
        results['crop_quadruple'] = np.array(
            new_crop_quadruple, dtype=np.float32)

        crop_bbox = np.array(final_bbox)
        results['crop_bbox'] = crop_bbox
        results['img_shape'] = (new_h, new_w)

        if not self.lazy:
            if 'keypoint' in results:
                results['keypoint'] = self._crop_kps(results['keypoint'],
                                                     crop_bbox)
            if 'imgs' in results:
                results['imgs'] = self._crop_imgs(results['imgs'], crop_bbox)
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')

            # record crop_bbox in lazyop dict to ensure only crop once in Fuse
            lazy_left, lazy_top, lazy_right, lazy_bottom = lazyop['crop_bbox']
            left = left * (lazy_right - lazy_left) / img_w
            right = right * (lazy_right - lazy_left) / img_w
            top = top * (lazy_bottom - lazy_top) / img_h
            bottom = bottom * (lazy_bottom - lazy_top) / img_h
            lazyop['crop_bbox'] = np.array([(lazy_left + left),
                                            (lazy_top + top),
                                            (lazy_left + right),
                                            (lazy_top + bottom)],
                                           dtype=np.float32)

        if 'gt_bboxes' in results:
            assert not self.lazy
            results = self._all_box_crop(results, results['crop_bbox'])

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'area_range={self.area_range}, '
                    f'aspect_ratio_range={self.aspect_ratio_range}, '
                    f'lazy={self.lazy})')
        return repr_str


@TRANSFORMS.register_module()
class suiji2Crop1(RandomCrop):
    """Fixed-size random crop that ignores area and aspect ratio ranges.

    Required keys in results are "img_shape", "crop_bbox", "imgs" (optional),
    "keypoint" (optional), added or modified keys are "imgs", "keypoint",
    "crop_bbox" and "lazy"; Required keys in "lazy" are "flip", "crop_bbox",
    added or modified key is "crop_bbox".

    Args:
        crop_size (int | tuple[int]): (w, h) of crop size.
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self, crop_size=256, lazy=False):
        self.crop_size = _pair(crop_size)
        self.lazy = lazy
        if not mmengine.is_tuple_of(self.crop_size, int):
            raise TypeError(f'Crop_size must be int or tuple of int, '
                            f'but got {type(crop_size)}')

    def transform(self, results):
        """Performs the fixed-size random crop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, self.lazy)
        if 'keypoint' in results:
            assert not self.lazy, ('Keypoint Augmentations are not compatible '
                                   'with lazy == True')

        img_h, img_w = results['img_shape']
        crop_w, crop_h = self.crop_size

        # Ensure crop size is not larger than image
        if crop_w > img_w or crop_h > img_h:
            raise ValueError(
                f"Crop size {self.crop_size} larger than image size {(img_w, img_h)}"
            )

        # Generate random position for fixed-size crop
        x_offset = random.randint(0, img_w - crop_w)
        y_offset = random.randint(0, img_h - crop_h)
        left, top, right, bottom = x_offset, y_offset, x_offset + crop_w, y_offset + crop_h

        new_h, new_w = bottom - top, right - left

        if 'crop_quadruple' not in results:
            results['crop_quadruple'] = np.array(
                [0, 0, 1, 1],  # x, y, w, h
                dtype=np.float32)

        x_ratio, y_ratio = left / img_w, top / img_h
        w_ratio, h_ratio = new_w / img_w, new_h / img_h

        old_crop_quadruple = results['crop_quadruple']
        old_x_ratio, old_y_ratio = old_crop_quadruple[0], old_crop_quadruple[1]
        old_w_ratio, old_h_ratio = old_crop_quadruple[2], old_crop_quadruple[3]
        new_crop_quadruple = [
            old_x_ratio + x_ratio * old_w_ratio,
            old_y_ratio + y_ratio * old_h_ratio, w_ratio * old_w_ratio,
            h_ratio * old_h_ratio
        ]
        results['crop_quadruple'] = np.array(
            new_crop_quadruple, dtype=np.float32)

        crop_bbox = np.array([left, top, right, bottom])
        results['crop_bbox'] = crop_bbox
        results['img_shape'] = (new_h, new_w)

        if not self.lazy:
            if 'keypoint' in results:
                results['keypoint'] = self._crop_kps(results['keypoint'],
                                                     crop_bbox)
            if 'imgs' in results:
                results['imgs'] = self._crop_imgs(results['imgs'], crop_bbox)
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')

            # record crop_bbox in lazyop dict to ensure only crop once in Fuse
            lazy_left, lazy_top, lazy_right, lazy_bottom = lazyop['crop_bbox']
            left = left * (lazy_right - lazy_left) / img_w
            right = right * (lazy_right - lazy_left) / img_w
            top = top * (lazy_bottom - lazy_top) / img_h
            bottom = bottom * (lazy_bottom - lazy_top) / img_h
            lazyop['crop_bbox'] = np.array([(lazy_left + left),
                                            (lazy_top + top),
                                            (lazy_left + right),
                                            (lazy_top + bottom)],
                                           dtype=np.float32)

        if 'gt_bboxes' in results:
            assert not self.lazy
            results = self._all_box_crop(results, results['crop_bbox'])

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'crop_size={self.crop_size}, '
                    f'lazy={self.lazy})')
        return repr_str


@TRANSFORMS.register_module()
class ST3DCrop(RandomCrop):
    """Random crop that specifics the area and height-weight ratio range.

    Required keys in results are "img_shape", "crop_bbox", "imgs" (optional),
    "keypoint" (optional), added or modified keys are "imgs", "keypoint",
    "crop_bbox" and "lazy"; Required keys in "lazy" are "flip", "crop_bbox",
    added or modified key is "crop_bbox".

    Args:
        area_range (Tuple[float]): The candidate area scales range of
            output cropped images. Default: (0.08, 1.0).
        aspect_ratio_range (Tuple[float]): The candidate aspect ratio range of
            output cropped images. Default: (3 / 4, 4 / 3).
        lazy (bool): Determine whether to apply lazy operation. Default: False.
        clip_len (int | None): Number of frames to sample temporally. If None,
            no temporal sampling is applied. Default: None.
    """

    def __init__(self,
                 area_range=(0.08, 1.0),
                 aspect_ratio_range=(3 / 4, 4 / 3),
                 lazy=False,
                 clip_len=32):
        self.area_range = area_range
        self.aspect_ratio_range = aspect_ratio_range
        self.lazy = lazy
        self.clip_len = clip_len
        if not mmengine.is_tuple_of(self.area_range, float):
            raise TypeError(f'Area_range must be a tuple of float, '
                            f'but got {type(area_range)}')
        if not mmengine.is_tuple_of(self.aspect_ratio_range, float):
            raise TypeError(f'Aspect_ratio_range must be a tuple of float, '
                            f'but got {type(aspect_ratio_range)}')

    @staticmethod
    def get_crop_bbox(img_shape,
                      area_range,
                      aspect_ratio_range,
                      max_attempts=10):
        """Get a crop bbox given the area range and aspect ratio range.

        Args:
            img_shape (Tuple[int]): Image shape
            area_range (Tuple[float]): The candidate area scales range of
                output cropped images. Default: (0.08, 1.0).
            aspect_ratio_range (Tuple[float]): The candidate aspect
                ratio range of output cropped images. Default: (3 / 4, 4 / 3).
                max_attempts (int): The maximum of attempts. Default: 10.
            max_attempts (int): Max attempts times to generate random candidate
                bounding box. If it doesn't qualified one, the center bounding
                box will be used.
        Returns:
            (list[int]) A random crop bbox within the area range and aspect
            ratio range.
        """
        assert 0 < area_range[0] <= area_range[1] <= 1
        assert 0 < aspect_ratio_range[0] <= aspect_ratio_range[1]

        img_h, img_w = img_shape
        area = img_h * img_w

        min_ar, max_ar = aspect_ratio_range
        aspect_ratios = np.exp(
            np.random.uniform(
                np.log(min_ar), np.log(max_ar), size=max_attempts))
        target_areas = np.random.uniform(*area_range, size=max_attempts) * area
        candidate_crop_w = np.round(np.sqrt(target_areas *
                                            aspect_ratios)).astype(np.int32)
        candidate_crop_h = np.round(np.sqrt(target_areas /
                                            aspect_ratios)).astype(np.int32)

        for i in range(max_attempts):
            crop_w = candidate_crop_w[i]
            crop_h = candidate_crop_h[i]
            if crop_h <= img_h and crop_w <= img_w:
                x_offset = random.randint(0, img_w - crop_w)
                y_offset = random.randint(0, img_h - crop_h)
                return x_offset, y_offset, x_offset + crop_w, y_offset + crop_h

        # Fallback
        crop_size = min(img_h, img_w)
        x_offset = (img_w - crop_size) // 2
        y_offset = (img_h - crop_size) // 2
        return x_offset, y_offset, x_offset + crop_size, y_offset + crop_size

    def transform(self, results):
        """Performs the RandomResizeCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, self.lazy)
        if 'keypoint' in results:
            assert not self.lazy, ('Keypoint Augmentations are not compatible '
                                   'with lazy == True')

        # === 改进的时间帧裁剪逻辑 ===
        if self.clip_len is not None:
            imgs = results['imgs']
            num_frames = len(imgs)

            if num_frames < self.clip_len:
                # 计算需要重复的帧数
                extra_frames = self.clip_len - num_frames
                # 创建重复序列：每个帧重复一次（初始状态）
                indices = list(range(num_frames))

                # 优先重复前面的帧（0到num_frames-2），直到满足总帧数要求
                # 跳过最后一帧（索引num_frames-1），因为它可能不需要重复
                for i in range(num_frames - 1):
                    if extra_frames > 0:
                        # 在当前位置插入重复帧
                        insert_index = indices.index(i) + 1
                        indices.insert(insert_index, i)
                        extra_frames -= 1

                # 如果仍然需要更多帧，则重复最后一帧
                if extra_frames > 0:
                    last_index = num_frames - 1
                    insert_index = indices.index(last_index) + 1
                    indices.insert(insert_index, last_index)
                    extra_frames -= 1

                # 如果还需要更多帧，则从开头继续重复
                while extra_frames > 0:
                    # 从第一个帧开始重复
                    for i in range(num_frames - 1):
                        if extra_frames > 0:
                            # 找到第一个出现位置并在其后插入重复帧
                            insert_index = indices.index(i) + 1
                            indices.insert(insert_index, i)
                            extra_frames -= 1
                        else:
                            break
            else:
                # 等间隔采样
                indices = np.linspace(0, num_frames - 1, self.clip_len, dtype=int).tolist()

            # 采样帧
            results['imgs'] = [imgs[i] for i in indices]

            # 采样关键点（若存在）
            if 'keypoint' in results:
                kp = results['keypoint']
                # 关键点形状可能是 [N, T, V, C] 或 [T, V, C]
                if kp.ndim == 4:  # [N, T, V, C]
                    results['keypoint'] = kp[:, indices, :, :]
                elif kp.ndim == 3:  # [T, V, C]
                    results['keypoint'] = kp[indices, ...]
        # === 结束改进 ===

        img_h, img_w = results['img_shape']

        left, top, right, bottom = self.get_crop_bbox(
            (img_h, img_w), self.area_range, self.aspect_ratio_range)
        new_h, new_w = bottom - top, right - left

        if 'crop_quadruple' not in results:
            results['crop_quadruple'] = np.array(
                [0, 0, 1, 1],  # x, y, w, h
                dtype=np.float32)

        x_ratio, y_ratio = left / img_w, top / img_h
        w_ratio, h_ratio = new_w / img_w, new_h / img_h

        old_crop_quadruple = results['crop_quadruple']
        old_x_ratio, old_y_ratio = old_crop_quadruple[0], old_crop_quadruple[1]
        old_w_ratio, old_h_ratio = old_crop_quadruple[2], old_crop_quadruple[3]
        new_crop_quadruple = [
            old_x_ratio + x_ratio * old_w_ratio,
            old_y_ratio + y_ratio * old_h_ratio, w_ratio * old_w_ratio,
            h_ratio * old_h_ratio
        ]
        results['crop_quadruple'] = np.array(
            new_crop_quadruple, dtype=np.float32)

        crop_bbox = np.array([left, top, right, bottom])
        results['crop_bbox'] = crop_bbox
        results['img_shape'] = (new_h, new_w)

        if not self.lazy:
            if 'keypoint' in results:
                results['keypoint'] = self._crop_kps(results['keypoint'],
                                                     crop_bbox)
            if 'imgs' in results:
                results['imgs'] = self._crop_imgs(results['imgs'], crop_bbox)
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')

            # record crop_bbox in lazyop dict to ensure only crop once in Fuse
            lazy_left, lazy_top, lazy_right, lazy_bottom = lazyop['crop_bbox']
            left = left * (lazy_right - lazy_left) / img_w
            right = right * (lazy_right - lazy_left) / img_w
            top = top * (lazy_bottom - lazy_top) / img_h
            bottom = bottom * (lazy_bottom - lazy_top) / img_h
            lazyop['crop_bbox'] = np.array([(lazy_left + left),
                                            (lazy_top + top),
                                            (lazy_left + right),
                                            (lazy_top + bottom)],
                                           dtype=np.float32)

        if 'gt_bboxes' in results:
            assert not self.lazy
            results = self._all_box_crop(results, results['crop_bbox'])

        results['clip_len'] = self.clip_len
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'area_range={self.area_range}, '
                    f'aspect_ratio_range={self.aspect_ratio_range}, '
                    f'lazy={self.lazy}, '
                    f'clip_len={self.clip_len})')
        return repr_str


@TRANSFORMS.register_module()
class doubleCrop(RandomCrop):
    """Enhanced RandomResizedCrop with normalization and random erasing.

    This class extends the basic RandomResizedCrop to include:
    - Normalization with mean and std
    - Random horizontal flipping
    - Random erasing
    - Generating two augmented versions of each frame and interleaving them

    Required keys in results are "img_shape", "crop_bbox", "imgs" (optional),
    "keypoint" (optional), added or modified keys are "imgs", "keypoint",
    "crop_bbox" and "lazy"; Required keys in "lazy" are "flip", "crop_bbox",
    added or modified key is "crop_bbox".

    Args:
        area_range (Tuple[float]): The candidate area scales range of
            output cropped images. Default: (0.08, 1.0).
        aspect_ratio_range (Tuple[float]): The candidate aspect ratio range of
            output cropped images. Default: (3 / 4, 4 / 3).
        lazy (bool): Determine whether to apply lazy operation. Default: False.
        mean (Sequence[float]): Mean values for normalization.
            Default: (0.485, 0.456, 0.406).
        std (Sequence[float]): Std values for normalization.
            Default: (0.229, 0.224, 0.225).
        random_flip (bool): Whether to apply random horizontal flip.
            Default: True.
        flip_ratio (float): Probability of implementing flip. Default: 0.5.
        random_erasing (bool): Whether to apply random erasing. Default: False.
        erase_prob (float): Probability that image will be randomly erased.
            Default: 0.5.
        min_area_ratio (float): Minimum erased area / input image area.
            Default: 0.02.
        max_area_ratio (float): Maximum erased area / input image area.
            Default: 1/3.
        aspect_range (sequence | float): Aspect ratio range of erased area.
            Default: (3/10, 10/3).
        mode (str): Fill method in erased area. Default: 'const'.
        fill_color (sequence | Number): Base color filled in erased area.
            Default: (128, 128, 128).
        fill_std (sequence | Number, optional): If set and mode is 'rand',
            fill erased area with random color from normal distribution.
            Default: None.
        num_augments (int): Number of augmented versions to generate from each frame.
            Default: 2.
    """

    def __init__(self,
                 area_range=(0.08, 1.0),
                 aspect_ratio_range=(3 / 4, 4 / 3),
                 lazy=False,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225),
                 random_flip=True,
                 flip_ratio=0.5,
                 random_erasing=False,
                 erase_prob=0.5,
                 min_area_ratio=0.02,
                 max_area_ratio=1/3,
                 aspect_range=(3/10, 10/3),
                 mode='const',
                 fill_color=(128, 128, 128),
                 fill_std=None,
                 num_augments=2):
        super().__init__(size=0, lazy=lazy)  # size will be set in transform
        self.area_range = area_range
        self.aspect_ratio_range = aspect_ratio_range
        self.lazy = lazy
        self.mean = mean
        self.std = std
        self.random_flip = random_flip
        self.flip_ratio = flip_ratio
        self.random_erasing = random_erasing
        self.num_augments = num_augments

        # Random erasing parameters
        self.erase_prob = erase_prob
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.aspect_range = aspect_range
        self.mode = mode
        self.fill_color = fill_color
        self.fill_std = fill_std

        if not mmengine.is_tuple_of(self.area_range, float):
            raise TypeError(f'Area_range must be a tuple of float, '
                            f'but got {type(area_range)}')
        if not mmengine.is_tuple_of(self.aspect_ratio_range, float):
            raise TypeError(f'Aspect_ratio_range must be a tuple of float, '
                            f'but got {type(aspect_ratio_range)}')

        if self.num_augments < 1:
            raise ValueError(f'num_augments must be at least 1, got {self.num_augments}')

    @staticmethod
    def get_crop_bbox(img_shape,
                      area_range,
                      aspect_ratio_range,
                      max_attempts=10):
        """Get a crop bbox given the area range and aspect ratio range.

        Args:
            img_shape (Tuple[int]): Image shape
            area_range (Tuple[float]): The candidate area scales range of
                output cropped images. Default: (0.08, 1.0).
            aspect_ratio_range (Tuple[float]): The candidate aspect
                ratio range of output cropped images. Default: (3 / 4, 4 / 3).
                max_attempts (int): The maximum of attempts. Default: 10.
            max_attempts (int): Max attempts times to generate random candidate
                bounding box. If it doesn't qualified one, the center bounding
                box will be used.
        Returns:
            (list[int]) A random crop bbox within the area range and aspect
            ratio range.
        """
        assert 0 < area_range[0] <= area_range[1] <= 1
        assert 0 < aspect_ratio_range[0] <= aspect_ratio_range[1]

        img_h, img_w = img_shape
        area = img_h * img_w

        min_ar, max_ar = aspect_ratio_range
        aspect_ratios = np.exp(
            np.random.uniform(
                np.log(min_ar), np.log(max_ar), size=max_attempts))
        target_areas = np.random.uniform(*area_range, size=max_attempts) * area
        candidate_crop_w = np.round(np.sqrt(target_areas *
                                            aspect_ratios)).astype(np.int32)
        candidate_crop_h = np.round(np.sqrt(target_areas /
                                            aspect_ratios)).astype(np.int32)

        for i in range(max_attempts):
            crop_w = candidate_crop_w[i]
            crop_h = candidate_crop_h[i]
            if crop_h <= img_h and crop_w <= img_w:
                x_offset = random.randint(0, img_w - crop_w)
                y_offset = random.randint(0, img_h - crop_h)
                return x_offset, y_offset, x_offset + crop_w, y_offset + crop_h

        # Fallback
        crop_size = min(img_h, img_w)
        x_offset = (img_w - crop_size) // 2
        y_offset = (img_h - crop_size) // 2
        return x_offset, y_offset, x_offset + crop_size, y_offset + crop_size

    def _normalize(self, imgs):
        """Normalize images with mean and std, ensuring float32 type."""
        # 确保 mean 和 std 是 float32 类型
        if isinstance(self.mean, list):
            mean = torch.tensor(self.mean, dtype=torch.float32)
        else:
            mean = torch.tensor(self.mean, dtype=torch.float32)

        if isinstance(self.std, list):
            std = torch.tensor(self.std, dtype=torch.float32)
        else:
            std = torch.tensor(self.std, dtype=torch.float32)

        normalized_imgs = []
        for img in imgs:
            # 确保图像是 float32 类型
            if img.dtype == torch.uint8:
                img = img.float()
                img = img / 255.0
            elif img.dtype == torch.float64:
                img = img.float()  # 将 float64 转换为 float32

            img = img - mean
            img = img / std
            normalized_imgs.append(img)

        return normalized_imgs

    def _random_erasing(self, imgs, img_h, img_w):
        """Apply random erasing to images."""
        if not self.random_erasing or random.random() > self.erase_prob:
            return imgs

        # Convert aspect_range to log space to equally handle width and height
        log_aspect_range = np.log(np.array(self.aspect_range, dtype=np.float32))
        aspect_ratio = np.exp(np.random.uniform(*log_aspect_range))

        area = img_h * img_w
        area *= np.random.uniform(self.min_area_ratio, self.max_area_ratio)

        h = min(int(round(np.sqrt(area * aspect_ratio))), img_h)
        w = min(int(round(np.sqrt(area / aspect_ratio))), img_w)
        top = random.randint(0, img_h - h) if img_h > h else 0
        left = random.randint(0, img_w - w) if img_w > w else 0

        # Fill the erased area
        if self.mode == 'const':
            fill_value = np.array(self.fill_color, dtype=np.uint8)
        elif self.fill_std is None:
            # Uniform distribution
            fill_value = np.random.uniform(0, 256, 3).astype(np.uint8)
        else:
            # Normal distribution
            fill_value = np.random.normal(self.fill_color, self.fill_std, 3)
            fill_value = np.clip(fill_value.astype(np.int32), 0, 255).astype(np.uint8)

        # Apply erasing to each image
        erased_imgs = []
        for img in imgs:
            img = img.copy()
            img[top:top + h, left:left + w] = fill_value
            erased_imgs.append(img)

        return erased_imgs

    def _random_flip(self, imgs, modality='RGB'):
        """Apply random horizontal flip to images."""
        if not self.random_flip or random.random() > self.flip_ratio:
            return imgs

        flipped_imgs = [mmcv.imflip(img, 'horizontal') for img in imgs]

        # Handle flow modality specially
        if modality == 'Flow':
            # The 1st frame of each 2 frames is flow-x
            for i in range(0, len(flipped_imgs), 2):
                flipped_imgs[i] = mmcv.iminvert(flipped_imgs[i])

        return flipped_imgs

    def _apply_augmentation(self, results, modality):
        """Apply a single augmentation to the results."""
        img_h, img_w = results['img_shape']

        # Get crop bbox
        left, top, right, bottom = self.get_crop_bbox(
            (img_h, img_w), self.area_range, self.aspect_ratio_range)
        new_h, new_w = bottom - top, right - left

        # Update crop quadruple
        if 'crop_quadruple' not in results:
            crop_quadruple = np.array([0, 0, 1, 1], dtype=np.float32)
        else:
            crop_quadruple = results['crop_quadruple'].copy()

        x_ratio, y_ratio = left / img_w, top / img_h
        w_ratio, h_ratio = new_w / img_w, new_h / img_h

        old_x_ratio, old_y_ratio = crop_quadruple[0], crop_quadruple[1]
        old_w_ratio, old_h_ratio = crop_quadruple[2], crop_quadruple[3]
        new_crop_quadruple = [
            old_x_ratio + x_ratio * old_w_ratio,
            old_y_ratio + y_ratio * old_h_ratio,
            w_ratio * old_w_ratio,
            h_ratio * old_h_ratio
        ]
        crop_quadruple = np.array(new_crop_quadruple, dtype=np.float32)

        crop_bbox = np.array([left, top, right, bottom])
        img_shape = (new_h, new_w)

        # Crop images
        if 'imgs' in results:
            imgs = self._crop_imgs(results['imgs'], crop_bbox)

            # Apply random flip
            imgs = self._random_flip(imgs, modality)

            # Normalize images
            #imgs = self._normalize(imgs)

            # Apply random erasing
            imgs = self._random_erasing(imgs, new_h, new_w)
        else:
            imgs = None

        # Crop keypoints
        if 'keypoint' in results:
            keypoint = self._crop_kps(results['keypoint'], crop_bbox)
        else:
            keypoint = None

        # Crop bounding boxes
        if 'gt_bboxes' in results:
            gt_bboxes = self._box_crop(results['gt_bboxes'], crop_bbox)
        else:
            gt_bboxes = None

        if 'proposals' in results and results['proposals'] is not None:
            proposals = self._box_crop(results['proposals'], crop_bbox)
        else:
            proposals = None

        return {
            'imgs': imgs,
            'keypoint': keypoint,
            'gt_bboxes': gt_bboxes,
            'proposals': proposals,
            'crop_bbox': crop_bbox,
            'img_shape': img_shape,
            'crop_quadruple': crop_quadruple
        }

    def transform(self, results):
        """Performs the Enhanced RandomResizeCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, self.lazy)
        if 'keypoint' in results:
            assert not self.lazy, ('Keypoint Augmentations are not compatible '
                                   'with lazy == True')

        modality = results.get('modality', 'RGB')

        # Apply augmentation multiple times
        augmented_results = []
        for i in range(self.num_augments):
            # For each augmentation, we need to use the original input
            # So we create a copy of the original results for each augmentation
            aug_result = self._apply_augmentation(results.copy(), modality)
            augmented_results.append(aug_result)

        # Merge the augmented results by interleaving frames
        # For images, we interleave frames from different augmentations
        if 'imgs' in results:
            original_frame_count = len(results['imgs'])
            all_imgs = []

            # For each original frame position, add all augmented versions
            for frame_idx in range(original_frame_count):
                for aug_idx in range(self.num_augments):
                    if augmented_results[aug_idx]['imgs'] is not None:
                        all_imgs.append(augmented_results[aug_idx]['imgs'][frame_idx])
            results['imgs'] = all_imgs

        # For keypoints, we interleave frames from different augmentations
        if 'keypoint' in results:
            original_frame_count = results['keypoint'].shape[0]
            all_keypoints = []

            for frame_idx in range(original_frame_count):
                for aug_idx in range(self.num_augments):
                    if augmented_results[aug_idx]['keypoint'] is not None:
                        all_keypoints.append(augmented_results[aug_idx]['keypoint'][frame_idx])

            if all_keypoints:
                results['keypoint'] = np.stack(all_keypoints, axis=0)

        # For bounding boxes, we interleave frames from different augmentations
        if 'gt_bboxes' in results:
            original_frame_count = results['gt_bboxes'].shape[0]
            all_gt_bboxes = []

            for frame_idx in range(original_frame_count):
                for aug_idx in range(self.num_augments):
                    if augmented_results[aug_idx]['gt_bboxes'] is not None:
                        all_gt_bboxes.append(augmented_results[aug_idx]['gt_bboxes'][frame_idx])

            if all_gt_bboxes:
                results['gt_bboxes'] = np.stack(all_gt_bboxes, axis=0)

        # For proposals, we interleave frames from different augmentations
        if 'proposals' in results and results['proposals'] is not None:
            original_frame_count = results['proposals'].shape[0]
            all_proposals = []

            for frame_idx in range(original_frame_count):
                for aug_idx in range(self.num_augments):
                    if augmented_results[aug_idx]['proposals'] is not None:
                        all_proposals.append(augmented_results[aug_idx]['proposals'][frame_idx])

            if all_proposals:
                results['proposals'] = np.stack(all_proposals, axis=0)

        # Use the crop information from the first augmentation
        # (this might need to be adjusted depending on your use case)
        if augmented_results:
            results['crop_bbox'] = augmented_results[0]['crop_bbox']
            results['img_shape'] = augmented_results[0]['img_shape']
            results['crop_quadruple'] = augmented_results[0]['crop_quadruple']

        # If we have labels and we're doing multiple augmentations,
        # we need to repeat the labels accordingly
        if 'label' in results and self.num_augments > 1:
            results['label'] = np.repeat(results['label'], self.num_augments)

        results['clip_len'] = results['clip_len'] * 2
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'area_range={self.area_range}, '
                    f'aspect_ratio_range={self.aspect_ratio_range}, '
                    f'lazy={self.lazy}, '
                    f'mean={self.mean}, '
                    f'std={self.std}, '
                    f'random_flip={self.random_flip}, '
                    f'flip_ratio={self.flip_ratio}, '
                    f'random_erasing={self.random_erasing}, '
                    f'erase_prob={self.erase_prob}, '
                    f'min_area_ratio={self.min_area_ratio}, '
                    f'max_area_ratio={self.max_area_ratio}, '
                    f'aspect_range={self.aspect_range}, '
                    f'mode={self.mode}, '
                    f'fill_color={self.fill_color}, '
                    f'fill_std={self.fill_std}, '
                    f'num_augments={self.num_augments})')
        return repr_str

@TRANSFORMS.register_module()
class DAFL:
    """ Mixup/Cutmix that applies different params to each element or whole batch

    Args:
        mixup_alpha (float): mixup alpha value, mixup is active if > 0.
        cutmix_alpha (float): cutmix alpha value, cutmix is active if > 0.
        cutmix_minmax (List[float]): cutmix min/max image ratio, cutmix is active and uses this vs alpha if not None.
        prob (float): probability of applying mixup or cutmix per batch or element
        switch_prob (float): probability of switching to cutmix instead of mixup when both are active
        mode (str): how to apply mixup/cutmix params (per 'batch', 'pair' (pair of elements), 'elem' (element)
        correct_lam (bool): apply lambda correction when cutmix bbox clipped by image borders
        label_smoothing (float): apply label smoothing to the mixed target tensor
        num_classes (int): number of classes for target
    """

    def __init__(self, mixup_alpha=1., cutmix_alpha=0., cutmix_minmax=None, prob=1.0, switch_prob=0.5,
                 mode='batch', correct_lam=True, label_smoothing=0.1, num_classes=8):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_minmax = cutmix_minmax
        if self.cutmix_minmax is not None:
            assert len(self.cutmix_minmax) == 2
            # force cutmix alpha == 1.0 when minmax active to keep logic simple & safe
            self.cutmix_alpha = 1.0
        self.mix_prob = prob
        self.switch_prob = switch_prob
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.mode = mode
        self.correct_lam = correct_lam  # correct lambda based on clipped area for cutmix
        self.mixup_enabled = True  # set to false to disable mixing (intended to be set by train loop)

    def _params_per_batch(self):
        lam = 1.
        use_cutmix = False
        if self.mixup_enabled and np.random.rand() < self.mix_prob:
            if self.mixup_alpha > 0. and self.cutmix_alpha > 0.:
                use_cutmix = np.random.rand() < self.switch_prob
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha) if use_cutmix else \
                    np.random.beta(self.mixup_alpha, self.mixup_alpha)
            elif self.mixup_alpha > 0.:
                lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            elif self.cutmix_alpha > 0.:
                use_cutmix = True
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            else:
                assert False, "One of mixup_alpha > 0., cutmix_alpha > 0., cutmix_minmax not None should be true."
            lam = float(lam_mix)
        return lam, use_cutmix

    def _cutmix_bbox_and_lam(self, shape, lam, ratio_minmax=None, correct_lam=True):
        """ Generate bbox and apply lambda correction for cutmix """
        if ratio_minmax is not None:
            raise NotImplementedError("ratio_minmax not implemented for video cutmix")

        H, W = shape[-2], shape[-1]
        r = np.sqrt(1. - lam)
        w = int(W * r)
        h = int(H * r)
        x = np.random.randint(W)
        y = np.random.randint(H)

        x1 = np.clip(x - w // 2, 0, W)
        y1 = np.clip(y - h // 2, 0, H)
        x2 = np.clip(x + w // 2, 0, W)
        y2 = np.clip(y + h // 2, 0, H)

        bbox = (y1, y2, x1, x2)

        if correct_lam:
            # Adjust lambda based on the actual clipped area
            area = (y2 - y1) * (x2 - x1)
            lam = 1. - area / float(H * W)

        return bbox, lam

    def _mix_batch(self, x):
        """Apply mixup/cutmix to video data in NCTHW format"""
        assert len(x.shape) == 5, "Input must be in NCTHW format"

        lam, use_cutmix = self._params_per_batch()
        if lam == 1.:
            return x, lam

        # Flip the batch along the batch dimension for mixing
        x_flipped = x.flip(0)

        if use_cutmix:
            # Generate cutmix bbox - same for all frames in a video
            bbox, lam = self._cutmix_bbox_and_lam(x.shape, lam,
                                                  ratio_minmax=self.cutmix_minmax,
                                                  correct_lam=self.correct_lam)
            y1, y2, x1, x2 = bbox

            # Apply cutmix to all frames
            x[:, :, :, y1:y2, x1:x2] = x_flipped[:, :, :, y1:y2, x1:x2]
        else:
            # Apply mixup to the entire batch
            x = x * lam + x_flipped * (1 - lam)

        return x, lam

    def __call__(self, results):
        """Apply mixup/cutmix to video data and labels"""
        assert 'imgs' in results, "Input must contain 'imgs' key"
        assert 'label' in results, "Input must contain 'label' key"

        # Convert to tensor if needed
        if not isinstance(results['imgs'], torch.Tensor):
            imgs = torch.from_numpy(np.stack(results['imgs']))
        else:
            imgs = results['imgs']

        # Ensure NCTHW format
        if len(imgs.shape) == 4:  # Assume TCHW
            imgs = imgs.unsqueeze(0)  # Add batch dimension

        # Apply mixing
        mixed_imgs, lam = self._mix_batch(imgs)

        # Convert back to numpy if original was numpy
        if not isinstance(results['imgs'], torch.Tensor):
            mixed_imgs = mixed_imgs.squeeze(0).numpy()  # Remove batch dimension
            results['imgs'] = [mixed_imgs[i] for i in range(mixed_imgs.shape[0])]
        else:
            results['imgs'] = mixed_imgs.squeeze(0)  # Remove batch dimension

        # Mix labels
        label = results['label']
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label)

        # Apply label mixing
        one_hot = torch.zeros(self.num_classes)
        one_hot[label] = 1
        mixed_label = one_hot * lam + torch.ones_like(one_hot) * (1 - lam) * (1 / self.num_classes)

        results['label'] = mixed_label

        return results


@TRANSFORMS.register_module()
class fencengCrop(RandomCrop):
    """Random crop that specifics the area and height-weight ratio range.

    Required keys in results are "img_shape", "crop_bbox", "imgs" (optional),
    "keypoint" (optional), added or modified keys are "imgs", "keypoint",
    "crop_bbox" and "lazy"; Required keys in "lazy" are "flip", "crop_bbox",
    added or modified key is "crop_bbox".

    Args:
        area_range (Tuple[float]): The candidate area scales range of
            output cropped images. Default: (0.08, 1.0).
        aspect_ratio_range (Tuple[float]): The candidate aspect ratio range of
            output cropped images. Default: (3 / 4, 4 / 3).
        lazy (bool): Determine whether to apply lazy operation. Default: False.
        class_weights (Dict[int, float]): Weight for each class to adjust crop
            sampling. Higher weight means more focus on central/important regions.
            Default: None.
        minority_class_id (int): Class ID of the minority class. Default: 1.
    """

    def __init__(self,
                 area_range=(0.08, 1.0),
                 aspect_ratio_range=(3 / 4, 4 / 3),
                 lazy=False,
                 class_weights=None,
                 minority_class_id=1):
        self.area_range = area_range
        self.aspect_ratio_range = aspect_ratio_range
        self.lazy = lazy
        self.class_weights = class_weights
        self.minority_class_id = minority_class_id

        if not mmengine.is_tuple_of(self.area_range, float):
            raise TypeError(f'Area_range must be a tuple of float, '
                            f'but got {type(area_range)}')
        if not mmengine.is_tuple_of(self.aspect_ratio_range, float):
            raise TypeError(f'Aspect_ratio_range must be a tuple of float, '
                            f'but got {type(aspect_ratio_range)}')

        # Set default class weights if not provided
        if self.class_weights is None:
            self.class_weights = {0: 1.0, 1: 1.5}  # Higher weight for minority class

    @staticmethod
    def get_crop_bbox(img_shape,
                      area_range,
                      aspect_ratio_range,
                      max_attempts=10,
                      focus_center=False):
        """Get a crop bbox given the area range and aspect ratio range.

        Args:
            img_shape (Tuple[int]): Image shape
            area_range (Tuple[float]): The candidate area scales range of
                output cropped images. Default: (0.08, 1.0).
            aspect_ratio_range (Tuple[float]): The candidate aspect
                ratio range of output cropped images. Default: (3 / 4, 4 / 3).
            max_attempts (int): The maximum of attempts. Default: 10.
            focus_center (bool): If True, bias crop towards center of image.
                Default: False.
        Returns:
            (list[int]) A random crop bbox within the area range and aspect
            ratio range.
        """
        assert 0 < area_range[0] <= area_range[1] <= 1
        assert 0 < aspect_ratio_range[0] <= aspect_ratio_range[1]

        img_h, img_w = img_shape
        area = img_h * img_w

        min_ar, max_ar = aspect_ratio_range
        aspect_ratios = np.exp(
            np.random.uniform(
                np.log(min_ar), np.log(max_ar), size=max_attempts))
        target_areas = np.random.uniform(*area_range, size=max_attempts) * area
        candidate_crop_w = np.round(np.sqrt(target_areas *
                                            aspect_ratios)).astype(np.int32)
        candidate_crop_h = np.round(np.sqrt(target_areas /
                                            aspect_ratios)).astype(np.int32)

        for i in range(max_attempts):
            crop_w = candidate_crop_w[i]
            crop_h = candidate_crop_h[i]
            if crop_h <= img_h and crop_w <= img_w:
                if focus_center:
                    # Bias towards center regions for important samples
                    center_x = img_w // 2
                    center_y = img_h // 2

                    # Calculate max offset to keep crop within image
                    max_x_offset = max(0, img_w - crop_w)
                    max_y_offset = max(0, img_h - crop_h)

                    # Generate offsets with bias towards center
                    x_offset = int(np.clip(
                        np.random.normal(center_x - crop_w / 2, max_x_offset / 4),
                        0, max_x_offset))
                    y_offset = int(np.clip(
                        np.random.normal(center_y - crop_h / 2, max_y_offset / 4),
                        0, max_y_offset))
                else:
                    x_offset = random.randint(0, img_w - crop_w)
                    y_offset = random.randint(0, img_h - crop_h)

                return x_offset, y_offset, x_offset + crop_w, y_offset + crop_h

        # Fallback
        crop_size = min(img_h, img_w)
        x_offset = (img_w - crop_size) // 2
        y_offset = (img_h - crop_size) // 2
        return x_offset, y_offset, x_offset + crop_size, y_offset + crop_size

    def transform(self, results):
        """Performs the RandomResizeCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, self.lazy)
        if 'keypoint' in results:
            assert not self.lazy, ('Keypoint Augmentations are not compatible '
                                   'with lazy == True')

        img_h, img_w = results['img_shape']

        # Determine if this is a minority class sample
        is_minority = False
        if 'label' in results and self.class_weights is not None:
            label = results['label']
            if label == self.minority_class_id:
                is_minority = True

        # Adjust crop strategy based on class weight
        focus_center = False
        if is_minority and self.class_weights.get(self.minority_class_id, 1.0) > 1.0:
            focus_center = True

            # For minority class, use slightly larger area range to capture more context
            adjusted_area_range = (
                max(self.area_range[0] * 0.8, 0.05),  # Larger minimum area
                min(self.area_range[1] * 1.1, 1.0)  # Larger maximum area
            )
        else:
            adjusted_area_range = self.area_range

        left, top, right, bottom = self.get_crop_bbox(
            (img_h, img_w), adjusted_area_range, self.aspect_ratio_range,
            focus_center=focus_center)
        new_h, new_w = bottom - top, right - left

        if 'crop_quadruple' not in results:
            results['crop_quadruple'] = np.array(
                [0, 0, 1, 1],  # x, y, w, h
                dtype=np.float32)

        x_ratio, y_ratio = left / img_w, top / img_h
        w_ratio, h_ratio = new_w / img_w, new_h / img_h

        old_crop_quadruple = results['crop_quadruple']
        old_x_ratio, old_y_ratio = old_crop_quadruple[0], old_crop_quadruple[1]
        old_w_ratio, old_h_ratio = old_crop_quadruple[2], old_crop_quadruple[3]
        new_crop_quadruple = [
            old_x_ratio + x_ratio * old_w_ratio,
            old_y_ratio + y_ratio * old_h_ratio, w_ratio * old_w_ratio,
            h_ratio * old_h_ratio
        ]
        results['crop_quadruple'] = np.array(
            new_crop_quadruple, dtype=np.float32)

        crop_bbox = np.array([left, top, right, bottom])
        results['crop_bbox'] = crop_bbox
        results['img_shape'] = (new_h, new_w)

        if not self.lazy:
            if 'keypoint' in results:
                results['keypoint'] = self._crop_kps(results['keypoint'],
                                                     crop_bbox)
            if 'imgs' in results:
                results['imgs'] = self._crop_imgs(results['imgs'], crop_bbox)
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')

            # record crop_bbox in lazyop dict to ensure only crop once in Fuse
            lazy_left, lazy_top, lazy_right, lazy_bottom = lazyop['crop_bbox']
            left = left * (lazy_right - lazy_left) / img_w
            right = right * (lazy_right - lazy_left) / img_w
            top = top * (lazy_bottom - lazy_top) / img_h
            bottom = bottom * (lazy_bottom - lazy_top) / img_h
            lazyop['crop_bbox'] = np.array([(lazy_left + left),
                                            (lazy_top + top),
                                            (lazy_left + right),
                                            (lazy_top + bottom)],
                                           dtype=np.float32)

        if 'gt_bboxes' in results:
            assert not self.lazy
            results = self._all_box_crop(results, results['crop_bbox'])

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'area_range={self.area_range}, '
                    f'aspect_ratio_range={self.aspect_ratio_range}, '
                    f'lazy={self.lazy}, '
                    f'class_weights={self.class_weights}, '
                    f'minority_class_id={self.minority_class_id})')
        return repr_str

@TRANSFORMS.register_module()
class ViolentAwareCrop(BaseTransform):
    """��Ա��������ı����Բü�"""

    def __init__(self,
                 violent_crop_range=(0.6, 1.0),  # ������������60%~100%����
                 normal_crop_range=(0.08, 1.0),  # ������������ü�
                 crop_size=224,
                 lazy=False):
        self.violent_crop_range = violent_crop_range
        self.normal_crop_range = normal_crop_range
        self.crop_size = crop_size
        self.lazy = lazy

    def transform(self, results):
        label = results.get('label', 0)
        img_h, img_w = results['img_shape']

        # ѡ��ü�����
        if label == 1:
            # ��������������ƫ��ü�
            scale = np.random.uniform(*self.violent_crop_range)
            crop_h = int(img_h * scale)
            crop_w = int(img_w * scale)
            # �ü����������ڻ������ġ�15%��Χ��
            center_x = img_w // 2
            center_y = img_h // 2
            x_offset = np.clip(
                np.random.randint(center_x - int(0.15 * img_w),
                                  center_x + int(0.15 * img_w)),
                0, img_w - crop_w
            )
            y_offset = np.clip(
                np.random.randint(center_y - int(0.15 * img_h),
                                  center_y + int(0.15 * img_h)),
                0, img_h - crop_h
            )
        else:
            # ��������������ü�
            scale = np.random.uniform(*self.normal_crop_range)
            crop_h = int(img_h * scale)
            crop_w = int(img_w * scale)
            x_offset = np.random.randint(0, img_w - crop_w)
            y_offset = np.random.randint(0, img_h - crop_h)

        # ִ�вü������������أ�
        crop_bbox = [x_offset, y_offset, x_offset + crop_w, y_offset + crop_h]
        # ...���ü��߼���MMAction2���нӿڶ��룬ʡ��ʵ�֣�
        return results

@TRANSFORMS.register_module()
class STSZCrop(BaseTransform):
    """Crop the images into 10 crops (top, bottom, left, right, center + flip)
    after temporal sampling with fixed interval.

    First sample frames with a fixed temporal interval, then crop the top,
    bottom, left, right and center parts of the image with the same given
    crop_size, and flip it horizontally.
    Required keys are "imgs", "img_shape", added or modified keys are "imgs",
    "crop_bbox" and "img_shape".

    Args:
        crop_size(int | tuple[int]): (w, h) of crop size.
        temporal_interval (int): Interval for temporal sampling.
            Default: 1 (no sampling).
        num_frames (int): Number of frames to sample. If None, uses all available
            frames after interval sampling. Default: None.
    """

    def __init__(self, crop_size, temporal_interval=1, num_frames=48):
        self.crop_size = _pair(crop_size)
        self.temporal_interval = temporal_interval
        self.num_frames = num_frames
        if not mmengine.is_tuple_of(self.crop_size, int):
            raise TypeError(f'Crop_size must be int or tuple of int, '
                            f'but got {type(crop_size)}')

    def _temporal_sampling(self, frames):
        """Perform temporal sampling with fixed interval.

        If the sampled frames exceed the total frames, sample from the middle
        and expand to both sides with repetition of adjacent frames.

        Args:
            frames (list): List of input frames.

        Returns:
            list: Sampled frames.
        """
        total_frames = len(frames)

        # If no specific number of frames requested, use interval sampling
        if self.num_frames is None:
            return frames[::self.temporal_interval]

        # Calculate how many frames we can get with interval sampling
        interval_frames = frames[::self.temporal_interval]

        # If we have enough frames with interval sampling
        if len(interval_frames) >= self.num_frames:
            # Just take the first num_frames
            return interval_frames[:self.num_frames]

        # If we don't have enough frames with interval sampling
        # We need to sample from the middle and expand to both sides
        # with repetition of adjacent frames

        # Start from the middle frame
        mid_idx = total_frames // 2
        sampled_frames = [frames[mid_idx]]

        # Expand to both sides with repetition of adjacent frames
        # We'll alternate between left and right
        left_ptr = mid_idx - 1
        right_ptr = mid_idx + 1
        expand_left = True

        while len(sampled_frames) < self.num_frames:
            if expand_left and left_ptr >= 0:
                # Add left frame and repeat it
                sampled_frames.insert(0, frames[left_ptr])
                if len(sampled_frames) < self.num_frames:
                    sampled_frames.insert(0, frames[left_ptr])  # Repeat the frame
                left_ptr -= 1
            elif not expand_left and right_ptr < total_frames:
                # Add right frame and repeat it
                sampled_frames.append(frames[right_ptr])
                if len(sampled_frames) < self.num_frames:
                    sampled_frames.append(frames[right_ptr])  # Repeat the frame
                right_ptr += 1
            else:
                # If we can't expand in the current direction, try the other direction
                expand_left = not expand_left

                # If both directions are exhausted, we need to repeat the entire sequence
                if left_ptr < 0 and right_ptr >= total_frames:
                    # Repeat the entire sampled sequence until we have enough frames
                    while len(sampled_frames) < self.num_frames:
                        # Repeat the sequence from the beginning
                        for i in range(len(sampled_frames)):
                            if len(sampled_frames) >= self.num_frames:
                                break
                            sampled_frames.append(sampled_frames[i])
                    break

            # Alternate direction for next iteration
            expand_left = not expand_left

        # Trim to exact number of requested frames
        return sampled_frames[:self.num_frames]

    def transform(self, results):
        """Performs the TenCrop augmentation with temporal sampling.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, False)

        if 'gt_bboxes' in results or 'proposals' in results:
            warnings.warn('TenCrop cannot process bounding boxes')

        # Temporal sampling with fixed interval
        imgs = results['imgs']
        imgs = self._temporal_sampling(imgs)

        if len(imgs) == 0:
            raise ValueError('No frames left after temporal sampling. '
                             'Please check your temporal_interval value.')

        img_h, img_w = imgs[0].shape[:2]
        crop_w, crop_h = self.crop_size

        # Calculate center position
        center_x = (img_w - crop_w) // 2
        center_y = (img_h - crop_h) // 2

        # Define five regions: top, bottom, left, right, center
        offsets = [
            (center_x, 0),  # top (centered horizontally)
            (center_x, img_h - crop_h),  # bottom (centered horizontally)
            (0, center_y),  # left (centered vertically)
            (img_w - crop_w, center_y),  # right (centered vertically)
            (center_x, center_y),  # center
        ]

        img_crops = list()
        crop_bboxes = list()
        for x_offset, y_offset in offsets:
            # Crop the region
            crop = [
                img[y_offset:y_offset + crop_h, x_offset:x_offset + crop_w]
                for img in imgs
            ]
            # Generate horizontal flip
            flip_crop = [np.flip(c, axis=1).copy() for c in crop]
            bbox = [x_offset, y_offset, x_offset + crop_w, y_offset + crop_h]
            # Add original and flipped crops
            img_crops.extend(crop)
            img_crops.extend(flip_crop)
            crop_bboxes.extend([bbox for _ in range(len(imgs) * 2)])

        crop_bboxes = np.array(crop_bboxes)
        results['imgs'] = img_crops
        results['crop_bbox'] = crop_bboxes
        results['img_shape'] = results['imgs'][0].shape[:2]

        results['clip_len'] = 24
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}(crop_size={self.crop_size}, '
                   f'temporal_interval={self.temporal_interval}, '
                   f'num_frames={self.num_frames})')
        return repr_str
