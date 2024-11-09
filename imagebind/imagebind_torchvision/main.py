import math
from typing import Tuple

import numpy as np
import torch

try:
    import cv2
except ImportError:
    _HAS_CV2 = False
else:
    _HAS_CV2 = True


class UniformTemporalSubsample(torch.nn.Module):
    """
    ``nn.Module`` wrapper for ``pytorchvideo.transforms.functional.uniform_temporal_subsample``.
    """

    def __init__(self, num_samples: int, temporal_dim: int = -3):
        """
        Args:
            num_samples (int): The number of equispaced samples to be selected
            temporal_dim (int): dimension of temporal to perform temporal subsample.
        """
        super().__init__()
        self._num_samples = num_samples
        self._temporal_dim = temporal_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): video tensor with shape (C, T, H, W).
        """
        return self.uniform_temporal_subsample(x, self._num_samples, self._temporal_dim)

    def uniform_temporal_subsample(
        self, x: torch.Tensor, num_samples: int, temporal_dim: int = -3
    ) -> torch.Tensor:
        """
        Uniformly subsamples num_samples indices from the temporal dimension of the video.
        When num_samples is larger than the size of temporal dimension of the video, it
        will sample frames based on nearest neighbor interpolation.

        Args:
            x (torch.Tensor): A video tensor with dimension larger than one with torch
                tensor type includes int, long, float, complex, etc.
            num_samples (int): The number of equispaced samples to be selected
            temporal_dim (int): dimension of temporal to perform temporal subsample.

        Returns:
            An x-like Tensor with subsampled temporal dimension.
        """
        t = x.shape[temporal_dim]
        assert num_samples > 0 and t > 0
        # Sample by nearest neighbor interpolation if num_samples > t.
        indices = torch.linspace(0, t - 1, num_samples)
        indices = torch.clamp(indices, 0, t - 1).long()
        return torch.index_select(x, temporal_dim, indices)


class ShortSideScale(torch.nn.Module):
    """
    ``nn.Module`` wrapper for ``pytorchvideo.transforms.functional.short_side_scale``.
    """

    def __init__(
        self, size: int, interpolation: str = "bilinear", backend: str = "pytorch"
    ):
        super().__init__()
        self._size = size
        self._interpolation = interpolation
        self._backend = backend

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): video tensor with shape (C, T, H, W).
        """
        return self.short_side_scale(x, self._size, self._interpolation, self._backend)

    def short_side_scale(
        self,
        x: torch.Tensor,
        size: int,
        interpolation: str = "bilinear",
        backend: str = "pytorch",
    ) -> torch.Tensor:
        """
        Determines the shorter spatial dim of the video (i.e. width or height) and scales
        it to the given size. To maintain aspect ratio, the longer side is then scaled
        accordingly.
        Args:
            x (torch.Tensor): A video tensor of shape (C, T, H, W) and type torch.float32.
            size (int): The size the shorter side is scaled to.
            interpolation (str): Algorithm used for upsampling,
                options: nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area'
            backend (str): backend used to perform interpolation. Options includes
                `pytorch` as default, and `opencv`. Note that opencv and pytorch behave
                differently on linear interpolation on some versions.
                https://discuss.pytorch.org/t/pytorch-linear-interpolation-is-different-from-pil-opencv/71181
        Returns:
            An x-like Tensor with scaled spatial dims.
        """  # noqa
        assert len(x.shape) == 4
        assert x.dtype == torch.float32
        assert backend in ("pytorch", "opencv")
        c, t, h, w = x.shape
        if w < h:
            new_h = int(math.floor((float(h) / w) * size))
            new_w = size
        else:
            new_h = size
            new_w = int(math.floor((float(w) / h) * size))
        if backend == "pytorch":
            return torch.nn.functional.interpolate(
                x, size=(new_h, new_w), mode=interpolation, align_corners=False
            )
        elif backend == "opencv":
            return self._interpolate_opencv(
                x, size=(new_h, new_w), interpolation=interpolation
            )
        else:
            raise NotImplementedError(f"{backend} backend not supported.")

    @torch.jit.ignore
    def _interpolate_opencv(
        self, x: torch.Tensor, size: Tuple[int, int], interpolation: str
    ) -> torch.Tensor:
        if not _HAS_CV2:
            raise ImportError(
                "opencv is required to use opencv transforms. Please "
                "install with 'pip install opencv-python'."
            )

        _opencv_pytorch_interpolation_map = {
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "bilinear": cv2.INTER_AREA,
            "bicubic": cv2.INTER_CUBIC,
        }
        assert interpolation in _opencv_pytorch_interpolation_map
        new_h, new_w = size
        img_array_list = [
            img_tensor.squeeze(0).numpy()
            for img_tensor in x.permute(1, 2, 3, 0).split(1, dim=0)
        ]
        resized_img_array_list = [
            cv2.resize(
                img_array,
                (new_w, new_h),  # The input order for OpenCV is w, h.
                interpolation=_opencv_pytorch_interpolation_map[interpolation],
            )
            for img_array in img_array_list
        ]
        img_array = np.concatenate(
            [np.expand_dims(img_array, axis=0) for img_array in resized_img_array_list],
            axis=0,
        )
        img_tensor = torch.from_numpy(np.ascontiguousarray(img_array))
        img_tensor = img_tensor.permute(3, 0, 1, 2)
        return img_tensor
