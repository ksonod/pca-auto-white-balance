"""
Auto white balance
"""

from enum import Enum
import numpy as np


class AutoWhiteBalanceMethods(Enum):
    NONE = "none"
    GRAYWORLD = "gray world"
    PCA = "PCA"


class AutoWhiteBalance:
    def __init__(self, awb_method: AutoWhiteBalanceMethods, verbose=False):
        self.awb_method = awb_method
        self.wb_gain = np.array([1, 1, 1])
        self.verbose = verbose

    def __str__(self):
        return f"{self.awb_method.value}"

    def __call__(self, r, g, b, saturation_mask):
        if self.awb_method == AutoWhiteBalanceMethods.GRAYWORLD:
            self.wb_gain = self.apply_gray_world_awb(r, g, b, saturation_mask)
        elif self.awb_method == AutoWhiteBalanceMethods.PCA:
            raise NotImplementedError("TODO")  # TODO:

        r = r * self.wb_gain[0]
        g = g * self.wb_gain[1]
        b = b * self.wb_gain[2]

        if self.verbose:
            print(f"- AWB gain: {self.wb_gain}")

        return r, g, b

    def apply_gray_world_awb(
            self, r: np.ndarray, g: np.ndarray, b: np.ndarray, saturation_mask: np.ndarray
    ) -> np.ndarray:

        mean_r = np.mean(r[~saturation_mask])
        mean_g = np.mean(g[~saturation_mask])
        mean_b = np.mean(b[~saturation_mask])

        return np.array([mean_g/mean_r, 1.0, mean_g/mean_b])
