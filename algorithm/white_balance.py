from enum import Enum
import numpy as np


class AutoWhiteBalanceMethods(Enum):
    NONE = "none"
    GRAYWORLD = "grayworld"
    PCA = "PCA"


class AutoWhiteBalance:
    def __init__(self, awb_method):
        self.awb_method = awb_method
        self.wb_gain = np.array([1, 1, 1])

    def __call__(self, r, g, b):
        if self.awb_method == AutoWhiteBalanceMethods.GRAYWORLD:
            self.wb_gain = self.apply_gray_world_awb(r, g, b)
        elif self.awb_method == AutoWhiteBalanceMethods.PCA:
            raise NotImplementedError("TODO")  # TODO:

        r = r * self.wb_gain[0]
        g = g * self.wb_gain[1]
        b = b * self.wb_gain[2]

        return r, g, b

    @staticmethod
    def apply_gray_world_awb(r, g, b):
        mean_g = np.mean(g)
        return np.array([mean_g/np.mean(r), 1.0, mean_g/np.mean(b)])
