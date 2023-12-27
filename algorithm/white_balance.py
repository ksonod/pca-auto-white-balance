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
    def __init__(self, awb_method: AutoWhiteBalanceMethods, awb_params: dict, verbose=False):
        self.awb_method = awb_method
        self.awb_params = awb_params
        self.wb_gain = np.array([1, 1, 1])
        self.verbose = verbose

    def __str__(self):
        return f"{self.awb_method.value}"

    def __call__(self, r, g, b, saturation_mask):
        if self.awb_method == AutoWhiteBalanceMethods.GRAYWORLD:
            self.wb_gain = self.apply_gray_world_awb(r, g, b, saturation_mask)
        elif self.awb_method == AutoWhiteBalanceMethods.PCA:
            self.wb_gain = self.apply_pca_based_method(
                r, g, b, saturation_mask
            )

        r = r * self.wb_gain[0]
        g = g * self.wb_gain[1]
        b = b * self.wb_gain[2]

        if self.verbose:
            print(f"- AWB gain: {self.wb_gain}")

        return r, g, b

    @staticmethod
    def apply_gray_world_awb(
            r: np.ndarray, g: np.ndarray, b: np.ndarray, saturation_mask: np.ndarray
    ) -> np.ndarray:

        mean_r = np.mean(r[~saturation_mask])
        mean_g = np.mean(g[~saturation_mask])
        mean_b = np.mean(b[~saturation_mask])

        return np.array([mean_g/mean_r, 1.0, mean_g/mean_b])

    def apply_pca_based_method(
            self, r: np.ndarray, g: np.ndarray, b: np.ndarray, saturation_mask: np.ndarray
    ) -> np.ndarray:
        """
        This is a function to use the algorithm developed in the following paper:
        - D. Cheng, D. K. Prasad, and M. S. Brown, Illuminant estimation for color consistency: why spatial domain
         methods work and the role of the color distribution, J. Opt. Soc. Am. A 31. 1049-1058, 2014

        :param r: Numpy array of the red component
        :param g: Numpy array of the green component
        :param b: Numpy array of the blue component
        :param saturation_mask: saturation mask to exclude some pixels
        :return: Numpy array with (3, )-shape of white balance gain
        """

        signal_scale = self.awb_params["saturation_value"]

        # Data points in the RGB space.
        ix = np.array([
            r[~saturation_mask],
            g[~saturation_mask],
            b[~saturation_mask]
        ]).T / signal_scale

        # Center in the RGB space.
        i0 = np.mean(ix, axis=0).reshape(-1, 1)

        # Scalar distance
        dx = (np.dot(ix, i0) / np.linalg.norm(ix, axis=1).reshape(-1, 1) / np.linalg.norm(i0)).flatten()
        dx_sorted = np.sort(dx)

        # Top and bottom n% of data will be selected.
        idx = int(np.round(dx.shape[0] * self.awb_params["pca_pickup_percentage"]))
        ix_selected = ix[(dx < dx_sorted[idx]) + (dx_sorted[-idx] < dx), :]

        # Conduct principle component analysis (PCA)
        sigma = np.matmul(ix_selected.T, ix_selected)/ix_selected.shape[0]
        eigen_vals, eigen_vecs = np.linalg.eig(sigma)
        principle_vec = np.abs(eigen_vecs[:, np.argmax(eigen_vals)])

        return np.array([
            principle_vec[1]/principle_vec[0],
            1.0,
            principle_vec[1]/principle_vec[2]
        ])
