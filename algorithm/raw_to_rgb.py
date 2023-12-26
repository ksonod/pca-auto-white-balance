import numpy as np
from enum import Enum
from white_balance import AutoWhiteBalance


class BayerPattern(Enum):
    BGGR = "bggr"
    RGBG = "rgbg"
    GRBG = "grbg"
    RGGB = "rggb"


class Raw2RGB:
    def __init__(self, config):
        self.black_level = config["black_level"]
        self.white_level = config["white_level"]
        self.bayer_pattern = config["bayer_pattern"]
        self.auto_white_balance = AutoWhiteBalance(config["auto_white_balance_method"])
        self.use_color_correction_matrix = config["use_color_correction_matrix"]
        self.gamma = config["gamma"]
        self.color_correction_coef = config["color_enhancement_coef"]

    def __call__(self, raw_img):
        raw_img = self.subtract_black_level(raw_img)
        r, g, b = self.demosaic_raw(raw_img)
        r, g, b = self.auto_white_balance(r, g, b)
        r, g, b = self.apply_color_correction_matrix(r, g, b)
        r, g, b = self.apply_gamma_corection(r, g, b, gamma=self.gamma, maximum_input_value=self.white_level-self.black_level)
        r, g, b = self.apply_color_enhancement(r, g, b)
        rgb = np.stack([r, g, b], axis=2)
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return rgb

    def subtract_black_level(self, raw_img):
        processed_img = raw_img - self.black_level
        processed_img[processed_img < 0] = 0
        return processed_img

    def get_rgb_component_from_raw_mosaiced_image(self, raw_img):
        if self.bayer_pattern == BayerPattern.BGGR:
            r = raw_img[::2, ::2]
            gr = raw_img[::2, 1::2]
            gb = raw_img[1::2, ::2]
            b = raw_img[1::2, 1::2]
        else:
            raise NotImplementedError("Only BGGR Bayer pattern is supported.")
        return r, gr, gb, b

    def demosaic_raw(self, raw_img):
        # TODO: Implement an appropriate demosaicing algorithm.
        r, gr, gb, b = self.get_rgb_component_from_raw_mosaiced_image(raw_img)
        g = np.copy(gr)

        return r, g, b

    def apply_color_correction_matrix(self, r, g, b):
        if self.use_color_correction_matrix:
            cc_matrix = np.array([
                [1.3368, - 0.4129, - 0.0596],
                [-0.4370, 1.4319, - 0.6943],
                [- 0.1554, - 0.2949, 1.4746]
            ])
            cc_matrix = cc_matrix.transpose()  # TODO: remove it.
            cc_offset = np.array([58.7671, 61.5549, 64.4064])
        else:
            cc_matrix = np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ])
            cc_offset = np.array([0.0, 0.0, 0.0])

        r = r * cc_matrix[0, 0] + g * cc_matrix[0, 1] + b * cc_matrix[0, 2] + cc_offset[0]
        g = r * cc_matrix[1, 0] + g * cc_matrix[1, 1] + b * cc_matrix[1, 2] + cc_offset[1]
        b = r * cc_matrix[2, 0] + g * cc_matrix[2, 1] + b * cc_matrix[2, 2] + cc_offset[2]

        r[r < 0] = 0
        g[g < 0] = 0
        b[b < 0] = 0

        return r, g, b

    def apply_gamma_corection(self, r, g, b, gamma=2.2, maximum_input_value=4095 - 200):
        coef = (maximum_input_value) ** (1 / gamma)
        rgb = np.stack([r, g, b], axis=2)
        rgb = 255 / coef * rgb ** (1 / gamma)
        rgb = np.round(np.clip(rgb, 0, 255)).astype(np.uint8)
        return rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    def apply_color_enhancement(self, r, g, b):

        # YCbCr to RGB with color enhancement
        mat1 = np.array([
            [1, 0, 1.402 * self.color_correction_coef],
            [1, -0.3441 * self.color_correction_coef, -0.7141 * self.color_correction_coef],
            [1, 1.772 * self.color_correction_coef, 0]
        ])

        # RGB to YCbCr
        mat2 = np.array([
            [0.299, 0.587, 0.114],
            [-0.169, -0.331, 0.5],
            [0.5, -0.419, -0.081]
        ])

        mat = np.matmul(mat1, mat2)

        r_ce = r * mat[0, 0] + g * mat[0, 1] + b * mat[0, 2]
        g_ce = r * mat[1, 0] + g * mat[1, 1] + b * mat[1, 2]
        b_ce = r * mat[2, 0] + g * mat[2, 1] + b * mat[2, 2]

        return r_ce, g_ce, b_ce
