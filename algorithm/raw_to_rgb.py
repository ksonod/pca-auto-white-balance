import numpy as np
from enum import Enum
from scipy.io import loadmat
import cv2
from algorithm.white_balance import AutoWhiteBalance


class BayerPattern(Enum):
    BGGR = "bggr"
    GBRG = "gbrg"
    GRBG = "grbg"
    RGGB = "rggb"


class ColorCorrectionMatrix(Enum):
    NONE = "none"
    PREDIFINED = "prefefined"
    READ_MAT = "read_mat"


class Raw2RGB:
    def __init__(self, config):

        if config["no_processing"]:
            # Disable CCM, gamma, and color enhancement.
            config["color_correction_matrix"] = {
                "method": ColorCorrectionMatrix.NONE,
            }
            config["gamma"] = 1.0
            config["color_enhancement_coef"] = 1.0

        self.black_level = config["black_level"]
        self.white_level = int(config["white_level"])
        self.bayer_pattern = config["bayer_pattern"]
        self.auto_white_balance = AutoWhiteBalance(config["auto_white_balance_method"], config["verbose"])
        self.color_correction_matrix = config["color_correction_matrix"]
        self.gamma = config["gamma"]
        self.color_correction_coef = config["color_enhancement_coef"]
        self.verbose = config["verbose"]

    def __call__(self, raw_img):
        if self.verbose:
            print(f"- AWB: {self.auto_white_balance.__str__()} | "
                  f"CCM: {self.color_correction_matrix['method'].value} | "
                  f"Gamma: {self.gamma} | "
                  f"CE: {self.color_correction_coef} | "
                  f"White level {self.white_level}")

        raw_img = self.subtract_black_level(raw_img)
        demosaic_img = self.demosaic_raw(raw_img)
        r, g, b = self.auto_white_balance(demosaic_img[:, :, 0], demosaic_img[:, :, 1], demosaic_img[:, :, 2])
        r, g, b = self.apply_color_correction_matrix(r, g, b)
        r, g, b = self.apply_gamma_corection(r, g, b)
        r, g, b = self.apply_color_enhancement(r, g, b)
        rgb = np.stack([r, g, b], axis=2)
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        return rgb

    def subtract_black_level(self, raw_img):
        processed_img = raw_img - self.black_level
        processed_img[processed_img < 0] = 0
        return processed_img

    def demosaic_raw(self, raw_img):
        if self.bayer_pattern == BayerPattern.RGGB:
            demosaic_img = cv2.cvtColor(raw_img, cv2.COLOR_BayerRGGB2RGB)
        elif self.bayer_pattern == BayerPattern.BGGR:
            demosaic_img = cv2.cvtColor(raw_img, cv2.COLOR_BayerBGGR2RGB)
        elif self.bayer_pattern == BayerPattern.GRBG:
            demosaic_img = cv2.cvtColor(raw_img, cv2.COLOR_BayerGRBG2RGB)
        elif self.bayer_pattern == BayerPattern.GBRG:
            demosaic_img = cv2.cvtColor(raw_img, cv2.COLOR_BayerGBRG2RGB)
        else:
            raise NotImplementedError("Choose a right CFA pattern.")

        return demosaic_img

    def apply_color_correction_matrix(self, r, g, b):
        if self.color_correction_matrix["method"] == ColorCorrectionMatrix.PREDIFINED:
            # User can modify this matrix.
            cc_matrix = np.array([
                [1.4, -0.5, -0.1],
                [- 0.1, 1.1, 0.01],
                [0.01, -0.64, 1.6]
            ])
            cc_offset = np.array([0, 0, 0])
        elif self.color_correction_matrix["method"] == ColorCorrectionMatrix.READ_MAT:
            ccm = loadmat(self.color_correction_matrix["matfile_path"])
            cc_matrix = ccm["ccm"][:3, :].transpose()
            cc_offset = ccm["ccm"][3, :]
        elif self.color_correction_matrix["method"] == ColorCorrectionMatrix.NONE:
            cc_matrix = np.identity(3)
            cc_offset = np.zeros(3)
        else:
            raise NotImplementedError("Choose a right CCM option.")

        if self.verbose:
            print("- CCM -\n"
                  f" {cc_matrix} \n -- CCM Offset \n"
                  f" {cc_offset}")

        r_ccm = r * cc_matrix[0, 0] + g * cc_matrix[0, 1] + b * cc_matrix[0, 2] + cc_offset[0]
        b_ccm = r * cc_matrix[1, 0] + g * cc_matrix[1, 1] + b * cc_matrix[1, 2] + cc_offset[1]
        g_ccm = r * cc_matrix[2, 0] + g * cc_matrix[2, 1] + b * cc_matrix[2, 2] + cc_offset[2]

        r_ccm[r_ccm < 0] = 0
        g_ccm[g_ccm < 0] = 0
        b_ccm[b_ccm < 0] = 0

        return r, g, b

    def apply_gamma_corection(self, r, g, b):
        maximum_input_value = self.white_level - self.black_level
        coef = maximum_input_value ** (1 / self.gamma)
        rgb = np.stack([r, g, b], axis=2)
        rgb = 255 / coef * rgb ** (1 / self.gamma)
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
