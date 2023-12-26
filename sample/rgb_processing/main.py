import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from algorithm.white_balance import AutoWhiteBalanceMethods
from algorithm.raw_to_rgb import Raw2RGB, BayerPattern, ColorCorrectionMatrix
from pathlib import Path

"""
References
- https://www.kaggle.com/datasets/tenxengineers/auto-white-balance-awb
- https://www.flir.com/support-center/iis/machine-vision/application-note/using-color-correction/ 
- https://www.mathworks.com/help/images/ref/esfrchart.measurecolor.html 
"""


INPUT_FILE = {
     "raw_image": Path(r"../../data/sample_image/AlphaISP_2592x1536_12bits_RGGB_Scene12.raw")
}

CONFIG = {
    "raw_image_shape": (1536, 2592),  # height x width
    "bayer_pattern": BayerPattern.RGGB,
    "auto_white_balance_method": AutoWhiteBalanceMethods.GRAYWORLD,
    "black_level": 200,
    "white_level": 4095,
    "color_correction_matrix": {
        "method": ColorCorrectionMatrix.READ_MAT,
        "matfile_path": Path(r"../color_correction_matrix/ccm.mat")
    },
    "gamma": 2.2,  # 2.2
    "color_enhancement_coef": 1.9,
    "no_processing": False,  # only True when you want to get input data for CCM construction.
    "save_img": False,
    "verbose": True,
}


def run_scripts(input_file, config):

    raw_img = np.fromfile(input_file["raw_image"], dtype=np.uint16)
    raw_img = raw_img.reshape(config["raw_image_shape"])
    raw_to_rgb = Raw2RGB(config)
    rgb = raw_to_rgb(raw_img)

    if config["save_img"]:
        im = Image.fromarray(rgb)
        im.save(input_file["raw_image"].parent / "processed_rgb.png")

    plt.figure()
    plt.imshow(rgb)
    plt.show()


if __name__ == '__main__':
    run_scripts(input_file=INPUT_FILE, config=CONFIG)
