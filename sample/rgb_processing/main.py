"""
Main script to convert raw images to standard RGB images with different AWB methods.

"""
import matplotlib.pyplot as plt
from PIL import Image
from algorithm.white_balance import AutoWhiteBalanceMethods
from algorithm.raw_to_rgb import Raw2RGB, BayerPattern, ColorCorrectionMatrix, read_raw_file
from pathlib import Path


# Path to an input raw image data.
INPUT_FILE = {
     "raw_image": Path(r"../../data/sample_image/AlphaISP_2592x1536_12bits_RGGB_Scene12.raw")
}

# Config dictionary
CONFIG = {
    "raw_image_shape": (1536, 2592),  # height x width
    "bayer_pattern": BayerPattern.RGGB,  # Pattern of a color filter array
    "auto_white_balance": {
        "method": AutoWhiteBalanceMethods.PCA,  # Currently, GRAYWORLD (reference) and PCA are available.
        "params": {
            "pca_pickup_percentage": 0.035  # Top and bottom n% will be considered. 3.5% is recommended.
        }
    },
    "black_level": 200,
    "white_level": 4095,
    "color_correction_matrix": {
        "method": ColorCorrectionMatrix.READ_MAT,
        "matfile_path": Path(r"../color_correction_matrix/ccm.mat")
    },
    "gamma": 2.2,  # 2.2
    "color_enhancement_coef": 2.0,

    # Set no_processing to be True (no gamma, CCM, or CE) when you want to get input data for CCM construction.
    "no_processing": False,
    "save_img": False,
    "verbose": False,
}


def run_scripts(input_file: dict, config: dict):

    raw_img = read_raw_file(input_file["raw_image"], config["raw_image_shape"])
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
