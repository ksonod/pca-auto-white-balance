import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from white_balance import AutoWhiteBalanceMethods
from raw_to_rgb import Raw2RGB, BayerPattern

"""
References
- https://www.kaggle.com/datasets/tenxengineers/auto-white-balance-awb
- https://www.flir.com/support-center/iis/machine-vision/application-note/using-color-correction/ 
- https://www.mathworks.com/help/images/ref/esfrchart.measurecolor.html 
"""


INPUT_FILE = {
     "raw_image": r"../../data/sample/AlphaISP_2592x1536_12bits_RGGB_Scene12.raw"
}

CONFIG = {
    "raw_image_shape": (1536, 2592),  # height x width
    "bayer_pattern": BayerPattern.BGGR,
    "auto_white_balance_method": AutoWhiteBalanceMethods.GRAYWORLD,
    "black_level": 200,
    "white_level": 4095,
    "use_color_correction_matrix": True,
    "gamma": 2.2,  # 2.2
    "color_enhancement_coef": 1.0,  # 1.75
    "save_img": False
}


def run_scripts(input_file, config):

    raw_img = np.fromfile(input_file["raw_image"], dtype=np.uint16)
    raw_img = raw_img.reshape(config["raw_image_shape"])
    raw_to_rgb = Raw2RGB(config)
    rgb = raw_to_rgb(raw_img)

    if config["save_img"]:
        im = Image.fromarray(rgb)
        im.save("processed_rgb.png")

    plt.figure()
    plt.imshow(rgb)
    plt.show()
    plt.show()


if __name__ == '__main__':
    run_scripts(input_file=INPUT_FILE, config=CONFIG)
