import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from PIL import Image

"""
References
- https://www.kaggle.com/datasets/tenxengineers/auto-white-balance-awb
- https://www.flir.com/support-center/iis/machine-vision/application-note/using-color-correction/ 
- https://www.mathworks.com/help/images/ref/esfrchart.measurecolor.html 
"""

class BayerPattern(Enum):
    BGGR = "bggr"
    RGBG = "rgbg"
    GRBG = "grbg"
    RGGB = "rggb"

INPUT_FILE = {
    "raw_image": r"/Users/kotarosonoda/Documents/pythonProject/archive/AlphaISP - AWB Dataset/AlphaISP - AWB Dataset/RAW Data/AlphaISP_2592x1536_12bits_RGGB_Scene12.raw"
}

CONFIG = {
    "raw_image_shape": (1536, 2592),  # height x width
    "bayer_pattern": BayerPattern.BGGR,
    "black_level": 200,
    "white_level": 4095,
    "apply_color_correction_matrix": True,
    "gamma": 2.2,
    "color_enhancement": 1.3,  # 1.75
    "save_img": False
}

def raw_to_r_g_b_components(raw_img: np.ndarray, bayer_pattern: BayerPattern, black_level=200):
    if bayer_pattern == BayerPattern.BGGR:
        r = raw_img[::2, ::2] - black_level
        r[r < 0] = 0

        gr = raw_img[::2, 1::2] - black_level
        gr[gr < 0] = 0

        gb = raw_img[1::2, ::2] - black_level
        gb[gb < 0] = 0

        b = raw_img[1::2, 1::2] - black_level
        b[b < 0] = 0
    else:
        raise NotImplementedError("Only BGGR Bayer pattern is supported.")

    return r, gr, gb, b


def apply_gamma_corection(img, gamma=2.2, maximum_input_value=4095-200):
    coeff = (maximum_input_value)**(1/gamma)
    return np.round(np.clip(255/coeff * img**(1/gamma), 0, 255)).astype(np.uint8)


def main(input_file, config):

    raw_img = np.fromfile(input_file["raw_image"], dtype=np.uint16)
    raw_img = raw_img.reshape(config["raw_image_shape"])

    r, gr, gb, b = raw_to_r_g_b_components(raw_img=raw_img, bayer_pattern=BayerPattern.BGGR, black_level=config["black_level"])
    mean_g = np.mean([gr[gr!=config["white_level"]], gb[gb!=config["white_level"]]])
    mean_r = np.mean(r[r!=config["white_level"]])
    mean_b = np.mean(b[b!=config["white_level"]])

    wb_gain = np.array(
        [
            mean_g / mean_r,
            1.0,
            mean_g / mean_b
        ]
    )

    r = r * wb_gain[0]
    b = b * wb_gain[2]

    if config["apply_color_correction_matrix"]:
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


    R = r * cc_matrix[0,0] + gb * cc_matrix[0,1] + b * cc_matrix[0,2] + cc_offset[0]
    G = r * cc_matrix[1,0] + gb * cc_matrix[1,1] + b * cc_matrix[1,2] + cc_offset[1]
    B = r * cc_matrix[2,0] + gb * cc_matrix[2,1] + b * cc_matrix[2,2] + cc_offset[2]

    # Avoid saturation
    R[config["white_level"]-config["black_level"] < R] = config["white_level"]-config["black_level"]
    G[config["white_level"]-config["black_level"] < G] = config["white_level"]-config["black_level"]
    B[config["white_level"]-config["black_level"] < B] = config["white_level"]-config["black_level"]

    # Gamma
    R = apply_gamma_corection(R, gamma=config["gamma"], maximum_input_value=config["white_level"]-config["black_level"])
    G = apply_gamma_corection(G, gamma=config["gamma"], maximum_input_value=config["white_level"]-config["black_level"])
    B = apply_gamma_corection(B, gamma=config["gamma"], maximum_input_value=config["white_level"]-config["black_level"])

    # Chromatic enhancement
    g = config["color_enhancement"]
    mat1 = np.array([
        [1, 0, 1.402 * g],
        [1, -0.3441 * g, -0.7141 * g],
        [1, 1.772 * g, 0]
    ])
    mat2 = np.array([
        [0.299, 0.587, 0.114],
        [-0.169, -0.331, 0.5],
        [0.5, -0.419, -0.081]
    ])

    mat = np.matmul(mat1, mat2)
    Rp = R * mat[0,0] + G * mat[0,1] + B * mat[0,2]
    Gp = R * mat[1,0] + G * mat[1,1] + B * mat[1,2]
    Bp = R * mat[2,0] + G * mat[2,1] + B * mat[2,2]

    # Data format
    Rp = np.clip(Rp, 0, 255).astype(np.uint8)
    Gp = np.clip(Gp, 0, 255).astype(np.uint8)
    Bp = np.clip(Bp, 0, 255).astype(np.uint8)

    rgb = np.stack([Rp, Gp, Bp],axis=2)
    im = Image.fromarray(rgb)

    if config["save_img"]:
        im.save("processed_rgb.png")

    plt.figure()
    plt.imshow(rgb)
    plt.show()
    plt.show()


if __name__ == '__main__':
    main(input_file=INPUT_FILE, config=CONFIG)
