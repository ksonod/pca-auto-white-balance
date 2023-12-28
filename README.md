# About this Repository
![image info](./docs/images/rgb_image_data_processing.png)

This repository provides:
- Python version of the auto-white-balance (AWB) method based on [principle component analysis](https://en.wikipedia.org/wiki/Principal_component_analysis) (PCA) [[1](https://opg.optica.org/josaa/abstract.cfm?uri=josaa-31-5-1049)].
- Python RGB data processing pipeline to convert raw mosaiced images to standard RGB images (i.e., demosaicing, AWB, color correction matrix, gamma correction, and color enhancement).
- Matlab script to construct a color correction matrix (CCM) using [Macbeth chart](https://en.wikipedia.org/wiki/ColorChecker).  

*I am not the author of the aforementioned paper. I have built this tool only for learning purposes.


# How to Use
1. Open [main.py](https://github.com/ksonod/pca-auto-white-balance/blob/main/sample/rgb_processing/main.py) in `./sample/rgb_processing` and specify config dictionaries.
2. Run `./sample/rgb_processing/main.py`. A processed RGB image will be visualized.

If a color correction matrix is not available, [create_color_correction_matrix.m](https://github.com/ksonod/pca-auto-white-balance/blob/main/sample/color_correction_matrix/create_color_correction_matrix.m) in `./sample/color_correction_matrix` can be used to make an original one for a specific system. For this purpose, images containing a color checker are needed.

# Sample Data
Sample raw data in `./data/sample_image` is taken from [[2](https://www.kaggle.com/datasets/tenxengineers/auto-white-balance-awb)].

# References
[[1](https://opg.optica.org/josaa/abstract.cfm?uri=josaa-31-5-1049)] D. Cheng, D. K. Prasad, and M. S. Brown, Illuminant estimation for color consistency: why spatial domain methods work and the role of the color distribution, J. Opt. Soc. Am. A 31. 1049-1058, 2014  
[[2](https://www.kaggle.com/datasets/tenxengineers/auto-white-balance-awb)] 10xEngineers. (2023). Auto White Balance (AWB) Dataset. Kaggle.com. https://www.kaggle.com/datasets/tenxengineers/auto-white-balance-awb [Accessed]: Dec. 26, 2023
