# thresholding
Create a binary mask by thresholding an image.

This routine uses two "high-pass" filters to threshold an image. The filters account for spatial variability in image intensity (signal). The results are a binary mask that can be further processed using the python-watershed routine.

![threshold](https://github.com/zbenson94/thresholding/blob/12bacfb0fe9fcf262eb1c17174884881f27f3d2d/threshold.png)


# Usage:

from thresholding import Threshold

import skimage.io as skio

######  # Common input parameters
thresh_parms = {
    
    'sigma':           8,
    
    'boxcar_size':     10
    
}

###### # Load binary image
img = skio.imread(path_to_image)


img_thresholded  = Threshold(img, **thresh_parms)





# Installation Instructions (if python is installed)

python -m venv venv

source venv/bin/activate

python -m pip install --upgrade pip

pip install -r requirements.txt



