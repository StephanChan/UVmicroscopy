# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 15:07:11 2024

@author: admin
"""

import numpy as  np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d as cov2

def gkern(l=5, sig=1.):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

size = 1024
image = np.zeros([size,size])

image[200:800,400:500]=1
image[400:500,200:800]=1

image = cov2(image,gkern(20,1),'same')
noise = np.random.rand(size,size)*0.2
image = image+noise

plt.figure()
plt.imshow(image)
plt.show()

import numpy as  np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d as cov2
def get_sharpness(image):
    kernel = np.zeros([5,1])
    kernel[0,:]=0.5
    kernel[2,:]=-0.5

    fst_drv = np.abs(cov2(image,kernel,'same'))
    # plt.figure()
    # plt.imshow(fst_drv)
    # plt.show()
    M=np.max(fst_drv)
    sharpness = np.mean(fst_drv[fst_drv>M/2])
    return sharpness

