from __future__ import print_function
from __future__ import division

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def equalizar_histograma(imagem):
    img_to_yuv = cv.cvtColor(imagem, cv.COLOR_BGR2YUV)
    img_to_yuv[:, :, 0] = cv.equalizeHist(img_to_yuv[:, :, 0])
    hist_equalization_result = cv.cvtColor(img_to_yuv, cv.COLOR_YUV2BGR)
    cv.imshow('Original / Equalizada', np.hstack([imagem, hist_equalization_result]))
    cv.waitKey(0)


def gerar_histograma(imagem):
    img = cv.cvtColor(imagem, cv.COLOR_BGR2GRAY)
    h_eq = cv.equalizeHist(img)
    plt.figure()
    plt.title("Histograma Equalizado")
    plt.xlabel("Intensidade")
    plt.ylabel("Qtde de Pixels")
    plt.hist(h_eq.ravel(), 256, [0, 256])
    plt.xlim([0, 256])
    plt.show()
    plt.figure()
    plt.title("Histograma Original")
    plt.xlabel("Intensidade")
    plt.ylabel("Qtde de Pixels")
    plt.hist(img.ravel(), 256, [0, 256])
    plt.xlim([0, 256])
    plt.show()
    cv.waitKey(0)
