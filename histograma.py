from __future__ import print_function
from __future__ import division

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def cdf(data):
    data_size = len(data)

    # Set bins edges
    data_set = sorted(set(data))
    bins = np.append(data_set, data_set[-1] + 1)

    # Use the histogram function to bin the data
    counts, bin_edges = np.histogram(data, bins=bins, density=False)

    counts = counts.astype(float) / data_size

    # Find the cdf
    cdf = np.cumsum(counts)

    # Plot the cdf
    plt.plot(bin_edges[0:-1], cdf, linestyle='--', marker="o", color='b')
    plt.ylim((0, 1))
    plt.ylabel("CDF")
    plt.grid(True)

    plt.show()


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

    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    plt.plot(cdf_normalized, color='r')
    plt.hist(img.flatten(), 256, [0, 256], color='b')
    plt.xlim([0, 256])
    plt.legend(('CDF', 'Histograma Original'), loc='upper left')
    plt.show()

    hist, bins = np.histogram(h_eq.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    plt.plot(cdf_normalized, color='r')
    plt.hist(h_eq.flatten(), 256, [0, 256], color='b')
    plt.xlim([0, 256])
    plt.legend(('CDF', 'Histograma Equalizado'), loc='upper left')
    plt.show()
