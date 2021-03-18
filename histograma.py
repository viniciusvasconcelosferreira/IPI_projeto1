from __future__ import print_function
from __future__ import division

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def equalizar_histograma(imagem):
    # Equalização do histograma por meio da conversão da imagem para
    # o espaço de cor YUV (YUV é um sistema de codificação de cores
    # normalmente usado como parte de um pipeline de imagens coloridas)
    # e equalização do canal Y para converter para RGB

    # converte a imagem de um espaço de cor para outro
    img_to_yuv = cv.cvtColor(imagem, cv.COLOR_BGR2YUV)
    # método de equalização de histograma no canal Y
    img_to_yuv[:, :, 0] = cv.equalizeHist(img_to_yuv[:, :, 0])
    # converter o canal Y para RGB (BGR no OpenCV)
    hist_equalization_result = cv.cvtColor(img_to_yuv, cv.COLOR_YUV2BGR)
    cv.imshow('Original / Equalizada', np.hstack([imagem, hist_equalization_result]))
    cv.waitKey(0)


def gerar_histograma(imagem):
    # conversão para escala de cinza
    img = cv.cvtColor(imagem, cv.COLOR_BGR2GRAY)
    # equalização do histograma da imagem
    h_eq = cv.equalizeHist(img)
    # geração do histograma
    plt.figure()
    # Legendas do histograma
    plt.title("Histograma Equalizado")
    plt.xlabel("Intensidade")
    plt.ylabel("Qtde de Pixels")
    # criação do histograma a partir da matriz da imagem
    plt.hist(h_eq.ravel(), 256, [0, 256])
    # define o intervalo do eixo 'x'
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
    # Retorne uma cópia da matriz em uma dimensão.
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    # Retorna a soma cumulativa dos elementos
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    plt.plot(cdf_normalized, color='r')
    plt.hist(img.flatten(), 256, [0, 256], color='b')
    plt.xlim([0, 256])
    plt.legend(('CDF', 'Histograma Original'), loc='upper left')
    plt.xlabel("Intensidade")
    plt.ylabel("Qtde de Pixels")
    plt.show()

    hist, bins = np.histogram(h_eq.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    plt.plot(cdf_normalized, color='r')
    plt.hist(h_eq.flatten(), 256, [0, 256], color='b')
    plt.xlim([0, 256])
    plt.legend(('CDF', 'Histograma Equalizado'), loc='upper left')
    plt.xlabel("Intensidade")
    plt.ylabel("Qtde de Pixels")
    plt.show()
