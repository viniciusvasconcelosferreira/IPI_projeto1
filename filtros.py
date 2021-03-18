from __future__ import print_function
from __future__ import division

import cv2 as cv
import numpy as np

"""
As quatro máscaras que serão utilizadas para o filtro Laplaciano:
    Centro negativo: remove bordas exteriores
    Centro positivo: remove bordas interiores
    Oito:
        Inclusão da leitura da diagonal da imagem
"""
quatro_negativo = [0, 1, 0], [1, -4, 1], [0, 1, 0]
quatro_positivo = [0, -1, 0], [-1, 4, -1], [0, -1, 0]
oito_negativo = [1, 1, 1], [1, -8, 1], [1, 1, 1]
oito_positivo = [-1, -1, -1], [-1, 8, -1], [-1, -1, -1]


def imprime_imagem(titulo, imagem):
    # Monta a imagem novamente e imprime
    cv.imshow(titulo, imagem)
    cv.waitKey(0)
    cv.destroyAllWindows()


def cria_nova_imagem(altura, largura):
    # Cria uma nova imagem vazia com base nas dimensões passadas
    imagem = np.empty((altura, largura, 3), np.uint8)
    return imagem


def calcula_valor(imagem, mascara, x, y):
    # Define os valores de pixel que serão utilizados no cálculo
    valores = []
    mi = -1

    for i in range(x - 1, x + 2):
        mi += 1
        mj = 0
        for j in range(y - 1, y + 2):
            if i < 0 or j < 0 or i >= imagem.shape[0] or j >= imagem.shape[1]:
                continue
            valores.append(imagem[i, j] * mascara[mi][mj])
            mj += 1

    valor = 0
    for i in range(len(valores)):
        valor += np.uint8(valores[i])
    # valor /= 9
    # Filtro se baseia numa matriz 3x3
    valor = np.true_divide(valor, 9)
    # Valor máximo da intensidade atingida por um pixel qualquer
    valor += 255
    if np.any(valor < 0):
        return 0
    return valor


def laplaciano(imagem, mascara):
    # Executa a filtragem laplaciana na imagem
    nova_imagem = cria_nova_imagem(imagem.shape[0], imagem.shape[1])

    for i in range(imagem.shape[0]):
        for j in range(imagem.shape[1]):
            nova_imagem[i, j] = calcula_valor(imagem, mascara, i, j)

    return nova_imagem


def gradiente(imagem):
    # Cálcula o aguçamento usando gradiente
    nova_imagem = cria_nova_imagem(imagem.shape[0], imagem.shape[1])

    for i in range(1, imagem.shape[0] - 2):
        for j in range(1, imagem.shape[1] - 2):
            gx = imagem[i + 1, j - 1] + 2 * imagem[i + 1, j] + imagem[i + 1, j + 1]
            gx -= imagem[i - 1, j - 1] + 2 * imagem[i - 1, j] + imagem[i - 1, j + 1]

            gy = imagem[i - 1, j + 1] + 2 * imagem[i, j + 1] + imagem[i + 1, j + 1]
            gy -= imagem[i - 1, j - 1] + 2 * imagem[i, j - 1] + imagem[i + 1, j - 1]

            nova_imagem[i, j] = gx

    return nova_imagem


def ajuste_gama(image, gama):
    # construir uma tabela de pesquisa mapeando os valores de pixel [0, 255] para
    # ajustar seus valores gama
    invGamma = 1.0 / gama
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # aplique a correção de gama usando a tabela de pesquisa
    return cv.LUT(image, table)
