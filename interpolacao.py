import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from numpy import uint8


def imprime_imagem(titulo, imagem):
    # Monta a imagem novamente e imprime
    cv.imshow(titulo, imagem)
    cv.waitKey(0)
    cv.destroyAllWindows()


def cria_nova_imagem(altura, largura):
    # Cria uma nova imagem vazia com base nas dimensões passadas
    return np.zeros((altura, largura, 3), uint8)


def vizinho_reducao(imagem):
    # Reduz a imagem usando interpolação por vizinho mais próximo
    nova_imagem = cria_nova_imagem(int(imagem.shape[0] / 2), int(imagem.shape[1] / 2))

    for i in range(nova_imagem.shape[0]):
        for j in range(nova_imagem.shape[1]):
            nova_imagem[i, j] = imagem[i + i, j + j]

    return nova_imagem


def vizinho_ampliacao(imagem):
    # Amplia a imagem usando a interpolação por vizinho mais próximo
    nova_imagem = cria_nova_imagem(imagem.shape[0] * 2, imagem.shape[1] * 2)

    for i in range(imagem.shape[0]):
        for j in range(imagem.shape[1]):
            nova_imagem[i + i, j + j] = imagem[i, j]

    for i in range(0, nova_imagem.shape[0], 2):
        for j in range(0, nova_imagem.shape[1], 2):
            nova_imagem[i, j + 1] = nova_imagem[i, j]
            nova_imagem[i + 1, j] = nova_imagem[i, j]
            nova_imagem[i + 1, j + 1] = nova_imagem[i, j]

    return nova_imagem


def bicubica_reducao(imagem):
    # Reduz a imagem usando a interpolação bicubica
    nova_imagem = cria_nova_imagem(int(imagem.shape[0] / 2), int(imagem.shape[1] / 2))

    for i in range(nova_imagem.shape[0]):
        for j in range(nova_imagem.shape[1]):
            if j == nova_imagem.shape[1] - 1 and i != nova_imagem.shape[0] - 1:
                # Borda direita
                valor = np.int_(imagem[i + i, j + j]) + np.int_(imagem[i + i, j + j + 1]) + np.int_(
                    imagem[i + i, j + j + 1])
                valor += np.int_(imagem[i + i + 1, j + j]) + np.int_(imagem[i + i + 1, j + j + 1]) + np.int_(
                    imagem[i + i + 1, j + j + 1])
                valor += np.int_(imagem[i + i + 2, j + j]) + np.int_(imagem[i + i + 2, j + j + 1]) + np.int_(
                    imagem[i + i + 2, j + j + 1])
                valor = np.true_divide(valor, 9)
            elif i == nova_imagem.shape[0] - 1 and j != nova_imagem.shape[1] - 1:
                # Borda inferior
                valor = np.int_(imagem[i + i, j + j]) + np.int_(imagem[i + i, j + j + 1]) + np.int_(
                    imagem[i + i, j + j + 2])

                valor += np.int_(imagem[i + i + 1, j + j]) + np.int_(imagem[i + i + 1, j + j + 1]) + np.int_(
                    imagem[i + i + 1, j + j + 2])
                valor += np.int_(imagem[i + i + 1, j + j]) + np.int_(imagem[i + i + 1, j + j + 1]) + np.int_(
                    imagem[i + i + 1, j + j + 2])
                valor = np.true_divide(valor, 9)
            elif i == nova_imagem.shape[0] - 1 and j == nova_imagem.shape[1] - 1:
                # Canto inferior direito
                valor = np.int_(imagem[i + i, j + j]) + np.int_(imagem[i + i, j + j + 1]) + np.int_(
                    imagem[i + i, j + j + 1])
                valor += np.int_(imagem[i + i + 1, j + j]) + np.int_(imagem[i + i + 1, j + j + 1]) + np.int_(
                    imagem[i + i + 1, j + j + 1])
                valor += np.int_(imagem[i + i + 1, j + j]) + np.int_(imagem[i + i + 1, j + j + 1]) + np.int_(
                    imagem[i + i + 1, j + j + 1])
                valor = np.true_divide(valor, 9)
            else:
                # Todo o resto
                valor = np.int_(imagem[i + i, j + j]) + np.int_(imagem[i + i, j + j + 1]) + np.int_(
                    imagem[i + i, j + j + 2])

                valor += np.int_(imagem[i + i + 1, j + j]) + np.int_(imagem[i + i + 1, j + j + 1]) + np.int_(
                    imagem[i + i + 1, j + j + 2])
                valor += np.int_(imagem[i + i + 2, j + j]) + np.int_(imagem[i + i + 2, j + j + 1]) + np.int_(
                    imagem[i + i + 2, j + j + 2])
                valor = np.true_divide(valor, 9)

            nova_imagem[i, j] = valor

    return nova_imagem


def bicubico_ampliacao(imagem):
    # Amplia a imagem usando a interpolação bicubica
    nova_imagem = cria_nova_imagem(imagem.shape[0] * 2, imagem.shape[1] * 2)

    for i in range(imagem.shape[0]):
        for j in range(imagem.shape[1]):
            nova_imagem[i + i, j + j] = imagem[i, j]

    return nova_imagem
