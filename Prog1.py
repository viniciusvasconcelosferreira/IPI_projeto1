import interpolacao as funcao
import cv2 as cv
import filtros as filtro
import numpy as np


def dec_int():
    print('Imagens disponiveis: car.png / crowd.png / test80.jpg / university.png')
    imagem_escolhida = input('Escolha uma das imagens disponiveis (digite o nome da imagem junto com a extensão): ')
    imagem = cv.imread('imagens/' + imagem_escolhida)
    funcao.imprime_imagem('Original', imagem)
    nova_imagem_reduzida = funcao.vizinho_reducao(imagem)
    funcao.imprime_imagem('Redução por vizinho mais próximo', nova_imagem_reduzida)
    nova_imagem_ampliada = funcao.vizinho_ampliacao(imagem)
    funcao.imprime_imagem('Ampliação por vizinho mais próximo', nova_imagem_ampliada)


def edge_improv_laplace():
    print('Imagens disponiveis: car.png / crowd.png / test80.jpg / university.png')
    imagem_escolhida = input('Escolha uma das imagens disponiveis (digite o nome da imagem junto com a extensão): ')
    imagem = cv.imread('imagens/' + imagem_escolhida)
    filtro.imprime_imagem('Original', imagem)
    nova_imagem_4_negativo = filtro.laplaciano(imagem, filtro.quatro_negativo)
    # print('Máscara com o 4 negativo no centro')
    # filtro.imprime_imagem('Máscara com o 4 negativo no centro', nova_imagem_4_negativo)

    nova_imagem_4_positivo = filtro.laplaciano(imagem, filtro.quatro_positivo)
    # print('Máscara com o 4 positivo no centro')
    # filtro.imprime_imagem('Máscara com o 4 positivo no centro', nova_imagem_4_positivo)

    nova_imagem_8_negativo = filtro.laplaciano(imagem, filtro.oito_negativo)
    # print('Máscara com o 8 negativo no centro')
    # filtro.imprime_imagem('Máscara com o 8 negativo no centro', nova_imagem_8_negativo)

    nova_imagem_8_positivo = filtro.laplaciano(imagem, filtro.oito_positivo)
    # print('Máscara com o 8 positivo no centro')
    # filtro.imprime_imagem('Máscara com o 8 positivo no centro', nova_imagem_8_positivo)

    nova_imagem_4_negativo_legenda = cv.putText(nova_imagem_4_negativo, 'Mascara 4 negativo', (10, 50),
                                                cv.FONT_HERSHEY_SIMPLEX, 1,
                                                (0, 0, 255), 2, cv.LINE_AA)
    nova_imagem_4_positivo_legenda = cv.putText(nova_imagem_4_positivo, 'Mascara 4 positivo', (10, 50),
                                                cv.FONT_HERSHEY_SIMPLEX, 1,
                                                (0, 0, 255), 2, cv.LINE_AA)
    nova_imagem_8_negativo_legenda = cv.putText(nova_imagem_8_negativo, 'Mascara 8 negativo', (10, 50),
                                                cv.FONT_HERSHEY_SIMPLEX, 1,
                                                (0, 0, 255), 2, cv.LINE_AA)
    nova_imagem_8_positivo_legenda = cv.putText(nova_imagem_8_positivo, 'Mascara 8 positivo', (10, 50),
                                                cv.FONT_HERSHEY_SIMPLEX, 1,
                                                (0, 0, 255), 2, cv.LINE_AA)

    filtro_quatro = np.hstack((nova_imagem_4_negativo_legenda, nova_imagem_4_positivo_legenda))
    filtro_oito = np.hstack((nova_imagem_8_negativo_legenda, nova_imagem_8_positivo_legenda))
    horizontal_imagens = np.vstack((filtro_quatro, filtro_oito))
    cv.imshow('Imagens Modificadas', horizontal_imagens)
    cv.waitKey(0)


def edge_improv_gradient():
    print('Imagens disponiveis: car.png / crowd.png / test80.jpg / university.png')
    imagem_escolhida = input('Escolha uma das imagens disponiveis (digite o nome da imagem junto com a extensão): ')
    imagem = cv.imread('imagens/' + imagem_escolhida)
    # filtro.imprime_imagem('Original', imagem)
    nova_imagem = filtro.gradiente(imagem)
    # filtro.imprime_imagem('Gradiente', nova_imagem)
    cv.imshow("Original / Gradiente", np.hstack([imagem, nova_imagem]))
    cv.waitKey(0)


def sub_menu():
    print('======== MENU DE FILTROS ========')
    print('1 - Laplaciano')
    print('2 - Gradiente')
    opcao = int(input('Escolha uma das opções acima: '))

    if opcao == 1:
        edge_improv_laplace()
    elif opcao == 2:
        edge_improv_gradient()
    else:
        print('Opção inválida!')


def edge_improv():
    sub_menu()


def menu():
    print('======== MENU DE OPÇÕES ========')
    print('========    QUESTÃO 1   ========')
    print('1 - Interpolação por vizinho mais próximo')
    print('2 - Filtro de aguçamento')
    print('3 - Interpolação cúbica')
    print('4 - Melhoramento da interpolação do vizinho mais próximo')
    opcao = int(input('Escolha uma das opções acima: '))

    if opcao == 1:
        dec_int()
    elif opcao == 2:
        edge_improv()
    elif opcao == 3:
        print('Opção em construção!')
    elif opcao == 4:
        print('Opção em construção!')
    else:
        print('Opção inválida!')


if __name__ == '__main__':
    menu()
