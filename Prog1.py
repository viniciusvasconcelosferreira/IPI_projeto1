import interpolacao as funcao
import cv2 as cv
import filtros as filtro


def dec_int():
    print('Imagens disponiveis: car.png / crowd.png / test80.jpg / university.png')
    imagem_escolhida = input('Escolha uma das imagens disponiveis (digite o nome da imagem junto com a extensão): ')
    imagem = cv.imread('imagens/' + imagem_escolhida)
    funcao.imprime_imagem('Original', imagem)
    nova_imagem_reduzida = funcao.vizinho_reducao(imagem)
    funcao.imprime_imagem('Redução por vizinho mais próximo', nova_imagem_reduzida)
    nova_imagem_ampliada = funcao.vizinho_ampliacao(imagem)
    funcao.imprime_imagem('Ampliação por vizinho mais próximo', nova_imagem_ampliada)


def edge_improv():
    print('Aqui novamente')


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
