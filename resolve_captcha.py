from keras.models import load_model
from helpers import resize_to_fit
from imutils import paths
import numpy as np
import cv2
import pickle
from trata_captchas import tratar_imagens

def quebrar_captha():
    # importar o modelo que treinamos e também o tradutor (traduz as letras para numero para a IA entender)
    with open ("rotulos_modelo.dat", "rb") as arquivo_tradutor:
        lb = pickle.load(arquivo_tradutor)

    modelo = load_model("modelo_CAPTCHA_treinado.hdf5")

    # usar o modelo para resolver os CAPTCHAS:
    # trata as imagens originais
    tratar_imagens("captchas_resolver", pasta_destino="captchas_resolver")

    # ler os arquivos da pasta "captchas_resolver"
    ### codigo copiado do arquivo "separa_letras" ###
    arquivos = list(paths.list_images("captchas_resolver"))
    for arquivo in arquivos:
        imagem = cv2.imread(arquivo)
        # a leitura da imagem é feita em RGB, portanto precisamos tranformar novamente em escala de cinza
        imagem = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)
        # em preto e branco
        _, nova_imagem = cv2.threshold(imagem, 0, 255, cv2.THRESH_BINARY_INV)

        contornos, _ = cv2.findContours(nova_imagem, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)  # esse método irá separar as letras achando os contornos de cada área com muitos pixels. O EXTERNAL trabalha encontrando os pixels de fora pra dentro, ate encontrar a letra.

        # precisamos dizer que as letras sao apenas as areas de pixels com uma grande area, caso contrario, ele ira definir cada pixel minusculo na tela como sendo uma letra
        lista_regiao_letras = []

        for contorno in contornos:
            (x, y, largura, altura) = cv2.boundingRect(
                contorno)  # bounding rect nos da a posiçao de x, y, largura e altura do contorno realizado em forma de retangulo ao redor da letra
            area = cv2.contourArea(contorno)  # nos dá a area do contorno
            if area > 115:
                lista_regiao_letras.append((x, y, largura, altura))

        lista_regiao_letras = sorted(lista_regiao_letras, key=lambda x: x[0])   # ordenar as letras da forma que aparecem, da esquerda pra direita, ou seja, de acordo com o eixo x

        # desenhar os contornos e separar cada letra em um arquivo individual
        imagem_final = cv2.merge([imagem] * 3)  # tranformando a imagem novamente em RGB para salvar os arquivos
        previsao = []   # adicionara as letras conforme ele for identificando no captcha

        for retangulo in lista_regiao_letras:
            x, y, largura, altura = retangulo  # pegando as caracteristicas de cada retangulo de contorno
            imagem_letra = imagem[y - 2:y + altura + 2,
                           x - 2:x + largura + 2]  # basicamente, estamos pegando a distancia de uma diagonal a outra (do ponto superio esquerdo ao ponto inferior direito). Tiramos 2 das medidas iniciais e adicionamos 2 nas finais para darmos uma folga maior para o contorno.

            # dar a letra para a IA descobrir qual letra é essa
            imagem_letra = resize_to_fit(imagem_letra, 20, 20)  # ajeitar o tamanho da imagem

            # tratamento para o Keras funcionar (adição de uma 4ª dimensão à imagem através de índices
            imagem_letra = np.expand_dims(imagem_letra, axis=2)
            imagem_letra = np.expand_dims(imagem_letra, axis=0)

            letra_prevista = modelo.predict(imagem_letra)
            letra_prevista = lb.inverse_transform(letra_prevista)[0] # traduzindo novamente, mas dessa vez de números para letras (resposta da IA para humano)
            previsao.append(letra_prevista)

        texto_previsao = "".join(previsao)  # retorna as letras previstas (antes armazenadas numa lista) em texto

        print(texto_previsao)
        return texto_previsao # comentar caso tiver mais de um captcha para serem resolvidos
        ###

if __name__ == "__main__":
    quebrar_captha()