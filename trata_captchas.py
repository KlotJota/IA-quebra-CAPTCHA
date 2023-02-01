import cv2
import os
import glob
from PIL import Image

# nesse arquivo, faremos a leitura de todas as imagens na pasta bdcaptcha

def tratar_imagens(pasta_origem, pasta_destino='finalizadas'):  # demos um valor padrão ao parâmetro "pasta_destino"
    arquivos = glob.glob(f"{pasta_origem}/*")   # glob retorna todos os arquivos do caminho informado
    for arquivo in arquivos:
        # mesmo codigo do arquivo "teste_metodo"
        imagem = cv2.imread(arquivo)    # caminho da imagem escolhida

        # transformar a imagem em uma escala de cinza
        imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)

        _, imagem_tratada = cv2.threshold(imagem_cinza, 127, 255, cv2.THRESH_TRUNC or cv2.THRESH_OTSU)
        nome_imagem = os.path.basename(arquivo) # pega o nome original do arquivo (telanovaX), ja que iremos manter esse mesmo nome na imagem finalizada
        cv2.imwrite(f'{pasta_destino}/{nome_imagem}', imagem_tratada)

    arquivos = glob.glob(f"{pasta_destino}/*")    # ele olhará a pasta_destino agora, no caso a "finalizadas"
    for arquivo in arquivos:
        imagem = Image.open(arquivo)  # pegamos a imagem que ficou melhor tratada, no caso a 3
        imagem = imagem.convert("L")
        imagem2 = Image.new("L", imagem.size, 255)  # criamos uma imagem com o fundo branco

        # trataremos a imagem novamente, onde, caso o cinza for mais escuro, esse pixel ficará preto, e onde for mais claro, se tornará branco

        for x in range(imagem.size[1]):  # para cada pixel na largura da imagem
            for y in range(imagem.size[0]):  # para cada pixel na altura da imagem
                cor_pixel = imagem.getpixel((y, x))
                if cor_pixel < 115:
                    imagem2.putpixel((y, x), 0)  # quando tivermos um pixel cinza escuro (mais escuros que 120), substituiremos esse mesmo pixel por um da cor preta

        nome_imagem = os.path.basename(arquivo)
        imagem2.save(f'{pasta_destino}/{nome_imagem}')

if __name__ == "__main__":
    tratar_imagens('bdcaptcha') # parâmetro "pasta_origem" (no caso é a bdcaptcha)

