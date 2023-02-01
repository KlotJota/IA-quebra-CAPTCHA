import cv2
import os
import glob

arquivos = glob.glob('finalizadas/*')
for arquivo in arquivos:
    imagem = cv2.imread(arquivo)
    # a leitura da imagem é feita em RGB, portanto precisamos tranformar novamente em escala de cinza
    imagem = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)
    # em preto e branco
    _, nova_imagem = cv2.threshold(imagem, 0, 255, cv2.THRESH_BINARY_INV)

    contornos, _ = cv2.findContours(nova_imagem, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # esse método irá separar as letras achando os contornos de cada área com muitos pixels. O EXTERNAL trabalha encontrando os pixels de fora pra dentro, ate encontrar a letra.

    # precisamos dizer que as letras sao apenas as areas de pixels com uma grande area, caso contrario, ele ira definir cada pixel minusculo na tela como sendo uma letra
    lista_regiao_letras = []

    for contorno in contornos:
        (x, y, largura, altura) = cv2.boundingRect(contorno)   # bounding rect nos da a posiçao de x, y, largura e altura do contorno realizado em forma de retangulo ao redor da letra
        area = cv2.contourArea(contorno)   # nos dá a area do contorno
        if area > 115:
            lista_regiao_letras.append((x, y, largura, altura))

    # isso irá prevenir que o codigo encontre letra a mais ou a menos no captcha, pois caso isso aconteça poderá atrapalhar no aprendizado da IA.
    if len(lista_regiao_letras) != 5:
        continue

    # desenhar os contornos e separar cada letra em um arquivo individual
    imagem_final = cv2.merge([imagem] * 3)  # tranformando a imagem novamente em RGB para salvar os arquivos

    i = 0
    for retangulo in lista_regiao_letras:
        x, y, largura, altura = retangulo   # pegando as caracteristicas de cada retangulo de contorno
        imagem_letra = imagem[y-2:y+altura+2, x-2:x+largura+2]  # basicamente, estamos pegando a distancia de uma diagonal a outra (do ponto superio esquerdo ao ponto inferior direito). Tiramos 2 das medidas iniciais e adicionamos 2 nas finais para darmos uma folga maior para o contorno.
        i += 1
        nome_arquivo = os.path.basename(arquivo).replace(".png", f"letra{i}.png")   # substituimos o nome inicial ".png" por letraX
        cv2.imwrite(f'letras/{nome_arquivo}', imagem_letra)
        cv2.rectangle(imagem_final, (x-2, y-2), (x+largura + 2, y+altura + 2), (255, 0, 0), 1)
    nome_arquivo = os.path.basename(arquivo)
    cv2.imwrite(f'identificado/{nome_arquivo}', imagem_final)