import cv2
import os
import numpy as np
import pickle
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense
from helpers import resize_to_fit

dados = []
rotulos = []
pasta_base_letras = "base_letras"

imagens = paths.list_images(pasta_base_letras)  # faz uma lista com todas as imagens contidas nesse diretório e seus respectivos rotulos de qual letra se refere

for arquivo in imagens:
    rotulo = arquivo.split(os.path.sep)[-2] # se refere à contra barra. Com isso, temos uma lista com as 3 informaçoes da imagem, a pasta, a letra, e a imagem. Precisamo apenas do 2º item, a letra
    imagem = cv2.imread(arquivo)
    imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # padronizar o tamanho da imagem em 20px x 20px
    imagem = resize_to_fit(imagem, 20, 20)

    # keras precisa que a imagem tenha 3 dimensoes. Vamos transformar as nossas de 2D para 3D
    imagem = np.expand_dims(imagem, axis=2) # em uma imagem 2d: [valor, valor, ], estamos adicionando a dimensao de indice 2

    # adicionar à lista de dados e rótulos
    rotulos.append(rotulo)
    dados.append(imagem)

dados = np.array(dados, dtype="float") / 255    # vamos padronizar os valores para que, qualquer valor de 0 a 255 (escala de pixels) se transforme em algo na escala de 0 a 1
rotulos = np.array(rotulos)

# separação em dados de treino (75%) e dados de teste (25%). Iremos das os de treino para a IA aprender e depois os de teste para ver como ela se sai em sua aplicação.
(X_train, X_test, Y_train, Y_test) = train_test_split(dados, rotulos, test_size=0.25, random_state=0)    # x = dados / y = rotulos

# Converter usando OneHotEncoding (TRADUTOR; mais sobre no arquivo word)
lb = LabelBinarizer().fit(Y_train)  # passando os rotulos para ele
Y_train = lb.transform(Y_train)
Y_test = lb.transform(Y_test)

# salvar o LabelBinarizer em um arquivo utilizando o pickle (para podermos traduzir os valores para letras novamente)
with open('rotulos_modelo.dat', 'wb') as arquivo_pickle:
    pickle.dump(lb, arquivo_pickle)

# criação e treinamento da IA
modelo = Sequential()   # rede neural de varias camadas

# criar as camadas da rede neural
modelo.add(Conv2D(20, (5, 5), padding="same", input_shape=(20, 20, 1), activation="relu"))
modelo.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# criar a 2º camada
modelo.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
modelo.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# criar mais uma camada
modelo.add(Flatten())
modelo.add(Dense(500, activation="relu"))   # temos 500 nós na camada intermediária

# camada de saída
modelo.add(Dense(26, activation="softmax")) # 26 possibilidades de saída (26 letras do alfabeto)

# compilar todas as camadas
modelo.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# treinar a IA
modelo.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=26, epochs=10, verbose=1)  # passamos os dados de treino, e o validation nos diz como vai ser verificado se estra sendo um bom treino, nele passamos os dados de teste (os 25% de imagens que deixamos separados justamente para isso). Verbose e uma barra de progressao de treinamente e epochs é quantas iterações ele vai fazer sobre a base dados para treinar a IA.

# salvar o modelo em um arquivo
modelo.save("modelo_CAPTCHA_treinado.hdf5")
