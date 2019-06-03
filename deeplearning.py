import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


mnist = tf.keras.datasets.mnist #dataset usado
(treino_x, treino_y),(teste_x, teste_y) = mnist.load_data()  # associa as imgs para treino_x/teste_x e os rotulos para treino_y/teste_y

treino_x = tf.keras.utils.normalize(treino_x, axis=1)  # normaliza a data entre 0 e 1
teste_x = tf.keras.utils.normalize(teste_x, axis=1)  # normaliza a data entre 0 e 1

modelo = tf.keras.models.Sequential()  # modelo basico feedfoward
modelo.add(tf.keras.layers.Flatten())  # converte as imagens de 28x28 para 1x784
modelo.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # camada totalmente conectada com 128 unidades, e funcao de ativacao relu
modelo.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # camada totalmente conectada com 128 unidades, e funcao de ativacao relu
modelo.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))  # camada de saida de dados com 10 unidades para 10 classes, utilizando
                                                                # softmax para distribuicao de probabilidade

modelo.compile(optimizer='adam',  # otimizador adam
              loss='sparse_categorical_crossentropy',  # como o erro sera calculado, uma RNA tenta minimizar perda
              metrics=['accuracy'])  # o que manter sobre os dados

modelo.fit(treino_x, treino_y, epochs=3)  # treinamento do modelo

perda_val, precisao_val = modelo.evaluate(teste_x, teste_y)  # avalia a data fora da amostra com o modelo
print(perda_val)  # perda do modelo (erro)
print(precisao_val)  # precisao do modelo

modelo.save('modelo.model') #salva em arquivo
novo_modelo = tf.keras.models.load_model('modelo.model') #carrega o arquivo

previsao = novo_modelo.predict(teste_x)
print(np.argmax(previsao[1]))

plt.imshow(teste_x[1],cmap=plt.cm.binary)
plt.show()