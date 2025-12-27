import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models

#Define a função build_model que recebe num_classes (101 para Food‑101) e img_size(224x224)
#e aqui onde vamos criar o nosso backbone
#layers.Input(...) cria o tensor de entrada do modelo com shape (224,224,3). 3 = canais de cor = RGB
def build_model(num_classes: int, img_size: int = 224):
    inputs = layers.Input(shape=(img_size, img_size, 3))

    #backbone do modelo. removemos a cabeca com include_top=False. weights carrega pesos pré-treinados de ImageNet, é aqui onde usamos o tranfer learning
    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3),
    )

    #congelamos o backbone para a primeira fase de treino
    base_model.trainable = False

    # -------------------------------aqui criamos a nova cabeça do modelo-------------------------------

    # inputs é o tensor da imagem com o tamanho (224, 224, 3).
    # O backbone (EfficientNet-B0 sem a cabeça original) passa a imagem por várias camadas Conv2D, pooling, etc.
    # Ou seja, o backbone devolve um tensor de "features", reduzindo a altura e a largura e aumentando o número de canais/features.
    # Por exemplo, algo como (7, 7, 1280): 7x7 de "mapa" para cada um dos 1280 canais.
    # Cada um desses canais é um "detetor" diferente de padrões (bordas, texturas, formas específicas, etc.).

    x = base_model(inputs, training=False)

    # O GlobalAveragePooling2D pega em cada canal (cada um dos 1280 mapas 7×7) e faz a média de todos os píxeis desse mapa.
    # Ficamos com 1 número por canal, ou seja, o tensor passa de (7, 7, 1280) para (1280,).
    x = layers.GlobalAveragePooling2D()(x)

    # apagar 30% das features no treino para evitar overfitting.
    x = layers.Dropout(0.3)(x)

    # converter o vetor (1280,) em probabilidades para cada classe
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    # Junta tudo num modelo Keras e devolve:
    # model: rede completa (backbone + head).
    # base_model: referência ao backbone para controlar trainable na fase de fine-tuning.
    model = models.Model(inputs, outputs)
    return model, base_model

