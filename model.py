import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models

#Define a função build_model que recebe num_classes (101 para Food‑101) e img_size.
#layers.Input(...) cria o tensor de entrada do modelo com shape (224,224,3). 3 = canais de cor = RGB
def build_model(num_classes: int, img_size: int = 224):
    inputs = layers.Input(shape=(img_size, img_size, 3))

#backbone do modelo. removemos a cabeca com include_top=False. weights carrega pesos pré-treinados de ImageNet, é aqui onde usamos o tranfer learning
base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(img_size, img_size, 3),
)
