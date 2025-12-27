import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
import json, os

#Explicacao do modelo e da tarefa em doc.txt
#tamanho das imagens que o modelo aceita
IMG_SIZE = 224
#recebe 32 imagens de cada vez ao treinar antes de alterar peso(nao sei bem explicar mas a cada 32 imagens ele aplica o que aprendeu)
BATCH_SIZE = 32
# epochs e a quantidade de vezes o modelo recebe o dataset de treino. neste caso 5
#primeira faze de treino. Só a "cabeça" nova que vamos criar mais a frente, com o backbone congelado.
EPOCHS_HEAD = 5
#treinamos o backbone que veio do modelo + a cabeça nova, so para o afinar para a nova tarefa
EPOCHS_FINE = 10

#carregar dataset e guardar dados de treino e de validaçao. o dataset ja tem esses splits disponiveis. train para treino, validation para validar no final/testes.
def load_data():
    #baralha as imagens com shuffle_files, as_supervized Faz com que cada elemento do dataset seja um par (image, label) em vez de um dicionário com várias chaves ({"image": ..., "label": ...}).
    #with_info faz com que a função devolva também um objeto ds_info com informação do dataset (número de classes, nomes das classes, etc.). isto vai ser usado mais a frente.

    (ds_train, ds_val), ds_info = tfds.load(
        "food101",
        split=["train", "validation"],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    #estamos a ir busacr o numero de classses e as classes em si. NUM_CLASSES=101(fod101 duhh) e label_names = ["chicken", "apple pie"..., etc 
    # A ordem da lista e IMPORTANTISSIMA, porqie cada classe tem o seu indice especifico
    num_classes = ds_info.features["label"].num_classes
    label_names = ds_info.features["label"].names

    #aqui criamos uma pasta artifacts, que vai guardar as classes em json, para depois implementar no server e garantir que a ordem fica consistente. 
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/food101_labels.json", "w") as f:
        json.dump(label_names, f)

    return ds_train, ds_val, num_classes, label_names


#def para processar as imagens para um tamanho que o modelo aceite: B0: 224x224 px https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/
#devolve a imagem formatada e a sua label
def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    #converte a img para float32, normaliza e reescala para a config esperada pelo modelo 
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    return image, label


#def para randomizar ainda mais as imagens. mudar a luminosidade, o contraste e virar imagens para naoo ficar sensivel a horizontalidade e a luminosidade. 
#devolve a imagem editada e a sua label 
def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.1)
    image = tf.image.random_contrast(image, 0.9, 1.1)
    return image, label


def build_datasets():
    #aqui aplicamos as funcoes que criamos atras no dataset e dividimos em treino e validação :) 
    # o num_parallel_calls=tf.data.AUTOTUNE deixa o TensorFlow decidir quantos workers usar para paralelizar este map, aproveitando CPU
    ds_train, ds_val, num_classes, label_names = load_data()

    train_ds = ds_train.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

    #aqui criamos batches de 32 imagens, como explicado acima. o prefetch faz com que o pipeline prepare o próximo batch enquanto a GPU está a treinar no batch atual.
    #shuffle mantém um buffer de 1000 elementos e baralha a ordem em que são consumidos
    train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    #aqui nao usamos augment por ser de validacao. o resto e o mesmo
    val_ds = ds_val.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, num_classes, label_names
