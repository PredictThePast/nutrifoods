from tensorflow import keras

# epochs e a quantidade de vezes o modelo recebe o dataset de treino. neste caso 5
#primeira faze de treino. Só a "cabeça" nova que vamos criar mais a frente, com o backbone congelado.
EPOCHS_HEAD = 5
#treinamos o backbone que veio do modelo + a cabeça nova, so para o afinar para a nova tarefa
EPOCHS_FINE = 10

model, base_model = build_model(num_classes=101, img_size=IMG_SIZE)

#aqui vamos compilar o modelo. 
#O otimizador é o algoritmo que atualiza os pesos para minimizar a loss.
#adam é um otimizador adaptativo: ajusta automaticamente o "passo" (learning rate efetivo) de cada peso com base nos gradientes que ve ao longo do treino, o que o torna estável e rápido para redes profundas.
# o conjunto de pesos codifica o "conhecimento" do modelo: 
# Que padroes de píxeis indicam um certo tipo de borda, textura, forma, etc. Como combinar essas features para chegar a classe correta.
#A learning rate é um número que controla o tamanho do passo que o modelo dá cada vez que atualiza os pesos durante o treino
#esre 1e-3 e exatamente a taxa de quanto os pesos sao ajustados em resposta ao erro que o modelo cometeu naquele batch

#a loss é a funçao que mede "quao errado" o modelo esta em cada prediction e é aquilo que o otimizador tenta minimizar.
#O modelo devolve um vetor com 101 probabilidades ao prever uma imagem, de soma 1. A
# usamos sparse_categorical_crossentropy:
# ela recebe o indice da classe verdadeira (ex. 7) e o vetor de probabilidades previsto, e calcula a cross-entropy olhando apenas para a probabilidade na classe correta.
#metricas servem para acompanhar o desempenho, mas nao sao diretamente otimizadas (quem manda é a loss).
#accuracy em classificação multiclasse é simplesmente a percentagem de exemplos em que a classe com maior probabilidade prevista pelo modelo coincide com a classe verdadeira.

#--------------------------------aqui compilamos a cabeca--------------------------------------
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

#aqui treinamos so a cabeca :) 5 epochs
model.fit(
    train_ds,
    epochs=EPOCHS_HEAD,
    validation_data=val_ds,
)

#"descongelamos" o backbone para fazermos fine-tuning
base_model.trainable = True

#--------------------------------aqui compilamos a cabeca e o backbone juntos--------------------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

#aqui treinamos a cabeca com o backbone :) 10 epochs
model.fit(
    train_ds,
    epochs=EPOCHS_FINE,
    validation_data=val_ds,
)