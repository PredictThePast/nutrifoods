from tensorflow import keras
import tensorflow as tf
import json
from pathlib import Path
from data_processing import build_datasets, IMG_SIZE, BATCH_SIZE
from model import build_model
import matplotlib.pyplot as plt


# epochs e a quantidade de vezes o modelo recebe o dataset de treino. neste caso 5
#primeira faze de treino. Só a "cabeça" nova que vamos criar mais a frente, com o backbone congelado.
EPOCHS_HEAD = 5
#treinamos o backbone que veio do modelo + a cabeça nova, so para o afinar para a nova tarefa
EPOCHS_FINE = 10

#importar os dados
train_ds, val_ds, num_classes, label_names = build_datasets()

model, base_model = build_model(num_classes=num_classes, img_size=IMG_SIZE)

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

#aqui treinamos a cabeca com o backbone :) 10 epochs. guardamos numa var para depois podermos ver os dados de treino em graficos mais abaixo
history_fine = model.fit(
    train_ds,
    epochs=EPOCHS_FINE,
    validation_data=val_ds,
)


# Avaliar no conjunto de validação
final_loss, final_acc = model.evaluate(val_ds)
final_acc_pct = final_acc * 100.0

summary = {
    "final_val_loss": float(final_loss),
    "final_val_accuracy": float(final_acc),
    "final_val_accuracy_pct": float(final_acc_pct),
}

Path("artifacts").mkdir(exist_ok=True)
with open("artifacts/training_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"Accuracy final na validação: {final_acc_pct:.2f}%")



#Graficos com 
loss = history_fine.history["loss"]
val_loss = history_fine.history["val_loss"]
acc = history_fine.history["accuracy"]
val_acc = history_fine.history["val_accuracy"]

epochs = range(1, len(loss) + 1)

# Plot de loss
plt.figure(figsize=(8, 4))
plt.plot(epochs, loss, label="Train loss")
plt.plot(epochs, val_loss, label="Val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("artifacts/loss_curve_fine.png")
plt.close()

# Plot de accuracy
plt.figure(figsize=(8, 4))
plt.plot(epochs, acc, label="Train accuracy")
plt.plot(epochs, val_acc, label="Val accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("artifacts/accuracy_curve_fine.png")
plt.close()