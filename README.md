## NutriFoods – Data Model

Este é o _data model_ **NutriFoods**.  
O objetivo é receber uma imagem de comida e classificar que comida é, usando o _dataset_ **Food‑101**.

Este projeto foi feito em colaboração com **NutriFlex**.

---

## Modelo utilizado

O modelo utilizado é o **_EfficientNet‑B0_**.  
Foi escolhido o **_B0_** porque o servidor de produção tem gráficos integrados e não tem GPU dedicada, por isso é importante usar um modelo leve.  
O _EfficientNet‑B0_ vem pré‑treinado em 1000 classes diferentes (_ImageNet_), o que permite reutilizar esse conhecimento. Por causa disso, foi usado _transfer learning_ em cima de _supervised learning_.

---

## Conceitos principais

### _Transfer learning_

_Transfer learning_ é quando um modelo é treinado primeiro numa tarefa e depois é reutilizado noutra tarefa relacionada.  
Exemplo: um modelo treinado para distinguir cães de gatos pode ser reaproveitado para classificar a _raça_ do animal, aproveitando as _features_ já aprendidas e poupando tempo de treino.

Neste projeto:

- O _EfficientNet‑B0_ foi inicialmente treinado em _ImageNet_ (1000 classes genéricas).  
- Esse modelo é reutilizado como _backbone_ para classificação de comida no _Food‑101_.

### _Supervised learning_

_Supervised learning_ é quando o modelo aprende a partir de um conjunto de dados de treino _rotulado_, ou seja, cada imagem vem com a classe correta (_label_).  
No _Food‑101_, cada imagem vem com o nome do prato, e a rede aprende a mapear _imagem → classe de comida_.

---

## Adaptação do _EfficientNet‑B0_ ao Food‑101

Antes de aplicar _transfer learning_, foi feita a seguinte adaptação:

- A “_cabeça_” original do _EfficientNet‑B0_ (camada final com 1000 saídas) foi removida.  
- Foi adicionada uma nova _cabeça_ com **101 saídas**, correspondentes às classes do _dataset_ **Food‑101**.  
- O _backbone_ foi mantido, reaproveitando a capacidade de extrair _features_ úteis: bordas, texturas, formas e padrões visuais.

O treino é feito em duas fases:

1. Treinar apenas a nova _cabeça_, mantendo o _backbone_ congelado.  
2. Fazer _fine‑tuning_ de algumas camadas finais do _backbone_, para especializar melhor o modelo em comida.

Assim, reutilizam‑se as capacidades de análise de imagens já existentes no _EfficientNet‑B0_, reduzindo o custo de treino e melhorando a performance com menos dados.

---

## Avisos

Se estiver em Windows, é recomendado treinar o modelo usando WSL, pois o TensorFlow 2.11+ não tem suporte nativo para GPUs em Windows. Se for o caso, recomendo utilizar _pip install "tensorflow[and_cuda]"_.
