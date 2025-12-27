Este e o data model nutrifoods. O objetivo e recener um imagem de comida em classificar qual é, treinado com o dataset food101.

O modelo que vamos usar e o EfficentNetB0. Decidi usar o B0 porque o servidor tem graficos integrados e nao tem GPU dedicada. 
E um modelo pre treinado em 1000 classes diferentes. Por causa disso, vamos usar transfer learning e supervized learning.

Transfer learning e quando um modelo e treinado primeiro em uma tarefa, e depois e reutilizado para outra. Um exemplo seria um modelo que classifica entre caes e gatos.
Podemos depois treinar esse modelo para outra tarefa, como por exemplo, classificar a raça do animal. Aproveitamos caracteristicas do modelo e poder computacional.

Supervized training e quando um algoritmo aprende a partir de um conjunto de dados de treino rotulados.

Antes de usar tranfer learning, removi a "cabeça" do modelo(o que classifica por 1000 classes) e colocar uma que tem 101 saídas (para o dataset food101). 
Deixei o backbone, que deixa a rede extrair features úteis: bordas, texturas, formas, padrões visuais, e uma nova uma cabeça para a nova tarefa. 
Assim reutilizei as capacidades de analise de imagens existentes deste modelo.
