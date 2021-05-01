"""
ATIVIDADE 2

1. Copiar o algoritmo KNN em R descrito nas vídeo aulas, incluir comentários e executar no conjunto de dados escolhido para o EP1 ou para um outro conjunto de dados do repositório "UCI Machine Learning Repository" que tenha como tarefa "classificação" e que o tipo de atributos seja inteiro ou real.

2. Escolher e implementar uma outra métrica de distância.

3. Adicionar um novo parámetro ao algoritmo para poder escolher entre a distância euclidiana e a outra distância implementada.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


def euclidianDistance(v1: np.array, v2: np.array):
	# Verificação de erro
	if v1.shape != v2.shape:
		raise IOError('array shapes are not the same')

	# Norma da diferença entre os dois vetores
	return np.linalg.norm(v1-v2)


def manhattanDistance(v1: np.array, v2: np.array):
	# Verificação de erro
	if v1.shape != v2.shape:
		raise IOError('array shapes are not the same')

	# Somatório do módulo da diferência dos valores na i-ésima posição dos vetores
	total = 0
	for index in range(len(v1)):
		total = total + abs(v1[index] - v2[index])

	return total


def cos_similarity(v1, v2):
	return 1 - (np.dot(v1, v2)/(np.linalg.norm(v1) * np.linalg.norm(v2)))


def knn(dataset: pd.DataFrame, query: np.array, k: int = 1, distance: str = 'eu'):
	y = dataset['label']
	X = dataset.drop('label', axis=1)

	#  Calcular as distâncias entre o exemplo novo query e cada exemplo do dataset X
	distances = []
	for index, row in X.iterrows():
		#  Armazenar em uma tupla o par ordenado (dist(query, Xi), label(Xi))
		if distance == 'eu':
			t = (euclidianDistance(row.values, query), y[index])
		elif distance == 'man':
			t = (manhattanDistance(row.values, query), y[index])
		elif distance == 'cos':
			t = (cos_similarity(row.values, query), y[index])
		else:
			raise IOError(f'No distance found for {distance}')

		distances.append(t)

	# Ordenar (crescente) a lista de tuplas criada anteriormente por meio das distâncias
	# Recuperar somente as K primeiras distâncias (as menores)
	sorted_by_distance = sorted(distances, key=lambda tup: tup[0])[0:k]

	# Recuperar somente os ids
	ids = [i[1] for i in sorted_by_distance]

	# Votação
	U = np.unique(y)
	votes = np.zeros(len(U))

	# Somar os votos para cada label
	for i in range(len(U)):
		votes[i] = sum(1 for j in ids if U[i] == j)

	# Retorno a classe mais votoado e o numero de votos
	return U[np.argmax(votes)], np.amax(votes).astype(int)


if __name__ == '__main__':
	data = pd.read_csv('./blood-transfusion/transfusion.csv')
	features = data.columns[0:len(data.columns)-1]

	# HISTOGRAMA
	data[features].hist(figsize=(10, 4))
	plt.show()

	# MATRIZ DE CORRELAÇÃO
	corr_matrix = data[features].corr()
	sns.heatmap(corr_matrix, cmap='YlGnBu', linewidths=.5, linecolor='white', xticklabels=['R', 'F', 'M', 'T'], yticklabels=['R', 'F', 'M', 'T'])

	# CORRELAÇÃO ENTRE FREQUENCIA E QUANTIDADE DOADA
	sns.pairplot(data[['Frequency', 'Quantity']])

	# RELAÇÃO ENTRE A FREQUENCIA DE DOAÇÃO E MESES DESDE A PRIMEIRA DOAÇÃO
	sns.lmplot('Frequency', 'MonthsSinceFirstDonation', data=data, hue='label', fit_reg=False)

	# RELAÇÃO ENTRE A FREQUENCIA DE DOAÇÃO E MESES DESDE A DOAÇÃO MAIS RECENTE
	sns.lmplot('Frequency', 'MonthsSinceLastDonation', data=data, hue='label', fit_reg=False)

	plt.show()

	# DIVIDIR CONJUNTO DE DADOS EM TREINO E TESTE
	data_train, data_test = train_test_split(data, test_size=0.2, random_state=707878)

	print(f'train data has shape {data_train.shape}')
	print(f'test data has shape {data_test.shape}')

	tp, tn, fp, fn = 0, 0, 0, 0
	accuracy_overtime = []

	for indice, linha in data_test.iterrows():
		true_label = linha['label']
		target = linha.drop('label')

		# VALIDAR ALGORITMO PARA CADA LINHA DO CONJUNTO DE TESTE
		classification, num_votes = knn(data_train, target.values, k=10, distance='man')
		print(f'Class {classification} with {num_votes} votes for {target.values}')
		print(f'Real class: {true_label}')

		# classe positiva = 1 (doou sangue), classe negativa = 0 (nao doou sangue)

		if true_label == classification == 1:
			tp = tp + 1
		elif true_label == classification == 0:
			tn = tn + 1
		elif true_label == 0 and classification == 1:
			fp = fp + 1
		elif true_label == 1 and classification == 0:
			fn = fn + 1

		# CALCULAR ACUŔACIA
		current_accuracy = (tp + tn)/(tp + tn + fp + fn)
		accuracy_overtime.append(current_accuracy)

	plt.plot(range(len(data_test)), accuracy_overtime)
	plt.xlabel('Num of tests')
	plt.ylabel('Accuracy')
	plt.title('Accuracy variation over time')
	plt.show()
