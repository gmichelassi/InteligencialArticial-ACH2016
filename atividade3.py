import pandas as pd
import numpy as np

DELTA = 0.1


# FUNCOES DE ATIVAÇÃO
def binaria_limiar(net, delta):
    if net >= delta:
        return 1
    else:
        return 0


def bipolar_limiar(net, delta):
    if net >= delta:
        return 1
    else:
        return -1


def treinamento(X: pd.DataFrame, y: np.array, learning_rate: float, max_it: int, thresholdError: float, activation):
    # Definir uma seed para o gerador de números aleatórios do numpy
    seed = np.random.RandomState(1234567890)
    # Inicializar o vetor w de pesos e o bias
    w = seed.rand(X.shape[1])
    bias = seed.rand()

    print(f'PESOS INICIAIS: {w}')
    print(f'BIAS INICIAL: {bias}')
    print('')
    t = 0
    somaErro = 0
    while t < max_it and somaErro <= thresholdError:
        somaErro = 0
        for index, row in X.iterrows():
            # Calcular o valor de net
            # Utilizamos a função sum() do python que soma os valores retornados por um iterador
            # Passamos como iterador um laço for, que pra cada valor i nos pesos w e pra cada linha j, multiplica i * j
            # Por fim somamos o bias
            net = sum(i * j for i, j in zip(w, row)) + bias

            # Descobrimos o valor predizido utilizando a função de ativação f
            y0 = activation(net, delta=DELTA)
            # Calculamos o erro e o erro²
            erro = y[index] - y0
            somaErro = somaErro + erro**2

            # Por fim atualizamos os valores dos pesos w
            w_att = []
            count = 0
            for wi in w:
                new_w = wi - learning_rate * erro * (-row[count])
                w_att.append(new_w)
                count = count + 1
            w = np.array(w_att)

            # E atualizamos o valor do bias
            bias = bias - learning_rate * erro * (-1)

        print(f'ÉPOCA {t}: PESOS:{w}')
        print(f'ÉPOCA {t}: BIAS: {bias}')
        print('')
        t = t + 1

    return w, bias


def teste(x, w, bias, activation):
    net = sum(i * j for i, j in zip(w, x)) + bias
    return activation(net, delta=DELTA)


if __name__ == '__main__':
    print('DATASET OR - TREINAMENTO E TESTE')
    X = pd.read_csv('./data_ativ3/dataset-or.csv')
    y = X['label']
    x = X.drop('label', axis=1)

    w, bias = treinamento(x, y, 0.01, 10, 10, binaria_limiar)

    exemplo_teste = [1, 0]
    label_teste = 1

    resp = teste(exemplo_teste, w, bias, binaria_limiar)
    print(f'O exemplo predito foi {resp} e o valor verdadeiro era {label_teste}')

    print('')

    print('DATASET CLARO/ESCURO - TREINAMENTO E TESTE')
    X = pd.read_csv('./data_ativ3/claro-escuro.csv')
    y = X['label']
    x = X.drop('label', axis=1)
    w, bias = treinamento(x, y, 0.01, 10, 10, bipolar_limiar)

    exemplo_teste = [-1, 1, -1, -1]
    label_teste = -1

    resp = teste(exemplo_teste, w, bias, bipolar_limiar)
    print(f'O exemplo predito foi {resp} e o valor verdadeiro era {label_teste}')
