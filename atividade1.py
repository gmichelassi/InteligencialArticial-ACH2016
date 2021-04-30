"""
1. Escolher um dos seguintes 3 conjuntos de dados, o qual será utilizado no EP1.
2. Fazer a leitura do conjunto de dados  escolhido usando  bibliotecas prontas ou funções das linguagens de programação Python ou R, por exemplo a biblioteca Pandas do Python.
3. Verificar se existem dados faltantes no conjunto de dados escolhido, caso sim, pesquise como lidar com eles.
4. Se o conjunto de dados escolhido é desbalanceado, aplique alguma técnica de rebalanceamento existente nas bibliotecas do Python ou R.
5. Aplicar alguma técnica de normalização nos dados.
6. Fazer uma análise exploratória do conjunto de dados escolhido (calcular médias, medianas, variâncias, min, max)
"""
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE


def load_dataset():
    X = pd.read_csv('./spambase/spambase.csv')
    y = X['label'].to_numpy()
    X = X.drop('label', axis=1)  # axis = 1 pra retirar coluna, axis = 0 pra linha

    return X, y


def normalizar(X):  # reescala linear
    df_final = []

    # iterar sobre cada coluna
    for (feature, data) in X.iteritems():
        column = []
        maxVal = data.max()
        minVal = data.min()

        diff = maxVal - minVal
        # iterar sobre cada valor em cada coluna
        for i in data:
            L = (i - minVal)/diff
            column.append(L)
        df_final.append(column)

    X_novo = pd.DataFrame(df_final).T
    X_novo.columns = X.columns
    return X_novo


def main():
    # Carregar dataset
    X, y = load_dataset()
    print(X.shape)

    print("----- Informações Gerais sobre o conjunto de dados original -----")
    print(X.info())

    print("----- Descrição detalhada sobre o conjunto de dados original -----")
    print(X.describe())

    print("----- Realizando o balanceamento de classes -----")
    X_novo, y_novo = SMOTE(random_state=787070).fit_resample(X, y)
    print(X_novo.shape)

    print("----- Realizando a reescala linear ------")
    X_normal = normalizar(X_novo)

    print("----- Informações Gerais sobre o conjunto de dados normalizado e balanceado -----")
    print(X_normal.info())

    print("----- Descrição detalhada sobre o conjunto de dados normalizado e balanceado -----")
    print(X_normal.describe())


if __name__ == '__main__':
    main()
