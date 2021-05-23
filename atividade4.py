import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

LABELS = {
    'glass': {
        1: 'building_windows_float_processed',
        2: 'building_windows_non_float_processed',
        3: 'vehicle_windows_float_processed',
        5: 'containers',
        6: 'tableware',
        7: 'headlamps'
    },
    'transfusion': {
        0: 'nao',
        1: 'sim'
    }
}


# Funcao de ativacao -> sigmoid
def activation(net: np.array) -> np.array:
    return 1 / 1 + np.exp(-net)


# Derivada da funcao de ativacao
def derivative_activation(f_net: np.array) -> np.array:
    return f_net * (1 - f_net)


# Construcao do modelo
def mlp_architeture(input_len=2, hidden_len=2, output_len=1) -> (np.array, np.array):
    seed = np.random.RandomState(12387890)

    # Constroi matrizes de valores aleatorios para os pesos seguindo uma distribuicao uniforme
    weights_hidden = seed.uniform(low=0, high=1, size=(input_len + 1, hidden_len))
    weights_output = seed.uniform(low=0, high=1, size=(hidden_len + 1, output_len))

    return weights_hidden, weights_output


def feedfoward(sample_row: np.array, weights_hidden: np.array, weights_output: np.array) \
        -> (np.array, np.array, np.array, np.array):

    resampled_row = np.reshape(sample_row, (1, sample_row.shape[0]))

    # Multiplica os valores das entradas pelos pesos da camada escondida, incluindo os BIAS
    hidden_net = np.matmul(np.append(resampled_row, 1), weights_hidden)
    f_hidden_net = activation(hidden_net)

    # Re-shaping arrays -> Para compatibilidade entre o R e o Python
    resampled_hidden_net = np.reshape(hidden_net, (1, hidden_net.shape[0]))
    resampled_f_hidden_net = np.reshape(f_hidden_net, (1, f_hidden_net.shape[0]))

    # Multiplica os f_net da hidden pelos pelos da camada de saida, incluindo os BIAS
    output_net = np.matmul(np.append(resampled_f_hidden_net, 1), weights_output)
    f_output_net = activation(output_net)

    # Re-shaping arrays
    resampled_output_net = np.reshape(output_net, (1, output_net.shape[0]))
    resampled_f_output_net = np.reshape(f_output_net, (1, f_output_net.shape[0]))

    return resampled_hidden_net, resampled_f_hidden_net, resampled_output_net, resampled_f_output_net


def backpropagation(x: pd.DataFrame, y: np.array, learning_rate=0.001, epocas=100, threshold=1e-3) \
        -> (np.array, np.array):
    squared_error = 2 * threshold
    epochs = 0

    # Montar arquitetura da rede neural
    weights_hidden, weights_output = mlp_architeture(x.shape[1], 3, len(np.unique(y)))

    print(f"Initial weights: \n hidden \n {weights_hidden}, \n output \n {weights_output}")

    while squared_error > threshold and epochs < epocas:
        squared_error = 0

        for index, row in x.iterrows():

            expected_output = y[index]

            # Aplica o algoritmo feedfoward
            hidden_net, f_hidden_net, output_net, f_output_net = feedfoward(row.values, weights_hidden, weights_output)

            # Calcula o erro e o erro quadratico
            error = expected_output - f_output_net

            squared_error += sum(i ** 2 for i in error[0])

            # Atualizando pesos
            # Primeiro calculamos os valores de delta output e delta hidden
            # Depois fazemos a atualizacao de fato, com os valores finais dos deltas
            # Sao feitos passos intermediarios para compatibilidade entre R e Python

            delta_output = error * derivative_activation(f_output_net)

            weights_output_without_bias = weights_output[:weights_hidden.shape[1]]

            output_and_weights = np.matmul(delta_output, np.transpose(weights_output_without_bias))

            delta_hidden = derivative_activation(f_hidden_net) * output_and_weights

            f_hidden_net_with_one = np.append(f_hidden_net, 1)
            f_hidden_net_with_one = np.reshape(f_hidden_net_with_one, (1, len(f_hidden_net_with_one)))

            row_values_with_one = np.append(row.values, 1)
            row_values_with_one = np.reshape(row_values_with_one, (1, len(row_values_with_one)))

            # Atualizacao de fato dos pesos da rede neural
            weights_output += learning_rate * np.matmul(np.transpose(f_hidden_net_with_one), delta_output)
            weights_hidden += learning_rate * np.matmul(np.transpose(row_values_with_one), delta_hidden)

        squared_error /= x.shape[0]
        print(f"Época {epochs}: Mean squared error -> {squared_error}")
        epochs += 1

    return weights_hidden, weights_output


def load_dataset(dataset) -> (pd.DataFrame, np.array):
    x = pd.read_csv(f'./{dataset}/{dataset}.csv')
    y = x['label'].to_numpy()
    x = x.drop('label', axis=1)

    return x, y


# Essa funcao recupera os dados do treinamento realizado via backpropagation e constroi a rede neural para predizer
def feedfoward_train(sample_row: np.array, weights_hidden: np.array, weights_output: np.array) -> np.array:
    # Hidden
    hidden_net = np.matmul(np.append(sample_row, 1), weights_hidden)
    f_hidden_net = activation(hidden_net)

    # Output
    output_net = np.matmul(np.append(f_hidden_net, 1), weights_output)
    f_output_net = activation(output_net)

    return f_output_net


# Utiliza a rede neural e os pesos treinados para predizer sobre um valor nao visto
def predict(dataset: str, row: np.array, weights_hidden: np.array, weights_output: np.array) -> str:
    f_output_net = feedfoward_train(row, weights_hidden, weights_output)

    max_value_index = np.argmax(f_output_net)

    predict_value = list(LABELS[dataset].values())[max_value_index]

    return predict_value


def main():
    dataset = 'transfusion'
    learning_rate = 0.1
    epocas = 1000
    x, y = load_dataset(dataset)

    # Divide o dataset em treino e teste, com 0.3 dos dados para teste
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=707878)

    print(y_train)

    weights_hidden, weights_output = backpropagation(x_train.reset_index(), y_train,
                                                     learning_rate=learning_rate, epocas=epocas)

    print(f"Final weights: \n hidden \n {weights_hidden}, \n output \n {weights_output}")

    y_true, y_pred = [], []
    for indice, linha in x_test.reset_index().iterrows():
        true_label = y_test[indice]

        y_true.append(LABELS[dataset][true_label])
        y_pred.append(predict(dataset, linha, weights_hidden, weights_output))

    # Calcula a acuracia usando a funcao do SCIKIT-LEARN
    accuracy = accuracy_score(y_true, y_pred)

    print(f"Accuracy {accuracy}")


if __name__ == '__main__':
    main()
