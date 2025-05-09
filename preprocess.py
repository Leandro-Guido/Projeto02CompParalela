import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Definir os nomes das colunas conforme especificado
colunas = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
]

# Carregar o dataset
data = pd.read_csv('adults/adult.data', header=None, names=colunas, na_values=' ?')

print(f"Dataset original: {data.shape}")

data = data.dropna()  # Remover linhas com valores ausentes
print(f"Dataset após remoção de NaN: {data.shape}")

data = data.drop_duplicates()  # Remover duplicatas
print(f"Dataset após remoção de duplicatas: {data.shape}")

# Separar as variáveis numéricas e categóricas
numericas = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
categoricas = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'income']

# Inicializar o MinMaxScaler para normalizar as variáveis numéricas
scaler = MinMaxScaler()

# Normalizar as variáveis numéricas
data[numericas] = scaler.fit_transform(data[numericas])

# Inicializar o LabelEncoder para codificar as variáveis categóricas
encoder = LabelEncoder()

# Codificar as variáveis categóricas
for coluna in categoricas:
    data[coluna] = encoder.fit_transform(data[coluna].astype(str))

# Salvar o dataset pré-processado em um novo arquivo CSV
data.to_csv('adult.csv', header=False, index=False)
