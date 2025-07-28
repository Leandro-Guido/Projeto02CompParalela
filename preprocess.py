import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# definir os nomes das colunas conforme especificado
colunas = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
]

# carregar o dataset
data = pd.read_csv('adults/adult.data', header=None, names=colunas, na_values=' ?')
print(f"dataset original: {data.shape}")

# pre-processamento
data = data.dropna()
data = data.drop_duplicates()
print(f"dataset apos limpeza: {data.shape}")

# normalização e codificação
numericas = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
categoricas = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'income']

scaler = MinMaxScaler()
data[numericas] = scaler.fit_transform(data[numericas])

encoder = LabelEncoder()
for coluna in categoricas:
    data[coluna] = encoder.fit_transform(data[coluna].astype(str))

# função para salvar subconjuntos
def salvar_dataset(fracao, nome_saida):
    subset = data.sample(frac=fracao, random_state=42)
    subset.to_csv(nome_saida, header=False, index=False)
    print(f"{nome_saida}: {subset.shape}")

# salvar os diferentes datasets
salvar_dataset(1.0, 'adult_1.csv')   # 100%
salvar_dataset(0.5, 'adult_0.5.csv') # 50%
salvar_dataset(0.25, 'adult_0.25.csv') # 25%
salvar_dataset(0.125, 'adult_0.125.csv') # 12.5%

print("Todos os datasets foram salvos.")
