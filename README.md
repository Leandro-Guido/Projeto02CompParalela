# Projeto01CompParalela

O [dataset](https://www.kaggle.com/datasets/camnugent/california-housing-prices?resource=download) ([housing.csv](housing.csv)) contém informações sobre imóveis na Califórnia com base no censo de 1990. Inclui estatísticas como localização, idade das residências, número de cômodos, população, renda e valor mediano das casas. Os dados usados para o agrupamento foi latitude, longitude e o valor mediano das casas.

## Sequêncial
```
python preprocess.py
g++ -o kmeans.exe kmeans.cpp
./kmeans.exe
python show.py
```
