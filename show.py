import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# carregar os dados do K-Means (output.csv) com as colunas normalizadas
df = pd.read_csv('output.csv', header=4, names=['longitude', 'latitude', 'median_house_value', 'cluster'])

# criar um gráfico 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# plotar os pontos com diferentes cores por cluster
scatter = ax.scatter(df['longitude'], df['latitude'], df['median_house_value'], c=df['cluster'], cmap='viridis')

# adicionar título e rótulos aos eixos
ax.set_title('K-Means Clustering em 3D')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_zlabel('Median House Value')

# adicionar barra de cores
cbar = fig.colorbar(scatter)
cbar.set_label('Cluster')

# exibir o gráfico
plt.show()
