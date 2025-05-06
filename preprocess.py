import pandas as pd

# Incluindo median_house_value na normalização
home_data = pd.read_csv('housing.csv', usecols=['longitude', 'latitude', 'median_house_value'])

pd.DataFrame(home_data, columns=['longitude', 'latitude', 'median_house_value']).to_csv('housing_pre.csv', index=False)
