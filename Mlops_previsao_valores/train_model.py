import pandas as pd
from sklearn.linear_model import LinearRegression 
import pickle

# Carregar os dados

data = pd.read_csv('Mlops_previsao_valores/Data/dataset.csv')

# Separar os dados

x = data[['tamanho']] #Entrada
y = data['preco'] #Saída

# Treinar o modelo

model = LinearRegression()
model.fit(x,y)

# Salvar o modelo

with open('model.pkl', 'wb') as f:
    pickle.dump(model,f)

print('Modelo treinado e salvo com sucesso!')