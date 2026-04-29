import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Cria os dados fictícios (Tamanho em m² e Preço em R$)
X = np.array([[40], [50], [60], [70], [80], [90], [100], [120]])
y = np.array([200000, 255000, 310000, 345000, 410000, 440000, 510000, 620000])

# Separa os dados em Treino e Teste
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Cria o Modelo
modelo = LinearRegression()

# Treina o Modelo
modelo.fit(X_treino, y_treino)

# Faz as previsões com a base de TESTE
previsoes = modelo.predict(X_teste)

# Calcula as métricas
mae = mean_absolute_error(y_teste, previsoes)
r2 = r2_score(y_teste, previsoes)

print(f"Erro Médio (MAE): R$ {mae:.2f}")
print(f"Acurácia do Modelo (R²): {r2:.2%}")

# Faz uma previsão
tamanho_novo = [[85]]
preco_previsto = modelo.predict(tamanho_novo)

print(f"Para um imóvel de 85m², a previsão de preço é: R$ {preco_previsto[0]:,.2f}")
