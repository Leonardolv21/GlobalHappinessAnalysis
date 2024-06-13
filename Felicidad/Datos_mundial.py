import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

df_mundial = pd.read_csv('C:/Users/Richard-P/Algoritmica-Avanzada/Felicidad/2015.csv');

columnas = {
    "Country": "País",
    "Region": "Región",
    "Happiness Rank": "Rango de Felicidad",
    "Happiness Score": "Puntuación de Felicidad",
    "Standard Error": "Error Estándar",
    "Economy (GDP per Capita)": "Economía (PIB per cápita)",
    "Family": "Familia",
    "Health (Life Expectancy)": "Salud (Esperanza de Vida)",
    "Freedom": "Libertad",
    "Trust (Government Corruption)": "Confianza (Corrupción Gubernamental)",
    "Generosity": "Generosidad",
    "Dystopia Residual": "Residual de Distopía"
}
df_renombrado = df_mundial.rename(columns=columnas)


# PREPARAR DATOS
# Seleccionar las características y la variable objetivo
# X = df_renombrado[['Economía (PIB per cápita)', 'Familia', 'Salud (Esperanza de Vida)', 'Libertad', 'Confianza (Corrupción Gubernamental)', 'Generosidad', 'Residual de Distopía']]
# y = df_renombrado['Puntuación de Felicidad']

# # Dividir los datos en conjuntos de entrenamiento y prueba
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Normalizar las características
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# Entrenar el modelo KNN
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# PREDICCIONES Y EVALUAR DATOS
# Realizar predicciones
# y_pred = knn.predict(X_test_scaled)

# # Evaluar el modelo
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print("Error Cuadrático Medio (MSE):", mse)
# print("Coeficiente de Determinación (R²):", r2)


# Validar que todas las columnas en el DataFrame tienen un mapeo en 'columnas'
# missing_columns = [col for col in df_mundial.columns if col not in columnas]
# if missing_columns:
#     print(f"Las siguientes columnas no tienen una traducción definida: {missing_columns}")
# else:
#     # Renombrar las columnas del DataFrame utilizando el mapeo
#     df_renombrado = df_mundial.rename(columns=columnas)

#     # Configurar pandas para mostrar todas las columnas
#     pd.set_option('display.max_columns', None)
    
#     # Imprimir el DataFrame con columnas traducidas
#     print(df_renombrado)
# print(df_mundial.head())
# print(df_renombrado.head())
# print(df_mundial.describe())
# # ------------------------------
# # CALCULA LA MATRIZ CON CORRELACIONES DE MAYOR A MENOR
# columnas_numericas = df_mundial.select_dtypes(include=['float', 'int'])

# # Calculamos la matriz de correlación
# correlaciones = columnas_numericas.corr()

# # Mostramos las correlaciones con la columna 'Puntuación de Felicidad'
# print(correlaciones['Happiness Score'].sort_values(ascending=False))
# # -------------------------------
# # calcula y muestra el promedio de las puntuaciones de felicidad para cada región en el conjunto de datos
# # obtener una visión general del nivel promedio de felicidad en cada región
# # import matplotlib.pyplot as plt

# # # Asegúrate de que el DataFrame esté ordenado o agregado correctamente si es necesario
# # df_mundial.groupby('Region')['Happiness Score'].mean().plot(kind='bar', figsize=(15, 5))

# # plt.title('Puntuación de Felicidad por Región')
# # plt.xlabel('Región')
# # plt.ylabel('Puntuación de Felicidad')
# # plt.show()
# # --------------------------------
# # ANALISIS DE CUAL PAIS ES MENOS O MAS FELICES
# paises_mas_felices = df_mundial.nlargest(10, 'Happiness Score')
# paises_menos_felices = df_mundial.nsmallest(10, 'Happiness Score')

# print("Países más felices:")
# print(paises_mas_felices[['Region', 'Happiness Score']])

# print("\nPaíses menos felices:")
# print(paises_menos_felices[['Region', 'Happiness Score']])
