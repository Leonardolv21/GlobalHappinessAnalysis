import pandas as pd

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

# Validar que todas las columnas en el DataFrame tienen un mapeo en 'columnas'
missing_columns = [col for col in df_mundial.columns if col not in columnas]
if missing_columns:
    print(f"Las siguientes columnas no tienen una traducción definida: {missing_columns}")
else:
    # Renombrar las columnas del DataFrame utilizando el mapeo
    df_renombrado = df_mundial.rename(columns=columnas)

    # Configurar pandas para mostrar todas las columnas
    pd.set_option('display.max_columns', None)
    
    # Imprimir el DataFrame con columnas traducidas
    print(df_renombrado)
# print(df_mundial.head())
# print(df_renombrado.head())