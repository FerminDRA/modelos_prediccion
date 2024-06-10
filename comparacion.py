import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pathlib
from joblib import dump, load
from sklearn import preprocessing


df = pd.read_csv('data/weatherhistoryszeged_20.csv')
df["Precip_Type"] = df["Precip_Type"].fillna(df["Precip_Type"].mode()[0])
le = preprocessing.LabelEncoder()
le.fit(df["Summary"])
df["Summary"] = le.transform(df["Summary"])
le.fit(df["Precip_Type"])
df["Precip_Type"] = le.transform(df["Precip_Type"])



#df['Date'] = pd.to_datetime(df['Date'])
df["Date"] = pd.to_datetime(df["Date"], format = "%Y-%m-%d %H:%M:%S.%f %z") 

df["year"] = df["Date"].apply(lambda x: x.year)
df["month"] = df["Date"].apply(lambda x: x.month)
df["day"] = df["Date"].apply(lambda x: x.day)


#df['month'] = df['Date'].dt.month


X = df[['month']]  # o puedes usar 'timestamp' si prefieres graficar por fechas
y = df['Temperature']

years = df["year"].unique()
#palette = sns.color_palette("husl", len(years))
#year_color_map = {year: palette[i] for i, year in enumerate(years)}
#colors = df["year"].map(year_color_map)

plt.figure(figsize=(25, 20))
#plt.scatter(X, y, label='Datos originales', alpha=0.5)
plt.scatter(X, y, label='Datos originales', alpha=0.5)
plt.xlabel('Fecha')
plt.ylabel('Temperatura')
plt.title('Diagrama de dispersión de la temperatura')

model = load(pathlib.Path('models/Linear regression.joblib'))
model2 = load(pathlib.Path('models/Random Forest Regression.joblib'))
model3 = load(pathlib.Path('models/Decision Tree regression.joblib'))
model4 = load(pathlib.Path('models/Ridge regression.joblib'))
model5 = load(pathlib.Path('models/Elastic Net regression.joblib'))

y = df["Temperature"]
X = df.drop(["Temperature","Apparent_Temperature","Daily_Summary","Date","Loud_Cover"], axis = 1)

y_pred = model.predict(X)

plt.scatter(X["month"], y_pred, color='green', label='Predicciones Regresión Lineal', alpha=0.5)

y_pred = model2.predict(X)

plt.scatter(X["month"], y_pred, color='purple', label='Predicciones Regresión Lineal', alpha=0.5)

y_pred = model3.predict(X)

plt.scatter(X["month"], y_pred, color='gray', label='Predicciones Regresión Lineal', alpha=0.5)

y_pred = model4.predict(X)

plt.scatter(X["month"], y_pred, color='black', label='Predicciones Regresión Lineal', alpha=0.5)

y_pred = model5.predict(X)

plt.scatter(X["month"], y_pred, color='red', label='Predicciones Regresión Lineal', alpha=0.5)
#plt.scatter(X, y_pred, color='green', label='Predicciones Random Forest', alpha=0.5)

model_names = ['Linear regression', 'Random Forest Regression', 'Decision Tree regression','Ridge regression','Elastic Net regression']
colors = ['green','purple','gray','black','red']

# Crear gráfico

# Agregar datos o gráficos aquí

# Crear leyenda personalizada
legend_elements = [plt.Line2D([0], [0], marker='o', color=color, label=name, markersize=10) for name, color in zip(model_names, colors)]
plt.legend(handles=legend_elements)
#handles = [plt.Line2D([0], [0], marker='o', color='w', markersize=10) for i in range(len(years))]
#plt.legend(handles, years, title="Año")

#handles = [plt.Line2D([0], [0], marker='o', color='w', markersize=10) for i in range(len(years))]
#plt.legend(handles, years, title="Año")
plt.savefig('test.png')
