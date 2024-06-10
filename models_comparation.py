import numpy as np 
import pandas as pd 
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from joblib import dump
import pathlib
from joblib import load
from math import sqrt


dataset = pd.read_csv("data/weatherhistoryszeged_80.csv")
print(dataset["Precip_Type"].mode()[0])

dataset["Precip_Type"] = dataset["Precip_Type"].fillna(dataset["Precip_Type"].mode()[0]) 
dataset["Date"] = pd.to_datetime(dataset["Date"], format = "%Y-%m-%d %H:%M:%S.%f %z") 
#print(dataset.head())

{column: len(dataset[column].unique()) for column in dataset.columns}


dataset = dataset.drop(["Loud_Cover","Daily_Summary"], axis=1) 
#dataset.corr() 
dataset = dataset.drop(["Apparent_Temperature"], axis=1) 

{column: len(dataset) for column in dataset.columns}
len((dataset.columns))

X = dataset

{column: len(X[column].unique()) for column in X.columns}

dataset["year"] = dataset["Date"].apply(lambda x: x.year)
dataset["month"] = dataset["Date"].apply(lambda x: x.month)
dataset["day"] = dataset["Date"].apply(lambda x: x.day)

dataset = dataset.drop(["Date"], axis=1)

le = preprocessing.LabelEncoder()
le.fit(dataset["Summary"])

list(le.classes_)

dataset["Summary"] = le.transform(dataset["Summary"])
#print(dataset.head())

le.fit(dataset["Precip_Type"])
dataset["Precip_Type"] = le.transform(dataset["Precip_Type"])

#print(dataset.info())

y = dataset["Temperature"]
X = dataset.drop(["Temperature"], axis = 1)

sc = StandardScaler()
sc.fit(X)
X = pd.DataFrame(sc.transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)

models = { 
                "Linear regression": LinearRegression(),
                 "Ridge regression": Ridge(),
                 "Lasso regression": Lasso(),
           "Elastic Net regression": ElasticNet(),
   "K-nearest Neighbors regression": KNeighborsRegressor(),
         "Decision Tree regression": DecisionTreeRegressor(),
'Support Vector Machine regression': SVR(),
         "Random Forest Regression": RandomForestRegressor()
}

for name, model in models.items():
    y_pred = model.fit(X_train, y_train)
    print(name + " Trained")
    dump(y_pred, pathlib.Path(f'models/{name}.joblib'))


#carpeta_modelos = pathlib.Path('models')
#archivos_modelos = carpeta_modelos.glob('*.joblib')
#
##models = {}
#for archivo in archivos_modelos:
#    nombre_modelo = archivo.stem
#    models = load(archivo)
#    y_pred = models.predict(X_test)
#    #print(f"{nombre_modelo} R^2: {r2_score(y_test, y_pred):.8f}")
#    #print("adnoaso")
#    print(nombre_modelo + " R^2: {:.8f}".format(r2_score(y_test, y_pred)))
#    print(nombre_modelo  + " RMSE: {:.8f}".format(np.sqrt(mean_squared_error(y_test, y_pred))))