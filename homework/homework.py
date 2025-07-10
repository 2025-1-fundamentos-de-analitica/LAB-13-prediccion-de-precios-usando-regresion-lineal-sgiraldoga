#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error
import pickle
import zipfile
import gzip
import json
import os
import pandas as pd


def clean_data(df):
    df = df.copy()
    df['Age'] = 2021 - df['Year']
    df = df.drop(columns=['Year', 'Car_Name'])
    df = df.dropna()

    return df

def model():

    categories = ["Fuel_Type", "Selling_type", "Transmission"]  
    numerics = [
        "Selling_Price", "Driven_kms", "Age", "Owner"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categories),
            ('scaler', MinMaxScaler(), numerics)
        ],
        remainder='passthrough'
    )

    selectkbest = SelectKBest(score_func=f_regression)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ("selectkbest", selectkbest),
        ('classifier', LinearRegression())
    ])

    return pipeline

def hyperparameters(model, n_splits, x_train, y_train, scoring):
    estimator = GridSearchCV(
        estimator=model,
        param_grid = {
            "selectkbest__k": range(1, 13),
        },
        cv=n_splits,
        refit=True,
        scoring=scoring
    )
    estimator.fit(x_train, y_train)

    return estimator

def metrics(model, x_train, y_train, x_test, y_test):

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    train_metrics = {
        'type': 'metrics',
        'dataset': 'train',
        'r2': r2_score(y_train, y_train_pred),
        'mse': mean_squared_error(y_train, y_train_pred),
        'mad': median_absolute_error(y_train, y_train_pred)
    }

    test_metrics = {
        'type': 'metrics',
        'dataset': 'test',
        'r2': r2_score(y_test, y_test_pred),
        'mse': mean_squared_error(y_test, y_test_pred),
        'mad': median_absolute_error(y_test, y_test_pred)
    }

    return train_metrics, test_metrics

def save_model(model):
    os.makedirs('files/models', exist_ok=True)

    with gzip.open('files/models/model.pkl.gz', 'wb') as f:
        pickle.dump(model, f)

def save_metrics(metrics):
    os.makedirs('files/output', exist_ok=True)

    with open("files/output/metrics.json", "w") as f:
        for metric in metrics:
            json_line = json.dumps(metric)
            f.write(json_line + "\n")


with zipfile.ZipFile('files/input/test_data.csv.zip', 'r') as zip:
    with zip.open('test_data.csv') as f:
        df_Test = pd.read_csv(f)

with zipfile.ZipFile('files/input/train_data.csv.zip', 'r') as zip:
    with zip.open('train_data.csv') as f:
        df_Train = pd.read_csv(f)


if __name__ == '__main__':
    df_Test = clean_data(df_Test)
    df_Train = clean_data(df_Train)

    x_train, y_train = df_Train.drop('Present_Price', axis=1), df_Train['Present_Price']
    x_test, y_test = df_Test.drop('Present_Price', axis=1), df_Test['Present_Price']

    model_pipeline = model()

    model_pipeline = hyperparameters(model_pipeline, 10, x_train, y_train, 'neg_mean_absolute_error')

    save_model(model_pipeline)

    train_metrics, test_metrics = metrics(model_pipeline, x_train, y_train, x_test, y_test)

    save_metrics([train_metrics, test_metrics])