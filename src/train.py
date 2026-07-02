from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import joblib
import os

def train_model(X_train, y_train, X_test, y_test, model_type='linear'):
    """
    Entrena un modelo de regresión y evalúa su desempeño.
    
    Args:
        X_train: Características de entrenamiento.
        y_train: Etiquetas de entrenamiento.
        X_test: Características de prueba.
        y_test: Etiquetas de prueba.
        model_type: Tipo de modelo a entrenar ('linear', 'ridge', 'lasso', 'elasticnet').
    
    Returns:
        model: Modelo entrenado.
        metrics: Diccionario con métricas de evaluación (RMSE, R2).
    """
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'ridge':
        model = RidgeCV(alphas=[0.1, 1.0, 10.0])
    elif model_type == 'lasso':
        model = LassoCV(alphas=[0.1, 1.0, 10.0])
    elif model_type == 'elasticnet':
        model = ElasticNetCV(alphas=[0.1, 1.0, 10.0], l1_ratio=[0.1, 0.5, 0.9])
    else:
        raise ValueError("Tipo de modelo no soportado. Use 'linear', 'ridge', 'lasso' o 'elasticnet'.")

    # Entrenamiento del modelo
    model.fit(X_train, y_train)

    # Predicciones
    y_pred = model.predict(X_test)

    # Evaluación del modelo
    rmse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metricas = {
        'RMSE': rmse,
        'R2': r2
    }

    return model, metricas

if __name__ == "__main__":
    X_train_path = 'C:\\Users\\phf14_dcq7b60\\Desktop\\Video_Games ML Project\\data\\processed\\vg_price_forecasting_definitive.csv\\X_train.csv'
    y_train_path = 'C:\\Users\\phf14_dcq7b60\\Desktop\\Video_Games ML Project\\data\\processed\\vg_price_forecasting_definitive.csv\\y_train.csv'
    X_test_path = 'C:\\Users\\phf14_dcq7b60\\Desktop\\Video_Games ML Project\\data\\processed\\vg_price_forecasting_definitive.csv\\X_test.csv'
    y_test_path = 'C:\\Users\\phf14_dcq7b60\\Desktop\\Video_Games ML Project\\data\\processed\\vg_price_forecasting_definitive.csv\\y_test.csv'
    directorio_modelos = 'C:\\Users\\phf14_dcq7b60\\Desktop\\Video_Games ML Project\\models'

    try:
        # 2. ¡LEER LOS DATOS CON PANDAS ANTES DE ENTRENAR!
        print("Cargando los datos...")
        X_train = pd.read_csv(X_train_path)
        X_test = pd.read_csv(X_test_path)
        
        # Usamos .values.ravel() en las 'y' para convertir la columna de Pandas 
        # en un array 1D simple, que es el formato que Scikit-Learn prefiere.
        y_train = pd.read_csv(y_train_path).values.ravel()
        y_test = pd.read_csv(y_test_path).values.ravel()

        # 3. Entrenar los modelos pasando los DataFrames (no las rutas)
        # Puedes iterar sobre una lista para entrenar y guardar todos de golpe
        tipos_modelos = ['linear', 'ridge', 'lasso', 'elasticnet']
        
        os.makedirs(directorio_modelos, exist_ok=True)

        for tipo in tipos_modelos:
            modelo_entrenado, metricas = train_model(X_train, y_train, X_test, y_test, model_type=tipo)
            
            # Guardamos cada modelo entrenado para usarlo después en predict.py
            ruta_modelo = os.path.join(directorio_modelos, f'modelo_{tipo}.joblib')
            joblib.dump(modelo_entrenado, ruta_modelo)
            
        print("Todos los modelos han sido entrenados y guardados con éxito.")

    except Exception as e:
        print(f"Ocurrió un error: {e}")