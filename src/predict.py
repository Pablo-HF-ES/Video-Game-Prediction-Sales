import pandas as pd
import joblib
import os

def predict(nuevos_datos_path, directorio_modelos, output_path, tipo_modelo='ridge'):
    """
    Carga un modelo entrenado y su pipeline de preprocesamiento para 
    hacer predicciones sobre datos completamente nuevos.
    """
    print(f"Cargando nuevos datos desde: {nuevos_datos_path}")
    
    # 1. Leer los datos nuevos
    nuevos_datos = pd.read_csv(nuevos_datos_path)
    
    # IMPORTANTE: Hacemos la misma limpieza básica que hicimos en features.py
    nuevos_datos.fillna(value=0, inplace=True)
    nuevos_datos_numericos = nuevos_datos.select_dtypes(exclude=['object'])
    nuevos_datos_numericos = nuevos_datos_numericos.drop(columns=['launch_price_decay'], errors='ignore')  # Por si acaso
    
    # 2. Cargar los Artefactos (Pipeline y Modelo)
    ruta_pipeline = os.path.join(directorio_modelos, 'preprocessing_pipeline.joblib')
    ruta_modelo = os.path.join(directorio_modelos, f'modelo_{tipo_modelo}.joblib')
    
    print("Cargando pipeline y modelo entrenado...")
    pipeline = joblib.load(ruta_pipeline)
    modelo = joblib.load(ruta_modelo)
    
    # 3. Transformar los datos nuevos
    # ¡USAMOS SOLO .transform(), NUNCA .fit_transform() en predecir!
    X_nuevos_transformados = pipeline.transform(nuevos_datos_numericos)
    
    # 4. Hacer la predicción
    print("Calculando predicciones...")
    predicciones = modelo.predict(X_nuevos_transformados)
    
    # 5. Añadir las predicciones al DataFrame original para que sea legible
    nuevos_datos['prediccion_launch_price_decay'] = predicciones
    
    # 6. Guardar los resultados para el cliente / equipo de negocio
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nuevos_datos.to_csv(output_path, index=False)
    
    print(f"\n¡Éxito! Las predicciones se han guardado en: {output_path}")
    
    # Mostramos una pequeña muestra en consola
    print("\nMuestra de las predicciones:")
    print(nuevos_datos[['prediccion_launch_price_decay']].head())

if __name__ == "__main__":
    # 1. Ruta donde tu equipo o cliente deja los datos nuevos a predecir
    # (Créate un CSV de prueba falso llamado 'nuevos_videojuegos.csv' para probar esto)
    RUTA_DATOS_NUEVOS = r'C:\Users\phf14_dcq7b60\Desktop\Video_Games ML Project\data\new\new_data.csv'
    
    # 2. Ruta de tus modelos y pipeline
    DIRECTORIO_MODELOS = r'C:\Users\phf14_dcq7b60\Desktop\Video_Games ML Project\models'
    
    # 3. Ruta donde guardaremos el informe final con las predicciones
    RUTA_RESULTADOS = r'C:\Users\phf14_dcq7b60\Desktop\Video_Games ML Project\reports\predicciones_finales.csv'
    
    # Ejecutamos la predicción (puedes cambiar 'ridge' por 'linear', 'lasso', etc.)
    predict(RUTA_DATOS_NUEVOS, DIRECTORIO_MODELOS, RUTA_RESULTADOS, tipo_modelo='ridge')