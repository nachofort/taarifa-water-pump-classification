import pandas as pd
import pickle

from limpieza import pipeline_preprocesado

if __name__ == "__main__":
    
    print("1. Cargando datos del examen oficial (Test set)...")
    # Cargamos los datos crudos que queremos predecir
    df_test = pd.read_csv('TestSetValues.csv')
    ids_test = df_test['id'] # Guardamos los IDs a salvo para el documento final
    
    print("2. Aplicando el pipeline de limpieza...")
    # Pasamos los datos crudos por tus funciones
    x_test_preprocessed = pipeline_preprocesado(df_test)
    
    print("3. Cargando el modelo entrenado (.pkl)...")
    # Abrimos tu modelo (Asegúrate de haberlo guardado desde Jupyter con este nombre)
    with open('modelo_taarifa.pkl', 'rb') as file:
        modelo_rf = pickle.load(file)
        
    print("4. Alineando columnas y haciendo predicciones...")
    # TRUCO DE EXPERTO: 
    # Como usamos 'pd.get_dummies', el Test puede tener menos columnas que el Train.
    # Usamos 'feature_names_in_' para decirle al Test que se adapte exactamente 
    # a las columnas que el modelo aprendió cuando estudió.
    columnas_entrenamiento = modelo_rf.feature_names_in_
    x_test_preprocessed = x_test_preprocessed.reindex(columns=columnas_entrenamiento, fill_value=0)
    
    # El modelo adivina las respuestas
    predicciones = modelo_rf.predict(x_test_preprocessed)
    

    df_entrega = pd.DataFrame({
        'id': ids_test, 
        'status_group': predicciones
    })
    
    df_entrega.to_csv('submission_taarifa.csv', index=False)
    
    print("\n✅ ¡Ejecución completada! Archivo 'submission_taarifa.csv' generado con éxito.")