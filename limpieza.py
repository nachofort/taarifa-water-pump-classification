

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


df=pd.read_csv('TrainingSetValues.csv')

"""## 1 Análisis exploratorio"""

df.head()

print(df['id'].duplicated().sum())

df.info()

df.describe()

df.describe(percentiles= [0.25, 0.5, 0.75, 0.9, 0.95,0.99])

for i in df.columns:
    print(i)
    print(df[i].value_counts())

print(df.nunique())

## Valores nulos

def limpiar_datos_cat(df):

    df_clean = df.copy()
    cols_drop = ['id', 'subvillage', 'recorded_by', 'scheme_name', 'extraction_type', 'extraction_type_group', 'region_code', 'district_code',
                 'management', 'payment_type', 'water_quality', 'quantity_group', 'source_class', 'source', 'waterpoint_type' ]
    df_clean = df_clean.drop(columns=cols_drop)
    df_clean['funder'] = df_clean['funder'].fillna('Unknown')
    df_clean['installer'] = df_clean['installer'].fillna('Unknown')
    df_clean['scheme_management'] = df_clean['scheme_management'].fillna('Other')
    df_clean['public_meeting'] = df_clean['public_meeting'].fillna(True)
    df_clean['permit'] = df_clean['permit'].fillna(True)
    df_clean['scheme_management'] = df_clean['scheme_management'].fillna('Other')
    df_clean['public_meeting'] = df_clean['public_meeting'].fillna(True)
    df_clean['permit'] = df_clean['permit'].fillna(True)
    return df_clean

def limpiar_datos_num(df):

    df_clean = df.copy()
    umbral_confianza = 150
    df_clean['has_amount_tsh'] = (df_clean['amount_tsh'] > 0).astype(int)
    df_clean['has_construction_year'] = (df_clean['construction_year'] > 0).astype(int)
    df_clean['has_population'] = (df_clean['population'] > 0).astype(int)

    df_clean['construction_year'] = df_clean['construction_year'].replace(0, np.nan)
    conteos_year = df_clean.groupby('region')['construction_year'].count()
    regiones_fiables_year = conteos_year[conteos_year >= umbral_confianza].index  
    falta_year = df_clean['construction_year'].isna()
    es_fiable_year = df_clean['region'].isin(regiones_fiables_year)
    df_clean.loc[falta_year & es_fiable_year, 'construction_year'] = df_clean.groupby('region')['construction_year'].transform('median')
    df_clean.loc[falta_year & ~es_fiable_year, 'construction_year'] = df_clean['construction_year'].median()

    df_clean['population'] = df_clean['population'].replace(0, np.nan)
    conteos_pop = df_clean.groupby('region')['population'].count()
    regiones_fiables_pop = conteos_pop[conteos_pop >= umbral_confianza].index   
    falta_pop = df_clean['population'].isna()
    es_fiable_pop = df_clean['region'].isin(regiones_fiables_pop)  
    df_clean.loc[falta_pop & es_fiable_pop, 'population'] = df_clean.groupby('region')['population'].transform('median')
    df_clean.loc[falta_pop & ~es_fiable_pop, 'population'] = df_clean['population'].median()
    
    df_clean['longitude'] = pd.to_numeric(df_clean['longitude'], errors='coerce')
    df_clean['latitude'] = pd.to_numeric(df_clean['latitude'], errors='coerce')
    df_clean['longitude'] = df_clean['longitude'].replace(0.0, np.nan)
    df_clean['latitude'] = df_clean['latitude'].replace(0.0, np.nan)
    df_clean['longitude'] = df_clean['longitude'].fillna(df_clean.groupby('region')['longitude'].transform('median'))
    df_clean['latitude'] = df_clean['latitude'].fillna(df_clean.groupby('region')['latitude'].transform('median'))
    df_clean['longitude'] = df_clean['longitude'].fillna(df_clean['longitude'].median())
    df_clean['latitude'] = df_clean['latitude'].fillna(df_clean['latitude'].median())
    
    df_clean['has_amount_tsh'] = (df_clean['amount_tsh'] > 0).astype(int)
    df_clean = df_clean.drop(columns=['num_private'])

    return df_clean

def Cardinalidad(df):
    df_agrupado = df.copy()
    def agrupar_funder(nombre):
        if pd.isna(nombre): return 'Unknown'
        name = str(nombre).lower()
        if name in ['0', 'unknown', 'not known', 'none', '-']: return 'Unknown'
        if any(w in name for w in ['gov', 'ministry', 'council', 'lga', 'serikali', 'municipal', 'district', 'halmashauri', 'water board']): return 'Government'
        if any(w in name for w in ['church', 'catholic', 'roman', 'rc ', 'mission', 'diocese', 'kkkt', 'mosque', 'islam', 'kanisa', 'baptist', 'anglican', 'pentecost']): return 'Religious'
        if any(w in name for w in ['unicef', 'unice', 'world bank', 'w.b', 'w0rld', 'danida', 'vision', 'tasaf', 'oxfam', 'wateraid', 'hesawa', 'rwssp', 'amref', 'dwsp', 'tcrs', 'care', 'fund', 'red cross']): return 'International_NGO'
        if any(w in name for w in ['germany', 'japan', 'netherlands', 'nethalan', 'holland', 'china', 'usaid', 'finland', 'swedish', 'italy', 'british', 'belgian', 'kuwait', 'france']): return 'Foreign_Government'
        return 'Private_Local_Other'

    def agrupar_installer(nombre):
        if pd.isna(nombre): return 'Unknown'
        name = str(nombre).lower()
        if name in ['0', 'unknown', 'not known', 'none', '-']: return 'Unknown'
        if any(w in name for w in ['dwe', 'rwe', 'mwe', 'gov', 'ministry', 'council', 'lga', 'serikali', 'municipal', 'district', 'halmashauri', 'water board', 'idara', 'wizara', 'department']): return 'Government'
        if any(w in name for w in ['church', 'catholic', 'roman', 'rc ', 'mission', 'diocese', 'kkkt', 'mosque', 'islam', 'kanisa', 'baptist', 'anglican', 'pentecost', 'kkt']) or name == 'rc': return 'Religious'
        if any(w in name for w in ['unicef', 'unice', 'world bank', 'w.b', 'danida', 'vision', 'tasaf', 'oxfam', 'wateraid', 'hesawa', 'rwssp', 'amref', 'dwsp', 'tcrs', 'care', 'fund', 'red cross', 'twesa', 'acra']): return 'International_NGO'
        if any(w in name for w in ['germany', 'japan', 'netherlands', 'holland', 'china', 'usaid', 'finland', 'swedish', 'italy', 'british', 'belgian', 'kuwait', 'france', 'norad', 'jica', 'koica', 'wachina']): return 'Foreign_Government'
        return 'Private_Local_Other'

    def agrupar_scheme(scheme):
        if pd.isna(scheme): return 'Other'
        scheme = str(scheme)
        if scheme in ['VWC', 'WUG', 'WUA', 'SWC']: return 'Community'
        elif scheme in ['Water authority', 'Water Board', 'Parastatal']: return 'Government'
        elif scheme in ['Company', 'Private operator', 'Trust']: return 'Private'
        else: return 'Other'

    df_agrupado['funder_grouped'] = df_agrupado['funder'].apply(agrupar_funder)
    df_agrupado['installer_grouped'] = df_agrupado['installer'].apply(agrupar_installer)
    df_agrupado['scheme_grouped'] = df_agrupado['scheme_management'].apply(agrupar_scheme)

    lga_str = df_agrupado['lga'].astype(str).str.lower()
    df_agrupado['lga_type'] = np.where(lga_str.str.contains('urban'), 'Urban',
                              np.where(lga_str.str.contains('rural'), 'Rural', 'Other'))

    columnas_a_borrar = ['funder', 'installer', 'scheme_management', 'lga', 'ward', 'wpt_name']
    df_agrupado = df_agrupado.drop(columns=[col for col in columnas_a_borrar if col in df_agrupado.columns])
    return df_agrupado

def transformacion_numeros(df):
    df_clean = df.copy()
    for col in ['population', 'construction_year']:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype('int64')
            
    if 'date_recorded' in df_clean.columns:
        df_clean['date_recorded'] = pd.to_datetime(df_clean['date_recorded'])
        df_clean['year_recorded'] = df_clean['date_recorded'].dt.year.astype('int64')
        df_clean = df_clean.drop(columns=['date_recorded'])
    return df_clean

def OneHot_Encoding(df):
    df_clean = df.copy()
    columnas_para_ohe = [
        'basin', 'region', 'public_meeting', 'permit',
        'extraction_type_class', 'management_group', 'payment', 
        'quality_group', 'quantity', 'source_type', 'waterpoint_type_group',
        'funder_grouped', 'installer_grouped', 'scheme_grouped', 'lga_type'
    ]
    cols_ohe_existentes = [col for col in columnas_para_ohe if col in df_clean.columns]
    df_clean = pd.get_dummies(df_clean, columns=cols_ohe_existentes, drop_first=True, dtype=int)
    return df_clean

def pipeline_preprocesado(df, columnas_entrenamiento=None):
    df_clean = limpiar_datos_cat(df)
    df_clean = limpiar_datos_num(df_clean)
    df_clean = Cardinalidad(df_clean)
    df_clean = transformacion_numeros(df_clean)
    df_clean = OneHot_Encoding(df_clean)

    if columnas_entrenamiento is not None:
        df_clean = df_clean.reindex(columns=columnas_entrenamiento, fill_value=0)
    return df_clean

"""### 2. Preprocesado de los datos"""

# 1. Cargamos las respuestas (etiquetas)
df_etiquetas = pd.read_csv('TrainingSetLabels.csv')

# 2. Pasamos nuestro DataFrame crudo (df) por la máquina de limpieza
print("Limpiando datos...")
df_procesado = pipeline_preprocesado(df)

"""### 3.Muestreo estratificado"""

X_train, X_test, y_train, y_test = train_test_split(
    df_procesado,                     # Usamos el df_procesado que acabamos de crear
    df_etiquetas['status_group'],     # Usamos las etiquetas cargadas
    test_size=0.2,
    random_state=42,
    stratify=df_etiquetas['status_group'])


"""### 4.Modelado"""

print("Entrenando Random Forest...")
modelo = RandomForestClassifier(
    n_estimators=300, 
    random_state=42, 
    class_weight='balanced',
    n_jobs=-1 # Añadido para que use todos los núcleos y vaya más rápido
)
modelo.fit(X_train, y_train)

# Predecimos y evaluamos
y_pred = modelo.predict(X_test)

print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))

"""### 5. Exportar el Modelo a Producción"""
import pickle

# Guardamos el modelo entrenado
pkl_filename = "modelo_taarifa.pkl"

with open(pkl_filename, 'wb') as file:
    pickle.dump(modelo, file)




