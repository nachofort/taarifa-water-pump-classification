# 💧 Mantenimiento Predictivo de Pozos de Agua en Tanzania (Proyecto Taarifa)

## 📌 Resumen del Proyecto
Este proyecto utiliza Machine Learning para predecir el estado de funcionamiento de las bombas de agua en Tanzania. El objetivo de negocio es optimizar las operaciones de mantenimiento, identificando qué pozos están a punto de fallar para enviar técnicos antes de que colapsen, ahorrando costes y garantizando el acceso al agua.

**Resultado destacado:** Precisión global (Accuracy) del **80.20%** en el conjunto de prueba oficial de DrivenData, demostrando un modelo altamente robusto frente a datos ruidosos.

## 🚧 El Reto de los Datos
El dataset presenta un fuerte desbalanceo de clases. Contiene miles de pozos funcionales o completamente rotos, pero una clase crítica minoritaria: `functional needs repair` (pozos que sacan agua pero necesitan reparación urgente).

### Pipeline de Preprocesado:
* **Limpieza:** Imputación inteligente de nulos (ej. año de construcción y población poblados con la mediana regional).
* **Agrupación por Cardinalidad:** Reducción de cientos de entidades financiadoras e instaladores en categorías manejables (`Government`, `International_NGO`, etc.).
* **Feature Engineering:** Creación de nuevas variables de impacto temporal e hídrico.

## 🧠 Experimentación y Decisión del Modelo
Durante la fase de modelado, se realizó un experimento comparando algoritmos de ensamblaje:

1.  **XGBoost (Gradient Boosting):** Alcanzó una alta precisión global, pero fracasó en la detección de la clase minoritaria crítica (Recall del 21% para pozos que necesitan reparación).
2.  **Random Forest (Bagging):** **El modelo ganador.** Al configurar 300 estimadores y forzar el hiperparámetro `class_weight='balanced'`, el algoritmo priorizó la clase minoritaria. 

**Decisión Técnica:** Se optó por el modelo Random Forest más simple y robusto. Aunque XGBoost es estadísticamente más complejo, el Random Forest equilibrado casi duplicó la capacidad de detectar bombas en estado crítico, alineándose perfectamente con el objetivo de negocio.

## 📁 Estructura del Repositorio

├── data/                   # Datasets crudos (no subidos por peso/privacidad)
├── notebooks/              # Cuadernos Jupyter con el EDA y la experimentación
│   └── 01_exploracion_y_modelado.ipynb
├── src/                    # Scripts de producción
│   ├── limpieza.py         # Librería personalizada de preprocesado
│   └── main.py             # Script ejecutable de inferencia
├── submission_taarifa.csv  # Archivo de salida final
└── README.md
