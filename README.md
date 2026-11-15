# 🏀 NBA Predictive Model 2026-27
> **Sistema avanzado de predicción de la NBA basado en Machine Learning (XGBoost) y métricas de eficiencia (Pace & Ratings).**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-red.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Descripción General
Este proyecto es un sistema completo de pronósticos deportivos diseñado para predecir los partidos de la temporada **2026-27 de la NBA**. Utiliza dos modelos de Machine Learning para predecir el **Ganador** y la **Línea de Puntos Totales (Over/Under)**, integrando un Dashboard interactivo para visualizar las probabilidades y detectar "Value Bets".

## 📊 Rendimiento del Modelo (Validación Temporal)
| Métrica | Resultado | Descripción |
| :--- | :---: | :--- |
| 🎯 **Precisión Ganador** | **64.7%** | Predicción del equipo ganador (Local vs Visitante). |
| 📉 **Error Puntos (MAE)** | **12.50 pts** | Error absoluto medio prediciendo los puntos totales del partido. |
| 📈 **ROI Simulado** | **+12.78%** | Retorno de inversión en Backtesting usando líneas de mercado. |
| ⚙️ **Sigma (Desviación)** | **13.87** | Utilizado para calcular la probabilidad matemática de Over/Under. |

## 🧠 Arquitectura y Características
El sistema de features fue diseñado para capturar el contexto real de la NBA:
* **Rolling Averages (5 y 10 partidos)**: Puntos anotados, recibidos y descanso (Back-to-Backs).
* **Métricas Avanzadas**: Cálculo de **Pace** (posesiones por 48 min) y **Ratings** (eficiencia ofensiva/defensiva por cada 100 posesiones).
* **Historial H2H**: Enfrentamientos directos de los últimos 10 partidos.
* **Modelos**: 
  - `XGBoost Classifier` (Ganador).
  - `XGBoost Regressor` (Puntos Totales).

## 🚀 Instalación y Uso Local

1. Clona el repositorio:
   ```bash
   git clone https://github.com/tu-usuario/nba-predictor-2026.git
   cd nba-predictor-2026
   ```

2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

3. Ejecuta el Dashboard (Streamlit):
   ```bash
   streamlit run app.py
   ```
   *Se abrirá automáticamente en tu navegador en `http://localhost:8501`.*

## 📂 Estructura del Proyecto
```text
nba-predictor-2026/
│
├── app.py                        # Código del Dashboard (Streamlit)
├── requirements.txt              # Dependencias del proyecto
├── README.md                     # Documentación
│
├── data/
│   ├── nba_dataset_final_v2.csv  # Dataset limpio con métricas avanzadas
│   └── nba_h2h.csv               # Historial de enfrentamientos
│
├── models/
│   └── nba_models_v2.pkl         # Modelos XGBoost entrenados
│
└── predictions/
    └── predicciones_2026_2027.csv # Predicciones finales de la temporada
```

## 💡 ¿Cómo funciona el Dashboard?
1. **Selecciona una fecha** en la barra lateral.
2. **Ajusta la línea de apuestas** (ej. 220.5 pts) para ver cómo cambia la probabilidad de Over/Under.
3. El sistema analizará la forma reciente, el ritmo de juego y el H2H.
4. **Alerta de Value Bet**: Si el modelo detecta más de un **60% de probabilidad** en un mercado, lo resaltará en verde indicando una posible oportunidad de apuesta.

---
*⚠️ **Descargo de responsabilidad**: Este proyecto es con fines educativos y de análisis de datos. No garantiza ganancias en apuestas deportivas.*
