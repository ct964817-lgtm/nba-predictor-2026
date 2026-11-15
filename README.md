# 🏀 NBA Predictive Model 2026-27

Sistema de predicción de la NBA basado en Machine Learning (XGBoost) para predecir el **Ganador** y la **Línea de Puntos (Over/Under)**.

## 📊 Rendimiento del Modelo
- **Precisión Ganador**: 64.7% (Validación temporal).
- **Error de Puntos (MAE)**: 12.50 puntos.
- **ROI Simulado**: +12.78% (Backtesting).

## 🧠 Características del Modelo
- Datos: +73,000 partidos históricos.
- Features: Rolling averages (5 y 10 partidos), descanso, Pace (posesiones) y Ratings (eficiencia).
- Modelos: XGBoost Classifier (Ganador) y XGBoost Regressor (Totales).

## 🚀 Cómo ejecutarlo localmente
1. Instala las dependencias: `pip install -r requirements.txt`
2. Ejecuta el dashboard: `streamlit run app.py`
