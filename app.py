
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import pickle

st.set_page_config(page_title="NBA Predictor 2026-27", page_icon="🏀", layout="wide")
st.cache_data.clear()  # Limpiar caché al iniciar

st.title("🏀 NBA Predictor 2026-27 (v2 - Pace & Ratings)")
st.markdown("Sistema avanzado con métricas de eficiencia y ritmo de juego.")

@st.cache_data
def load_data():
    df = pd.read_csv("predicciones_nba_2026_2027.csv", parse_dates=['date'])
    h2h = pd.read_csv("nba_h2h.csv")
    df = df.merge(h2h, on='gameId', how='left')
    return df

@st.cache_resource
def load_models():
    # Ahora carga el modelo v2
    with open('nba_models_v2.pkl', 'rb') as f:
        models = pickle.load(f)
    return models

try:
    df = load_data()
    models = load_models()
    sigma = models['sigma']
except Exception as e:
    st.error(f"Error cargando archivos: {e}")
    st.stop()

st.sidebar.header("Filtros")
linea_apuestas = st.sidebar.number_input("Línea de apuestas (Puntos Totales)", min_value=180.0, max_value=280.0, value=220.5, step=0.5)

fechas_disponibles = sorted(df['date'].dt.date.unique())
fecha_seleccionada = st.sidebar.selectbox("Selecciona una fecha", fechas_disponibles)

partidos_dia = df[df['date'].dt.date == fecha_seleccionada].copy()

if len(partidos_dia) == 0:
    st.warning("No hay partidos para esta fecha.")
else:
    st.subheader(f"Partidos del {fecha_seleccionada}")
    
    for _, row in partidos_dia.iterrows():
        pred_total = row['pred_total_pts']
        # Usamos el nuevo Sigma (13.87) para mayor precisión
        prob_over = 1 - norm.cdf(linea_apuestas, loc=pred_total, scale=sigma)
        prob_under = 1 - prob_over
        
        with st.container():
            col1, col2, col3 = st.columns([2, 2, 2])
            
            with col1:
                st.markdown(f"### {row['away_team']} @ {row['home_team']}")
                st.caption(f"Game ID: {row['gameId']}")
                
                st.markdown("**Historial Directo (Últimos 10)**")
                if row['h2h_played'] > 0:
                    st.write(f"⚔️ {row['home_team']}: **{row['h2h_home_wins']}** victorias")
                    st.write(f"⚔️ {row['away_team']}: **{row['h2h_away_wins']}** victorias")
                    st.caption(f"Último partido: {row['h2h_last_match']}")
                else:
                    st.write("No hay historial reciente.")
                
            with col2:
                st.markdown("**Ganador**")
                st.progress(float(row['p_home']))
                st.write(f"🏠 {row['home_team']}: **{row['p_home']:.1%}**")
                st.write(f"✈️ {row['away_team']}: **{row['p_away']:.1%}**")
                
            with col3:
                st.markdown(f"**Puntos Totales (Línea: {linea_apuestas})**")
                st.write(f"📊 Predicción: **{pred_total:.1f} pts**")
                st.write(f"⬆️ Over: **{prob_over:.1%}**")
                st.write(f"⬇️ Under: **{prob_under:.1%}**")
                
                if prob_over > 0.60:
                    st.success("🔥 VALUE BET: OVER")
                elif prob_under > 0.60:
                    st.success("🔥 VALUE BET: UNDER")
            
            st.divider()

st.sidebar.markdown("---")
st.sidebar.markdown("**Modelo v2:** XGBoost + Pace + Ratings.")
st.sidebar.markdown(f"**Precisión Ganador:** ~64.7%")
st.sidebar.markdown(f"**MAE Puntos:** 12.5 pts (Sigma: {sigma:.2f})")
