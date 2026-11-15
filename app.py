
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import pickle
import os

st.set_page_config(page_title="NBA Predictor 2026-27", page_icon="🏀", layout="wide")
st.title("🏀 NBA Predictor 2026-27")
st.markdown("Sistema avanzado con métricas de eficiencia, historial y estado de plantillas.")

@st.cache_data
def load_data():
    df = pd.read_csv("predicciones_nba_2026_2027.csv", parse_dates=['date'])
    h2h = pd.read_csv("nba_h2h.csv")
    df = df.merge(h2h, on='gameId', how='left')
    return df

@st.cache_data
def load_logos():
    if os.path.exists("logos_nba.csv"):
        return pd.read_csv("logos_nba.csv")
    return pd.DataFrame(columns=['TeamName', 'LogoURL'])

@st.cache_data
def load_injuries():
    if os.path.exists("lesiones_nba.csv"):
        df = pd.read_csv("lesiones_nba.csv")
        if len(df) > 0:
            return df
    return pd.DataFrame(columns=['team', 'player_name', 'current_status', 'reason'])

@st.cache_resource
def load_models():
    with open('nba_models_v2.pkl', 'rb') as f:
        models = pickle.load(f)
    return models

try:
    df = load_data()
    logos = load_logos()
    lesiones = load_injuries()
    models = load_models()
    sigma = models['sigma']
except Exception as e:
    st.error(f"Error cargando archivos: {e}")
    st.stop()

logo_dict = dict(zip(logos['TeamName'], logos['LogoURL']))

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
        prob_over = 1 - norm.cdf(linea_apuestas, loc=pred_total, scale=sigma)
        prob_under = 1 - prob_over
        
        with st.container():
            col1, col2, col3 = st.columns([2, 2, 2])
            
            with col1:
                # LOGOS
                c1, c2 = st.columns([1, 3])
                with c1:
                    if row['away_team'] in logo_dict:
                        st.image(logo_dict[row['away_team']], width=50)
                with c2:
                    st.markdown(f"**{row['away_team']}**")
                
                st.markdown("<p style='text-align: center; margin: 0;'>@</p>", unsafe_allow_html=True)
                
                c3, c4 = st.columns([1, 3])
                with c3:
                    if row['home_team'] in logo_dict:
                        st.image(logo_dict[row['home_team']], width=50)
                with c4:
                    st.markdown(f"**{row['home_team']}**")
                
                st.caption(f"Game ID: {row['gameId']}")
                
                # H2H
                st.markdown("**Historial Directo (Últimos 10)**")
                if row['h2h_played'] > 0:
                    st.write(f"⚔️ {row['home_team']}: **{row['h2h_home_wins']}** victorias")
                    st.write(f"⚔️ {row['away_team']}: **{row['h2h_away_wins']}** victorias")
                else:
                    st.write("No hay historial reciente.")
                
                # LESIONES
                st.markdown("**Reporte de Lesiones**")
                eq_local = row['home_team']
                eq_visita = row['away_team']
                
                inj_local = lesiones[lesiones['team'].str.contains(eq_local.split()[-1], case=False, na=False)]
                inj_visita = lesiones[lesiones['team'].str.contains(eq_visita.split()[-1], case=False, na=False)]
                
                if len(inj_local) > 0:
                    st.caption(f"🏠 {eq_local}:")
                    for _, inj in inj_local.iterrows():
                        st.caption(f"- {inj['player_name']} ({inj['current_status']})")
                
                if len(inj_visita) > 0:
                    st.caption(f"✈️ {eq_visita}:")
                    for _, inj in inj_visita.iterrows():
                        st.caption(f"- {inj['player_name']} ({inj['current_status']})")
                
                if len(inj_local) == 0 and len(inj_visita) == 0:
                    st.caption("Sin datos de lesiones actuales.")
                
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
st.sidebar.markdown("**Modelo v3:** XGBoost + Pace + Ratings.")
st.sidebar.markdown(f"**Precisión Ganador:** ~64.7%")
st.sidebar.markdown(f"**MAE Puntos:** 12.5 pts")
