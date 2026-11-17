
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import pickle
import os

st.set_page_config(page_title="NBA Predictor 2026-27", page_icon="🏀", layout="wide")
st.title("🏀 NBA Predictor 2026-27")
st.markdown("Sistema Pro: Ajuste por BPM (Impacto Real del Jugador).")

@st.cache_data
def load_data():
    df = pd.read_csv("predicciones_nba_2026_2027.csv", parse_dates=['date'])
    h2h = pd.read_csv("nba_h2h.csv")
    df = df.merge(h2h, on='gameId', how='left')
    return df

@st.cache_data
def load_injuries():
    if os.path.exists("lesiones_nba.csv"):
        df = pd.read_csv("lesiones_nba.csv")
        if len(df) > 0: return df
    return pd.DataFrame(columns=['team', 'player_name', 'current_status', 'reason'])

@st.cache_data
def load_bpm():
    if os.path.exists("bpm_jugadores.csv"):
        return pd.read_csv("bpm_jugadores.csv")
    return pd.DataFrame(columns=['player_name', 'bpm'])

@st.cache_resource
def load_models():
    with open('nba_models_v2.pkl', 'rb') as f:
        return pickle.load(f)

try:
    df = load_data()
    lesiones = load_injuries()
    bpm_df = load_bpm()
    models = load_models()
    sigma = models['sigma']
except Exception as e:
    st.error(f"Error cargando archivos: {e}")
    st.stop()

# Diccionario de BPM
bpm_dict = dict(zip(bpm_df['player_name'], bpm_df['bpm'])) if len(bpm_df) > 0 else {}

logos = {'Atlanta Hawks': 'atl', 'Boston Celtics': 'bos', 'Brooklyn Nets': 'bkn', 'Charlotte Hornets': 'cha', 'Chicago Bulls': 'chi', 'Cleveland Cavaliers': 'cle', 'Dallas Mavericks': 'dal', 'Denver Nuggets': 'den', 'Detroit Pistons': 'det', 'Golden State Warriors': 'gs', 'Houston Rockets': 'hou', 'Indiana Pacers': 'ind', 'LA Clippers': 'lac', 'Los Angeles Lakers': 'lal', 'Memphis Grizzlies': 'mem', 'Miami Heat': 'mia', 'Milwaukee Bucks': 'mil', 'Minnesota Timberwolves': 'min', 'New Orleans Pelicans': 'no', 'New York Knicks': 'ny', 'Oklahoma City Thunder': 'okc', 'Orlando Magic': 'orl', 'Philadelphia 76ers': 'phi', 'Phoenix Suns': 'phx', 'Portland Trail Blazers': 'por', 'Sacramento Kings': 'sac', 'San Antonio Spurs': 'sa', 'Toronto Raptors': 'tor', 'Utah Jazz': 'utah', 'Washington Wizards': 'wsh'}

def get_logo_html(team_name, height=40):
    for nombre, abbr in logos.items():
        if nombre in team_name or team_name in nombre:
            return f'<img src="https://a.espncdn.com/i/teamlogos/nba/500/{abbr}.png" height="{height}" style="vertical-align: middle; margin-right: 10px;">'
    return ""

def get_ultimos_totales(team_name, n=5):
    mask = (df['home_team'] == team_name) | (df['away_team'] == team_name)
    return df[mask].sort_values('date').tail(n).copy()[['date', 'pred_total_pts']].set_index('date')

st.sidebar.header("Filtros")
linea_apuestas = st.sidebar.number_input("Línea Total Puntos", value=220.5, step=0.5)
linea_spread = st.sidebar.number_input("Línea Spread (Local)", value=-4.5, step=0.5)

fechas_disponibles = sorted(df['date'].dt.date.unique())
fecha_seleccionada = st.sidebar.selectbox("Selecciona una fecha", fechas_disponibles)

partidos_dia = df[df['date'].dt.date == fecha_seleccionada].copy()

if len(partidos_dia) == 0:
    st.warning("No hay partidos para esta fecha.")
else:
    st.subheader(f"Partidos del {fecha_seleccionada}")
    
    for _, row in partidos_dia.iterrows():
        eq_local = row['home_team']
        eq_visita = row['away_team']
        
        inj_local = lesiones[lesiones['team'].str.contains(eq_local.split()[-1], case=False, na=False)]
        inj_visita = lesiones[lesiones['team'].str.contains(eq_visita.split()[-1], case=False, na=False)]
        
        # --- LÓGICA DE AJUSTE POR BPM REAL ---
        bpm_perdido_local = 0
        for _, inj in inj_local.iterrows():
            if inj['current_status'] in ['Out', 'Day-To-Day']:
                bpm = bpm_dict.get(inj['player_name'], 0.0) # Si no lo encuentra, impacto 0
                factor_duda = 0.5 if inj['current_status'] == 'Day-To-Day' else 1.0
                bpm_perdido_local += (bpm * factor_duda)
        
        bpm_perdido_visita = 0
        for _, inj in inj_visita.iterrows():
            if inj['current_status'] in ['Out', 'Day-To-Day']:
                bpm = bpm_dict.get(inj['player_name'], 0.0)
                factor_duda = 0.5 if inj['current_status'] == 'Day-To-Day' else 1.0
                bpm_perdido_visita += (bpm * factor_duda)
        
        # Convertir BPM perdido a puntos (1 BPM ≈ 0.8 puntos)
        penalty_local = max(0, bpm_perdido_local * 0.8)
        penalty_visita = max(0, bpm_perdido_visita * 0.8)
        
        # Ajuste de probabilidad de Ganador
        ajuste_prob = (penalty_visita - penalty_local) * 0.015 
        p_home_ajustada = max(0.05, min(0.95, row['p_home'] + ajuste_prob))
        p_away_ajustada = 1 - p_home_ajustada
        
        # Totales
        pred_total = row['pred_total_pts'] - (penalty_local + penalty_visita)
        prob_over = 1 - norm.cdf(linea_apuestas, loc=pred_total, scale=sigma)
        
        # Handicap y Puntos por Equipo
        margen = (p_home_ajustada - 0.5) * 20
        pts_local = (pred_total / 2) + (margen / 2)
        pts_visita = (pred_total / 2) - (margen / 2)
        prob_local_cubre = 1 - norm.cdf(linea_spread, loc=margen, scale=sigma * 0.8)
        
        with st.container():
            col1, col2, col3 = st.columns([2, 2, 2])
            with col1:
                st.markdown(f"<div style='display:flex; align-items:center;'>{get_logo_html(row['away_team'])} <b>{row['away_team']}</b></div>", unsafe_allow_html=True)
                st.markdown("<div style='margin-left: 20px; color:gray;'>@</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='display:flex; align-items:center;'>{get_logo_html(row['home_team'])} <b>{row['home_team']}</b></div>", unsafe_allow_html=True)
                st.markdown("---")
                st.markdown("**Lesiones**")
                if len(inj_local) > 0:
                    st.caption(f"🏠 {eq_local}:")
                    for _, inj in inj_local.iterrows(): st.caption(f"- {inj['player_name']} ({inj['current_status']})")
                if len(inj_visita) > 0:
                    st.caption(f"✈️ {eq_visita}:")
                    for _, inj in inj_visita.iterrows(): st.caption(f"- {inj['player_name']} ({inj['current_status']})")
            with col2:
                st.markdown("**Ganador (Ajustado por BPM)**")
                st.progress(float(p_home_ajustada))
                st.write(f"🏠 {eq_local}: **{p_home_ajustada:.1%}**")
                st.write(f"✈️ {eq_visita}: **{p_away_ajustada:.1%}**")
                if abs(ajuste_prob) > 0.005: st.caption(f"*(Base: {row['p_home']:.1%})*")
                st.markdown("---")
                st.markdown(f"**Spread ({linea_spread})**")
                st.write(f"🏠 Local: **{prob_local_cubre:.1%}**")
            with col3:
                st.markdown(f"**Totales (Línea: {linea_apuestas})**")
                st.write(f"📊 Pred: **{pred_total:.1f} pts**")
                st.write(f"⬆️ Over: **{prob_over:.1%}**")
                st.write(f"⬇️ Under: **{1-prob_over:.1%}**")
                if prob_over > 0.60: st.success("🔥 VALUE: OVER")
                elif (1-prob_over) > 0.60: st.success("🔥 VALUE: UNDER")
                st.markdown("---")
                st.markdown("**Puntos por Equipo**")
                st.write(f"🏠 {eq_local}: **{pts_local:.1f}**")
                st.write(f"✈️ {eq_visita}: **{pts_visita:.1f}**")
            st.divider()
