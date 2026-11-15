
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import pickle
import os

st.set_page_config(page_title="NBA Predictor 2026-27", page_icon="🏀", layout="wide")
st.title("🏀 NBA Predictor 2026-27")
st.markdown("Sistema avanzado con métricas de eficiencia, historial, lesiones y logos.")

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
    lesiones = load_injuries()
    models = load_models()
    sigma = models['sigma']
except Exception as e:
    st.error(f"Error cargando archivos: {e}")
    st.stop()

# DICCIONARIO DE LOGOS
logos = {
    'Atlanta Hawks': 'atl', 'Boston Celtics': 'bos', 'Brooklyn Nets': 'bkn',
    'Charlotte Hornets': 'cha', 'Chicago Bulls': 'chi', 'Cleveland Cavaliers': 'cle',
    'Dallas Mavericks': 'dal', 'Denver Nuggets': 'den', 'Detroit Pistons': 'det',
    'Golden State Warriors': 'gs', 'Houston Rockets': 'hou', 'Indiana Pacers': 'ind',
    'LA Clippers': 'lac', 'Los Angeles Lakers': 'lal', 'Memphis Grizzlies': 'mem',
    'Miami Heat': 'mia', 'Milwaukee Bucks': 'mil', 'Minnesota Timberwolves': 'min',
    'New Orleans Pelicans': 'no', 'New York Knicks': 'ny', 'Oklahoma City Thunder': 'okc',
    'Orlando Magic': 'orl', 'Philadelphia 76ers': 'phi', 'Phoenix Suns': 'phx',
    'Portland Trail Blazers': 'por', 'Sacramento Kings': 'sac', 'San Antonio Spurs': 'sa',
    'Toronto Raptors': 'tor', 'Utah Jazz': 'utah', 'Washington Wizards': 'wsh'
}

def get_logo_html(team_name, height=40):
    for nombre, abbr in logos.items():
        if nombre in team_name or team_name in nombre:
            url = f"https://a.espncdn.com/i/teamlogos/nba/500/{abbr}.png"
            return f'<img src="{url}" height="{height}" style="vertical-align: middle; margin-right: 10px;">'
    return ""

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
        # 1. Buscar lesiones de ambos equipos
        eq_local = row['home_team']
        eq_visita = row['away_team']
        
        inj_local = lesiones[lesiones['team'].str.contains(eq_local.split()[-1], case=False, na=False)]
        inj_visita = lesiones[lesiones['team'].str.contains(eq_visita.split()[-1], case=False, na=False)]
        
        # Contar lesionados de gravedad (Out o Day-To-Day)
        out_local = len(inj_local[inj_local['current_status'].isin(['Out', 'Day-To-Day'])])
        out_visita = len(inj_visita[inj_visita['current_status'].isin(['Out', 'Day-To-Day'])])
        
        # 2. Ajustar la predicción de puntos según las lesiones
        # Restamos 1.5 puntos por cada jugador 'Out' y 0.5 por 'Day-To-Day'
        penalty_local = (out_local * 1.5)
        penalty_visita = (out_visita * 1.5)
        
        pred_total_base = row['pred_total_pts']
        pred_total = pred_total_base - (penalty_local + penalty_visita)
        
        # 3. Calcular probabilidades con el total ajustado
        prob_over = 1 - norm.cdf(linea_apuestas, loc=pred_total, scale=sigma)
        prob_under = 1 - prob_over
        
        with st.container():
            col1, col2, col3 = st.columns([2, 2, 2])
            
            with col1:
                logo_away = get_logo_html(row['away_team'])
                logo_home = get_logo_html(row['home_team'])
                
                st.markdown(f"<div style='display:flex; align-items:center; margin-bottom:5px;'>{logo_away} <span style='font-size:18px; font-weight:bold;'>{row['away_team']}</span></div>", unsafe_allow_html=True)
                st.markdown("<div style='margin-left: 20px; font-size:12px; color:gray;'>@</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='display:flex; align-items:center;'>{logo_home} <span style='font-size:18px; font-weight:bold;'>{row['home_team']}</span></div>", unsafe_allow_html=True)
                
                st.caption(f"Game ID: {row['gameId']}")
                st.markdown("---")
                
                st.markdown("**Historial Directo (Últimos 10)**")
                if row['h2h_played'] > 0:
                    st.write(f"⚔️ {row['home_team']}: **{row['h2h_home_wins']}** victorias")
                    st.write(f"⚔️ {row['away_team']}: **{row['h2h_away_wins']}** victorias")
                else:
                    st.write("No hay historial reciente.")
                
                st.markdown("---")
                st.markdown("**Reporte de Lesiones**")
                
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
                
                # Mostrar el ajuste visualmente
                if penalty_local + penalty_visita > 0:
                    st.write(f"📊 Predicción: **{pred_total:.1f} pts** *(Ajustado de {pred_total_base:.1f} por lesiones)*")
                else:
                    st.write(f"📊 Predicción: **{pred_total:.1f} pts**")
                    
                st.write(f"⬆️ Over: **{prob_over:.1%}**")
                st.write(f"⬇️ Under: **{prob_under:.1%}**")
                
                if prob_over > 0.60:
                    st.success("🔥 VALUE BET: OVER")
                elif prob_under > 0.60:
                    st.success("🔥 VALUE BET: UNDER")
            
            st.divider()

st.sidebar.markdown("---")
st.sidebar.markdown("**Modelo v3:** XGBoost + Pace + Ratings + Lesiones.")
st.sidebar.markdown(f"**Precisión Ganador:** ~64.7%")
st.sidebar.markdown(f"**MAE Puntos:** 12.5 pts")
