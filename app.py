
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import pickle
import os

st.set_page_config(page_title="NBA Predictor 2026-27", page_icon="🏀", layout="wide")
st.title("🏀 NBA Predictor 2026-27")
st.markdown("Sistema avanzado: Totales, Ganador (ajustado por lesiones), Handicap y Puntos por Equipo.")

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

@st.cache_data
def load_ppg():
    if os.path.exists("ppg_jugadores.csv"):
        return pd.read_csv("ppg_jugadores.csv")
    return pd.DataFrame(columns=['player_name', 'ppg'])

@st.cache_resource
def load_models():
    with open('nba_models_v2.pkl', 'rb') as f:
        models = pickle.load(f)
    return models

try:
    df = load_data()
    lesiones = load_injuries()
    ppg_df = load_ppg()
    models = load_models()
    sigma = models['sigma']
except Exception as e:
    st.error(f"Error cargando archivos: {e}")
    st.stop()

ppg_dict = dict(zip(ppg_df['player_name'], ppg_df['ppg'])) if len(ppg_df) > 0 else {}

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

def get_ultimos_totales(team_name, n=5):
    mask = (df['home_team'] == team_name) | (df['away_team'] == team_name)
    team_games = df[mask].sort_values('date').tail(n).copy()
    return team_games[['date', 'pred_total_pts']].set_index('date')

st.sidebar.header("Filtros")
linea_apuestas = st.sidebar.number_input("Línea Total Puntos", min_value=180.0, max_value=280.0, value=220.5, step=0.5)
linea_spread = st.sidebar.number_input("Línea Spread (Local)", min_value=-25.0, max_value=25.0, value=-4.5, step=0.5)

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
        
        # 1. Calcular penalización en PUNTOS
        penalty_local = 0
        for _, inj in inj_local.iterrows():
            if inj['current_status'] in ['Out', 'Day-To-Day']:
                jugador = inj['player_name']
                ppg = ppg_dict.get(jugador, 10.0)
                factor_duda = 0.5 if inj['current_status'] == 'Day-To-Day' else 1.0
                if ppg >= 25: penalty_local += (ppg * 0.25 * factor_duda)
                elif ppg >= 15: penalty_local += (ppg * 0.15 * factor_duda)
                else: penalty_local += (ppg * 0.05 * factor_duda)
        
        penalty_visita = 0
        for _, inj in inj_visita.iterrows():
            if inj['current_status'] in ['Out', 'Day-To-Day']:
                jugador = inj['player_name']
                ppg = ppg_dict.get(jugador, 10.0)
                factor_duda = 0.5 if inj['current_status'] == 'Day-To-Day' else 1.0
                if ppg >= 25: penalty_visita += (ppg * 0.25 * factor_duda)
                elif ppg >= 15: penalty_visita += (ppg * 0.15 * factor_duda)
                else: penalty_visita += (ppg * 0.05 * factor_duda)
        
        # 2. Calcular probabilidad de Ganador AJUSTADA por lesiones
        # La probabilidad base del modelo es row['p_home']
        # Si el local tiene más penalización, su probabilidad baja.
        # Usamos una escala donde 10 puntos de penalización = ~10% menos de probabilidad.
        ajuste_prob = (penalty_visita - penalty_local) * 0.01 
        
        p_home_ajustada = row['p_home'] + ajuste_prob
        # Mantener la probabilidad entre 5% y 95% para evitar valores absurdos
        p_home_ajustada = max(0.05, min(0.95, p_home_ajustada))
        p_away_ajustada = 1 - p_home_ajustada
        
        # 3. Calcular totales
        pred_total_base = row['pred_total_pts']
        pred_total = pred_total_base - (penalty_local + penalty_visita)
        
        prob_over = 1 - norm.cdf(linea_apuestas, loc=pred_total, scale=sigma)
        prob_under = 1 - prob_over
        
        # 4. Calcular Handicap y Puntos por Equipo
        margen_esperado = (p_home_ajustada - 0.5) * 20 
        pts_local_estimados = (pred_total / 2) + (margen_esperado / 2)
        pts_visita_estimados = (pred_total / 2) - (margen_esperado / 2)
        
        sigma_spread = sigma * 0.8 
        prob_local_cubre = 1 - norm.cdf(linea_spread, loc=margen_esperado, scale=sigma_spread)
        prob_visita_cubre = 1 - prob_local_cubre
        
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
                st.markdown("**Ganador (Ajustado)**")
                st.progress(float(p_home_ajustada))
                st.write(f"🏠 {row['home_team']}: **{p_home_ajustada:.1%}**")
                st.write(f"✈️ {row['away_team']}: **{p_away_ajustada:.1%}**")
                
                # Mostrar si hubo ajuste
                if abs(ajuste_prob) > 0.005:
                    st.caption(f"*(Ajustado por lesiones. Base era {row['p_home']:.1%})*")
                
                st.markdown("---")
                st.markdown(f"**Handicap (Spread: {linea_spread})**")
                st.write(f"🏠 {eq_local} ({linea_spread}): **{prob_local_cubre:.1%}**")
                st.write(f"✈️ {eq_visita} ({-linea_spread:+}): **{prob_visita_cubre:.1%}**")
                
            with col3:
                st.markdown(f"**Puntos Totales (Línea: {linea_apuestas})**")
                
                if penalty_local + penalty_visita > 0.5:
                    st.write(f"📊 Predicción: **{pred_total:.1f} pts** *(Ajustado de {pred_total_base:.1f})*")
                else:
                    st.write(f"📊 Predicción: **{pred_total:.1f} pts**")
                    
                st.write(f"⬆️ Over: **{prob_over:.1%}**")
                st.write(f"⬇️ Under: **{prob_under:.1%}**")
                
                if prob_over > 0.60:
                    st.success("🔥 VALUE BET: OVER")
                elif prob_under > 0.60:
                    st.success("🔥 VALUE BET: UNDER")
                    
                st.markdown("---")
                st.markdown("**Puntos por Equipo (Estimados)**")
                st.write(f"🏠 {eq_local}: **{pts_local_estimados:.1f} pts**")
                st.write(f"✈️ {eq_visita}: **{pts_visita_estimados:.1f} pts**")
            
            st.markdown("---")
            st.markdown("**📈 Tendencia de Puntos Totales (Últimos 5 partidos)**")
            
            tendencia_local = get_ultimos_totales(eq_local)
            tendencia_visita = get_ultimos_totales(eq_visita)
            
            if len(tendencia_local) > 0 or len(tendencia_visita) > 0:
                chart_data = pd.DataFrame()
                if len(tendencia_local) > 0:
                    chart_data[eq_local] = tendencia_local['pred_total_pts'].values
                if len(tendencia_visita) > 0:
                    chart_data[eq_visita] = tendencia_visita['pred_total_pts'].values
                st.line_chart(chart_data)
            else:
                st.caption("No hay suficientes datos para graficar.")
            
            st.divider()

st.sidebar.markdown("---")
st.sidebar.markdown("**Modelo v8:** ML + Lesiones (Puntos y Ganador) + Spread.")
st.sidebar.markdown(f"**Precisión Ganador Base:** ~64.7%")
st.sidebar.markdown(f"**MAE Puntos:** 12.4 pts")
