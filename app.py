
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import pickle
import os

st.set_page_config(page_title="NBA Predictor 2026-27", page_icon="🏀", layout="wide")
st.title("🏀 NBA Predictor 2026-27")
st.markdown("Sistema Pro: Impacto On/Off, Varianza, Kelly Criterion y Métricas.")

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
def load_net_rating():
    if os.path.exists("net_rating_jugadores.csv"):
        return pd.read_csv("net_rating_jugadores.csv")
    return pd.DataFrame(columns=['player_name', 'net_rating'])

@st.cache_data
def load_varianza():
    if os.path.exists("varianza_equipos.csv"):
        return pd.read_csv("varianza_equipos.csv")
    return pd.DataFrame(columns=['team_name', 'std_dev_points'])

@st.cache_data
def load_def_ratings():
    if os.path.exists("def_ratings.csv"):
        return pd.read_csv("def_ratings.csv")
    return pd.DataFrame(columns=['team_name', 'def_rating'])

@st.cache_resource
def load_models():
    with open('nba_models_v2.pkl', 'rb') as f:
        return pickle.load(f)

try:
    df = load_data()
    lesiones = load_injuries()
    net_df = load_net_rating()
    var_df = load_varianza()
    def_df = load_def_ratings()
    models = load_models()
    sigma_base = models['sigma']
except Exception as e:
    st.error(f"Error cargando archivos: {e}")
    st.stop()

net_dict = dict(zip(net_df['player_name'], net_df['net_rating'])) if len(net_df) > 0 else {}
var_dict = dict(zip(var_df['team_name'], var_df['std_dev_points'])) if len(var_df) > 0 else {}
def_dict = dict(zip(def_df['team_name'], def_df['def_rating'])) if len(def_df) > 0 else {}

logos = {'Atlanta Hawks': 'atl', 'Boston Celtics': 'bos', 'Brooklyn Nets': 'bkn', 'Charlotte Hornets': 'cha', 'Chicago Bulls': 'chi', 'Cleveland Cavaliers': 'cle', 'Dallas Mavericks': 'dal', 'Denver Nuggets': 'den', 'Detroit Pistons': 'det', 'Golden State Warriors': 'gs', 'Houston Rockets': 'hou', 'Indiana Pacers': 'ind', 'LA Clippers': 'lac', 'Los Angeles Lakers': 'lal', 'Memphis Grizzlies': 'mem', 'Miami Heat': 'mia', 'Milwaukee Bucks': 'mil', 'Minnesota Timberwolves': 'min', 'New Orleans Pelicans': 'no', 'New York Knicks': 'ny', 'Oklahoma City Thunder': 'okc', 'Orlando Magic': 'orl', 'Philadelphia 76ers': 'phi', 'Phoenix Suns': 'phx', 'Portland Trail Blazers': 'por', 'Sacramento Kings': 'sac', 'San Antonio Spurs': 'sa', 'Toronto Raptors': 'tor', 'Utah Jazz': 'utah', 'Washington Wizards': 'wsh'}

def get_logo_html(team_name, height=40):
    for nombre, abbr in logos.items():
        if nombre in team_name or team_name in nombre:
            return f'<img src="https://a.espncdn.com/i/teamlogos/nba/500/{abbr}.png" height="{height}" style="vertical-align: middle; margin-right: 10px;">'
    return ""

def get_ultimos_totales(team_name, n=5):
    mask = (df['home_team'] == team_name) | (df['away_team'] == team_name)
    return df[mask].sort_values('date').tail(n).copy()[['date', 'pred_total_pts']].set_index('date')

# --- BARRA LATERAL CON MÉTRICAS VISIBLES ---
st.sidebar.header("Filtros")
linea_apuestas = st.sidebar.number_input("Línea Total Puntos", value=220.5, step=0.5)
linea_spread = st.sidebar.number_input("Línea Spread (Local)", value=-4.5, step=0.5)
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Métricas del Modelo")
st.sidebar.markdown(f"**Precisión Histórica:** ~64.7%")
st.sidebar.markdown(f"**Error Promedio (MAE):** 12.5 pts")
st.sidebar.markdown(f"**Sigma Base:** {sigma_base:.2f}")
bankroll = st.sidebar.number_input("Tu Bankroll ($)", value=1000, step=100)
st.sidebar.markdown("---")

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
        
        # 1. IMPACTO ON/OFF REAL
        net_perdido_local = 0
        for _, inj in inj_local.iterrows():
            if inj['current_status'] in ['Out', 'Day-To-Day']:
                net = net_dict.get(inj['player_name'], 0.0)
                factor_duda = 0.5 if inj['current_status'] == 'Day-To-Day' else 1.0
                net_perdido_local += (net * factor_duda)
        
        net_perdido_visita = 0
        for _, inj in inj_visita.iterrows():
            if inj['current_status'] in ['Out', 'Day-To-Day']:
                net = net_dict.get(inj['player_name'], 0.0)
                factor_duda = 0.5 if inj['current_status'] == 'Day-To-Day' else 1.0
                net_perdido_visita += (net * factor_duda)
        
        penalty_local = max(0, net_perdido_local * 0.8)
        penalty_visita = max(0, net_perdido_visita * 0.8)
        
        ajuste_prob = (penalty_visita - penalty_local) * 0.015 
        p_home_ajustada = max(0.05, min(0.95, row['p_home'] + ajuste_prob))
        p_away_ajustada = 1 - p_home_ajustada
        
        # 2. AJUSTE DEFENSIVO
        def_local = def_dict.get(eq_local, 110.0)
        def_visita = def_dict.get(eq_visita, 110.0)
        def_total = def_local + def_visita
        
        if def_total > 225:
            defensa_estado = "🔥 Defensas Débiles (Favorece Over)"
            color_def = "red"
            ajuste_defensa = 3.0
        elif def_total < 215:
            defensa_estado = "🛡️ Defensas Fuertes (Favorece Under)"
            color_def = "blue"
            ajuste_defensa = -3.0
        else:
            defensa_estado = "⚖️ Defensas Promedio"
            color_def = "gray"
            ajuste_defensa = 0.0
        
        # 3. VARIANZA (CONSISTENCIA)
        std_local = var_dict.get(eq_local, 15.0)
        std_visita = var_dict.get(eq_visita, 15.0)
        std_promedio = (std_local + std_visita) / 2
        
        # Si la varianza es alta, el Sigma sube (menos confianza)
        sigma_partido = sigma_base * (std_promedio / 15.0) 
        sigma_partido = max(sigma_base, min(sigma_base * 1.5, sigma_partido)) # Limitar el ajuste
        
        pred_total = row['pred_total_pts'] + ajuste_defensa - (penalty_local + penalty_visita)
        
        prob_over = 1 - norm.cdf(linea_apuestas, loc=pred_total, scale=sigma_partido)
        prob_under = 1 - prob_over
        
        margen = (p_home_ajustada - 0.5) * 20
        pts_local = (pred_total / 2) + (margen / 2)
        pts_visita = (pred_total / 2) - (margen / 2)
        
        prob_local_cubre = 1 - norm.cdf(linea_spread, loc=margen, scale=sigma_partido * 0.8)
        prob_visita_cubre = 1 - prob_local_cubre
        
        # 4. KELLY CRITERION
        # Asumimos una cuota estándar de -110 (1.90) para el cálculo del Kelly
        cuota_decimal = 1.90
        b = cuota_decimal - 1
        
        # Kelly para Over
        p_over = prob_over
        q_over = 1 - p_over
        kelly_over = max(0, (b * p_over - q_over) / b)
        
        # Kelly para Under
        p_under = prob_under
        q_under = 1 - p_under
        kelly_under = max(0, (b * p_under - q_under) / b)
        
        # Half-Kelly (más conservador)
        kelly_final = max(kelly_over, kelly_under) / 2
        apuesta_sugerida = bankroll * kelly_final
        
        # SEMÁFORO DE CONFIANZA
        prob_max = max(p_home_ajustada, p_away_ajustada)
        dif_puntos = abs(pts_local - pts_visita)
        
        if prob_max >= 0.65 or dif_puntos >= 8.0:
            estado = "🟢 ALTA CONFIANZA"
            color = "green"
        elif prob_max >= 0.55:
            estado = "🟡 CONFIANZA MEDIA"
            color = "orange"
        else:
            estado = "🔴 PARTIDO PAREJO"
            color = "red"
        
        with st.container():
            col1, col2, col3 = st.columns([2, 2, 2])
            
            with col1:
                st.markdown(f"<div style='background-color:{color}; padding:5px; border-radius:5px; color:white; text-align:center; font-weight:bold; margin-bottom:10px;'>{estado}</div>", unsafe_allow_html=True)
                
                st.markdown(f"<div style='display:flex; align-items:center;'>{get_logo_html(row['away_team'])} <b>{row['away_team']}</b></div>", unsafe_allow_html=True)
                st.markdown("<div style='margin-left: 20px; color:gray;'>@</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='display:flex; align-items:center;'>{get_logo_html(row['home_team'])} <b>{row['home_team']}</b></div>", unsafe_allow_html=True)
                st.caption(f"Game ID: {row['gameId']}")
                
                st.markdown("---")
                st.markdown("**Historial Directo (Últimos 10)**")
                if row['h2h_played'] > 0:
                    st.write(f"⚔️ {row['home_team']}: **{row['h2h_home_wins']}** victorias")
                    st.write(f"⚔️ {row['away_team']}: **{row['h2h_away_wins']}** victorias")
                else:
                    st.write("No hay historial reciente.")
                
                st.markdown("---")
                st.markdown("**Lesiones**")
                if len(inj_local) > 0:
                    st.caption(f"🏠 {eq_local}:")
                    for _, inj in inj_local.iterrows(): st.caption(f"- {inj['player_name']} ({inj['current_status']})")
                if len(inj_visita) > 0:
                    st.caption(f"✈️ {eq_visita}:")
                    for _, inj in inj_visita.iterrows(): st.caption(f"- {inj['player_name']} ({inj['current_status']})")
                
            with col2:
                st.markdown("**Ganador (Ajustado por On/Off)**")
                st.progress(float(p_home_ajustada))
                st.write(f"🏠 {eq_local}: **{p_home_ajustada:.1%}**")
                st.write(f"✈️ {eq_visita}: **{p_away_ajustada:.1%}**")
                if abs(ajuste_prob) > 0.005: st.caption(f"*(Base: {row['p_home']:.1%})*")
                
                st.markdown("---")
                st.markdown(f"**Spread ({linea_spread})**")
                if linea_spread > 0:
                    st.write(f"🏠 {eq_local} (+{linea_spread}): **{prob_local_cubre:.1%}**")
                    st.write(f"✈️ {eq_visita} (-{linea_spread}): **{prob_visita_cubre:.1%}**")
                else:
                    st.write(f"🏠 {eq_local} ({linea_spread}): **{prob_local_cubre:.1%}**")
                    st.write(f"✈️ {eq_visita} (+{abs(linea_spread)}): **{prob_visita_cubre:.1%}**")
                
            with col3:
                st.markdown(f"**Totales (Línea: {linea_apuestas})**")
                st.markdown(f"<div style='background-color:{color_def}; padding:5px; border-radius:5px; color:white; text-align:center; font-weight:bold; margin-bottom:10px;'>{defensa_estado}</div>", unsafe_allow_html=True)
                
                st.write(f"📊 Pred: **{pred_total:.1f} pts**")
                st.write(f"⬆️ Over: **{prob_over:.1%}**")
                st.write(f"⬇️ Under: **{prob_under:.1%}**")
                
                if prob_over > 0.60: 
                    st.success("🔥 VALUE: OVER")
                    if kelly_over > 0:
                        st.info(f"💰 Apuesta sugerida (Half-Kelly): **${apuesta_sugerida:.2f}**")
                elif prob_under > 0.60: 
                    st.success("🔥 VALUE: UNDER")
                    if kelly_under > 0:
                        st.info(f"💰 Apuesta sugerida (Half-Kelly): **${apuesta_sugerida:.2f}**")
                
                st.markdown("---")
                st.markdown("**Puntos por Equipo**")
                st.write(f"🏠 {eq_local}: **{pts_local:.1f}**")
                st.write(f"✈️ {eq_visita}: **{pts_visita:.1f}**")
            
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

st.sidebar.markdown("**Modelo v14:** On/Off, Varianza, Kelly y Métricas.")
