import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from scipy.stats import t as t_dist

# --- 0. CONFIGURACIÓN ---
st.set_page_config(page_title="Financial Wealth Optimizer Pro", layout="wide")

st.markdown("""
    <style>
    th { text-align: center !important; font-weight: bold !important; text-transform: uppercase; color: #1E88E5; }
    td { text-align: center !important; }
    .stMetric { background-color: #f8f9fb; padding: 15px; border-radius: 10px; border-left: 5px solid #1E88E5; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. TASA LIBRE DE RIESGO ---
def obtener_risk_free_live():
    try:
        tnx = yf.Ticker("^TNX")
        data = tnx.history(period="1d")
        if data.empty: return 0.042
        return data['Close'].iloc[-1] / 100
    except:
        return 0.042

# --- 2. MOTOR DE SIMULACIÓN (PASOS 1, 2 Y 3) ---
def generar_simulacion_diaria(mu_h_anual, cov_h_anual, n_sims, dist_type):
    n_assets = len(mu_h_anual)
    N_steps = 252 
    
    # Inputs diarios
    mu_d = mu_h_anual / 252
    cov_d = cov_h_anual / 252
    sigmas_d = np.sqrt(np.diag(cov_d))
    
    L = np.linalg.cholesky(cov_d + 1e-12 * np.eye(n_assets))
    
    retornos_diarios_sim = np.zeros((N_steps, n_sims, n_assets))
    
    # Drift MBG: mu - 0.5 * sigma^2 (Corrección Hull)
    drift_diario = mu_d - 0.5 * (sigmas_d**2)
    
    nu, gamma = 5, 1.3 
    for t in range(N_steps):
        if dist_type == 'MBG':
            z = np.random.standard_normal((n_assets, n_sims))
        elif dist_type == 'T-Student':
            z = t_dist.rvs(df=nu, size=(n_assets, n_sims))
            z = z / np.sqrt(nu / (nu - 2))
        elif dist_type == 'T-Skewed':
            Y = t_dist.rvs(df=nu, size=(n_assets, n_sims))
            Z_raw = np.where(Y >= 0, Y / gamma, Y * gamma)
            z = (Z_raw - Z_raw.mean(axis=1, keepdims=True)) / Z_raw.std(axis=1, keepdims=True)
        
        shock = L @ z
        retornos_diarios_sim[t] = drift_diario + shock.T
        
    # --- PASO 3: EXTRACCIÓN DE MÉTRICAS SIMULADAS ---
    # Promedio aritmético diario de CADA ACTIVO (Promediando todos los días y todos los escenarios)
    mu_sim_d = retornos_diarios_sim.mean(axis=(0, 1))
    
    # ANUALIZACIÓN DIRECTA (Como pediste: Promedio Diario * 252)
    mu_sim_anual = mu_sim_d * 252
    
    # COVARIANZA ANUALIZADA (std diario * sqrt(252))
    reshaped_rets = retornos_diarios_sim.reshape(-1, n_assets)
    cov_sim_anual = np.cov(reshaped_rets, rowvar=False) * 252
    
    return mu_sim_anual, cov_sim_anual, retornos_diarios_sim

# --- 3. OPTIMIZADOR ---
def optimizar_portfolio(mu_sim, cov_sim, rets_diarios, rf_rate, asset_names, objetivo, min_weight, capital):
    # Aquí alimentamos al optimizador exactamente con el promedio anualizado de la simulada
    mu_s = pd.Series(mu_sim, index=asset_names)
    cov_s = pd.DataFrame(cov_sim, index=asset_names, columns=asset_names)
    bounds = (min_weight if min_weight else 0.0, 1.0)
    
    ef = EfficientFrontier(mu_s, cov_s, weight_bounds=bounds)
    try:
        if "Sharpe" in objetivo:
            ef.max_sharpe(risk_free_rate=rf_rate)
        else:
            ef.min_volatility()
        weights = ef.clean_weights()
    except:
        ef.min_volatility()
        weights = ef.clean_weights()
    
    ret_p, vol_p, _ = ef.portfolio_performance(risk_free_rate=rf_rate)
    
    # Proyección Monetaria Final (Basada en capitalización real)
    pesos_arr = np.array([weights[t] for t in asset_names])
    ret_acum_p = np.exp(rets_diarios.sum(axis=0) @ pesos_arr) - 1
    mu_p_wealth = ret_acum_p.mean()
    vaR_pct = np.percentile(ret_acum_p, 5)
    
    return {
        "pesos": weights, "retorno_esperado": ret_p, "volatilidad_esperada": vol_p, 
        "vaR_pct": vaR_pct, "ganancia_esperada_monetaria": mu_p_wealth * capital,
        "resultado_monetario_peor_caso": capital * vaR_pct,
        "capital_final_peor_caso": capital * (1 + vaR_pct),
        "capital_potencial": capital * (1 + mu_p_wealth),
        "ret_acum_p": ret_acum_p
    }

# --- 4. INTERFAZ ---
st.title("🚀 financial_wealth: Portfolio Intelligence")

with st.sidebar:
    st.header("⚙️ Parámetros")
    cap_inicial = st.number_input("Capital ($)", min_value=100.0, value=10000.0)
    n_simulaciones = st.slider("Simulaciones", 1000, 10000, 2000)
    tickers_str = st.text_input("Tickers", "AAPL, MSFT, NVDA, GGAL, MELI, TSLA, AMZN")
    tickers = [t.strip().upper() for t in tickers_str.split(",")]
    col_a, col_b = st.columns(2)
    with col_a: f_inicio = st.date_input("Inicio", pd.to_datetime("2021-01-01"))
    with col_b: f_fin = st.date_input("Fin", pd.to_datetime("today"))
    dist_modelo = st.selectbox("Modelo", ["MBG", "T-Student", "T-Skewed"])
    obj_input = st.radio("Objetivo", ["Max Sharpe Ratio", "Min Volatility"])
    restr_w = st.checkbox("Mínimo 5%", value=True)

if st.button("Simular y Analizar"):
    with st.spinner("Sincronizando retornos diarios simulados..."):
        rf = obtener_risk_free_live()
        raw_df = yf.download(tickers, start=f_inicio, end=f_fin)
        if raw_df.empty: st.stop()
        
        # Extracción de datos
        if 'Adj Close' in raw_df.columns: data = raw_df['Adj Close']
        elif 'Close' in raw_df.columns: data = raw_df['Close']
        else: data = raw_df.xs('Adj Close', axis=1, level=0) if 'Adj Close' in raw_df.columns.levels[0] else raw_df.xs('Close', axis=1, level=0)

        data = data.ffill().dropna()
        log_returns = np.log(data / data.shift(1)).dropna()
        mu_h = log_returns.mean() * 252
        cov_h = risk_models.sample_cov(data)
        
        final_tickers = data.columns.tolist()

        # SIMULACIÓN (PASOS 1, 2 Y 3)
        mu_sim, cov_sim, rets_diarios = generar_simulacion_diaria(mu_h.values, cov_h.values, n_simulaciones, dist_modelo)
        
        # OPTIMIZACIÓN (PASO 4)
        res = optimizar_portfolio(mu_sim, cov_sim, rets_diarios, rf, final_tickers, obj_input, 0.05 if restr_w else None, cap_inicial)

        if res:
            st.success("✅ Datos Sincronizados Correctamente")
            
            # FILA 1: MÉTRICAS
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Retorno Esperado", f"{res['retorno_esperado']:.2%}")
            m2.metric("Volatilidad Anual", f"{res['volatilidad_esperada']:.2%}")
            m3.metric("Ratio de Sharpe", f"{(res['retorno_esperado']-rf)/res['volatilidad_esperada']:.2f}")
            m4.metric("VaR 95% (Anual)", f"{res['vaR_pct']:.2%}")

            # FILA 2: AUDITORÍA (Aquí verás que coincide con el promedio diario * 252)
            st.subheader("🎯 Eficiencia Individual Simulada")
            vols_sim_ind = np.sqrt(np.diag(cov_sim))
            df_ind = pd.DataFrame({
                "Retorno Anual %": mu_sim * 100,
                "Volatilidad Anual %": vols_sim_ind * 100,
                "Sharpe": (mu_sim - rf) / vols_sim_ind
            }, index=final_tickers)
            st.table(df_ind.sort_values(by="Retorno Anual %", ascending=False).style.format("{:.2f}"))

            st.divider()

            # FILA 3: FRONTERA EFICIENTE (SINCRONIZADA)
            st.subheader("📈 Frontera Eficiente de Markowitz")
            col_fe, col_pie = st.columns([2, 1])
            with col_fe:
                n_port = 1000
                p_r, p_v = [], []
                for _ in range(n_port):
                    w = np.random.random(len(final_tickers)); w /= np.sum(w)
                    p_r.append(np.dot(w, mu_sim))
                    p_v.append(np.sqrt(np.dot(w.T, np.dot(cov_sim, w))))
                
                target_rets = np.linspace(min(mu_sim), max(mu_sim), 30)
                frontier_v = []
                for r in target_rets:
                    ef_line = EfficientFrontier(pd.Series(mu_sim, index=final_tickers), pd.DataFrame(cov_sim, index=final_tickers, columns=final_tickers))
                    try:
                        ef_line.efficient_return(r)
                        frontier_v.append(ef_line.portfolio_performance()[1])
                    except: frontier_v.append(None)

                fig_fe, ax_fe = plt.subplots(figsize=(10, 6))
                ax_fe.scatter(p_v, p_r, c=(np.array(p_r)/np.array(p_v)), marker='o', s=10, alpha=0.3, cmap='viridis')
                
                valid_v = [v for v in frontier_v if v is not None]; valid_r = [r for v, r in zip(frontier_v, target_rets) if v is not None]
                ax_fe.plot(valid_v, valid_r, color='black', linestyle='--', linewidth=2)

                for i, t in enumerate(final_tickers):
                    ax_fe.scatter(vols_sim_ind[i], mu_sim[i], color='red', marker='X', s=100)
                    ax_fe.annotate(t, (vols_sim_ind[i], mu_sim[i]), xytext=(5,5), textcoords='offset points', fontweight='bold')
                
                ax_fe.scatter(res['volatilidad_esperada'], res['retorno_esperado'], color='gold', marker='*', s=400, edgecolor='black')
                st.pyplot(fig_fe)

            with col_pie:
                fig_pie, ax_pie = plt.subplots()
                pesos_plot = {k: v for k, v in res['pesos'].items() if v > 0.001}
                ax_pie.pie(pesos_plot.values(), labels=pesos_plot.keys(), autopct='%1.1f%%', startangle=140, colors=sns.color_palette("viridis", len(pesos_plot)))
                st.pyplot(fig_pie)
