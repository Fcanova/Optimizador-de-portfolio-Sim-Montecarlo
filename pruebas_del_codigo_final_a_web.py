import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from pypfopt.efficient_frontier import EfficientFrontier
from scipy.stats import t as t_dist

# --- 1. TASA LIBRE DE RIESGO (LIVE) ---
def obtener_risk_free_live():
    try:
        tnx = yf.Ticker("^TNX")
        data = tnx.history(period="1d")
        if data.empty: return 0.042
        return data['Close'].iloc[-1] / 100
    except:
        return 0.042

# --- 2. MOTOR DE SIMULACIÓN ---
def generar_simulacion_profesional(returns_h, n_sims, dist_type):
    n_assets = returns_h.shape[1]
    mu_h = returns_h.mean().values * 252
    sigma_h = returns_h.cov().values * 252

    T_years = len(returns_h) / 252
    N_steps = int(T_years * 252)
    dt = 1 / 252

    L = np.linalg.cholesky(sigma_h + 1e-10 * np.eye(n_assets))
    var_diag = np.diag(sigma_h)
    drift = (mu_h - 0.5 * var_diag) * dt

    S = np.ones((N_steps + 1, n_sims, n_assets))
    nu, gamma = 5, 1.3

    for t in range(1, N_steps + 1):
        if dist_type == 'MBG':
            z = np.random.standard_normal((n_assets, n_sims))
        elif dist_type == 'T-Student':
            z = t_dist.rvs(df=nu, size=(n_assets, n_sims))
            z = z / np.sqrt(nu / (nu - 2))
        elif dist_type == 'T-Skewed':
            Y = t_dist.rvs(df=nu, size=(n_assets, n_sims))
            Z_raw = np.where(Y >= 0, Y / gamma, Y * gamma)
            z = (Z_raw - Z_raw.mean(axis=1, keepdims=True)) / Z_raw.std(axis=1, keepdims=True)

        corr_shock = L @ z
        incr = drift[:, None] + corr_shock * np.sqrt(dt)
        S[t] = S[t-1] * np.exp(incr.T)

    final_returns = S[-1] - 1
    mu_sim_annual = (1 + final_returns.mean(axis=0))**(1/T_years) - 1
    
    # Covarianza de los retornos diarios de la simulación
    rets_daily_sim = S[1:] / S[:-1] - 1
    rets_flat = rets_daily_sim.reshape(-1, n_assets)
    cov_sim_annual = np.cov(rets_flat, rowvar=False) * 252

    return mu_sim_annual, cov_sim_annual, final_returns

# --- 3. OPTIMIZADOR CON VaR RELATIVO ---
def optimizar_portfolio(mu_sim, cov_sim, rf_rate, asset_names, objetivo, min_weight):
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
        ef = EfficientFrontier(mu_s, cov_s, weight_bounds=(0, 1))
        ef.min_volatility()
        weights = ef.clean_weights()
    
    ret_p, vol_p, sharpe_p = ef.portfolio_performance(risk_free_rate=rf_rate)
    
    # --- VaR RELATIVO (Z * Volatilidad) ---
    z_score = 1.645
    var_95_anual = z_score * vol_p 
    
    return {
        "pesos": weights, 
        "retorno_esperado": ret_p, 
        "volatilidad_esperada": vol_p, 
        "sharpe_ratio": sharpe_p, 
        "var_95": var_95_anual
    }

# --- 4. INTEGRADOR ---
def ejecutar_analisis_portfolio(tickers, f_inicio, f_fin, n_simulaciones, distribucion, objetivo, min_weight):
    rf = obtener_risk_free_live()
    df = yf.download(tickers, start=f_inicio, end=f_fin)
    if df.empty or len(df) < 10: return None, None
    data = df['Adj Close'] if 'Adj Close' in df.columns else df['Close']
    returns_h = np.log(data / data.shift(1)).dropna()
    
    mu_sim, cov_sim, rets_f = generar_simulacion_profesional(returns_h, n_simulaciones, distribucion)
    res = optimizar_portfolio(mu_sim, cov_sim, rf, returns_h.columns.tolist(), objetivo, min_weight)
    return res, rets_f

# --- 5. INTERFAZ ---
st.set_page_config(page_title="Equity Optimizer Pro", layout="wide")
st.title("🚀 financial_wealth: Portfolio Intelligence")

with st.sidebar:
    st.header("⚙️ Configuración")
    tickers_str = st.text_input("Tickers", "AAPL, MSFT, NVDA, GGAL, MELI, GLD")
    tickers = [t.strip().upper() for t in tickers_str.split(",")]
    col1, col2 = st.columns(2)
    with col1: f_inicio = st.date_input("Inicio", value=pd.to_datetime("2021-01-01"))
    with col2: f_fin = st.date_input("Fin", value=pd.to_datetime("today"))
    dist_modelo = st.selectbox("Modelo", ["MBG", "T-Student", "T-Skewed"])
    obj_input = st.radio("Objetivo", ["Max Sharpe Ratio", "Min Volatility"])
    restr_w = st.checkbox("Mínimo 5% por activo", value=True)

if st.button("Simular y Optimizar"):
    with st.spinner("Ejecutando simulación de Monte Carlo..."):
        res, sims = ejecutar_analisis_portfolio(tickers, f_inicio, f_fin, 2000, dist_modelo, obj_input, 0.05 if restr_w else None)
        if res:
            st.success("✅ Análisis completado")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Retorno Esperado (Anual)", f"{res['retorno_esperado']:.2%}")
            m2.metric("Volatilidad (Anual)", f"{res['volatilidad_esperada']:.2%}")
            m3.metric("Ratio de Sharpe", f"{res['sharpe_ratio']:.2f}")
            m4.metric("VaR 95% (Relativo)", f"{res['var_95']:.2%}", help="Indica cuánto puede caer el retorno respecto al promedio esperado.")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            pesos_plot = {k: v for k, v in res['pesos'].items() if v > 0.001}
            ax1.pie(pesos_plot.values(), labels=pesos_plot.keys(), autopct='%1.1f%%', startangle=140, colors=sns.color_palette("viridis", len(pesos_plot)))
            ax1.set_title("Composición Óptima")
            
            pesos_arr = np.array(list(res['pesos'].values()))
            port_rets = sims @ pesos_arr
            sns.histplot(port_rets, kde=True, ax=ax2, color="#2E86C1")
            ax2.axvline(np.percentile(port_rets, 5), color='red', linestyle='--', label="Peor escenario (5%)")
            ax2.set_title("Distribución de Escenarios")
            ax2.legend()
            st.pyplot(fig)
            
            st.subheader("📋 Ponderación Detallada")
            st.table(pd.DataFrame.from_dict(res['pesos'], orient='index', columns=['%']).multiply(100).round(2))
