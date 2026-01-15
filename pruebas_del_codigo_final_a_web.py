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

# --- 2. MOTOR DE SIMULACIÓN (MONTE CARLO) ---
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
    rets_daily_sim = S[1:] / S[:-1] - 1
    rets_flat = rets_daily_sim.reshape(-1, n_assets)
    cov_sim_annual = np.cov(rets_flat, rowvar=False) * 252
    return mu_sim_annual, cov_sim_annual, final_returns

# --- 3. OPTIMIZADOR Y LÓGICA DE PÉRDIDA MÁXIMA ---
def optimizar_portfolio(mu_sim, cov_sim, rf_rate, asset_names, objetivo, min_weight, capital):
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
    
    # --- MÉTRICA DE RIESGO DE PÉRDIDA MÁXIMA ---
    # Usamos el VaR Absoluto (incluyendo retorno) para ver el resultado neto real.
    z_score = 1.645 # 95% Confianza
    peor_resultado_pct = ret_p - (z_score * vol_p)
    
    # Calculamos cuánto se gana o se pierde en dinero real respecto al capital inicial
    # Si peor_resultado_pct es negativo, hay pérdida de capital.
    resultado_monetario_peor_caso = capital * peor_resultado_pct
    
    return {
        "pesos": weights, 
        "retorno_esperado": ret_p, 
        "volatilidad_esperada": vol_p, 
        "sharpe_ratio": sharpe_p, 
        "peor_resultado_pct": peor_resultado_pct,
        "resultado_monetario_peor_caso": resultado_monetario_peor_caso,
        "ganancia_esperada_monetaria": ret_p * capital,
        "capital_final_peor_caso": capital + resultado_monetario_peor_caso
    }

# --- 4. INTEGRADOR ---
def ejecutar_analisis_portfolio(tickers, f_inicio, f_fin, n_simulaciones, distribucion, objetivo, min_weight, capital):
    rf = obtener_risk_free_live()
    df = yf.download(tickers, start=f_inicio, end=f_fin)
    if df.empty or len(df) < 10: return None, None
    data = df['Adj Close'] if 'Adj Close' in df.columns else df['Close']
    returns_h = np.log(data / data.shift(1)).dropna()
    mu_sim, cov_sim, rets_f = generar_simulacion_profesional(returns_h, n_simulaciones, distribucion)
    res = optimizar_portfolio(mu_sim, cov_sim, rf, returns_h.columns.tolist(), objetivo, min_weight, capital)
    return res, rets_f

# --- 5. INTERFAZ STREAMLIT ---
st.set_page_config(page_title="Portfolio Risk Analyzer", layout="wide")
st.title("🚀 financial_wealth: Portfolio Intelligence")

with st.sidebar:
    st.header("⚙️ Parámetros")
    capital = st.number_input("Capital a Invertir ($)", min_value=100.0, value=10000.0, step=100.0)
    tickers_str = st.text_input("Tickers", "AAPL, MSFT, NVDA, GGAL, MELI, GLD")
    tickers = [t.strip().upper() for t in tickers_str.split(",")]
    col1, col2 = st.columns(2)
    with col1: f_inicio = st.date_input("Inicio", value=pd.to_datetime("2021-01-01"))
    with col2: f_fin = st.date_input("Fin", value=pd.to_datetime("today"))
    dist_modelo = st.selectbox("Modelo", ["MBG", "T-Student", "T-Skewed"])
    obj_input = st.radio("Objetivo", ["Max Sharpe Ratio", "Min Volatility"])
    restr_w = st.checkbox("Mínimo 5% por activo", value=True)

if st.button("Simular y Analizar Riesgos"):
    with st.spinner("Calculando proyecciones anuales..."):
        res, sims = ejecutar_analisis_portfolio(tickers, f_inicio, f_fin, 2000, dist_modelo, obj_input, 0.05 if restr_w else None, capital)
        if res:
            st.success("✅ Análisis Completo")
            
            # --- SECCIÓN MONETARIA ---
            st.subheader(f"💵 Proyección Anual Neta (Inversión: ${capital:,.0f})")
            c1, c2, c3 = st.columns(3)
            
            c1.metric("Ganancia Esperada", f"+ ${res['ganancia_esperada_monetaria']:,.2f}", 
                      help="Lo que ganarías en un año normal (promedio de simulaciones).")
            
            # Lógica de color para el peor caso
            color_var = "inverse" if res['resultado_monetario_peor_caso'] < 0 else "normal"
            etiqueta_peor = "Pérdida Máxima Estándar" if res['resultado_monetario_peor_caso'] < 0 else "Ganancia Mínima Estándar"
            
            c2.metric(etiqueta_peor, f"${res['resultado_monetario_peor_caso']:,.2f}", 
                      delta="Escenario 95% Confianza", delta_color=color_var)
            
            c3.metric("Capital Final (Peor Escenario)", f"${res['capital_final_peor_caso']:,.2f}", 
                      help="Dinero total que tendrías en el bolsillo tras un año de estrés financiero.")

            # --- GRÁFICOS ---
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                # Gráfico de Barras Ganancia vs Pérdida
                fig_bar, ax_bar = plt.subplots(figsize=(8, 5))
                labels = ['Ganancia Esperada', 'Peor Escenario (VaR)']
                valores = [res['ganancia_esperada_monetaria'], res['resultado_monetario_peor_caso']]
                colores = ['#2ECC71', '#E74C3C']
                ax_bar.bar(labels, valores, color=colores)
                ax_bar.axhline(0, color='black', linewidth=0.8)
                ax_bar.set_ylabel("Dólares ($)")
                ax_bar.set_title("Potencial Anual: Éxito vs Riesgo")
                st.pyplot(fig_bar)

            with col_chart2:
                # Histograma de retornos
                fig_hist, ax_hist = plt.subplots(figsize=(8, 5))
                pesos_arr = np.array(list(res['pesos'].values()))
                port_rets_monetarios = (sims @ pesos_arr) * capital
                sns.histplot(port_rets_monetarios, kde=True, ax=ax_hist, color="#3498DB")
                ax_hist.axvline(res['resultado_monetario_peor_caso'], color='red', linestyle='--', label="Límite Crítico (VaR)")
                ax_hist.set_title("Distribución de Resultados Monetarios Finales")
                ax_hist.set_xlabel("Ganancia / Pérdida en $")
                ax_hist.legend()
                st.pyplot(fig_hist)
            
            st.subheader("📋 Composición del Portfolio")
            st.table(pd.DataFrame.from_dict(res['pesos'], orient='index', columns=['%']).multiply(100).round(2))
