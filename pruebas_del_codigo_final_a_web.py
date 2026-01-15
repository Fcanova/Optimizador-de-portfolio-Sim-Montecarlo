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
    rets_daily_sim = S[1:] / S[:-1] - 1
    rets_flat = rets_daily_sim.reshape(-1, n_assets)
    cov_sim_annual = np.cov(rets_flat, rowvar=False) * 252
    return mu_sim_annual, cov_sim_annual, final_returns

# --- 3. OPTIMIZADOR Y LÓGICA DE RIESGO ---
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
    
    # Riesgo Neto real
    z_score = 1.645
    peor_resultado_pct = ret_p - (z_score * vol_p)
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
st.set_page_config(page_title="Equity Optimizer Pro", layout="wide")
st.title("🚀 financial_wealth: Portfolio Intelligence")

with st.sidebar:
    st.header("⚙️ Parámetros")
    cap_inicial = st.number_input("Capital a Invertir ($)", min_value=100.0, value=10000.0)
    tickers_str = st.text_input("Tickers", "AAPL, MSFT, NVDA, GGAL, MELI, GLD")
    tickers = [t.strip().upper() for t in tickers_str.split(",")]
    col1, col2 = st.columns(2)
    with col1: f_inicio = st.date_input("Inicio", value=pd.to_datetime("2021-01-01"))
    with col2: f_fin = st.date_input("Fin", value=pd.to_datetime("today"))
    dist_modelo = st.selectbox("Modelo", ["MBG", "T-Student", "T-Skewed"])
    obj_input = st.radio("Objetivo", ["Max Sharpe Ratio", "Min Volatility"])
    restr_w = st.checkbox("Mínimo 5% por activo", value=True)

if st.button("Simular y Analizar"):
    with st.spinner("Ejecutando simulación de escenarios..."):
        res, sims = ejecutar_analisis_portfolio(tickers, f_inicio, f_fin, 2000, dist_modelo, obj_input, 0.05 if restr_w else None, cap_inicial)
        if res:
            st.success("✅ Análisis Completo")
            
            # FILA 1: MÉTRICAS PORCENTUALES
            st.subheader("📊 Métricas de Eficiencia (Anualizadas)")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Retorno Esperado", f"{res['retorno_esperado']:.2%}", 
                      help="Promedio ponderado de los retornos anuales esperados según la simulación.")
            m2.metric("Volatilidad Anual", f"{res['volatilidad_esperada']:.2%}", 
                      help="Desviación estándar de los retornos. Mide la variabilidad o riesgo de mercado.")
            m3.metric("Ratio de Sharpe", f"{res['sharpe_ratio']:.2f}", 
                      help="Relación entre el exceso de retorno y la volatilidad. Cuanto más alto, mejor es la eficiencia del riesgo.")
            m4.metric("VaR 95% Confianza", f"{res['peor_resultado_pct']:.2%}", 
                      help="Con un 95% de probabilidad perderías de manera estimada, como máximo esto.")

            # FILA 2: MÉTRICAS MONETARIAS
            st.subheader(f"💵 Proyección de Capital (${cap_inicial:,.0f})")
            c1, c2, c3 = st.columns(3)
            c1.metric("Ganancia Esperada", f"+ ${res['ganancia_esperada_monetaria']:,.2f}", 
                      help="Resultado monetario estimado en un escenario central (promedio).")
            
            color_delta = "inverse" if res['resultado_monetario_peor_caso'] < 0 else "normal"
            c2.metric("Resultado Neto Peor Caso", f"${res['resultado_monetario_peor_caso']:,.2f}", 
                      delta="Pérdida Estimada" if res['resultado_monetario_peor_caso'] < 0 else "Ganancia Mínima", 
                      delta_color=color_delta,
                      help="Monto en dólares que representa el peor escenario proyectado al 95% de confianza.")
            
            c3.metric("Capital Remanente", f"${res['capital_final_peor_caso']:,.2f}", 
                      help="Capital remanente en caso de que se efectivice la pérdida máxima esperada con un 95% de probabilidad.")

            # FILA 3: GRÁFICOS
            st.divider()
            g1, g2, g3 = st.columns([1, 1, 1])
            
            with g1:
                st.write("### Composición Óptima")
                fig_pie, ax_pie = plt.subplots()
                pesos_plot = {k: v for k, v in res['pesos'].items() if v > 0.001}
                ax_pie.pie(pesos_plot.values(), labels=pesos_plot.keys(), autopct='%1.1f%%', startangle=140, colors=sns.color_palette("viridis", len(pesos_plot)))
                st.pyplot(fig_pie)

            with g2:
                st.write("### Potencial: Éxito vs Riesgo")
                fig_bar, ax_bar = plt.subplots()
                labels = ['Ganancia Esp.', 'Peor Caso']
                valores = [res['ganancia_esperada_monetaria'], res['resultado_monetario_peor_caso']]
                ax_bar.bar(labels, valores, color=['#2ECC71', '#E74C3C'])
                ax_bar.axhline(0, color='black', linewidth=0.8)
                st.pyplot(fig_bar)

            with g3:
                st.write("### Distribución de Resultados")
                fig_hist, ax_hist = plt.subplots()
                pesos_arr = np.array(list(res['pesos'].values()))
                rets_monetarios = (sims @ pesos_arr) * cap_inicial
                sns.histplot(rets_monetarios, kde=True, ax=ax_hist, color="#3498DB")
                ax_hist.axvline(res['resultado_monetario_peor_caso'], color='red', linestyle='--')
                st.pyplot(fig_hist)
            
            st.subheader("📋 Tabla de Ponderaciones")
            st.table(pd.DataFrame.from_dict(res['pesos'], orient='index', columns=['%']).multiply(100).round(2))
