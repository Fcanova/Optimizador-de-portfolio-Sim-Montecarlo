Tenés toda la razón, Franco. Una frontera eficiente sin la línea que une los puntos es solo una nube de puntos. Para que se vea profesional, tenemos que trazar la curva técnica que une los portafolios óptimos para cada nivel de riesgo.

Para lograr esto, vamos a iterar sobre diferentes niveles de retorno objetivo y hallar la varianza mínima para cada uno. Esto dibujará la "envolvente" superior de la nube de puntos.

Aquí tenés el código final con la línea de la frontera trazada correctamente:

Python

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from pypfopt.efficient_frontier import EfficientFrontier
from scipy.stats import t as t_dist

# --- 1. TASA LIBRE DE RIESGO ---
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

# --- 3. OPTIMIZADOR ---
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
    z_score = 1.645
    peor_resultado_pct = ret_p - (z_score * vol_p)
    
    return {
        "pesos": weights, "retorno_esperado": ret_p, "volatilidad_esperada": vol_p, 
        "sharpe_ratio": sharpe_p, "peor_resultado_pct": peor_resultado_pct,
        "resultado_monetario_peor_caso": capital * peor_resultado_pct,
        "ganancia_esperada_monetaria": ret_p * capital,
        "capital_final_peor_caso": capital * (1 + peor_resultado_pct)
    }

# --- 4. INTERFAZ ---
st.set_page_config(page_title="Financial Wealth Optimizer", layout="wide")
st.title("🚀 financial_wealth: Portfolio Intelligence")

with st.sidebar:
    st.header("⚙️ Configuración")
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
    with st.spinner("Trazando frontera eficiente..."):
        rf = obtener_risk_free_live()
        df = yf.download(tickers, start=f_inicio, end=f_fin)
        data = df['Adj Close'] if 'Adj Close' in df.columns else df['Close']
        returns_h = np.log(data / data.shift(1)).dropna()
        mu_sim, cov_sim, sims = generar_simulacion_profesional(returns_h, 2000, dist_modelo)
        res = optimizar_portfolio(mu_sim, cov_sim, rf, tickers, obj_input, 0.05 if restr_w else None, cap_inicial)

        if res:
            # MÉTRICAS Y MONETARIAS (IGUAL QUE ANTES)
            st.subheader("📊 Métricas de Eficiencia y Capital")
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                m1, m2, m3 = st.columns(3)
                m1.metric("Retorno Esperado", f"{res['retorno_esperado']:.2%}")
                m2.metric("Volatilidad Anual", f"{res['volatilidad_esperada']:.2%}")
                m3.metric("Ratio de Sharpe", f"{res['sharpe_ratio']:.2f}")
            with col_m2:
                c1, c2, c3 = st.columns(3)
                c1.metric("Ganancia Esperada", f"+ ${res['ganancia_esperada_monetaria']:,.2f}")
                c2.metric("VaR 95% Confianza", f"{res['peor_resultado_pct']:.2%}", help="Probabilidad de pérdida máxima estimada.")
                c3.metric("Capital Remanente", f"${res['capital_final_peor_caso']:,.2f}")

            st.divider()
            
            # --- GRÁFICO DE FRONTERA EFICIENTE CON LÍNEA ---
            col_g1, col_g2 = st.columns([2, 1])
            
            with col_g1:
                st.write("### Frontera Eficiente de Markowitz")
                
                # 1. Generar nube de puntos aleatorios
                n_portfolios = 1000
                p_ret, p_vol = [], []
                for _ in range(n_portfolios):
                    w = np.random.random(len(tickers))
                    w /= np.sum(w)
                    p_ret.append(np.dot(w, mu_sim))
                    p_vol.append(np.sqrt(np.dot(w.T, np.dot(cov_sim, w))))
                
                # 2. Generar la línea de la Frontera (Curva)
                target_rets = np.linspace(min(mu_sim), max(mu_sim), 30)
                frontier_vol = []
                for r in target_rets:
                    ef_line = EfficientFrontier(pd.Series(mu_sim, index=tickers), pd.DataFrame(cov_sim, index=tickers, columns=tickers))
                    try:
                        ef_line.efficient_return(r)
                        frontier_vol.append(ef_line.portfolio_performance()[1])
                    except:
                        frontier_vol.append(None)
                
                fig_fe, ax_fe = plt.subplots(figsize=(10, 6))
                # Nube
                ax_fe.scatter(p_vol, p_ret, c=(np.array(p_ret)/np.array(p_vol)), marker='o', s=5, alpha=0.2, cmap='viridis')
                # Línea de la Frontera
                valid_line = [(v, r) for v, r in zip(frontier_vol, target_rets) if v is not None]
                if valid_line:
                    v_line, r_line = zip(*valid_line)
                    ax_fe.plot(v_line, r_line, color='black', linestyle='--', linewidth=2, label='Frontera Eficiente')
                
                # Activos e Individuales
                ax_fe.scatter(np.sqrt(np.diag(cov_sim)), mu_sim, color='red', marker='X', s=100, label='Activos')
                ax_fe.scatter(res['volatilidad_esperada'], res['retorno_esperado'], color='gold', marker='*', s=300, label='Tu Portfolio', edgecolor='black')
                
                ax_fe.set_xlabel("Riesgo (Volatilidad)")
                ax_fe.set_ylabel("Retorno Esperado")
                ax_fe.legend()
                st.pyplot(fig_fe)

            with col_g2:
                st.write("### Composición")
                fig_pie, ax_pie = plt.subplots()
                pesos_plot = {k: v for k, v in res['pesos'].items() if v > 0.001}
                ax_pie.pie(pesos_plot.values(), labels=pesos_plot.keys(), autopct='%1.1f%%', startangle=140, colors=sns.color_palette("viridis", len(pesos_plot)))
                st.pyplot(fig_pie)
