import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from pypfopt.efficient_frontier import EfficientFrontier
from scipy.stats import t as t_dist

# --- 0. CONFIGURACIÓN ---
st.set_page_config(page_title="Financial Wealth Optimizer Pro", layout="wide")

st.markdown("""
    <style>
    th { text-align: center !important; font-weight: bold !important; text-transform: uppercase; }
    td { text-align: center !important; }
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

# --- 2. MOTOR DE SIMULACIÓN ---
def generar_simulacion_profesional(returns_h, n_sims, dist_type):
    n_assets = returns_h.shape[1]
    # Anualización base
    mu_annual = returns_h.mean().values * 252
    sigma_annual = returns_h.cov().values * 252
    
    N_steps = 252
    dt = 1 / 252
    L = np.linalg.cholesky(sigma_annual + 1e-10 * np.eye(n_assets))
    
    drift = (mu_annual - 0.5 * np.diag(sigma_annual)) * dt
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
        
        shock = L @ z
        # shock ya viene escalado por la covarianza anualizada, ajustamos por sqrt(dt)
        incr = drift[:, None] + shock * np.sqrt(dt)
        S[t] = S[t-1] * np.exp(incr.T)
        
    final_returns = S[-1] - 1
    mu_sim = final_returns.mean(axis=0)
    cov_sim = np.cov(final_returns, rowvar=False)
    return mu_sim, cov_sim, final_returns

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
        ef.min_volatility()
        weights = ef.clean_weights()
    
    ret_p, vol_p, _ = ef.portfolio_performance(risk_free_rate=rf_rate)
    
    # VaR 95% corregido: Z=1.645
    va_r_95 = ret_p - (1.645 * vol_p)
    
    return {
        "pesos": weights, "retorno_esperado": ret_p, "volatilidad_esperada": vol_p, 
        "vaR_pct": va_r_95, "ganancia_esperada_monetaria": ret_p * capital,
        "resultado_monetario_peor_caso": capital * va_r_95,
        "capital_final_peor_caso": capital * (1 + va_r_95),
        "capital_potencial": capital * (1 + ret_p)
    }

# --- 4. INTERFAZ ---
st.title("🚀 financial_wealth: Portfolio Intelligence")

with st.sidebar:
    st.header("⚙️ Parámetros")
    cap_inicial = st.number_input("Capital a Invertir ($)", min_value=100.0, value=10000.0)
    n_simulaciones = st.slider("Simulaciones Monte Carlo", 1000, 10000, 2000, 500)
    tickers_str = st.text_input("Tickers", "AAPL, MSFT, NVDA, GGAL, MELI, TSLA")
    tickers = [t.strip().upper() for t in tickers_str.split(",")]
    col1, col2 = st.columns(2)
    with col1: f_inicio = st.date_input("Inicio", pd.to_datetime("2021-01-01"))
    with col2: f_fin = st.date_input("Fin", pd.to_datetime("today"))
    dist_modelo = st.selectbox("Modelo", ["MBG", "T-Student", "T-Skewed"])
    obj_input = st.radio("Objetivo", ["Max Sharpe Ratio", "Min Volatility"])
    restr_w = st.checkbox("Mínimo 5% por activo", value=True)

if st.button("Simular y Analizar"):
    with st.spinner("Procesando..."):
        rf = obtener_risk_free_live()
        raw_data = yf.download(tickers, start=f_inicio, end=f_fin)
        
        if isinstance(raw_data.columns, pd.MultiIndex):
            data = raw_data['Adj Close'] if 'Adj Close' in raw_data.columns.levels[0] else raw_data['Close']
        else:
            data = raw_data
            
        data = data.ffill().dropna()
        # Retornos logarítmicos
        returns_h = np.log(data / data.shift(1)).dropna()
        final_tickers = returns_h.columns.tolist()
        
        mu_sim, cov_sim, sims = generar_simulacion_profesional(returns_h, n_simulaciones, dist_modelo)
        res = optimizar_portfolio(mu_sim, cov_sim, rf, final_tickers, obj_input, 0.05 if restr_w else None, cap_inicial)

        if res:
            st.success("✅ Análisis Completo")
            
            # --- MÉTRICAS ---
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Retorno Esperado", f"{res['retorno_esperado']:.2%}")
            m2.metric("Volatilidad Anual", f"{res['volatilidad_esperada']:.2%}")
            m3.metric("Ratio de Sharpe", f"{(res['retorno_esperado']-rf)/res['volatilidad_esperada']:.2f}")
            m4.metric("VaR 95% Confianza", f"{res['vaR_pct']:.2%}")

            st.subheader(f"💵 Proyección de Capital (${cap_inicial:,.0f})")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Ganancia Esperada", f"+ ${res['ganancia_esperada_monetaria']:,.2f}")
            c2.metric("📈 Capital Potencial", f"${res['capital_potencial']:,.2f}", delta=f"+{res['retorno_esperado']:.1%}")
            c3.metric("Resultado Neto (VaR)", f"${res['resultado_monetario_peor_caso']:,.2f}")
            c4.metric("📉 Capital Remanente", f"${res['capital_final_peor_caso']:,.2f}", delta=f"${res['capital_final_peor_caso']-cap_inicial:,.2f}", delta_color="inverse")

            st.divider()

            col_plan, col_ind = st.columns(2)
            
            with col_plan:
                st.subheader("📋 Plan de Inversión")
                df_t = pd.DataFrame.from_dict(res['pesos'], orient='index', columns=['Ponderación (%)'])
                df_t['Ponderación (%)'] = df_t['Ponderación (%)'] * 100
                df_t['Monto a Invertir ($)'] = (df_t['Ponderación (%)'] / 100) * cap_inicial
                st.table(df_t.sort_values(by='Monto a Invertir ($)', ascending=False).style.format({'Ponderación (%)': '{:.2f}%', 'Monto a Invertir ($)': '$ {:,.2f}'}))

            with col_ind:
                st.subheader("🎯 Eficiencia Individual")
                # CORRECCIÓN DE VOLATILIDAD INDIVIDUAL
                vols_ind = np.sqrt(np.diag(cov_sim))
                sharpes_ind = (mu_sim - rf) / vols_ind
                df_ind = pd.DataFrame({"Retorno": mu_sim * 100, "Volatilidad": vols_ind * 100, "Sharpe": sharpes_ind}, index=final_tickers)
                st.table(df_ind.sort_values(by="Sharpe", ascending=False).style.format({"Retorno": "{:.1f}%", "Volatilidad": "{:.1f}%", "Sharpe": "{:.2f}"}))

            st.divider()

            # --- GRÁFICOS ---
            col_g1, col_g2 = st.columns([2, 1])
            with col_g1:
                st.write("### Frontera Eficiente de Markowitz")
                n_port = 1000
                p_r, p_v = [], []
                for _ in range(n_port):
                    w = np.random.random(len(final_tickers)); w /= np.sum(w)
                    p_r.append(np.dot(w, mu_sim)); p_v.append(np.sqrt(np.dot(w.T, np.dot(cov_sim, w))))
                
                fig_fe, ax_fe = plt.subplots(figsize=(10, 6))
                ax_fe.scatter(p_v, p_r, c=(np.array(p_r)/np.array(p_v)), marker='o', s=5, alpha=0.3, cmap='viridis')
                
                # LA ESTRELLITA (Punto Óptimo)
                ax_fe.scatter(res['volatilidad_esperada'], res['retorno_esperado'], color='red', marker='*', s=250, label='Portfolio Seleccionado', edgecolors='black', zorder=10)
                
                # Activos individuales
                for i, t in enumerate(final_tickers):
                    ax_fe.scatter(vols_ind[i], mu_sim[i], color='blue', marker='X', s=50, alpha=0.6)
                    ax_fe.annotate(t, (vols_ind[i], mu_sim[i]), xytext=(5,5), textcoords='offset points', fontsize=8)
                
                ax_fe.legend()
                st.pyplot(fig_fe)

            with col_g2:
                st.write("### Composición Visual")
                fig_pie, ax_pie = plt.subplots()
                pesos_plot = {k: v for k, v in res['pesos'].items() if v > 0.001}
                ax_pie.pie(pesos_plot.values(), labels=pesos_plot.keys(), autopct='%1.1f%%', startangle=140, colors=sns.color_palette("viridis", len(pesos_plot)))
                st.pyplot(fig_pie)
