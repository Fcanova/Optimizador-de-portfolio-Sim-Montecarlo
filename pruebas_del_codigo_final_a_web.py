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

# --- 2. MOTOR DE SIMULACIÓN DIARIA (CON CORRECCIÓN HULL/TIF) ---
def generar_simulacion_diaria(mu_h, cov_h, n_sims, dist_type):
    n_assets = len(mu_h)
    N_steps = 252 
    L = np.linalg.cholesky(cov_h/252 + 1e-8 * np.eye(n_assets))
    drift_diario = (mu_h / 252) 
    ret_logs_sim = np.zeros((N_steps, n_sims, n_assets))
    
    for t in range(N_steps):
        if dist_type == 'MBG':
            z = np.random.standard_normal((n_assets, n_sims))
        elif dist_type == 'T-Student':
            nu = 5
            z = t_dist.rvs(df=nu, size=(n_assets, n_sims))
            z = z / np.sqrt(nu / (nu - 2))
        elif dist_type == 'T-Skewed':
            nu, gamma = 5, 1.3
            Y = t_dist.rvs(df=nu, size=(n_assets, n_sims))
            Z_raw = np.where(Y >= 0, Y / gamma, Y * gamma)
            z = (Z_raw - Z_raw.mean(axis=1, keepdims=True)) / Z_raw.std(axis=1, keepdims=True)
        
        shock = L @ z
        ret_logs_sim[t] = drift_diario + shock.T
        
    mu_sim_log = ret_logs_sim.mean(axis=(0, 1)) * 252
    reshaped_rets = ret_logs_sim.reshape(-1, n_assets)
    cov_sim = np.cov(reshaped_rets, rowvar=False) * 252
    
    # Retorno Aritmético Esperado E[R] según Hull
    sigmas_sim = np.sqrt(np.diag(cov_sim))
    mu_sim_aritmetico = np.exp(mu_sim_log + 0.5 * (sigmas_sim**2)) - 1
    
    return mu_sim_log, mu_sim_aritmetico, cov_sim, ret_logs_sim

# --- 3. OPTIMIZADOR ---
def optimizar_portfolio(mu_sim_log, mu_aritmetico, cov_sim, ret_logs_sim, rf_rate, asset_names, objetivo, min_weight, capital):
    mu_s = pd.Series(mu_sim_log, index=asset_names)
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
    
    ret_p_log, vol_p, _ = ef.portfolio_performance(risk_free_rate=rf_rate)
    # Retorno del portfolio pasado a aritmético
    ret_p_arit = np.exp(ret_p_log + 0.5 * vol_p**2) - 1
    
    pesos_arr = np.array([weights[t] for t in asset_names])
    ret_final_sim_p = np.exp(ret_logs_sim.sum(axis=0) @ pesos_arr) - 1
    vaR_pct = np.percentile(ret_final_sim_p, 5)
    
    return {
        "pesos": weights, "retorno_esperado": ret_p_arit, "volatilidad_esperada": vol_p, 
        "vaR_pct": vaR_pct, "ganancia_esperada_monetaria": ret_p_arit * capital,
        "resultado_monetario_peor_caso": capital * vaR_pct,
        "capital_final_peor_caso": capital * (1 + vaR_pct),
        "capital_potencial": capital * (1 + ret_p_arit),
        "ret_final_sim_p": ret_final_sim_p,
        "mu_aritmetico": mu_aritmetico
    }

# --- 4. INTERFAZ ---
st.title("🚀 financial_wealth: Portfolio Intelligence")

with st.sidebar:
    st.header("⚙️ Parámetros")
    cap_inicial = st.number_input("Capital a Invertir ($)", min_value=100.0, value=10000.0)
    n_simulaciones = st.slider("Simulaciones Monte Carlo", 1000, 10000, 2000, 500)
    tickers_str = st.text_input("Tickers", "AAPL, MSFT, NVDA, GGAL, MELI, TSLA, AMZN")
    tickers = [t.strip().upper() for t in tickers_str.split(",")]
    col_a, col_b = st.columns(2)
    with col_a: f_inicio = st.date_input("Inicio", pd.to_datetime("2021-01-01"))
    with col_b: f_fin = st.date_input("Fin", pd.to_datetime("today"))
    dist_modelo = st.selectbox("Modelo", ["MBG", "T-Student", "T-Skewed"])
    obj_input = st.radio("Objetivo", ["Max Sharpe Ratio", "Min Volatility"])
    restr_w = st.checkbox("Mínimo 5% por activo", value=True)

if st.button("Simular y Analizar"):
    with st.spinner("Sincronizando Frontera con Hull..."):
        rf = obtener_risk_free_live()
        
        raw_df = yf.download(tickers, start=f_inicio, end=f_fin)
        if 'Adj Close' in raw_df.columns: data = raw_df['Adj Close']
        elif 'Close' in raw_df.columns: data = raw_df['Close']
        else: data = raw_df.xs('Adj Close', axis=1, level=0) if 'Adj Close' in raw_df.columns.levels[0] else raw_df.xs('Close', axis=1, level=0)

        data = data.ffill().dropna()
        log_returns = np.log(data / data.shift(1)).dropna()
        mu_h = log_returns.mean() * 252
        
        try: cov_h_clean = risk_models.CovarianceShrinkage(data).ledoit_wolf()
        except: cov_h_clean = risk_models.sample_cov(data)
        
        final_tickers = data.columns.tolist()

        # SIMULACIÓN
        mu_sim_log, mu_sim_arit, cov_sim, ret_logs = generar_simulacion_diaria(mu_h.values, cov_h_clean.values, n_simulaciones, dist_modelo)
        res = optimizar_portfolio(mu_sim_log, mu_sim_arit, cov_sim, ret_logs, rf, final_tickers, obj_input, 0.05 if restr_w else None, cap_inicial)

        if res:
            st.success("✅ Análisis Sincronizado Completado")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Retorno Esperado", f"{res['retorno_esperado']:.2%}")
            m2.metric("Volatilidad Anual", f"{res['volatilidad_esperada']:.2%}")
            m3.metric("Ratio de Sharpe", f"{(res['retorno_esperado']-rf)/res['volatilidad_esperada']:.2f}")
            m4.metric("VaR 95% (Anual)", f"{res['vaR_pct']:.2%}")

            st.subheader("🎯 Eficiencia Individual")
            vols_sim_ind = np.sqrt(np.diag(cov_sim))
            df_ind = pd.DataFrame({
                "Retorno Anual": res['mu_aritmetico'] * 100,
                "Volatilidad Anual": vols_sim_ind * 100,
                "Ratio de Sharpe": (res['mu_aritmetico'] - rf) / vols_sim_ind
            }, index=final_tickers)
            st.table(df_ind.sort_values(by="Retorno Anual", ascending=False).style.format("{:.2f}"))

            st.divider()

            # FRONTERA EFICIENTE SINCRONIZADA
            st.subheader("📈 Frontera Eficiente de Markowitz")
            col_fe, col_pie = st.columns([2, 1])
            with col_fe:
                n_port = 1000
                p_r_arit, p_v_vals = [], []
                for _ in range(n_port):
                    w = np.random.random(len(final_tickers)); w /= np.sum(w)
                    # Calculamos métricas del portfolio aleatorio en log y pasamos a aritmético
                    p_ret_log = np.dot(w, mu_sim_log)
                    p_vol = np.sqrt(np.dot(w.T, np.dot(cov_sim, w)))
                    p_r_arit.append(np.exp(p_ret_log + 0.5 * p_vol**2) - 1)
                    p_v_vals.append(p_vol)
                
                # Línea de la frontera
                target_rets_log = np.linspace(min(mu_sim_log), max(mu_sim_log), 30)
                frontier_v = []
                for r in target_rets_log:
                    ef_line = EfficientFrontier(pd.Series(mu_sim_log, index=final_tickers), pd.DataFrame(cov_sim, index=final_tickers, columns=final_tickers))
                    try:
                        ef_line.efficient_return(r)
                        frontier_v.append(ef_line.portfolio_performance()[1])
                    except: frontier_v.append(None)

                fig_fe, ax_fe = plt.subplots(figsize=(10, 6))
                scatter = ax_fe.scatter(p_v_vals, p_r_arit, c=(np.array(p_r_arit)/np.array(p_v_vals)), marker='o', s=10, alpha=0.3, cmap='viridis')
                plt.colorbar(scatter, label='Ratio de Sharpe')
                
                valid_v = [v for v in frontier_v if v is not None]
                valid_r = [np.exp(r + 0.5 * v**2) - 1 for v, r in zip(frontier_v, target_rets_log) if v is not None]
                ax_fe.plot(valid_v, valid_r, color='black', linestyle='--', linewidth=2, label="Frontera")

                for i, t in enumerate(final_tickers):
                    ax_fe.scatter(vols_sim_ind[i], res['mu_aritmetico'][i], color='red', marker='X', s=100)
                    ax_fe.annotate(t, (vols_sim_ind[i], res['mu_aritmetico'][i]), xytext=(5,5), textcoords='offset points', fontweight='bold')
                
                ax_fe.scatter(res['volatilidad_esperada'], res['retorno_esperado'], color='gold', marker='*', s=400, edgecolor='black', label="Portfolio Óptimo")
                ax_fe.set_xlabel("Riesgo (Volatilidad)"); ax_fe.set_ylabel("Retorno"); ax_fe.legend(); st.pyplot(fig_fe)

            with col_pie:
                st.write("### Composición")
                fig_pie, ax_pie = plt.subplots()
                pesos_plot = {k: v for k, v in res['pesos'].items() if v > 0.001}
                ax_pie.pie(pesos_plot.values(), labels=pesos_plot.keys(), autopct='%1.1f%%', startangle=140, colors=sns.color_palette("viridis", len(pesos_plot)))
                st.pyplot(fig_pie)

            st.divider()
            
            st.subheader(f"💵 Proyección Monetaria (${cap_inicial:,.0f})")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Ganancia Esperada", f"+ ${res['ganancia_esperada_monetaria']:,.2f}")
            c2.metric("📈 Capital Potencial", f"${res['capital_potencial']:,.2f}")
            c3.metric("Resultado Neto (VaR)", f"${res['resultado_monetario_peor_caso']:,.2f}")
            c4.metric("📉 Capital Remanente", f"${res['capital_final_peor_caso']:,.2f}", delta=f"${res['capital_final_peor_caso']-cap_inicial:,.2f}", delta_color="inverse")
