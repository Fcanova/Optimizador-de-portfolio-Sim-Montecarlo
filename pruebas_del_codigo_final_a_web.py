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
    th { text-align: center !important; font-weight: bold !important; text-transform: uppercase; color: #1E88E5; }
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

# --- 2. MOTOR DE SIMULACIÓN (POST-SIMULACIÓN) ---
def generar_simulacion_profesional(mu_h, cov_h, n_sims, dist_type):
    n_assets = len(mu_h)
    N_steps = 252
    dt = 1 / 252
    L = np.linalg.cholesky(cov_h + 1e-10 * np.eye(n_assets))
    
    drift = (mu_h - 0.5 * np.diag(cov_h)) * dt
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
    z_score = 1.645 
    vaR_pct = ret_p - (z_score * vol_p)
    
    return {
        "pesos": weights, "retorno_esperado": ret_p, "volatilidad_esperada": vol_p, 
        "vaR_pct": vaR_pct, "ganancia_esperada_monetaria": ret_p * capital,
        "resultado_monetario_peor_caso": capital * vaR_pct,
        "capital_final_peor_caso": capital * (1 + vaR_pct),
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
    col_a, col_b = st.columns(2)
    with col_a: f_inicio = st.date_input("Inicio", pd.to_datetime("2021-01-01"))
    with col_b: f_fin = st.date_input("Fin", pd.to_datetime("today"))
    dist_modelo = st.selectbox("Modelo", ["MBG", "T-Student", "T-Skewed"])
    obj_input = st.radio("Objetivo", ["Max Sharpe Ratio", "Min Volatility"])
    restr_w = st.checkbox("Mínimo 5% por activo", value=True)

if st.button("Simular y Analizar"):
    with st.spinner("Descargando y procesando datos..."):
        rf = obtener_risk_free_live()
        
        # --- DESCARGA ROBUSTA (FIX DEL KEYERROR) ---
        raw_data = yf.download(tickers, start=f_inicio, end=f_fin)
        
        # Extraemos precios de forma segura manejando el MultiIndex
        if isinstance(raw_data.columns, pd.MultiIndex):
            # Priorizamos Adj Close, si no Close
            if 'Adj Close' in raw_data.columns.levels[0]:
                data_prices = raw_data['Adj Close']
            else:
                data_prices = raw_data['Close']
        else:
            data_prices = raw_data
            
        # Llenamos huecos individualmente antes de calcular retornos
        data_prices = data_prices.ffill()
        log_returns = np.log(data_prices / data_prices.shift(1))
        
        # --- CUADRO DE MÉTRICAS INDIVIDUALES ---
        # Usamos skipna=True (por defecto en pandas) para que MELI no se vea afectado por otros
        mu_h = log_returns.mean() * 252
        vol_h = log_returns.std() * np.sqrt(252)
        cov_h = log_returns.cov() * 252
        
        st.subheader("📋 Métricas Anualizadas por Activo (Histórico)")
        df_metrics = pd.DataFrame({
            "Retorno Anual (%)": mu_h * 100,
            "Volatilidad Anual (%)": vol_h * 100,
            "Ratio de Sharpe": (mu_h - rf) / vol_h
        }).sort_values(by="Retorno Anual (%)", ascending=False)
        
        st.table(df_metrics.style.format({
            "Retorno Anual (%)": "{:.2f}%",
            "Volatilidad Anual (%)": "{:.2f}%",
            "Ratio de Sharpe": "{:.2f}"
        }))
        
        # --- OPTIMIZACIÓN Y GRÁFICOS ---
        final_tickers = log_returns.columns.tolist()
        mu_sim, cov_sim, sims = generar_simulacion_profesional(mu_h.values, cov_h.values, n_simulaciones, dist_modelo)
        res = optimizar_portfolio(mu_sim, cov_sim, rf, final_tickers, obj_input, 0.05 if restr_w else None, cap_inicial)

        if res:
            st.success("✅ Análisis Completo")
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Retorno Esperado", f"{res['retorno_esperado']:.2%}")
            m2.metric("Volatilidad Anual", f"{res['volatilidad_esperada']:.2%}")
            m3.metric("Ratio de Sharpe", f"{(res['retorno_esperado']-rf)/res['volatilidad_esperada']:.2f}")
            m4.metric("VaR 95% Confianza", f"{res['vaR_pct']:.2%}")

            st.divider()

            # TABLA DE TENENCIAS
            st.subheader("📋 Plan de Inversión Sugerido")
            df_t = pd.DataFrame.from_dict(res['pesos'], orient='index', columns=['Ponderación (%)'])
            df_t['Ponderación (%)'] = df_t['Ponderación (%)'] * 100
            df_t['Monto a Invertir ($)'] = (df_t['Ponderación (%)'] / 100) * cap_inicial
            st.table(df_t.sort_values(by='Monto a Invertir ($)', ascending=False).style.format({'Ponderación (%)': '{:.2f}%', 'Monto a Invertir ($)': '$ {:,.2f}'}))

            st.divider()

            # GRÁFICOS
            col_g1, col_g2 = st.columns([2, 1])
            with col_g1:
                st.write("### Frontera Eficiente de Markowitz")
                n_port = 800
                p_r, p_v = [], []
                for _ in range(n_port):
                    w = np.random.random(len(final_tickers)); w /= np.sum(w)
                    p_r.append(np.dot(w, mu_sim)); p_v.append(np.sqrt(np.dot(w.T, np.dot(cov_sim, w))))
                
                fig_fe, ax_fe = plt.subplots(figsize=(10, 6))
                ax_fe.scatter(p_v, p_r, c=(np.array(p_r)/np.array(p_v)), marker='o', s=5, alpha=0.3, cmap='viridis')
                
                for i, t in enumerate(final_tickers):
                    ax_fe.scatter(vol_h[i], mu_h[i], color='red', marker='X', s=80)
                    ax_fe.annotate(t, (vol_h[i], mu_h[i]), xytext=(5,5), textcoords='offset points', fontweight='bold')
                
                ax_fe.scatter(res['volatilidad_esperada'], res['retorno_esperado'], color='gold', marker='*', s=300, edgecolor='black', label="Portfolio")
                st.pyplot(fig_fe)

            with col_g2:
                st.write("### Composición Visual")
                fig_pie, ax_pie = plt.subplots()
                pesos_plot = {k: v for k, v in res['pesos'].items() if v > 0.001}
                ax_pie.pie(pesos_plot.values(), labels=pesos_plot.keys(), autopct='%1.1f%%', startangle=140, colors=sns.color_palette("viridis", len(pesos_plot)))
                st.pyplot(fig_pie)
            
            st.write("### Distribución de Resultados (Simulación)")
            
            fig_hist, ax_hist = plt.subplots(figsize=(12, 4))
            pesos_arr = np.array(list(res['pesos'].values()))
            rets_monetarios = (sims @ pesos_arr) * cap_inicial
            sns.histplot(rets_monetarios, kde=True, ax=ax_hist, color="#1E88E5")
            ax_hist.axvline(res['resultado_monetario_peor_caso'], color='red', linestyle='--')
            st.pyplot(fig_hist)
