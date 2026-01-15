import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from pypfopt.efficient_frontier import EfficientFrontier
from scipy.stats import t as t_dist

# --- 0. CONFIGURACIÓN Y ESTILO ---
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

# --- 2. MOTOR DE SIMULACIÓN (AZAR LIBERADO) ---
def generar_simulacion_profesional(returns_h, n_sims, dist_type):
    # Sin semilla fija para que los números se actualicen en cada simulación
    n_assets = returns_h.shape[1]
    mu_daily = returns_h.mean().values
    sigma_daily = returns_h.cov().values
    
    mu_annual = mu_daily * 252
    sigma_annual = sigma_daily * 252 
    
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
        
        corr_shock = L @ z
        incr = drift[:, None] + corr_shock * np.sqrt(dt)
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
        ef = EfficientFrontier(mu_s, cov_s, weight_bounds=(0, 1))
        ef.min_volatility()
        weights = ef.clean_weights()
    
    ret_p, vol_p, _ = ef.portfolio_performance(risk_free_rate=rf_rate)
    z_score = 1.645 
    vaR_pct = ret_p - (z_score * vol_p)
    
    return {
        "pesos": weights, 
        "retorno_esperado": ret_p, 
        "volatilidad_esperada": vol_p, 
        "vaR_pct": vaR_pct,
        "ganancia_esperada_monetaria": ret_p * capital,
        "resultado_neto_peor_caso": capital * vaR_pct,
        "capital_final_peor_caso": capital * (1 + vaR_pct),
        "capital_potencial": capital * (1 + ret_p)
    }

# --- 4. INTERFAZ ---
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
    with st.spinner("Ejecutando simulación y optimización..."):
        rf = obtener_risk_free_live()
        df = yf.download(tickers, start=f_inicio, end=f_fin)
        data = df['Adj Close'] if 'Adj Close' in df.columns else df['Close']
        returns_h = np.log(data / data.shift(1)).dropna()
        
        mu_sim, cov_sim, sims = generar_simulacion_profesional(returns_h, 2000, dist_modelo)
        res = optimizar_portfolio(mu_sim, cov_sim, rf, tickers, obj_input, 0.05 if restr_w else None, cap_inicial)

        if res:
            st.success("✅ Análisis Completo")
            
            # FILA 1: MÉTRICAS
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Retorno Esperado", f"{res['retorno_esperado']:.2%}", help="Retorno promedio anual simulado.")
            m2.metric("Volatilidad Anual", f"{res['volatilidad_esperada']:.2%}", help="Riesgo de mercado anualizado.")
            m3.metric("Ratio de Sharpe", f"{(res['retorno_esperado']-rf)/res['volatilidad_esperada']:.2f}", help="Eficiencia del riesgo.")
            m4.metric("VaR 95% Confianza", f"{res['vaR_pct']:.2%}", help="Peor escenario neto anual estimado.")

            # FILA 2: MONETARIAS
            st.subheader(f"💵 Proyección de Capital (${cap_inicial:,.0f})")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Ganancia Esperada", f"+ ${res['ganancia_esperada_monetaria']:,.2f}")
            c2.metric("📈 Capital Potencial", f"${res['capital_potencial']:,.2f}", delta=f"+{res['retorno_esperado']:.1%}")
            
            val_neto = res['resultado_neto_peor_caso']
            c3.metric("Resultado Neto (VaR)", f"${val_neto:,.2f}", delta="Pérdida Máxima" if val_neto < 0 else "Ganancia Mínima")
            
            diff = res['capital_final_peor_caso'] - cap_inicial
            c4.metric("📉 Capital Remanente", f"${res['capital_final_peor_caso']:,.2f}", delta=f"${diff:,.2f}", delta_color="inverse")

            st.divider()

            # TABLA DE TENENCIAS
            st.subheader("📋 Plan de Inversión (Tenencias)")
            df_t = pd.DataFrame.from_dict(res['pesos'], orient='index', columns=['Ponderación (%)'])
            df_t['Ponderación (%)'] = df_t['Ponderación (%)'] * 100
            df_t['Monto a Invertir ($)'] = (df_t['Ponderación (%)'] / 100) * cap_inicial
            df_t = df_t.sort_values(by='Monto a Invertir ($)', ascending=False)
            st.table(df_t.style.format({'Ponderación (%)': '{:.2f}%', 'Monto a Invertir ($)': '$ {:,.2f}'}))

            st.divider()

            # FILA 3: FRONTERA CON LÍNEA EFICIENTE
            col_g1, col_g2 = st.columns([2, 1])
            with col_g1:
                st.write("### Frontera Eficiente de Markowitz")
                n_port = 1000
                p_r, p_v = [], []
                for _ in range(n_port):
                    w = np.random.random(len(tickers)); w /= np.sum(w)
                    p_r.append(np.dot(w, mu_sim))
                    p_v.append(np.sqrt(np.dot(w.T, np.dot(cov_sim, w))))
                
                # CÁLCULO DE LA LÍNEA DE LA FRONTERA
                target_rets = np.linspace(min(mu_sim), max(mu_sim), 25)
                frontier_v = []
                for r in target_rets:
                    ef_line = EfficientFrontier(pd.Series(mu_sim, index=tickers), pd.DataFrame(cov_sim, index=tickers, columns=tickers))
                    try:
                        ef_line.efficient_return(r)
                        frontier_v.append(ef_line.portfolio_performance()[1])
                    except: frontier_v.append(None)
                
                fig_fe, ax_fe = plt.subplots(figsize=(10, 6))
                ax_fe.scatter(p_v, p_r, c=(np.array(p_r)/np.array(p_v)), marker='o', s=5, alpha=0.3, cmap='viridis')
                
                # Dibujar la línea punteada
                valid_v = [v for v in frontier_v if v is not None]
                valid_r = [r for v, r in zip(frontier_v, target_rets) if v is not None]
                ax_fe.plot(valid_v, valid_r, color='black', linestyle='--', linewidth=1.5, label="Frontera Eficiente")
                
                v_ind = np.sqrt(np.diag(cov_sim))
                ax_fe.scatter(v_ind, mu_sim, color='red', marker='X', s=80, label="Activos")
                for i, t in enumerate(tickers):
                    ax_fe.annotate(t, (v_ind[i], mu_sim[i]), xytext=(5,5), textcoords='offset points', fontweight='bold')
                
                ax_fe.scatter(res['volatilidad_esperada'], res['retorno_esperado'], color='gold', marker='*', s=300, edgecolor='black', label="Tu Portfolio")
                ax_fe.legend()
                st.pyplot(fig_fe)
                

            with col_g2:
                st.write("### Composición Visual")
                fig_pie, ax_pie = plt.subplots()
                pesos_plot = {k: v for k, v in res['pesos'].items() if v > 0.001}
                ax_pie.pie(pesos_plot.values(), labels=pesos_plot.keys(), autopct='%1.1f%%', startangle=140, colors=sns.color_palette("viridis", len(pesos_plot)))
                st.pyplot(fig_pie)

            # FILA 4: BARRAS E HISTOGRAMA
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                st.write("### Potencial: Esperado vs Peor Caso")
                fig_bar, ax_bar = plt.subplots()
                ax_bar.bar(['Ganancia Esp.', 'Resultado VaR'], [res['ganancia_esperada_monetaria'], res['resultado_neto_peor_caso']], color=['#2ECC71', '#E74C3C'])
                ax_bar.axhline(0, color='black', linewidth=0.8)
                st.pyplot(fig_bar)
            
            with col_b2:
                st.write("### Distribución de Probabilidades")
                fig_hist, ax_hist = plt.subplots()
                pesos_arr = np.array(list(res['pesos'].values()))
                rets_monetarios = (sims @ pesos_arr) * cap_inicial
                sns.histplot(rets_monetarios, kde=True, ax=ax_hist, color="#3498DB")
                ax_hist.axvline(res['resultado_neto_peor_caso'], color='red', linestyle='--', label="VaR 95%")
                st.pyplot(fig_hist)
