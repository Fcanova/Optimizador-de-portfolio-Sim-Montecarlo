import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from pypfopt.efficient_frontier import EfficientFrontier
from scipy.stats import t as t_dist

# Commented out IPython magic to ensure Python compatibility.
# %%writefile app.py
# import streamlit as st
# import pandas as pd
# import numpy as np
# import yfinance as yf
# import matplotlib.pyplot as plt
# import seaborn as sns
# from pypfopt.efficient_frontier import EfficientFrontier
# from scipy.stats import t as t_dist
# 
# # --- CAPÍTULO 1: DATA LIVE ---
# def obtener_risk_free_live():
#     try:
#         tnx = yf.Ticker("^TNX")
#         rate = tnx.history(period="1d")['Close'].iloc[-1]
#         return rate / 100
#     except:
#         return 0.042
# 
# # --- CAPÍTULO 2: MOTOR DE SIMULACIÓN ---
# def generar_simulacion_profesional(returns_h, n_sims, dist_type):
#     n_assets = returns_h.shape[1]
#     mu_vec = returns_h.mean().values * 252
#     sigma_annual = returns_h.cov().values * 252
# 
#     T_years = len(returns_h) / 252
#     N_steps = int(T_years * 252)
#     dt = 1 / 252
# 
#     L = np.linalg.cholesky(sigma_annual + 1e-10 * np.eye(n_assets))
#     var_diag = np.diag(sigma_annual)
#     drift = (mu_vec - 0.5 * var_diag) * dt
# 
#     S = np.ones((N_steps + 1, n_sims, n_assets))
#     nu, gamma = 5, 1.3
# 
#     for t in range(1, N_steps + 1):
#         if dist_type == 'MBG':
#             z = np.random.standard_normal((n_assets, n_sims))
#         elif dist_type == 'T-Student':
#             z = t_dist.rvs(df=nu, size=(n_assets, n_sims))
#             z = z / np.sqrt(nu / (nu - 2))
#         elif dist_type == 'T-Skewed':
#             Y = t_dist.rvs(df=nu, size=(n_assets, n_sims))
#             Z_raw = np.where(Y >= 0, Y / gamma, Y * gamma)
#             z = (Z_raw - Z_raw.mean(axis=1, keepdims=True)) / Z_raw.std(axis=1, keepdims=True)
# 
#         corr_shock = L @ z
#         incr = drift[:, None] + corr_shock * np.sqrt(dt)
#         S[t] = S[t-1] * np.exp(incr.T)
# 
#     final_returns = S[-1] - 1
#     # Fix: Usamos el T_years calculado para anualizar
#     cov_sim_annual = np.cov(final_returns, rowvar=False) * (1/T_years)
#     mu_sim_annual = (1 + final_returns.mean(axis=0))**(1/T_years) - 1
# 
#     return mu_sim_annual, cov_sim_annual, final_returns
# 
# # --- CAPÍTULO 3: OPTIMIZADOR ---
# def optimizar_portfolio(mu_anual, cov_anual, rf_rate, final_returns_sim, asset_names, objetivo='Max Sharpe', min_weight=None, T_years=1.0):
#     mu_anual = pd.Series(mu_anual, index=asset_names)
#     cov_anual = pd.DataFrame(cov_anual, index=asset_names, columns=asset_names)
#     bounds = (min_weight if min_weight else 0.0, 1.0)
#     ef = EfficientFrontier(mu_anual, cov_anual, weight_bounds=bounds)
# 
#     try:
#         if 'sharpe' in objetivo.lower():
#             raw_weights = ef.max_sharpe(risk_free_rate=rf_rate)
#         else:
#             raw_weights = ef.min_volatility()
#         weights = ef.clean_weights()
#     except:
#         ef = EfficientFrontier(mu_anual, cov_anual, weight_bounds=(0, 1))
#         weights = ef.min_volatility()
# 
#     ret_p, vol_p, sharpe_p = ef.portfolio_performance(risk_free_rate=rf_rate)
#     pesos_arr = np.array(list(weights.values()))
#     port_rets_totales = final_returns_sim @ pesos_arr
#     port_rets_anual = np.power(np.maximum(1 + port_rets_totales, 0.00001), 1 / T_years) - 1
#     var_95 = np.percentile(port_rets_anual, 5)
# 
#     return {"pesos": weights, "retorno_esperado": ret_p, "volatilidad_esperada": vol_p, "sharpe_ratio": sharpe_p, "var_95": var_95}
# 
# # --- CAPÍTULO 4: INTEGRADOR MEJORADO ---
# def ejecutar_analisis_portfolio(tickers, f_inicio, f_fin, n_simulaciones, distribucion, objetivo, min_weight):
#     rf = obtener_risk_free_live()
#     df = yf.download(tickers, start=f_inicio, end=f_fin)
# 
#     if df.empty or len(df) < 10:
#         st.error("❌ Datos insuficientes. Elegí un rango de fechas más amplio (mínimo 1 mes).")
#         return None, None
# 
#     data = df['Adj Close'] if 'Adj Close' in df.columns else df['Close']
#     returns_h = np.log(data / data.shift(1)).dropna()
# 
#     if returns_h.empty:
#         st.error("❌ Los activos no tienen datos para ese periodo.")
#         return None, None
# 
#     T_years = len(returns_h) / 252
#     if T_years < 0.01: T_years = 0.01 # Evita división por cero
# 
#     mu_s, cov_s, rets_f = generar_simulacion_profesional(returns_h, n_simulaciones, distribucion)
#     res = optimizar_portfolio(mu_s, cov_s, rf, rets_f, returns_h.columns.tolist(), objetivo, min_weight, T_years)
# 
#     return res, rets_f
# 
# # --- INTERFAZ STREAMLIT ---
# st.set_page_config(page_title="Equity Optimizer", layout="wide")
# st.title("🚀 financial_wealth: Portfolio Intelligence")
# 
# with st.sidebar:
#     st.header("⚙️ Parámetros")
#     tickers_str = st.text_input("Tickers (coma)", "AAPL, MSFT, NVDA, GGAL, MELI")
#     tickers = [t.strip().upper() for t in tickers_str.split(",")]
# 
#     # AGREGAMOS FECHA FINAL
#     col_f1, col_f2 = st.columns(2)
#     with col_f1:
#         f_inicio = st.date_input("Inicio", value=pd.to_datetime("2021-01-01"))
#     with col_f2:
#         f_fin = st.date_input("Fin", value=pd.to_datetime("today"))
# 
#     dist_modelo = st.selectbox("Distribución", ["MBG", "T-Student", "T-Skewed"])
#     obj_input = st.radio("Objetivo", ["Max Sharpe", "Min Volatility"])
#     restr_w = st.checkbox("Mínimo 5% por activo", value=True)
# 
# if st.button("Simular y Optimizar"):
#     if f_inicio >= f_fin:
#         st.error("La fecha de inicio debe ser anterior a la de fin.")
#     else:
#         with st.spinner("Procesando..."):
#             res, sims = ejecutar_analisis_portfolio(tickers, f_inicio, f_fin, 2000, dist_modelo, obj_input, 0.05 if restr_w else None)
# 
#             if res:
#                 c1, c2, c3, c4 = st.columns(4)
#                 c1.metric("Retorno Anual", f"{res['retorno_esperado']:.2%}")
#                 c2.metric("Volatilidad Anual", f"{res['volatilidad_esperada']:.2%}")
#                 c3.metric("Ratio Sharpe", f"{res['sharpe_ratio']:.2f}")
#                 c4.metric("VaR Anualizado", f"{res['var_95']:.2%}")
# 
#                 fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
#                 pesos_dict = {k: v for k, v in res['pesos'].items() if v > 0.001}
#                 ax1.pie(pesos_dict.values(), labels=pesos_dict.keys(), autopct='%1.1f%%', colors=sns.color_palette("magma", len(pesos_dict)))
#                 ax1.set_title("Distribución Óptima")
# 
#                 pesos_arr = np.array(list(res['pesos'].values()))
#                 port_rets_finales = sims @ pesos_arr
#                 sns.histplot(port_rets_finales, kde=True, ax=ax2, color="teal")
#                 ax2.axvline(res['var_95'], color='red', linestyle='--', label=f"VaR Anual: {res['var_95']:.2%}")
#                 ax2.set_title(f"Distribución de Retornos - Modelo: {dist_modelo}")
#                 ax2.legend()
#                 st.pyplot(fig)


