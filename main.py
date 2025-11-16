
import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from itertools import combinations

# Parámetros
CAPITAL_INICIAL = 1_000_000.0
COMISION = 0.00125  # 0.125% por operación (entrada o salida)
BORROW_ANUAL = 0.0025  # 0.25% anual
DIAS_TRADING = 252
BORROW_DIARIO = BORROW_ANUAL / DIAS_TRADING

VENTANA_JOHANSEN = 120  # días para recalcular cointegración en el backtest
VENTANA_ZSCORE = 30  # ventana para zscore
THRESHOLD_ENTRADA = 2.0
THRESHOLD_SALIDA = 0.05
ALLOCATION_FRAC = 0.8  # 80% del capital disponible


def descargar_datos(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        precios = data['Close']
    else:
        precios = data[['Close']]

    precios = precios.dropna(how='all')
    return precios


# ADF y rolling corr
def adf_test(series, maxlag=None):
    s = series.dropna()
    if len(s) < 10:
        return 1.0
    return adfuller(s, maxlag=maxlag, autolag='AIC')[1]


def rolling_corr(a, b, window=60):
    return a.rolling(window).corr(b)


# Johansen / VECM utilities
def ejecutar_johansen(df_pair, det_order=0, k_ar_diff=2):
    arr = df_pair.dropna().values
    if arr.shape[0] < 10:
        raise ValueError("No hay suficientes observaciones para Johansen")
    res = coint_johansen(arr, det_order, k_ar_diff)
    return res


def obtener_rank_johansen(df_pair, det_order=0, k_ar_diff=2, signif=0.05):
    res = ejecutar_johansen(df_pair, det_order, k_ar_diff)
    lr1 = res.lr1
    crit_95 = res.cvt[:, 1]
    rank = sum(lr1 > crit_95)
    return rank, res




# Kalman filters (column-vector safe)
class KalmanReg:
    def __init__(self, w_init=(0.0, 1.0), q=1e-5, r=1.0):
        self.w = np.array(w_init, dtype=float).reshape((2, 1))
        self.A = np.eye(2)
        self.Q = np.eye(2) * q
        self.R = np.array([[r]])
        self.P = np.eye(2)

    def predict(self):
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, x, y):
        C = np.array([[1.0, x]])
        S = C @ self.P @ C.T + self.R
        K = self.P @ C.T @ np.linalg.inv(S)
        y_pred = (C @ self.w).reshape(())
        innov = float(y - y_pred)
        self.w = self.w + K * innov
        self.P = (np.eye(2) - K @ C) @ self.P

    @property
    def params(self):
        return float(self.w[0, 0]), float(self.w[1, 0])


class KalmanRegVECM(KalmanReg):
    def __init__(self, w_init=(0.0, 1.0), q=1e-5, r=1.0, alpha=0.1):
        super().__init__(w_init, q, r)
        self.alpha = alpha

    def update_with_vecm(self, x, y, err_vecm):
        C = np.array([[1.0, x]])
        S = C @ self.P @ C.T + self.R
        K = self.P @ C.T @ np.linalg.inv(S)
        y_pred = (C @ self.w).reshape(())
        innov = float((y - y_pred) + self.alpha * err_vecm)
        self.w = self.w + K * innov
        self.P = (np.eye(2) - K @ C) @ self.P


# Selección de pares
def seleccionar_pares(df, min_corr=0.7, p_adf=0.05, corr_window=60):
    candidatos = []
    cols = df.columns
    for a, b in combinations(cols, 2):
        corr_series = rolling_corr(df[a], df[b], window=corr_window).dropna()
        mean_corr = corr_series.mean() if len(corr_series) > 0 else 0.0
        if mean_corr > min_corr:
            x = df[b].dropna()
            y = df[a].dropna()
            common_idx = x.index.intersection(y.index)
            if len(common_idx) < 10:
                continue
            x_cl = x.loc[common_idx]
            y_cl = y.loc[common_idx]
            # OLS simple para residuo (prices)
            X = sm.add_constant(x_cl)
            model = sm.OLS(y_cl, X).fit()
            alpha, beta = model.params
            resid = model.resid
            pval = adf_test(resid)
            if pval < p_adf:
                candidatos.append((a, b, mean_corr, pval))
    return pd.DataFrame(candidatos, columns=["A", "B", "Corr", "pADF"])


# Backtest
def backtest_pair(df_pair,
                  ventana_joh=VENTANA_JOHANSEN,
                  capital_inicial=CAPITAL_INICIAL,
                  comision=COMISION,
                  borrow_diario=BORROW_DIARIO,
                  allocation_frac=ALLOCATION_FRAC,
                  threshold_entrada=THRESHOLD_ENTRADA):

    df_pair = df_pair.dropna().copy()

    if df_pair.shape[1] != 2:
        raise ValueError(f"df_pair debe tener 2 columnas, pero tiene {df_pair.shape[1]}: {df_pair.columns.tolist()}")

    A, B = df_pair.columns
    kalman = KalmanRegVECM()

    alpha_list = []
    beta_list = []
    eigvec_list = []

    cash = capital_inicial
    posiciones = {'A': 0.0, 'B': 0.0}
    history = []
    trades = []
    vecm_params = None
    spread_list = []

    for t in range(len(df_pair)):
        fecha = df_pair.index[t]
        pa = float(df_pair.iloc[t][A])
        pb = float(df_pair.iloc[t][B])

        kalman.predict()

        if vecm_params is not None:
            err_vecm = float(vecm_params[0] * pa + vecm_params[1] * pb)
        else:
            err_vecm = 0.0

        if t > 0:
            kalman.update_with_vecm(pb, pa, err_vecm)
        else:
            kalman.update(pb, pa)

        alpha, beta = kalman.params
        alpha_list.append(alpha)
        beta_list.append(beta)

        spread = (pa * alpha + beta * pb)
        spread_list.append(spread)

        # CÁLCULO DE JOHANSEN 
        if t >= ventana_joh:
            sub = df_pair.iloc[t - ventana_joh:t][[A, B]].dropna()

            if sub.shape[0] >= 20:
                try:
                    joh = coint_johansen(sub.values, 0, 2)
                    vec = joh.evec[:, 0]

                    # Normalize
                    if abs(vec[1]) > 1e-12:
                        vec = vec / vec[1]

                    vecm_params = vec
                    eigvec_list.append(vec.tolist())

                except Exception:
                    eigvec_list.append([None, None])
            else:
                eigvec_list.append([None, None])
        else:
            eigvec_list.append([None, None])

        # Z-score
        if len(spread_list) >= VENTANA_ZSCORE:
            window = spread_list[-VENTANA_ZSCORE:]
            mu = np.mean(window)
            sigma = np.std(window, ddof=1)
            zscore = (spread - mu) / sigma if sigma > 0 else 0.0
        else:
            zscore = 0.0

        target_shares = {'A': posiciones['A'], 'B': posiciones['B']}

        if zscore > threshold_entrada:
            asign_total = allocation_frac * cash
            asign_cada = asign_total / 2
            target_shares['A'] = -(asign_cada / pa)
            target_shares['B'] = +(asign_cada / pb)

        elif zscore < -threshold_entrada:
            asign_total = allocation_frac * cash
            asign_cada = asign_total / 2
            target_shares['A'] = +(asign_cada / pa)
            target_shares['B'] = -(asign_cada / pb)

        elif abs(zscore) < THRESHOLD_SALIDA:
            target_shares['A'] = 0.0
            target_shares['B'] = 0.0

        # Ejecutar
        for tkr, price in [('A', pa), ('B', pb)]:
            delta = target_shares[tkr] - posiciones[tkr]
            if abs(delta) > 1e-12:
                trade_value = abs(delta) * price
                commission = trade_value * comision

                cash -= delta * price
                cash -= commission

                trades.append({
                    'fecha': fecha,
                    'ticker': (A if tkr == 'A' else B),
                    'delta_shares': delta,
                    'price': price,
                    'trade_value': trade_value,
                    'commission': commission,
                    'cash_after': cash
                })

                posiciones[tkr] = target_shares[tkr]

        borrow_cost = 0.0
        if posiciones['A'] < 0:
            borrow_cost += abs(posiciones['A']) * pa * borrow_diario
        if posiciones['B'] < 0:
            borrow_cost += abs(posiciones['B']) * pb * borrow_diario

        cash -= borrow_cost

        pv = cash + posiciones['A'] * pa + posiciones['B'] * pb

        fila_hist = {
            'fecha': fecha,
            'pv': pv,
            'cash': cash,
            'pos_A': posiciones['A'],
            'pos_B': posiciones['B'],
            'zscore': zscore,
            'borrow_cost': borrow_cost,
            'alpha': alpha,
            'beta': beta
        }

        # Eigenvectores (si ya tenemos vecm_params)
        if vecm_params is not None:
            fila_hist['vec1'] = vecm_params[0]
            fila_hist['vec2'] = vecm_params[1]
        else:
            fila_hist['vec1'] = np.nan
            fila_hist['vec2'] = np.nan

        history.append(fila_hist)


    df_hist = pd.DataFrame(history).set_index('fecha')

    # Guardamos alpha y beta
    df_hist["alpha"] = alpha_list
    df_hist["beta"] = beta_list



    trades_df = pd.DataFrame(trades)

    # Métricas
    df_hist["retornos"] = df_hist["pv"].pct_change().fillna(0)
    retorno_acum = df_hist["pv"].iloc[-1] / capital_inicial - 1
    vol_anual = df_hist["retornos"].std() * np.sqrt(252)
    sharpe = retorno_acum / vol_anual if vol_anual > 0 else np.nan

    metrics = {
        "retorno_acumulado": retorno_acum,
        "volatilidad_anualizada": vol_anual,
        "sharpe": sharpe,
        "n_trades": len(trades_df)
    }



    return df_hist, trades_df, metrics


def optimizar_threshold_entrada(df_pair,
                                lista_thresholds,
                                ventana_joh=VENTANA_JOHANSEN,
                                capital_inicial=CAPITAL_INICIAL,
                                comision=COMISION,
                                borrow_diario=BORROW_DIARIO,
                                allocation_frac=ALLOCATION_FRAC,
                                criterio='retorno_acumulado'):

    resultados = []
    mejor = None
    todos_hist = {}     # AQUÍ guardamos todos los históricos

    for th in lista_thresholds:

        df_hist, trades_df, metrics = backtest_pair(
            df_pair,
            ventana_joh=ventana_joh,
            capital_inicial=capital_inicial,
            comision=comision,
            borrow_diario=borrow_diario,
            allocation_frac=allocation_frac,
            threshold_entrada=th
        )

        # Se guarda el histórico completo para este threshold
        todos_hist[th] = df_hist.copy()

        # Se guarda métricas en una fila de resumen
        fila = {'threshold_entrada': th}
        fila.update(metrics)
        resultados.append(fila)

        valor = metrics.get(criterio, np.nan)
        if not np.isnan(valor):
            if (mejor is None) or (valor > mejor['valor']):
                mejor = {
                    'valor': valor,
                    'threshold_entrada': th,
                    'df_hist': df_hist.copy(),
                    'trades_df': trades_df.copy(),
                    'metrics': metrics
                }

    tabla_resultados = pd.DataFrame(resultados)

    return mejor, tabla_resultados







# Eigenvector
def plot_eigenvectors(df_hist):
    mask = df_hist["eig_A"].notna()

    plt.figure(figsize=(12,6))
    plt.plot(df_hist.loc[mask, "eig_A"].values, label="Eigenvector A")
    plt.plot(df_hist.loc[mask, "eig_B"].values, label="Eigenvector B")
    plt.title("Primer Eigenvector de Johansen a Través del Tiempo")
    plt.xlabel("Iteraciones del Backtest")
    plt.ylabel("Valor del Eigenvector")
    plt.grid(True)
    plt.legend()
    plt.show()
#######################
# Ejemplo de ejecución: pipeline mínima (selección en train, backtest en test)
if __name__ == "__main__":
    # 15 años atrás
    end = datetime.today()
    start = end - timedelta(days=15 * 365)

    #tickers = [
        #"AAPL", "MSFT", "GOOGL", "META", "IBM", "AMD", "INTC", "NVDA", "ORCL", "CSCO", "TXN",
        #"JPM", "BAC", "WFC", "C", "GS", "MS", "USB", "AXP",
        #"XOM", "CVX", "COP", "BP", "SHEL", "TTE",
       # "WMT", "TGT", "COST", "HD", "LOW", "MCD", "SBUX", "KO", "PEP", "PG",
      #  "UPS", "FDX", "CAT", "DE", "GE",
     #   "JNJ", "PFE", "MRK", "ABBV", "ABT",
    #    "BHP", "RIO", "FCX"
   # ]
    tickers = ['C','MS']
    precios = descargar_datos(tickers, start, end)

    # Split cronológico 60/20/20
    n = len(precios)
    train = precios.iloc[:int(0.6 * n)]
    test = precios.iloc[int(0.6 * n):int(0.8 * n)]
    val = precios.iloc[int(0.8 * n):]
    p = [train,  test,  val]
    p_l = ['train', 'test', 'val']
    print(f"Fechas: start={precios.index[0].date()} end={precios.index[-1].date()}")
    print(f"Tamños: total={n}, train={len(train)}, test={len(test)}, val={len(val)}")

    # Selección de pares en training
    pares = seleccionar_pares(train, min_corr=0.7, p_adf=0.05, corr_window=60)
    print("Pares seleccionados:\n", pares)

    # Backtest simple: ejecutar en el periodo test para cada par seleccionado

    #
    resultados_all = []
    resumen_metrics = []
    for _, row in pares.iterrows():
        for i in range(len(p)):
            print(f'Periodo de {p_l[i]}')
            print("=" * 50)
            A, B = row['A'], row['B']
            # usar solo la porción test
            df_pair_test = p[i][[A, B]].dropna()
            lista_thresholds = np.linspace(0.5, 2.5, 15)

            mejor, tabla = optimizar_threshold_entrada(
                df_pair_test,
                lista_thresholds,
                criterio='retorno_acumulado'  # o 'sharpe', etc.
            )
            print("Tabla resumen:\n", tabla)
            print("Mejor threshold_entrada:", mejor['threshold_entrada'])
            print("Metrics del mejor:", mejor['metrics'])
            df_hist_mejor = mejor['df_hist']
            print(df_hist_mejor)

            # Concatenar y mostrar equity finales

            plt.plot(df_hist_mejor.index, df_hist_mejor['pv'], alpha=0.6)
            plt.title(f'Equity curve por par periodo {p_l[i]}')
            plt.show()
            # 2) Z-Score del spread (DESPUÉS de equity)
            plt.figure(figsize=(12, 5))
            df_hist_mejor['zscore'].plot()
            trades_df = mejor['trades_df']
            theta = mejor['threshold_entrada']

            plt.axhline(theta, linestyle='--', label=f'Umbral Entrada (+θ={theta:.2f})')
            plt.axhline(-theta, linestyle='--', label=f'Umbral Entrada (-θ={theta:.2f})')
            plt.axhline(0, linestyle='--')

            plt.title(f"Z-Score del spread – periodo {p_l[i]}")
            plt.xlabel("Fecha")
            plt.ylabel("Z-Score")
            plt.grid()
            plt.legend()
            plt.show()

            # 3) Hedge ratio (beta) (DESPUÉS del z-score)
            plt.figure(figsize=(12, 5))
            df_hist_mejor['beta'].plot(label='Beta (hedge ratio)')
            # opcional: también alpha
            # df_hist_mejor['alpha'].plot(label='Alpha')

            plt.title(f"Hedge ratio (Kalman) – periodo {p_l[i]}")
            plt.xlabel("Fecha")
            plt.ylabel("Valor del parámetro")
            plt.grid()
            plt.legend()
            plt.show()
            # 4 Retornos por trade
            trades_df['retorno_trade'] = trades_df['delta_shares'] * trades_df['price']
            trades_df['retorno_trade'].hist(bins=30, figsize=(10, 5))
            plt.title("Distribución de Retornos por Trade")
            plt.grid()
            plt.show()
            # 5) Eigenvectores de Johansen (si se guardaron)
            if 'vec1' in df_hist_mejor.columns and 'vec2' in df_hist_mejor.columns:
                plt.figure(figsize=(12, 5))
                df_hist_mejor['vec1'].plot(label='Eigenvector componente 1')
                df_hist_mejor['vec2'].plot(label='Eigenvector componente 2')
                plt.title(f"Eigenvectores Johansen – periodo {p_l[i]}")
                plt.xlabel("Fecha")
                plt.ylabel("Valor")
                plt.grid()
                plt.legend()
                plt.show()