    # kalman 2
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import warnings
class KalmanFilterRegVECM():
    def __init__(self, w_init=(-3, 1.5), q=1e-4, r=25.0, alpha=1.0):
        """
        Igual que KalmanFilterReg pero:
        - alpha controla cuánto pesa el error de cointegración (vecm)
        """
        self.w = np.array(w_init, dtype=float)   # [w0, w1]
        self.A = np.eye(2)
        self.Q = np.eye(2) * q
        self.R = np.array([[r]])
        self.P = np.eye(2)
        self.alpha = alpha  # peso del error VECM

    def predict(self):
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, x, y, err_vecm):
        """
        x: precio B (regresor)
        y: precio A (dependiente)
        err_vecm: error de cointegración en este instante (por ejemplo b1*x + b2*y)
        """
        # matriz de observación
        C = np.array([[1.0, x]])

        # S = C P C^T + R
        S = C @ self.P @ C.T + self.R

        # K = P C^T S^{-1}
        K = self.P @ C.T @ np.linalg.inv(S)

        # error de regresión estándar
        err_reg = (y - C @ self.w)

        # error total: regresión + VECM
        innov = err_reg + self.alpha * err_vecm

        # actualizar parámetros
        self.w = self.w + K @ innov

        # actualizar covarianza
        self.P = (np.eye(2) - K @ C) @ self.P

    @property
    def params(self):
        # devuelve (w0, w1)
        return float(self.w[0]), float(self.w[1])
import yfinance as yf
ticker = ['AMZN','AAPL']
df = yf.download(tickers=ticker, period='15y', interval='1d').dropna()
df = df['Close']
price_A = df['AMZN']
price_B= df['AAPL']
n = len(price_A)
# 2. Inicializamos el filtro de Kalman

vecm_hat_l = []
kalman2 = KalmanFilterRegVECM(w_init=(-3, 1.5), q=1e-5, r=0.5, alpha=0.1)


vecm_norm_l = []
for i in range(252 ,n):
    eig = coint_johansen(df.iloc[i - 252:i, :], det_order=0, k_ar_diff=2)
    e1, e2 = eig.eig
    x1 = price_A[i]
    x2 = price_B[i]
    vecm = e1 * x1 + e2 * x2
    kalman2.predict()
    kalman2.update(x1, x2, vecm)
    e1_hat, e2_hat = kalman2.params
    vecm_hat = x1 * e1_hat + x2 * e2_hat
    vecm_hat_l.append(vecm_hat)
    window = 252
    if len(vecm_hat_l) >= window:
        vecm_window = np.array(vecm_hat_l[-window:])
    else:
        vecm_window = np.array(vecm_hat_l)

    # --- Calculas media y desviación estándar ---
    mean_vecm = np.mean(vecm_window)
    std_vecm = np.std(vecm_window, ddof=1)


    if std_vecm > 0:
        vecm_norm = float((vecm_hat - mean_vecm) / std_vecm)
    else:
        vecm_norm = 0.0
    vecm_norm_l.append(vecm_norm)
print(vecm_norm_l)
#eig = coint_johansen(df.iloc[1:252])
sen = []
for i in range(len(vecm_norm_l)):
    if vecm_norm_l[i] > 1:
        sen.append(1)
    elif vecm_norm_l[i] < -1:
        sen.append(2)
    elif abs(vecm_norm_l[i]) < 0.05:
        sen.append(3)
    else:
        sen.append(4)