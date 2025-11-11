from copyreg import clear_extension_cache

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import warnings

class KalmanFilterReg():
    def __init__(self, w_init=(-3, 1.5), q=1e-4, r=25.0):
        # estimación inicial de parámetros (intercepto y pendiente)
        self.w = np.array(w_init, dtype=float)   # [w0, w1]

        # matriz de transición (modelo estático: parámetros casi no cambian)
        self.A = np.eye(2)

        # ruido en la dinámica de los parámetros (process noise)
        self.Q = np.eye(2) * q

        # ruido en las observaciones (observation noise)
        self.R = np.array([[r]])

        # covarianza inicial de la estimación de w
        self.P = np.eye(2)

    def predict(self):
        # P_t|t-1 = A P_{t-1|t-1} A^T + Q
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, x, y):
        # C = [1, x_t]
        C = np.array([[1.0, x]])
        # S = C P C^T + R
        S = C @ self.P @ C.T + self.R
        # K = P C^T S^{-1}
        K = self.P @ C.T @ np.linalg.inv(S)

        # actualizar covarianza
        self.P = (np.eye(2) - K @ C) @ self.P

        # actualizar parámetros w
        self.w = self.w + K @ (y - C @ self.w)

    @property
    def params(self):
        # devuelve (w0, w1)
        return float(self.w[0]), float(self.w[1])
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
kalman = KalmanFilterReg(w_init=(-3, 1.5), q=1e-5, r=0.5)
spread = []
fair_value = []
vecm_hat_l = []
kalman2 = KalmanFilterRegVECM(w_init=(-3, 1.5), q=1e-5, r=0.5, alpha=0.1)
# Parámetros para la ventana de z-score
window = 15
zscore = []
hr = []
signals = []
vecm_norm_l = []
for i in range(n):
    warnings.filterwarnings('ignore')
    #kalman 1

    kalman.predict()

    x = price_B[i]  # activo B
    y = price_A[i]  # activo A

    kalman.update(x, y)
    w0, w1 = kalman.params
    hr.append(w1)
    y_fair = w0 + w1 * x
    fair_value.append(y_fair)

plt.figure(figsize=(10, 4))
plt.scatter(price_B, price_A, s=10, alpha=0.5, label='Datos')
plt.plot(price_B, fair_value, label='Línea Kalman (últimos params)', linewidth=2)
plt.xlabel('Precio B')
plt.ylabel('Precio A')
plt.legend()
plt.title('Relación A vs B con estimación de Kalman')
plt.show()