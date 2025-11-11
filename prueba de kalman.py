import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import yfinance as yf
ticker = [
    "ALFAA.MX", "ALSEA.MX", "AMXB.MX", "AC.MX", "BBAJIOO.MX", "GFNORTEO.MX", "BOLSAA.MX",
    "CEMEXCPO.MX", "CHDRAUIB.MX", "KOFUBL.MX", "VESTA.MX", "LIVEPOLC-1.MX", "FEMSAUBD.MX", "LABB.MX", "GENTERA.MX",
    "GRUMAB.MX", "OMAB.MX", "CUERVO.MX", "ASURB.MX", "BIMBOA.MX", "GCARSOA1.MX", "GCC.MX", "ELEKTRA.MX",
    "GMEXICOB.MX", "GFINBURO.MX", "KIMBERA.MX", "MEGACPO.MX", "ORBIA.MX", "PE&OLES.MX", "PINFRA.MX","GAPB.MX",
    "Q.MX", "RA.MX", "TLEVISACPO.MX", "WALMEX.MX","LACOMERUBC.MX"
]
df = yf.download(tickers=ticker, period='15y', interval='1d').dropna()
df = df['Close']
acc_coin=[]
for i in range(len(ticker)):
    price_A = df[ticker[i]]
    for j in range(len(ticker)):
        price_B= df[ticker[j]]
        if ticker[i] == ticker[j]:
            print('son la misma accion')
        else:
            # 1️⃣ Estimamos la regresión A = α + βB
            X = sm.add_constant(price_B)  # agrega el intercepto α
            model = sm.OLS(price_A, X).fit()
            alpha, beta = model.params
            print(f"α = {alpha:.4f}, β = {beta:.4f}")

            # 2️⃣ Calculamos los residuos (spread)
            residuals = model.resid
            # 3️⃣ Aplicamos la prueba ADF sobre el spread
            adf_result = adfuller(residuals)
            print("ADF Statistic:", adf_result[0])
            print("p-value:", adf_result[1])

            if adf_result[1] < 0.05:
                print("El spread es estacionario → series cointegradas.")
                temp = (ticker[i], ticker[j])
                acc_coin.append(temp)
            else:
                print("El spread NO es estacionario → no cointegradas.")
print(acc_coin)