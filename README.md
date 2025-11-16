# Proyecto de Pairs Trading con Filtro de Kalman

Este repositorio contiene el código principal de una estrategia de **pairs trading** que utiliza:

- Pruebas estadísticas (ADF / Johansen en algunas versiones)
- Estimación dinámica del **hedge ratio** con **Filtro de Kalman**
- Backtesting con cálculo de **equity curve**, métricas básicas de desempeño y gráficas por periodo (*train*, *test*, *val*)

> 💡 **Importante:** para reproducir la estrategia solo se necesita el archivo `main.py`.  
Para este solo se necesita el main, el resto fue para compartir los modelos de como usarlos entre yo y mi colaborador Jesus Garcia.

Por ultimo para usar el codigo solo se necesitan los requisitos de abajo y aparte solo correr el main, para repoducibilidad rapida comente la lista de tickers originales que hace la seleccion de pares y solo deje la que se uso al final del modelo, solo se necesita descomentar eso y borrar la linea abajo de tickers que solo contienen "C" "MS" y ya estara listo para correr el programa original

---

## Requisitos

- Python 3.9 o superior (recomendado)
- Acceso a internet si se descargan datos con `yfinance`
librerias usadas
beautifulsoup4==4.14.2
certifi==2025.10.5
cffi==2.0.0
charset-normalizer==3.4.4
contourpy==1.3.2
curl_cffi==0.13.0
cycler==0.12.1
fonttools==4.60.1
frozendict==2.4.6
idna==3.11
kiwisolver==1.4.9
matplotlib==3.10.7
multitasking==0.0.12
numpy==2.2.6
packaging==25.0
pandas==2.3.3
patsy==1.0.2
peewee==3.18.3
pillow==12.0.0
platformdirs==4.5.0
protobuf==6.33.0
pycparser==2.23
pyparsing==3.2.5
python-dateutil==2.9.0.post0
pytz==2025.2
requests==2.32.5
scipy==1.15.3
six==1.17.0
soupsieve==2.8
statsmodels==0.14.5
typing_extensions==4.15.0
tzdata==2025.2
urllib3==2.5.0
websockets==15.0.1
yfinance==0.2.66
---


   git clone https://github.com/usuario/repositorio-kalman-pairs.git
   cd repositorio-kalman-pairs

