# test del modulo pd_derivata

from pd_derivata import pandas_derivata_robusta
import pandas as pd
import plotly.express as px

dati = pd.read_csv('gsc_rasatura gluta.csv')
dati_plus = pandas_derivata_robusta(dati,'Time','Unsubtracted Weight')
print(dati_plus.head(10))
fig_dM = px.line(
    dati_plus,
    x='Sample Temperature',
    y='derivata',
    color_discrete_sequence=['red'],
)
fig_dM.show()
fig_M = px.line(
    dati_plus,
    x='Sample Temperature',
    y='Unsubtracted Weight',
    color_discrete_sequence=['red'],
)
fig_M.show()
print('OK')