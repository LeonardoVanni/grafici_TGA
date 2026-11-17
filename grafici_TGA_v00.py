# Grafica di file csv da TGA con una web-app streamlit
#
# Lo script è finalizzato a caricare le registrazioni csv della TGA
# per visualizzare l'andamento nel tempo delle grandezze registrate
# e le correlazioni con la temperatura
#
# L. Vanni, vers.0, 13/11/2026
#

# Librerie
import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
from pd_derivata import pandas_derivata_robusta

# funzioni ausiliarie
def unisci_fig(fig_1, fig_2):
    "Unisce le figure fig_1 e fig_2 producendo una figura streamlit multiplot, fig_U"
    
    # creazione di una figura con più grafici sovrapposti
    fig_U = make_subplots(specs=[[{'secondary_y': True}]])
    
    # Aggiunta delle tracce alla figura principale con doppio asse Y
    fig_U.add_trace(fig_1.data[0], secondary_y=False)  # Asse Y di sinistra
    fig_U.add_trace(fig_2.data[0], secondary_y=True)  # Asse Y di destra

    fig_U.update_traces(marker=dict(size=2))

    fig_U.update_xaxes(
        showspikes=True,
        spikemode='across',
        spikesnap='cursor',
        spikethickness=1,
    )
    fig_U.update_yaxes(
        showspikes=True,
        spikemode='across',
        spikesnap='cursor',
        spikethickness=1,
    )
    fig_U.update_layout(
        hovermode='x unified'
    )  # mostra tutti i valori alla stessa x in un unico tooltip

    fig_U.update_layout(
        font=dict(size=13, family='sans-serif'),
        legend=dict(
            orientation='h',  # Disposizione orizzontale
            yanchor='bottom',
            y=1.05,  # Posizionata appena sopra il bordo superiore del grafico
            xanchor='right',
            x=1,  # Allineata a destra
        ),
        template='plotly_white',  # Tema pulito dello sfondo
    )
    return fig_U

# Intestazione interfaccia
"""
Visualizzazione registrazioni TGA
============================
"""

# Acquisizione dei dati in pandas dataframe
# l'oggetto file_caricato è un oggetto nativo streamlit per i file
file_caricato = st.file_uploader("Carica il file csv con i dati della TGA", type=["csv"])

if file_caricato is not None:
    dati = pd.read_csv(file_caricato)

    # RIORGANIZZAZIONE DEI DATI
    dati.rename(columns={'Time':'Tempo [min]',
                         'Unsubtracted Weight':'massa [mg]',
                         'Program Temperature':'setpoint temperatura [°C]',
                         'Sample Temperature':'temperatura [°C]'
                         },
                inplace=True)
    dati.drop(columns=['Baseline Weight',
                       'Approx. Gas Flow',
                       'R25 Diagnostic Signal'],
                inplace=True)
    dati.pop(dati.columns[-1])

    # VISUALIZZAZIONE TABELLARE dei 'dati' caricati
    st.subheader("Riepilogo dati registrati")
    st.dataframe(
    dati,
    height=200,
    width='stretch',
    hide_index=True,
    key="tabella",
     )

    # ELABORAZIONE DEI DATI
    #
    # parametro di lisciatura derivate
    st.subheader('Lisciatura derivate')
    opzioni_liscia = ['nessuna', 'bassa', 'media', 'alta']
    lisciatura = st.select_slider('grado di lisciatura',options=opzioni_liscia)
    match lisciatura:
        case 'nessuna':
            smooth_window = 3
        case 'bassa':
            smooth_window = 31
        case 'media':
            smooth_window = 301
        case 'alta':
            smooth_window =3001      
    #
    # Calcolo della massa percentuale del campione, Mp
    M_0 = dati.loc[[0, 1, 2], 'massa [mg]'].mean() # massa iniziale del campione
    dati['M/M_0 [%]'] = dati['massa [mg]']/M_0*100

    # Derivata di Mp vs tempo
    dati_dMp_dt = pandas_derivata_robusta(dati,'Tempo [min]','M/M_0 [%]', smooth_window=smooth_window)

    # Derivata di TEMPERATURA vs tempo
    dati_dT_dt = pandas_derivata_robusta(dati,'Tempo [min]','temperatura [°C]', smooth_window=smooth_window)

    # GRAFICA DEI DATI
    #     
    ## GRAFICO TEMPERATURA vs tempo
    
    # 1. Creiamo la prima linea e abilitiamo esplicitamente la legenda con un nome
    fig_T = px.line(
        dati_dT_dt,
        x='Tempo [min]',
        y='temperatura [°C]',
        color_discrete_sequence=['red'],
    )
    fig_T.update_traces(name='Temperatura', showlegend=True)

    # 2. Creiamo la seconda linea e abilitiamo esplicitamente la legenda con un nome
    fig_dT = px.line(
        dati_dT_dt,
        x='Tempo [min]',
        y='derivata',
        color_discrete_sequence=['blue'],
    )
    fig_dT.update_traces(name='dT/dt', showlegend=True)
    fig_dT.update_traces(line=dict(dash='dash', width=2))

    # Unione delle due figure in un'unica figura con plot sovrapposti
    fig_TdT = unisci_fig(fig_T, fig_dT)

    # titoli della figura unione
    fig_TdT.update_layout(title='Temperatura del campione e velocità di riscaldamento')
    fig_TdT.update_xaxes(title_text='<b>tempo [min]</b>')
    fig_TdT.update_yaxes(title_text='<b>Temperatura [°C]</b>', secondary_y=False)
    fig_TdT.update_yaxes(title_text='<b>dT/dt [°C/min]</b>', secondary_y=True)

    st.plotly_chart(fig_TdT)

    ## GRAFICO MASSA vs tempo

    fig_M = px.line(
        dati,
        x='Tempo [min]',
        y='M/M_0 [%]',
        color_discrete_sequence=['red'],
    )
    fig_M.update_traces(name='Massa percentuale del campione', showlegend=True)

    fig_M.update_xaxes(
        showspikes=True,
        spikemode='across',
        spikesnap='cursor',
        spikethickness=1,
    )
    fig_M.update_layout(
        hovermode='x unified'
    )  # mostra tutti i valori alla stessa x in un unico tooltip

    fig_M.update_layout(
        title_text='Massa percentuale del campione nel tempo',
        font=dict(size=13, family='sans-serif'),
        legend=dict(
            orientation='h',  # Disposizione orizzontale
            yanchor='bottom',
            y=1.05,  # Posizionata appena sopra il bordo superiore del grafico
            xanchor='right',
            x=1,  # Allineata a destra
        ),
        template='plotly_white',  # Tema pulito dello sfondo
    )

    fig_M.update_xaxes(title_text='<b>tempo [min]</b>')
    fig_M.update_yaxes(title_text='<b>Massa/Massa inziale [%]</b>')


    st.plotly_chart(fig_M)

    ## GRAFICI MASSA% vs TEMPERATURA e derivata MASSA% vs tempo

    # 1. Creiamo la prima linea e abilitiamo esplicitamente la legenda con un nome
    fig_Mp = px.line(
        dati_dMp_dt,
        x='temperatura [°C]',
        y='M/M_0 [%]',
        color_discrete_sequence=['red'],
    )
    fig_Mp.update_traces(name='M/M_0 [%]', showlegend=True)

    # 2. Creiamo la seconda linea e abilitiamo esplicitamente la legenda con un nome
    fig_dMp_dt = px.line(
        dati_dMp_dt,
        x='temperatura [°C]',
        y='derivata',
        color_discrete_sequence=['blue'],
    )
    fig_dMp_dt.update_traces(name='d(M/M0%)/dt', showlegend=True)
    fig_dMp_dt.update_traces(line=dict(dash='dash', width=2))

    # Unione dei due grafici
    fig_Mp_dMp = unisci_fig(fig_Mp, fig_dMp_dt)
    fig_Mp_dMp.update_xaxes(title_text='<b>Temperatura [°C]</b>')
    fig_Mp_dMp.update_yaxes(title_text='<b>Massa campione [%]</b>', secondary_y=False)
    fig_Mp_dMp.update_yaxes(title_text='<b>d(M/M0%)/dt [%/min]</b>', secondary_y=True)

    st.plotly_chart(fig_Mp_dMp)