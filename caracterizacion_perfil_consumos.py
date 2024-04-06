# -*- coding: utf-8 -*-

"""
@author: Martin Municio, David
"""

# Este es un una herramienta diseñada para la caracterización eléctrica
# de perfiles de consumos eléctricos, que permite realizar balances energéticos 
# y económicos de instalaciones fotovoltaicas de autoconsumo individuales para
# consumidores residenciales, de acuerdo con el sistema tarifario español.


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec
import matplotlib.dates as mdates


# Configuracion de la terminal
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.expand_frame_repr', False)

# Definir colores para las graficas
jet= plt.get_cmap('Dark2')
colors = iter(jet(np.linspace(0,1,10)))
next(colors)
color1 = next(colors)
color2 = next(colors)
color3 = next(colors)
color4 = next(colors)
color5 = next(colors)
color6 = next(colors)


""" Definición de las funciones"""

def importar_consumo(perfil_consumo):
    # Crea un dataframe con el perfil de consumo
    df = pd.read_csv(perfil_consumo,
                     sep=';',
                     skiprows=1,
                     names=['tiempo', 'consumo'],
                     parse_dates=['tiempo'],
                     index_col=['tiempo'],
                     infer_datetime_format=True)
    df.index = [ts.replace(year=2023) for ts in df.index]
    # df.consumo = df.consumo/1000 #Si el input esta  en MWh
    return df

def importar_generacion(perfil_generación_pv):
    # Crea un dataframe con el perfil de generacion obtenido del software
    # de simulacion PVSyst. Las unidades de la energia generada son KWh
    df = pd.read_csv(perfil_generación_pv,
                     sep=';',
                     decimal=',',
                     skiprows=13,
                     names=['tiempo', 'e_generada'],
                     parse_dates=['tiempo'],
                     index_col=['tiempo'],
                     encoding='cp1252'
                     )
    df.replace({-0.001:0},
               inplace=True)
    df.index = [ts.replace(year=2023) for ts in df.index]
    return df

def calcular_exdecentes(df):
    # Añade los términos relacionados con el ahorro energetico, deduciendo 
    # del consumo la producción generada insitu.
    # Evalua el balance entre generacion y consumo y añade los valores
    # en dos nuevas columnas    
    for index,row in df.iterrows():
        if  row['e_generada'] == 0.0:
            df.loc[index, 'consumo_insitu'] = 0
            df.loc[index, 'consumo_con_fv'] = row['consumo']
            df.loc[index, 'excedentes'] = 0
        else:
            if row['e_generada'] > row['consumo']:
                df.loc[index, 'consumo_insitu'] = row['consumo']
                df.loc[index, 'consumo_con_fv'] = 0
                df.loc[index, 'excedentes'] = row['e_generada'] - row['consumo']
            if row['e_generada'] < row['consumo']:
                df.loc[index, 'consumo_insitu'] = row['e_generada'] 
                df.loc[index, 'consumo_con_fv'] = row['consumo'] - row['e_generada']
                df.loc[index, 'excedentes'] = 0

    return df

def balance_energetico(consumo, generacion):
    # Crea el dataframe con el balance energetico entre consumo y generacion
    df_consumo = importar_consumo(consumo)
    df_generacion = importar_generacion(generacion)
    # Junta los dataframes de generacion y consumo
    merge=pd.merge(df_generacion,df_consumo,
                   how='inner',
                   left_index=True,
                   right_index=True)
    merge.index = pd.to_datetime(merge.index, utc=True)
    # Calcula excedentes y consumo insitu
    merge = calcular_exdecentes(merge)
    return merge

def autoconsumo(df, mostrar):
    #Valores mensuales
    mes = df.groupby(pd.Grouper(freq="M")).sum()
    mes.index = mes.index.to_period('M').to_timestamp()
    mes['excedentes_%'] = mes.excedentes/mes.e_generada*100
    mes['autoconsumo'] = mes.consumo_insitu/mes.e_generada*100
    excedentes_anuales_per = mes.excedentes.sum()/mes.e_generada.sum()*100
    excedentes_anuales_tot = mes.excedentes.sum()
    autoconsumo_anual_total = mes.consumo_insitu.sum()/mes.e_generada.sum()*100
    # Boolean para desplegar la informacion sobre autoconsumo
    if mostrar:
        print('=========================================== Informacion Sobre Autoconsumo ===========================================')
        print(mes)
        print(f'\nExcedentes anuales: {excedentes_anuales_tot} Kwh \n{excedentes_anuales_per:.2f} % Con respecto a la generacion total ') 
        print(f'Autoconsumo Anual: {autoconsumo_anual_total:.2f} %\n')  
        print('=========================================================================================================================')
        # Pinta la grafica de autoconsumo0
        grafica_autoconsumo(mes)
    return mes

def autosuficiencia(df, mostrar):
    #Valores mensuales
    mes = df.groupby(pd.Grouper(freq="M")).sum()
    mes.index = mes.index.to_period('M').to_timestamp()
    mes['excedentes_%'] = mes.excedentes/mes.consumo*100
    mes['autosuficiencia'] = mes.consumo_insitu/mes.consumo*100
    excedentes_anuales_per = mes.excedentes.sum()/mes.consumo.sum()*100
    excedentes_anuales_tot = mes.excedentes.sum()
    autosuficiencia_anual_total = mes.consumo_insitu.sum()/mes.consumo.sum()*100
    # Boolean para desplegar la informacion sobre autosuficiencia
    if mostrar:
        print('=========================================== Informacion Sobre Autosuficiencia ============================================')
        print(mes)
        print(f'\nExcedentes anuales: {excedentes_anuales_tot} Kwh \n{excedentes_anuales_per:.2f} % Con respecto al consumo total ') 
        print(f'Autosuficienciia Anual: {autosuficiencia_anual_total:.2f} %\n')  
        print('=============================================================================================')     
        # Pinta La grafica
        grafica_autosuficiencia(mes)
    return mes
    
def añadir_discriminacion_horaria(df):
    #Añade una columna con la discriminación horaria en tarifa_horaria_2_0TD.csv
    df['fecha'] = df.index
    df['dia_semana'] = df.fecha.dt.day_name()
    # Añadir discriminación horaria. 
    df.loc[df.between_time('00:00:00', '07:00:00').index, 'franja_consumo'] = 'Valle'
    df.loc[df.between_time('08:00:00', '10:00:00').index, 'franja_consumo'] = 'Llano'
    df.loc[df.between_time('11:00:00', '14:00:00').index, 'franja_consumo'] = 'Punta'
    df.loc[df.between_time('15:00:00', '18:00:00').index, 'franja_consumo'] = 'Llano'
    df.loc[df.between_time('19:00:00', '22:00:00').index, 'franja_consumo'] = 'Punta'
    df.loc[df.between_time('23:00:00', '23:00:00').index, 'franja_consumo'] = 'Llano'
    # Añadir la discriminación horaria en el fin de semana
    df.loc[df.dia_semana=='Sunday', 'franja_consumo'] = 'Valle'
    df.loc[df.dia_semana=='Saturday', 'franja_consumo'] = 'Valle'
    # Elimina la columna de fecha, y dia ya que no se necesitan en adelante
    df.drop(['fecha', 'dia_semana'], axis=1, inplace=True)
    return df

def añadir_precio_facturacion(df, fichero):
    # Procesamiento de datos del fichero de entrada con los terminos de 
    # facturacion horaria de la tarifa 2.0TD
    # Valores en eur/MWh
    tar_df = pd.read_csv(fichero,
                         sep=';',
                         usecols=["datetime", "value"],
                         parse_dates=['datetime'],
                         index_col=['datetime'])
    tar_df.index = pd.to_datetime(tar_df.index, utc=True)
    tar_df['value'] = tar_df['value']/1000 #eur/KWh
    tar_df.rename(columns={'value':'precio_kwh'},
                  inplace=True)
    # Filtrar para 2023 y expresar valores en kwh
    filt_tar = tar_df.loc['2023-01-01':'2023-12-31']
    # Añadir al dataframe 
    merge=pd.merge(df,filt_tar,
                   how='inner',
                   left_index=True,
                   right_index=True)
    return merge

def añadir_precio_excedentes(df, fichero):
    # Procesamiento de  datos del fichero de entrada que contiene los
    # precios horarios de la compensacion por excedentes
    exce_df = pd.read_csv(fichero,
                          sep=';',
                          usecols=["datetime", "value"],
                          parse_dates=['datetime'],
                          index_col=['datetime'])
    exce_df.index = pd.to_datetime(exce_df.index, utc=True)
    exce_df['value'] = exce_df['value']/1000 #eur/KWh
    exce_df.rename(columns={'value':'precio_kwh_excedentes'},
                   inplace=True)
    # Filtrar para 2023 y expresar valores en kwh
    filt_exce = exce_df.loc['2023-01-01':'2023-12-31']
    # Añadir al dataframe 
    merge=pd.merge(df,filt_exce,
                   how='inner',
                   left_index=True,
                   right_index=True)
    return merge
    
def calcular_costes(df):
    # Esta funcion aplica los precios de facturacion y de compensacion por 
    # excedentes a los balances energeticos calculados previamente
    df['coste_sin_fv'] = df.consumo*df.precio_kwh
    df['coste_con_fv'] = df.consumo_con_fv*df.precio_kwh - df.excedentes*df.precio_kwh_excedentes
    df['valoracion_excedentes'] = df.excedentes*df.precio_kwh_excedentes
    return df
    
def calcular_ahorro_mensual(df, mostrar):
    mes = df.groupby(pd.Grouper(freq="M")).sum()
    mes.index = mes.index.to_period('M').to_timestamp()
    mes['ahorro_eur'] = mes.coste_sin_fv-mes.coste_con_fv
    mes['excedentes_relativos'] = mes.excedentes/mes.consumo
    ahorro_anual = mes.ahorro_eur.sum()
    excedentes_anuales = mes.excedentes.sum()
    valoracion_exce_anual = mes.valoracion_excedentes.sum()
    excedentes_anuales_relativos = valoracion_exce_anual/mes.coste_sin_fv.sum()*100
    # Filtro para el dataframee
    columnas = ['e_generada','consumo','ahorro_eur','excedentes','valoracion_excedentes']
    mes = mes[columnas]
    
    if mostrar:
        print('============================ Informacion Sobre Ahorro Económico =============================')
        print(mes)
        print(f'\nAhorro Anual: {ahorro_anual:.2f} Eur. ')
        print(f'Excedentes Anuales Totales: {excedentes_anuales:.2f} Kwh')
        print(f'Valoracion Excedentes Anuales= {valoracion_exce_anual} Eur')
        print(f'Excedentes Totales Anuales en Relacion al Consumo Electrico Sin Autoconsumo: {excedentes_anuales_relativos:.2f} %\n')
        print('============================================================================================')
        # Pinta la grafica
        grafica_ahorro(mes)
        demanda_generacion(mes)
    return mes

def curva_monotona_de_carga(df):
    # Crea una curva monotona de carga con y sin autoconsumo FV
    # Filtra y ordena los elementos de cada una de las columnas
    df_sin_fv = df[['consumo']].sort_values(by=['consumo'],
                                            ascending=False).reset_index(drop=True)
    df_con_fv = df[['consumo_con_fv']].sort_values(by=['consumo_con_fv'],
                                                   ascending=False).reset_index(drop=True)
    # Crea la grafica
    
    fig, ax = plt.subplots()
    
    plt.plot(df_sin_fv.index,
              df_sin_fv.consumo,
              color=color1,
              label = 'Consumo Sin Instalacion FV')
    
    plt.plot(df_con_fv.index,
              df_con_fv.consumo_con_fv,
              color=color2,
              label = 'Consumo Con Instalacion FV')
    
    ax.fill_between(df_sin_fv.index,
                    df_sin_fv.consumo,
                    df_con_fv.consumo_con_fv,
                    color=color1,
                    alpha=0.1)
    ax.fill_between(df_con_fv.index,
                    df_con_fv.consumo_con_fv,
                    color=color2,
                    alpha=0.1)
    
    plt.grid(visible=True, which='both', axis='both', linestyle='-')
    plt.legend()
    ax.set_xlabel('Horas [h]')
    ax.set_ylabel('Demanda [KWh]')
    ax.set_title('Curva Monótona de Carga')
    

def demanda_generacion(df, diario=False):
    
    fig,ax1 = plt.subplots()
    fig.set_figheight(8)
    fig.set_figwidth(12)
    
    fig.suptitle('Evolucion Anual del Consumo y la Generacion',
                 fontsize=14)
    
    
    ax1.plot(df.index,
              df['e_generada'],
              color='lightseagreen',
              label = 'Generacion FV',
              marker='.')
    
    ax1.plot(df.index,
             df.consumo,
             color='lightcoral',
             label = 'Consumo',
             marker='.')
    
    
    ax1.fill_between(df.index,
                    df['e_generada'],
                    color='lightseagreen',
                    alpha=0.1)
    ax1.fill_between(df.index,
                    df.consumo,
                    color='lightcoral',
                    alpha=0.1)

    # ax.legend(loc='upper right')
    ax1.set_ylabel('Energia KWh')

    ax1.grid(visible=True, which='both', axis='both', linestyle='-')
    fig.legend(bbox_to_anchor=(0.65,0.93), loc='upper left')
    if diario:
        ax1.set_xlabel('[Mes-Dia Hora]')
        ax1.set_title('Valores Diarios')
        
    else:
        ax1.set_xlabel('Mes')
        ax1.set_title('Valores Mensuales')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    fig.show()

def grafica_autoconsumo(df):
    
    autoconsumo_anual = df.consumo_insitu.sum()/df.e_generada.sum()*100
    excedentes_anuales_per = df.excedentes.sum()/df.e_generada.sum()*100
    

    fig = plt.figure()
    fig.set_figheight(8)
    fig.set_figwidth(12)
    
    spec = gridspec.GridSpec(ncols=2, nrows=1,
                             width_ratios=[3, 1], wspace=0.4,
                             hspace=0.5)
    
    fig.suptitle('Análisis del Autoconsumo Respecto Generacion',
                 fontsize=14)
    
    ax1 = fig.add_subplot(spec[0])
    
    ax1.plot(df.index,
              df['excedentes_%'],
              color=color1,
              label = 'Excedentes',
              marker='.')
    
    ax1.plot(df.index,
             df.autoconsumo,
             color=color2,
             label = 'Autoconsumo',
             marker='.')
    ax2 = fig.add_subplot(spec[1])
    
    ax2.bar([1,2],
            [autoconsumo_anual,excedentes_anuales_per],
            color=[color2,color1],
            alpha=0.4)
    
    ax1.fill_between(df.index,
                    df['excedentes_%'],
                    color=color1,
                    alpha=0.1)
    ax1.fill_between(df.index,
                    df.autoconsumo,
                    color=color2,
                    alpha=0.1)

    # ax.legend(loc='upper right')
    ax1.set_ylabel('[%]')
    ax1.set_xlabel('Mes')
    ax1.set_title('Valores Mensuales')
    ax2.set_title('Valores Anuales ')
    ax2.set_ylabel('[%]')
    
    ax1.grid(visible=True, which='both', axis='both', linestyle='-')
    ax2.get_xaxis().set_visible(False)
    fig.legend(bbox_to_anchor=(0.65,0.93), loc='upper left')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    fig.show()

def grafica_autosuficiencia(df):

    autosuficiencia_anual = df.consumo_insitu.sum()/df.consumo.sum()*100
    excedentes_anuales_per = df.excedentes.sum()/df.consumo.sum()*100
    
    #Crea una grafica de autoconsumo
    
    fig = plt.figure()
    fig.set_figheight(8)
    fig.set_figwidth(12)
    
    spec = gridspec.GridSpec(ncols=2, nrows=1,
                             width_ratios=[3, 1], wspace=0.4,
                             hspace=0.5)
    
    
    
    # fig, (ax1, ax2)= plt.subplots(1,2)
    fig.suptitle('Análisis del Autosuficiencia Respecto Consumo',
                  fontsize=14)
    ax1 = fig.add_subplot(spec[0])
    ax1.plot(df.index,
              df['excedentes_%'],
              color=color1,
              label = 'Excedentes',
              marker='.',)
    
    ax1.plot(df.index,
             df.autosuficiencia,
             color=color6,
             label = 'Autosuficiencia',
             marker='.')
    
    ax2 = fig.add_subplot(spec[1])
    
    ax2.bar([1,2],
            [autosuficiencia_anual,excedentes_anuales_per],
            color=[color6,color1],
            alpha=0.4)
    
    ax1.fill_between(df.index,
                    df['excedentes_%'],
                    color=color1,
                    alpha=0.1)
    ax1.fill_between(df.index,
                    df.autosuficiencia,
                    color=color6,
                    alpha=0.1)
    
    ax1.set_ylabel('[%]')
    ax1.set_xlabel('Mes')
    ax1.set_title('Valores Mensuales')
    ax2.set_title('Valores Anuales ')
    ax2.set_ylabel('[%]')
    ax1.grid(visible=True, which='both', axis='both', linestyle='-')
    ax2.get_xaxis().set_visible(False)
    fig.legend(bbox_to_anchor=(0.65,0.93), loc='upper left')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    
    fig.show()

def grafica_ahorro(df):
    
    ahorro_anual = df.ahorro_eur.sum()
    excedentes_anuales= df.valoracion_excedentes.sum()
    
    #Crea una grafica de autoconsumo
    fig = plt.figure()
    fig.set_figheight(8)
    fig.set_figwidth(12)
    
    spec = gridspec.GridSpec(ncols=2, nrows=1,
                             width_ratios=[3, 1], wspace=0.4,
                             hspace=0.5)
    
    
    
    # fig, (ax1, ax2)= plt.subplots(1,2)
    fig.suptitle('Análisis del Ahorro Debido al Autoconsumo',
                  fontsize=14)
    ax1 = fig.add_subplot(spec[0])
    ax1.plot(df.index,
              df['ahorro_eur'],
              color=color3,
              label = 'Ahorro Autoconsumo',
              marker='.',)
    
    ax1.plot(df.index,
              df.valoracion_excedentes,
              color=color1,
              label = 'Valoración Excedentes',
              marker='.')
    
    ax2 = fig.add_subplot(spec[1])
    
    ax2.bar([1,2],
            [ahorro_anual,excedentes_anuales],
            color=[color3,color1],
            alpha=0.4)
    
    ax1.fill_between(df.index,
                    df['ahorro_eur'],
                    color=color3,
                    alpha=0.1)
    ax1.fill_between(df.index,
                    df.valoracion_excedentes,
                    color=color1,
                    alpha=0.1)
    
    
    
    ax1.set_ylabel('Euros')
    ax1.set_xlabel('Mes')
    ax1.set_title('Valores Mensuales')
    ax2.set_title('Valores Anuales ')
    ax1.grid(visible=True, which='both', axis='both', linestyle='-')
    ax2.get_xaxis().set_visible(False)
    ax2.set_ylabel('[Euros]')
    fig.legend(bbox_to_anchor=(0.65,0.93), loc='upper left')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    
    fig.show()

    
    
""" Definición de los inputs """

perfi_consumo_horario = 'consumo_adaptado_entrega_1.csv'
perfil_generación_pv  = 'Simulation and Optimization 1_VC0_HourlyRes_autosuficiencia.csv'
tarifa_horaria = 'tarifa_horaria_2_0TD.csv'
excedentes = 'PrecioDeLaEnergiaExcedentariaDelAutoconsumoParaElMecanismoDeCompensacionSimplificada(PVPC).csv'
facturacion_pvpc = 'export_TerminoDeFacturacionDeEnergiaActivaDelPVPC20TD.csv'




if __name__ == '__main__':
    
    # 0. Procesar Los datos
    datos = balance_energetico(consumo=perfi_consumo_horario,
                               generacion=perfil_generación_pv)
    # 1.Calcular Autoconsumo
    autoconsumo = autoconsumo(datos, mostrar=True)
    # Calcular Autosuficiencia
    autosuficiencia = autosuficiencia(datos, mostrar=True)
    # Añadir Discriminación Horaria y precio de facturación
    datos = añadir_discriminacion_horaria(datos)
    datos = añadir_precio_facturacion(datos,
                                      fichero=facturacion_pvpc )
    datos = añadir_precio_excedentes(datos,
                                     fichero=excedentes )
    # Calcular Costes
    datos = calcular_costes(datos)
    ahorro = calcular_ahorro_mensual(datos, mostrar=True)
    
    curva_monotona_de_carga(datos)
    
    
    
    





