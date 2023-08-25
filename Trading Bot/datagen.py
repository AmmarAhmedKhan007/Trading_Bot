


#-------------------------------------------------------------load
import MetaTrader5 as mt5

import numpy as np
import pandas as pd
import tensorflow as tf
import talib
import math
import sys

        
from sklearn.preprocessing import MinMaxScaler
import joblib
#import matplotlib.pyplot as plt

from time import sleep, process_time
from datetime import datetime

from pathlib import Path


def gen_data(data_anzahl,symbol_load,Anzahl_Inputs,input_data, close_data, ma_data, Zeiteinheit ): 
    
    mt5.initialize()    
    #-------------------------------                            Inputs    
    
    DatenAnzahl     =   data_anzahl
    symbol          =   symbol_load
    SP              =   0
    EP              =   Anzahl_Inputs
    maxV            =   100                                       #np.amax(x_train)*(1+skal)  
    minV            =   -100                                      #np.amin(x_train)*(1-skal)
    
    InputX=[]
    x_train=[]
    
    account_info = mt5.account_info()
    
    if account_info == None:
        print(" ")
        print("no Connection to trading account!!!")
        print( f'Check - Real Trading Account / Accountinfo = {account_info}')
        print("Check latest Beta Version of your Metatrader5 - Sign in to a MetaQuotes Demo Account - ")
        print("then go to Help > Check for Beta-uptates")
        sys.exit()
        
    if Zeiteinheit == "4H":
        DataI = mt5.copy_rates_from_pos( symbol, mt5.TIMEFRAME_H4 ,1, DatenAnzahl+1000)    
    if Zeiteinheit == "1H":
        DataI = mt5.copy_rates_from_pos( symbol, mt5.TIMEFRAME_H1 ,1, DatenAnzahl+1000)
    if Zeiteinheit == "15min":
        DataI = mt5.copy_rates_from_pos( symbol, mt5.TIMEFRAME_M15 ,1, DatenAnzahl+1000)
    if Zeiteinheit == "10min":
        DataI = mt5.copy_rates_from_pos( symbol, mt5.TIMEFRAME_M10 ,1, DatenAnzahl+1000)
    if Zeiteinheit == "5min":
        DataI = mt5.copy_rates_from_pos( symbol, mt5.TIMEFRAME_M5 ,1, DatenAnzahl+1000)
    if Zeiteinheit == "3min":
        DataI = mt5.copy_rates_from_pos( symbol, mt5.TIMEFRAME_M3 ,1, DatenAnzahl+1000)
    if Zeiteinheit == "1min":
        DataI = mt5.copy_rates_from_pos( symbol, mt5.TIMEFRAME_M1 ,1, DatenAnzahl+1000)    
    
    DataI = pd.DataFrame(DataI)
    
    if DataI.empty:
        print(" ")
        print("Dataframe is empty")
        print("Daten wurden nicht richtig geladen")
        print( f'Check - Symbol Name  / Symbolname = {symbol}')
        sys.exit()
        
    DataI['time']=pd.to_datetime(DataI['time'], unit='s') 
    
    def min_func(ts):
        return ts.minute
    def hr_func(ts):
        return ts.hour
    
    times_hour       = DataI['time'].apply(hr_func)
    times_min        = DataI['time'].apply(min_func)
    
    times       = DataI['time']
    open        = DataI['open']
    high        = DataI['high']
    low         = DataI['low']
    close       = DataI['close'] 
    times       = np.asarray(times) 
    open        = np.asarray(open)   
    high        = np.asarray(high)
    low         = np.asarray(low)
    close       = np.asarray(close) 
    
    MA = talib.EMA(close, timeperiod=5)    
    MA  = MA [~np.isnan(MA)]    
    
    atr1 = talib.ATR(high, low, close, timeperiod=10)    
    atr1  = atr1 [~np.isnan(atr1)]    
    atr2 = talib.ATR(high, low, close, timeperiod=30)    
    atr2  = atr2 [~np.isnan(atr2)]    
    atr3 = talib.ATR(high, low, close, timeperiod=50)    
    atr3  = atr3 [~np.isnan(atr3)]    
     
    rocI1 =  talib.ROC(close, timeperiod=5)     
    rocI1  = rocI1 [~np.isnan(rocI1)]    
    rocI2 =  talib.ROC(close, timeperiod=12)     
    rocI2  = rocI2 [~np.isnan(rocI2)]  
    rocI3 =  talib.ROC(close, timeperiod=17)     
    rocI3  = rocI3 [~np.isnan(rocI3)]          
    rocI4 =  talib.ROC(close, timeperiod=50)     
    rocI4  = rocI4 [~np.isnan(rocI4)]            
    rocI5 =  talib.ROC(close, timeperiod=70)     
    rocI5  = rocI5 [~np.isnan(rocI5)]           
    rocI6 =  talib.ROC(close, timeperiod=100)     
    rocI6  = rocI6 [~np.isnan(rocI6)]  
    
    slowk1, slowd1 = talib.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)           
    slowd1  =slowd1 [~np.isnan(slowd1)]  
    slowk2, slowd2 = talib.STOCH(high, low, close, fastk_period=10, slowk_period=5, slowk_matype=0, slowd_period=5, slowd_matype=0)   
    slowd2  =slowd2 [~np.isnan(slowd2)]  
    slowk3, slowd3 = talib.STOCH(high, low, close, fastk_period=20, slowk_period=7, slowk_matype=0, slowd_period=5, slowd_matype=0)
    slowd3  = slowd3 [~np.isnan(slowd3)]        
         
    rsi1 =  talib.RSI(close, timeperiod=14)    
    rsi1  = rsi1 [~np.isnan(rsi1)]           
    rsi2 =  talib.RSI(close, timeperiod=28)     
    rsi2  = rsi2 [~np.isnan(rsi2)] 
    
    di1 = talib.PLUS_DI(high, low, close, timeperiod=10)    
    di1  = di1 [~np.isnan(di1)]     
    di2 = talib.PLUS_DI(high, low, close, timeperiod=30)
    di2  = di2 [~np.isnan(di2)] 
    
    PPO1 = talib.PPO(close, fastperiod=7, slowperiod=14, matype=0)   
    PPO1  = PPO1 [~np.isnan(PPO1)]    
    PPO2 = talib.PPO(close, fastperiod=20, slowperiod=60, matype=0)   
    PPO2  = PPO2 [~np.isnan(PPO2)]      
    
    willi1 = talib.WILLR(high, low, close, timeperiod=10)
    willi1 = willi1 [~np.isnan(willi1)] 
    willi2 = talib.WILLR(high, low, close, timeperiod=30)
    willi2 = willi2 [~np.isnan(willi2)]     
    willi3 = talib.WILLR(high, low, close, timeperiod=60)
    willi3 = willi3 [~np.isnan(willi3)]       
    
    
    #----trendindicatoren
        
    upperband1, middleband1, lowerband1 = talib.BBANDS(close, timeperiod=100, nbdevup=1, nbdevdn=1, matype=0)     
    upperband2, middleband2, lowerband2 = talib.BBANDS(close, timeperiod=100, nbdevup=2, nbdevdn=2, matype=0)
    upperband3, middleband3, lowerband3 = talib.BBANDS(close, timeperiod=100, nbdevup=3, nbdevdn=3, matype=0)
    
    upperband1 = upperband1 [~np.isnan(upperband1)]  
    upperband2 = upperband2 [~np.isnan(upperband2)] 
    upperband3 = upperband3 [~np.isnan(upperband3)] 
    
    middleband1 = middleband1 [~np.isnan(middleband1)]  
    middleband2 = middleband2 [~np.isnan(middleband2)]  
    middleband3 = middleband3 [~np.isnan(middleband3)] 
    
    lowerband1 = lowerband1 [~np.isnan(lowerband1)]  
    lowerband2 = lowerband2 [~np.isnan(lowerband2)]  
    lowerband3 = lowerband3 [~np.isnan(lowerband3)]  
     
    
    DMA1 = talib.DEMA(close, timeperiod=30)
    DMA2 = talib.DEMA(close, timeperiod=50)
    DMA3 = talib.DEMA(close, timeperiod=150)
    DMA4 = talib.DEMA(close, timeperiod=200)
    DMA5 = talib.DEMA(close, timeperiod=500) 
    
    DMA1 = DMA1 [~np.isnan(DMA1)] 
    DMA2 = DMA2 [~np.isnan(DMA2)] 
    DMA3 = DMA3 [~np.isnan(DMA3)] 
    DMA4 = DMA4 [~np.isnan(DMA4)] 
    DMA5 = DMA5 [~np.isnan(DMA5)] 
    
    
    Angle1 = talib.LINEARREG_ANGLE(close, timeperiod=30)
    Angle2 = talib.LINEARREG_ANGLE(close, timeperiod=50)
    Angle3 = talib.LINEARREG_ANGLE(close, timeperiod=200)
    Angle1 = Angle1 [~np.isnan(Angle1)] 
    Angle2 = Angle2 [~np.isnan(Angle2)] 
    Angle3 = Angle3 [~np.isnan(Angle3)] 
    
    
    cor1 = talib.CORREL(high, low, timeperiod=30)
    cor2 = talib.CORREL(high, low, timeperiod=100)
    cor1 = cor1[~np.isnan(cor1)] 
    cor2 = cor2[~np.isnan(cor2)] 
    
    #---------------------------------------------------Daten umdrehen
    times_hour  = times_hour [::-1]
    times_min  = times_min [::-1]
    
    MA = MA[::-1]
    close = close[::-1]
    open = open[::-1]
    high = high[::-1]
    low = low[::-1]
    
    times = times[::-1]      
    
    atr1 = atr1[::-1] 
    atr2 = atr2[::-1] 
    atr3 = atr3[::-1] 
    
    rocI1 = rocI1[::-1] 
    rocI2 = rocI2[::-1] 
    rocI3 = rocI3[::-1] 
    rocI4 = rocI4[::-1]   
    rocI5 = rocI5[::-1]    
    rocI6 = rocI6[::-1]    
    
    slowd1 = slowd1[::-1] 
    slowd2 = slowd2[::-1] 
    slowd3 = slowd3[::-1] 
    
    rsi1 = rsi1[::-1]     
    rsi2 = rsi2[::-1] 
    
    di1 = di1[::-1]     
    di2 = di2[::-1]
    
    PPO1 = PPO1[::-1]     
    PPO2 = PPO2[::-1]
    
    willi1 = willi1[::-1] 
    willi2 = willi2[::-1] 
    willi3 = willi3[::-1] 
        
    #----trendindicatoren
    upperband1 = upperband1[::-1]
    upperband2 = upperband2[::-1]
    upperband3 = upperband3[::-1]
    middleband1 = middleband1 [::-1]
    middleband2 = middleband2 [::-1]
    middleband3 = middleband3 [::-1]  
    lowerband1 = lowerband1[::-1] 
    lowerband2 = lowerband2[::-1] 
    lowerband3 = lowerband3[::-1]
    
    DMA1 = DMA1[::-1]
    DMA2 = DMA2[::-1]
    DMA3 = DMA3[::-1]
    DMA4 = DMA4[::-1]
    DMA5 = DMA5[::-1]
    
    Angle1 = Angle1[::-1]
    Angle2 = Angle2[::-1]
    Angle3 = Angle3[::-1]
    
    
    #---------------------------------------------------Daten hinzufugen
    for a in range(0,DatenAnzahl):           
       for i in range(SP, EP):                       
          InputX.append(rocI1[i+a]) 
       for i in range(SP, EP):                       
          InputX.append(rocI2[i+a]) 
       for i in range(SP, EP):                       
          InputX.append(rocI3[i+a]) 
       for i in range(SP, EP):                       
          InputX.append(rocI4[i+a]) 
       for i in range(SP, EP):                       
          InputX.append(rocI5[i+a]) 
       for i in range(SP, EP):                       
          InputX.append(rocI6[i+a])     
            
          
       np.transpose(InputX)
       x_train.append(InputX)
       InputX=[]
    
    x_train = x_train[::-1]  
          
    pd.DataFrame(x_train).to_csv(input_data,sep=';',header=False ,index=False)  
    
    InputY=[]
    y_train=[]
    
    for a in range(0,DatenAnzahl):                 
       for i in range(0, 1):                        
          InputY.append(close[i+a])    
        
       np.transpose(InputY)
       y_train.append(InputY)
       InputY=[]
       
    y_train = y_train[::-1]  
    pd.DataFrame(y_train).to_csv(close_data,sep=';',header=False ,index=False)  
    
    
    InputZ=[]
    z_train=[]
    
    for a in range(0,DatenAnzahl):                 
       for i in range(0, 1):                         
          #InputZ.append(MA[i+a])                   
          InputZ.append(close[i+a])     
        
       np.transpose(InputZ)
       z_train.append(InputZ)
       InputZ=[]
       
    z_train = z_train[::-1]  
    pd.DataFrame(z_train).to_csv(ma_data,sep=';',header=False ,index=False)  
    
    
    
mt5.shutdown()












































