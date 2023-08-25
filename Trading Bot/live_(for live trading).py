import MetaTrader5 as mt5
from functions import read_data
from datagen import gen_data

import numpy as np
import pandas as pd
import tensorflow as tf
import talib
import math
import time 
start = time.time()

from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential, initializers
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout
from tensorflow.python.client import device_lib 
        
from sklearn.preprocessing import MinMaxScaler
import joblib
#import matplotlib.pyplot as plt

from time import sleep, process_time
from datetime import datetime

from pathlib import Path
mt5.initialize()

#-----------------------------------------------------------------------------------------
#------------------------------------------------ Daten
#----------------------------------------------------------------------------------------- 
start_kapital       =   10000
lotsize             =   0.1
gebuehren           =   0.8           #0.6€ bei 0.1 lot €
#Symbol              =   "EURUSD"

data_anzahl         =   1        #Daten laden
data_verschieb      =   0
Anzahl_Inputs       =   5
Version             =   "V10_"
Dense_network       =   True         #if false = LSTM Network


#-----------------------------------------------------------------------------------------
#------------------------------------------------ Symbole
#-----------------------------------------------------------------------------------------
# Symbol=[]
# for s in range(0, 1):  
#    Symbol.append("US500") 
Symbol = np.array(["US500"])

   #Symbol.append("AUDCHF") 
  # Symbol.append("EURCHF")
   #Symbol.append("USDCHF") 
     

for Symbol in Symbol:   
    print(Symbol)   
    #-----------------------------------------------------------------------------------------
    #------------------------------------------------ Path
    #-----------------------------------------------------------------------------------------
    directory           =   str(Path.cwd())                         # Get the parent directory of the current working directory
    data_directory      =   'C:\\Users\\Administrator\\AppData\\Roaming\\MetaQuotes\\Terminal\\A8F185E7EED350E64E3329CD6D7497DC\\MQL5\\Files\\Data\\Data_KI'
    model_directory     =   'C:\\Users\\Administrator\\AppData\\Roaming\\MetaQuotes\\Terminal\\A8F185E7EED350E64E3329CD6D7497DC\\MQL5\\Files\\Data\\Model'
         
    input_data_path     =   data_directory + '/input_data_'+Symbol+'.csv'  
    close_data_path     =   data_directory + '/close_data_'+Symbol+'.csv'   
    scaler_Path         =   model_directory + '/scaler_'+Symbol 
    model_path          =   model_directory + '/model_'
    
    testdata_path       =   data_directory + '/test_input_data_'+Symbol+'.csv'
    
    
    live_directory      =   directory + "/live" 
    live_path           =   'C:\\Users\\Administrator\\AppData\\Roaming\\MetaQuotes\\Terminal\\A8F185E7EED350E64E3329CD6D7497DC\\MQL5\\Files\\Data\\Output_'+Symbol+'.csv'
    
    #-----------------------------------------------------------------------------------------
    #------------------------------------------------ Daten laden
    #----------------------------------------------------------------------------------------- 
    
    gen_data(data_anzahl,data_verschieb, Symbol, Anzahl_Inputs,input_data_path,close_data_path)
    
    model = Sequential()
    model = tf.keras.models.load_model(model_path + Version + Symbol)
    scaler = joblib.load(scaler_Path)
    
    #Initalisieren
    x_test = read_data(input_data_path, Dense_network)
    
    cls_data = pd.read_csv(close_data_path,sep=';',header=None)
    cls_data = cls_data.to_numpy().flatten() 
    
    x_test_scaled  = scaler.transform(x_test)    # für GRU / LSTM x_test  = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))           
    #np.savetxt(testdata_path,x_test_scaled,delimiter=';')  
    
    
    
    #-----------------------------------------------------------------------------------------
    #------------------------------------------------ Auswerten
    #-----------------------------------------------------------------------------------------
    
    predictions = model.predict(x_test_scaled)
    action = np.argmax(predictions, axis=1)
    print(action)

    
    # np.savetxt(live_path, action, delimiter=';')
    np.savetxt(live_path, action, delimiter=';', fmt='%d')
 
    

ende = time.time()
print('{:5.3f}s'.format(ende-start))
    
mt5.shutdown()
