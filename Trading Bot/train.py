import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
from tensorflow.keras import Sequential, initializers
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.python.client import device_lib 

from functions import *
from datagen import gen_data
import time 

import sys
import pandas as pd
import joblib
import random
from pathlib import Path


config = tf.compat.v1.ConfigProto(gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

def model_create(U4, U3, U2, U1, count_inputs):
    model = Sequential()      
    model.add(Dense(units=U4, input_dim=count_inputs, activation="relu"))
    model.add(Dropout(0.33))        
    model.add(Dense(units=U3, activation="relu"))
    model.add(Dense(units=U2, activation="relu"))
    model.add(Dense(units=U1, activation="relu"))
    model.add(Dense(action_size, activation="linear"))            #/softmax - linear - sigmoid
    model.compile(loss="mse", optimizer=Adam(lr=0.0001))    #/categorical_crossentropy - mse / binary_crossentropy
               
    print(model.summary()) 
    return model

def act(state, model):                 
    options = model.predict(state)
    return np.argmax(options[0])            

def expReplay(bz, memory, model, gamma):
    target_f_array = []
    state_array = []
    
    for state, action, reward, next_state, done in memory:
        target = reward
        if not done:
            target = reward + gamma * np.amax(model.predict(next_state)[0])

        target_f = model.predict([state])
        target_f[0][action] = target
        model.fit(state, target_f, epochs=1, verbose=0 )    
    
    
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
    
def read_data(input_data_path):
    data = pd.read_csv(input_data_path,sep=';',header=None)
    data = data.to_numpy() 
        
    return data

def scaled_data(input_data_path,scaler_Path, Skalierung ,count_data_train, Dense_network):
    data = pd.read_csv(input_data_path,sep=';',header=None)
    data = data.to_numpy() 
    
    traindata  = data[0:count_data_train]      #  [start:stop:step]
    
    #scalieren        
    if Skalierung == "MinMax":
        scalerX = MinMaxScaler(feature_range = (-1, 1))   
    
    if Skalierung == "Standard":
        scalerX = StandardScaler()  
        
    scalerX.fit(traindata)                            #trainieren      
    scaled_data  = scalerX.transform(data)   
    joblib.dump(scalerX,scaler_Path)  
    
    #------------------------------------------------------loeschen der Zeile        
    #scaled_data = np.delete(scaled_data, (0), axis=0)       #lösche die erste Zeile von mixmax scaling
    if Dense_network == False:    
        scaled_data = np.reshape(scaled_data, (scaled_data.shape[0],scaled_data.shape[1],1))  
        
    return scaled_data
    
# returns an an n-day state representation ending at time t
def getState(scaled_data, t):
    state = scaled_data[[t]]
    return state
  
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def get_trades(action, t, inventorysell , inventorybuy, cls_data,Bilanz,profit_historie,trades_historie, buy_flag, buy_flag_count, buy_flag_end, buy_flag_count_end, sell_flag, sell_flag_count, sell_flag_end, sell_flag_count_end ):      
    bought_price = []
    
      
    if (action == 0 or action == 2) and len(inventorybuy) > 0:  
        bought_price = inventorybuy.pop(0)
        
        profit_trade = get_profit_trade(bought_price,cls_data[t+1], lotsize, gebuehren) 
        Bilanz += profit_trade                                    
        profit_historie.append(Bilanz)                          
        trades_historie.append(profit_trade)
        
        buy_flag_end.append(cls_data[t+1])
        buy_flag_count_end.append(t+1)
        
        
    if (action == 0 or action == 1) and len(inventorysell) > 0:        
        bought_price = inventorysell.pop(0)
        
        profit_trade = (get_profit_trade(bought_price,cls_data[t+1], lotsize, gebuehren) *(-1))   
        Bilanz += profit_trade                                    
        profit_historie.append(Bilanz)                          
        trades_historie.append(profit_trade)
    
                
        sell_flag_end.append(cls_data[t+1])
        sell_flag_count_end.append(t+1)
                
    if action == 1 and len(inventorybuy) == 0:
        inventorybuy.append(cls_data[t+1]) 
        
        buy_flag.append(cls_data[t+1])
        buy_flag_count.append(t+1)               
        
    if action == 2 and len(inventorysell) == 0:  
        inventorysell.append(cls_data[t+1])
                                
        sell_flag.append(cls_data[t+1])
        sell_flag_count.append(t+1)   
        
    return Bilanz

def get_profit_trade(buyprice, buypriceend, lotsize, gebuehren):
    pipvalue = 0.86155          #0.8 = EURUSD // 0.008 = US500
    
    value = (buypriceend - buyprice) * 100000  # 5 nachkommerstellen
    value = value * pipvalue * lotsize # 1 pip = 0,8 usd oda so bei 1 lot, wir haben 0.1 lot  // 2 für die gebühren bei 0.1
    value = value - gebuehren
    return value

def get_reward(currentprice , lastprice, lotsize):
    pipvalue = 0.86155
    
    gewinn = (currentprice - lastprice) * 100000  # 5 nachkommerstellen bie EURUSD / 2 bei US500
    value = gewinn * pipvalue * lotsize # 1 pip = 0,8 usd oda so bei 1 lot, wir haben 0.1 lot  // 2 für die gebühren bei 0.1
    return value




#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def KI_Train(Symbol):        
    for Symbol in Symbol:   
        print(Symbol)     
        #-----------------------------------------------------------------------------------------
        #------------------------------------------------ Datei Pfad
        #-----------------------------------------------------------------------------------------
        directory           = str(Path.cwd())  # Get the parent directory of the current working directory
        test_directory      = directory + "/tests"
        history_data_path   = directory + "/History.csv"
        
        OrderPath           = new_test_order(test_directory)
             
        model_directory     = OrderPath + "/model"  
        screenshots_directory = OrderPath + "/screenshots" 
        data_directory      = OrderPath + "/data"       
            
        input_data_path     = data_directory + '/input_data_'+Symbol+'.csv'  
        close_data_path     = data_directory + '/close_data_'+Symbol+'.csv'  
        ma_data_path        = data_directory + '/ma_data_'+Symbol+'.csv'  
        
        scaler_Path         = model_directory + '/scaler_'+Version+Symbol
        model_path          = model_directory + '/model_' +Version+Symbol
        screenshots_path    = screenshots_directory + '/save_fig_'   
        
        
        add_history(history_data_path)    
        
        #----- load data
        gen_data(count_data_train + count_data_test ,Symbol, Anzahl_Inputs,input_data_path,close_data_path ,ma_data_path,Zeiteinheit )
        
        #Initalisieren
        data = scaled_data(input_data_path, scaler_Path, Skalierung,count_data_train, Dense_network)
        count_inputs = np.size(data, 1)
                       
        cls_data = pd.read_csv(close_data_path,sep=';',header=None)
        cls_data = cls_data.to_numpy().flatten() 
        l = len(data) - 1
        
        ma_data = pd.read_csv(ma_data_path,sep=';',header=None)
        ma_data = ma_data.to_numpy().flatten()    
               
        
        model = model_create(U4, U3, U2, U1, count_inputs)
        
        pf_array_train  = []
        pf_array_test  = []
        memory = []        
        #-----------------------------------------------------------------------------------------
        #------------------------------------------------ 
        #-----------------------------------------------------------------------------------------  
        for e in range(episode_count + 1):
            print ("Episode " + str(e) + "/" + str(episode_count))
            state = getState(data, 0)    
            
            start = time.time()     #mitstoppen
            #arrays
            inventorysell = []
            inventorybuy = []
            open_buytrades = 0
            open_selltrades = 0
            trades_historie = []
            profit_historie = []
            
            buy_flag = []
            buy_flag_count = []
            buy_flag_end = []
            buy_flag_count_end  = []
            
            sell_flag = []
            sell_flag_count = []
            sell_flag_end = []
            sell_flag_count_end = []
            
            profit_trade = 0
            Bilanz = start_kapital
            
            Kontostand_array = []
            Kontostand_array_count = []
            
            
            reward_bilanz = 0
            profit = 0 
            Kontostand_array.append(Bilanz)         
            Kontostand_array_count.append(0)
            
            trades_historie = []
            count_reward = 0
            ####### TRAIN 
            #------------------------------------------------------------------------- 
            #-------------------------------------------------------------------------          
            for t in range(0, count_data_train):
                action = act(state, model)
                next_state = getState(data, t + 1)
                if action    == 0:
                    """ halten """ 
                    reward = 0  
                                   
                elif action  == 1: 
                    """ kaufen """                    
                    reward = get_reward(ma_data[t+1], ma_data[t], lotsize) 
                                     
                elif action  == 2:  
                    """ verkaufen """  
                    reward = (get_reward(ma_data[t+1], ma_data[t], lotsize) *(-1)) 
                              
                    
                    
                Bilanz = get_trades(action,t, inventorysell , inventorybuy, cls_data ,Bilanz,profit_historie,trades_historie, buy_flag, buy_flag_count, buy_flag_end, buy_flag_count_end, sell_flag, sell_flag_count, sell_flag_end, sell_flag_count_end) 
                ####KI LEARNING 
                done = True if t == count_data_train - 1 else False
                
                
                if reward_system == "Bilanz":
                    reward_bilanz += reward
                    memory.append((state, action, reward_bilanz, next_state, done))  
                
                if reward_system == "Profit":
                    memory.append((state, action, reward, next_state, done))  
                    
                    
                state = next_state
                if len(memory) > batch_size: 
                    expReplay(batch_size, memory, model, gamma)  
                    memory = []          
                                
                
            anzahl_traintrade = len(trades_historie)        
            pf_train = get_profit_faktor(trades_historie)  
            SummeGewinne_train = sum(x for x in trades_historie if x > 0)
            SummeVerlierer_train = sum(x for x in trades_historie if x < 0)  
            Gewinn_train = SummeGewinne_train + SummeVerlierer_train  
                 
            ####### TEST
            #------------------------------------------------------------------------- 
            #-------------------------------------------------------------------------  
            trades_historie = []
            
            for t in range(count_data_train, l):
                action = act(state, model)
                next_state = getState(data, t + 1)
                
                if action    == 0:
                    """ halten """ 
                    reward = 0                     
                                   
                elif action  == 1: 
                    """ kaufen """                    
                    reward = get_reward(ma_data[t+1], ma_data[t], lotsize) 
                                     
                elif action  == 2:  
                    """ verkaufen """  
                    reward = (get_reward(ma_data[t+1], ma_data[t], lotsize) *(-1)) 
                    
                    
                Bilanz = get_trades(action,t, inventorysell , inventorybuy, cls_data ,Bilanz,profit_historie,trades_historie, buy_flag, buy_flag_count, buy_flag_end, buy_flag_count_end, sell_flag, sell_flag_count, sell_flag_end, sell_flag_count_end )   
               
                state = next_state      
                    
            anzahl_testtrade = len(trades_historie)        
            pf_test = get_profit_faktor(trades_historie)   
            SummeGewinne_test = sum(x for x in trades_historie if x > 0)
            SummeVerlierer_test = sum(x for x in trades_historie if x < 0)  
            Gewinn_test = SummeGewinne_test + SummeVerlierer_test
        
            pf_array_train.append(pf_train) 
            pf_array_test.append(pf_test)
            
            Auswertungname=  ["Anzahl Train",   "Anzahl Test"]
            Auswertung=      [anzahl_traintrade,round(Gewinn_train,2), round(SummeGewinne_train,2), round(SummeVerlierer_train,2),"/ /",  anzahl_testtrade,round(Gewinn_test,2), round(SummeGewinne_test,2), round(SummeVerlierer_test,2)]
        
            #########SPEICHERN UND AUSWERTEN 
            #-------------------------------------------------------------------------  
            #-------------------------------------------------------------------------        
            show_plot_chart(profit_historie, cls_data,ma_data, 
                             buy_flag, buy_flag_count, buy_flag_end, buy_flag_count_end, sell_flag, sell_flag_count, sell_flag_end, sell_flag_count_end,
                            e, Symbol,screenshots_directory,  screenshots_path, 
                                count_data_train, pf_array_train, pf_array_test,
                                Parametername, Parameter,
                                Auswertungname, Auswertung, Notiz, anzahl_traintrade
                                )
        
            model.save(model_path)  
                        
                    
                    
                    
                    
                    
                    
                    
                    
                    
            
            ende = time.time()
            print('{:5.3f}s'.format(ende-start))     #zeit
                                  
        #-----------------------------------------------------------------------------------------
        #------------------------------------------------ End
        #-----------------------------------------------------------------------------------------
         



if __name__ == "__main__":      
    #Inputs
    lotsize                 = 1 
    gebuehren               = 0.05           #0.6€ bei 0.1 lot € bei EURUSD  // 0.5 bei US500 - 1lot
    SL_Value                = 0.001
    start_kapital           = 10000
    Version                 = "Main_"
    save_plot               = False         #Jede Epoche eine auswertung ja / nein
    
    #KI
    Zeiteinheit             = "1H"
    count_data_train        = 500
    count_data_test         = 500
    Anzahl_Inputs           = 60
    episode_count           = 20       #Epochen
    reward_system           = "Profit"   #"Profit" "Bilanz" 
    
    batch_size              = 0
    gamma                   = 0.99
    memorylength            = 300  
    Dense_network           = True         #if false = LSTM Network
    train                   = True
    Skalierung              = "MinMax"      #/ "MinMax"   "Standard"
    
    U4                      = 256
    U3                      = 128
    U2                      = 64
    U1                      = 32
    
    action_size             = 3 
    
        
    Parametername=  ["RewardSystem","lotsize","gebuehren","count_data_train", "count_data_test","Anzahl_Inputs","episode_count","batch_size","gamma","U4","U3","U2","U1","Zeiteinheit","Skalierung"]
    Parameter=      [reward_system ,lotsize,   gebuehren ,  count_data_train, count_data_test,Anzahl_Inputs,episode_count,batch_size,gamma,U4,U3,U2,U1,Zeiteinheit, Skalierung]
    
    Notiz = [" Close - // Normal // mit Dropoout33 + Kernzel // Lernrate  0.0001 // linear " ]
    
    #-----------------------------------------------------------------------------------------
    #------------------------------------------------ Symbole
    #-----------------------------------------------------------------------------------------
    
    Symbol=[]
    for s in range(0, 1):   
         Symbol.append("USDCAD")   
         Symbol.append("EURUSD")    
         Symbol.append("EURGBP")    
         Symbol.append("EURAUD")    
         Symbol.append("GBPUSD")    
         Symbol.append("AUDUSD")  
    KI_Train(Symbol)           