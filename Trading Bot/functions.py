import numpy as np
import math

from pathlib import Path
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import random
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib


from csv import writer



trade_faktor = 0
highest_Kontostand = 0
gesamt_gewinn   = 0
SummeGewinne    = 0
SummeVerlierer  = 0
max_profitfaktor = 3
max_gewinn      = 100000

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def new_test_order(test_directory):
    dir = test_directory 
    ctfig = str(len([iq for iq in os.scandir(dir)]) + 1)   
    OrderPath = test_directory +"/Test"+ctfig
    os.mkdir(OrderPath)
    os.mkdir(OrderPath+"/model")
    os.mkdir(OrderPath+"/screenshots" )   
    os.mkdir(OrderPath+"/data" )   
    
    return OrderPath


def add_history(history_data_path):
    
    List=[10,20,30,40,50,60,70,80,90,100,110,120,130,140,150]
    with open(history_data_path,  'a', newline='') as f_object:  
        writer_object = writer(f_object , delimiter = ";") 
        writer_object.writerow(List)  
        f_object.close()
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def get_profit_faktor(trades_historie):    
    # SummeGewinne = sum(x for x in trades_historie if x > 0)
    # SummeVerlierer = sum(x for x in trades_historie if x < 0)
    SummeGewinne = np.sum(np.where(trades_historie > 0, trades_historie, 0))
    SummeVerlierer = np.sum(np.where(trades_historie < 0, trades_historie, 0))

    
    
    if SummeGewinne == 0 or SummeVerlierer == 0:
        profit_faktor = 0
    else:       
        profit_faktor = SummeGewinne / (SummeVerlierer *-1)
    
    return profit_faktor

def get_Drowdown(Kontostand):
    global highest_Kontostand
    
    if Kontostand > highest_Kontostand:
        highest_Kontostand = Kontostand
    
    drowdown = (highest_Kontostand-Kontostand)/highest_Kontostand * 100
    drowdown_umgedreht = 100 - drowdown
    
    return drowdown_umgedreht
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def show_plot_chart(plot_reward, close_price,ma_price, 
                     buy_flag, buy_flag_count, buy_flag_end, buy_flag_count_end, sell_flag, sell_flag_count, sell_flag_end, sell_flag_count_end,                   
                    epoche2, symbol ,screenshots_directory, screenshots_path,
                    count_data_train, pf_array_train, pf_array_test,
                    Parametername, Parameter, Auswertungname, Auswertung, Notiz, anzahl_traintrade):
    
    
    figure(figsize=(23.4, 13.1), dpi=200)
    # CHART    
    plt.axes([0.05, 0.6, 0.9, 0.35])          #left, bottom, width, height (range 0 to 1)    
    plt.plot(close_price, color = 'black', label = 'Close' ,linewidth=0.5)     
    plt.plot(ma_price, color = 'green', label = 'ma' ,linewidth=0.5)             
    plt.plot([count_data_train , count_data_train] ,[np.amax(close_price) ,np.amin(close_price) ] , color = 'blue' ,linewidth=1 , linestyle='dashed')
    
    plt.scatter(buy_flag_count, buy_flag, color = 'blue', marker='^' ,linewidth=0.3)
    plt.scatter(sell_flag_count, sell_flag, color = 'red', marker='v' ,linewidth=0.3) 
    
    

    for i in range(0,len(buy_flag_end)):    
        if buy_flag[i] < buy_flag_end[i]:
            plt.plot([buy_flag_count[i] , buy_flag_count_end[i]] ,[buy_flag[i],buy_flag_end[i]] , color = 'green' ,linewidth=1)
        elif buy_flag[i] > buy_flag_end[i]:
            plt.plot([buy_flag_count[i] , buy_flag_count_end[i]] ,[buy_flag[i],buy_flag_end[i]] , color = 'red' ,linewidth=1)


    for i in range(0,len(sell_flag_end)):    
        if sell_flag[i] > sell_flag_end[i]:
            plt.plot([sell_flag_count[i] , sell_flag_count_end[i]] ,[sell_flag[i],sell_flag_end[i]] , color = 'green' ,linewidth=1)
        elif sell_flag[i] < sell_flag_end[i]:
            plt.plot([sell_flag_count[i] , sell_flag_count_end[i]] ,[sell_flag[i],sell_flag_end[i]] , color = 'red' ,linewidth=1)
   
            
            
    plt.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.3)  
    plt.title('AI - Summary')
    plt.legend()      
    plt.ylabel('Profit')
    
    #KONTOSTAND
    plt.axes([0.05, 0.35, 0.9, 0.22])          #left, bottom, width, height (range 0 to 1)    
    plt.plot(plot_reward, color = 'grey', label = 'Bilanz' ,linewidth=1)  
    
    plt.plot([anzahl_traintrade , anzahl_traintrade] ,[np.amax(plot_reward) ,np.amin(plot_reward) ] , color = 'blue' ,linewidth=1 , linestyle='dashed')     
 
    
    plt.legend()      
    plt.ylabel('Gewinn') 
           
    
    #PF
    plt.axes([0.65, 0.05, 0.3, 0.20])          #left, bottom, width, height (range 0 to 1)  
    plt.plot(pf_array_train, color = 'blue', label = 'PF-Train' ,linewidth=1)  
    plt.plot(pf_array_test, color = 'green', label = 'PF-Test'  ,linewidth=1)  
    
    plt.legend()      
    plt.ylabel('Profitfaktor')
    
    
    
    
    #PARAMETER
    plt.gcf().text(0.05, 0.25, 'Inputs: ', fontsize=16)           #Uberschrift    
    plt.gcf().text(0.15, 0.25, symbol , fontsize=10)
    
    plt.gcf().text(0.05, 0.20, Parametername , fontsize=10)       
    plt.gcf().text(0.05, 0.18, Parameter , fontsize=10)
       
    
    plt.gcf().text(0.05, 0.15, Auswertungname , fontsize=10)       
    plt.gcf().text(0.05, 0.13, Auswertung , fontsize=10)
    
    plt.gcf().text(0.05, 0.11, "Notiz:" , fontsize=10)       
    plt.gcf().text(0.05, 0.08, Notiz , fontsize=10)
        
    dir = screenshots_directory
    ctfig = str(len([iq for iq in os.scandir(dir)]) + 1)
         
    mng = plt.get_current_fig_manager()
    #mng.window.state('zoomed')      
      
    plt.savefig(screenshots_path+ctfig+'.png')   
            
    plt.close()        
#------------------------------------------------------------------------------
    
    
    
    
    
    
    
    
    
    
    
    