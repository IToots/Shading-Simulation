# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 09:30:34 2023

@author: Rui Guilherme
"""

# To convert a jupyter notebook to a regular .py file: in the CMD line type "jupyter nbconvert [fileName].ipynb --to python"
import csv, importlib, itertools, math, os, sys, random
from matplotlib.patches import Rectangle
from mpl_toolkits import mplot3d
import seaborn as sb
import numpy as np
import pandas as pd
import sympy as sp
import scipy as sc
import scipy.integrate as it
import plotly.express as px
import plotly.io as pio
from scipy.interpolate import interp1d

from datetime import date, datetime, timedelta
from os import listdir
from os.path import isfile, join
from pathlib import Path
from pprint import pprint
from sklearn.metrics import mean_squared_error

from scipy.signal import find_peaks, peak_widths
from scipy.interpolate import UnivariateSpline,splprep,splev
from scipy.optimize import curve_fit, fsolve
from scipy import interpolate
from scipy.special import lambertw
from scipy.stats import linregress
from scipy.constants import (
    c, # speed of light in vacuum
    epsilon_0, # the electric constant (vacuum permittivity)
    pi,
    mu_0, # the magnetic constant
    h, # the Planck constant
    k, # Boltzmann constant
    elementary_charge, # elementary charge, q
)
Ɛ0 = epsilon_0
π = pi
μ0 = mu_0
q = elementary_charge

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.colors as colors

from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.interpolate import griddata
from matplotlib import cm
from tabulate import tabulate

#folder = 'D:\\Work\\HighLight Project\\Data\\Photoluminescence\\'
#folderRaw = 'D:\\Work\\HighLight Project\\Data\\Photoluminescence\\Raw&Backup\\'

tT = 273.15 + 25
tT = 300
# Calculation of the diode saturation current with temperature
def J0(T=tT, Is = 1.66e-12, k=0.14):
    ## Ref: https://dspace.mit.edu/bitstream/handle/1721.1/52058/rle_qpr_045_xi.pdf?sequence=1
    # T (K): temperature at which the diode is operated
    # Is (A): Reverse saturation current at 300K [1e-9 up to 1.66e-12 (Ideal value)]
    # k: constant given by the energy gap of the diode material [Si: 0.14, Ge: 0.09]
    
    ## Ref2: https://testbook.com/question-answer/for-every-10c-increase-in-temperature-the-re--5f71c1ca6af969983a541e47
        # Is*2**((T-300)/10)
    # The reverse saturation current of the diode increases with an increase in the temperature.
    # The rise in the reverse saturation current is 7% /°C for both germanium and silicon and 
    # approximately doubles for every 10°C rise in temperature.
    
    return Is*np.exp(k*(tT-298.15))

# Calculation of thermal voltage
def Vthermal(T=tT):
    # T (K): temperature at which the diode is operated    
    return k*T/q

# I-V curve calculation
def iv(Isc=0.035, Is=1e-12, Voc=None, T=tT, Rs=0.1, Rsh=1000):
    # Isc (A.cm2): short circuit current == Iph (A): photogenerated current source
    # Is (A.cm2): Reverse saturation current at 300K [1e-9 up to 1.66e-12 (Ideal value)]
    # Voc (V): Open circuit voltage
    # T (K): temperature at which the diode is operated
    # Rs (Ohm): Series resistance
    # Rsh (Ohm): Shunt resistance
    
    ## Calculation of the diode saturation current with temperature
    I0 = J0(T=T, Is = Is)
    
    ## Calculation of open circuit voltage   
    if Voc == None:
        Voc = (n*k*T/q)*np.log(Isc/I0+1) # As a funtion of temperature
    
    Vth = Vthermal(T=T) # Thermal voltage
    V = np.linspace(0,Voc,100) # Voltage range

    ## Calculation of SC current
    I = -V/(Rs+Rsh)-(
        lambertw(Rs*I0*Rsh*np.exp((Rsh*(Rs*Isc+Rs*I0+V))/(n*Vth*(Rs+Rsh)))/(n*Vth*(Rs+Rsh)))*n*Vth
    )/Rs+(Rsh*(I0+Isc))/(Rs+Rsh)
    
    ## Calculation of SC power and Maximum Power point (Vmp, Imp)
    P = V*I
    Pmax = np.max(P)
    index = (np.where(P == Pmax))
    Vmp, Imp, = V[index][0], I[index][0].real
    
    return Voc, Vmp, Isc, Imp, Pmax.real, V, I.real, P.real

# Function to obtain n and Is for a target Voc
def find_n_Is(Voc, T=tT):
    # Voc (V): Open circuit voltage
    # T (K): temperature at which the diode is operated
    
    # Create n and Is arrays and 
    n_ = np.arange(1, 2.001, 0.001)
    Is_1 = np.arange(1e-12, 1e-11, 1e-15)
    Is_2 = np.arange(1e-11, 1e-10, 1e-12)
    Is_3 = np.arange(1e-10, 1e-9, 1e-11)
    Is_ = np.concatenate((Is_1,Is_2,Is_3))
    
    # Create the corresponding matrix of n*Is combinations
    n, Is = np.meshgrid(n_, Is_) # Create 2D meshgrids
    I0 = J0(T=T, Is = Is)
    
    # Calculate Voc for all n*Is combinations
    Voc_matrix = (n*k*T/q)*np.log(Isc/I0+1)
    
    # Find best n*Is fit
    nearestVoc = Voc-Voc_matrix

    if nearestVoc.any() == 0:
        i,j = np.where(np.abs(nearestVoc) == 0)
    else:
        val = np.min(np.abs(nearestVoc))
        i,j = np.where(np.abs(nearestVoc) == val)
    
    best_n = n[i,j][0]
    best_Is = Is[i,j][0]
        
    # Plot data and optimum values for n and Is
    # plt.style.use('seaborn-notebook')
    plt.figure(figsize=(7,4.5))
    plt.xlabel('Ideality factor (n)', fontsize=16)
    plt.ylabel('Is (A/cm$^2$)', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tick_params(direction='in', left = True, top = True)

    high=np.max([np.abs(np.min(nearestVoc)),np.abs(np.max(nearestVoc))])
    v = np.linspace(-high, high, 200, endpoint=True)

    plt.scatter(best_n, best_Is, c='k', s=8, zorder=2)
    plt.annotate(f'n = {best_n:.3f}\nIs = {best_Is:.2e} A/cm$^2$', (best_n*1.025, best_Is), ha ='left', fontsize = 14)
    plt.contourf(n, Is, nearestVoc, v, cmap=plt.cm.seismic)

    cbar = plt.colorbar(format = '%.0e', 
                        ticks=np.linspace(-high, high, 5))
    
    cbar.set_label(label = r'$\Delta$ Voc (V)', fontsize=16)
    cbar.ax.tick_params(direction='in')
    plt.yscale('log')
    plt.show()

    return best_n, best_Is

# Function to obtain Rs and Rsh for a target IV curve
def findResistance(df, T=tT):
    # df: DataFrame with experimental data (should have at least V and Jsc data)
    # T (K): temperature at which the diode is operated
    
    # Array with range of resistance values
    rs = np.arange(0.01, 10.1, 0.1)
    rsh = np.arange(100, 10100, 100)

    I0 = J0(T=tT, Is = Is)
    
    Vth = Vthermal(T=T)
    Voc = (n*k*T/q)*np.log(Isc/I0+1) # As a funtion of temperature
    V = np.array(df['Voltage (V)'])
    
    # First iteration to find best Rs and Rsh
    def findRsRsh(rs, rsh, Vth, Voc, V, I0):
        i = []
        rr = []
        rmse = []
        
        for Rs in rs:
            for Rsh in rsh:
                rs_rsh = Rs+Rsh
                lbtw = lambertw(Rs*I0*Rsh*np.exp(
                    (Rsh*(Rs*Isc+Rs*I0+V))/(n*Vth*(rs_rsh)))/(n*Vth*(rs_rsh))).real
                I = -V/(rs_rsh)-(lbtw*n*Vth)/Rs+(Rsh*(I0+Isc))/(rs_rsh)
                i.append(I)
                rr.append([Rs,Rsh])

        for ii in i:
            rmse.append(mean_squared_error(df['Current (A)'], ii, squared=False))

        minrmse = np.min(rmse)
        pos = np.where(rmse==minrmse)[0][0]
        rmse[pos], rr[pos]
        
        return rr[pos][0], rr[pos][1]
    
    best_Rs, best_Rsh = findRsRsh(rs, rsh, Vth, Voc, V, I0)
    
    # Refinement
    rs = np.arange(best_Rs/2, best_Rs*1.5+0.01, 0.01)
    rsh = np.arange(best_Rsh/2, best_Rsh*1.5+1, 1)
    
    best_Rs, best_Rsh = findRsRsh(rs, rsh, Vth, Voc, V, I0)
    
    
    I0 = J0(T=T, Is = Is)
    
    Vth = Vthermal(T=T)
    Voc = (n*k*T/q)*np.log(Isc/I0+1) # As a funtion of temperature
    V = np.array(df['Voltage (V)'])
    
    # First iteration to find best Rs and Rsh
    def IIII(rs, rsh, Vth, Voc, V, I0):
        i = []
        rmse = []
        
        for Rs in rs:
            for Rsh in rsh:
                rs_rsh = Rs+Rsh
                lbtw = lambertw(Rs*I0*Rsh*np.exp(
                    (Rsh*(Rs*Isc+Rs*I0+V))/(n*Vth*(rs_rsh)))/(n*Vth*(rs_rsh))).real
                I = -V/(rs_rsh)-(lbtw*n*Vth)/Rs+(Rsh*(I0+Isc))/(rs_rsh)
                i.append(I)
                
        for ii in i:
            rmse.append(mean_squared_error(df['Current (A)'], ii, squared=False))
        
        return rmse
    
    x = rs
    y = rsh
    
    comb_array = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
    
    z = IIII(rs, rsh, Vth, Voc, V, I0)
    
    xgrid, ygrid = np.meshgrid(rs, rsh)
    ctr_f = griddata((comb_array), z, (xgrid, ygrid), method='linear')
    
    # plt.style.use('seaborn-notebook')
    plt.figure(figsize=(7,4.5))
    plt.xlabel('Sheet series resistance (Ω/cm$^2$)', fontsize=16)
    plt.ylabel('Sheet parallel resistance (Ω/cm$^2$)', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tick_params(direction='in', left = True, top = False)
    
    high=np.max([np.abs(np.min(z)),np.abs(np.max(z))])
    v = np.linspace(0, high, 60, endpoint=True)
    
    plt.scatter(best_Rs,best_Rsh, c='k', s=8, zorder=2)
    plt.annotate(f'Rs = {best_Rs:.3f} Ω/cm$^2$\nRsh = {best_Rsh:.2e} Ω/cm$^2$', (best_Rs*1.025, best_Rsh), ha ='left', fontsize = 13)
    plt.contourf(xgrid, ygrid, ctr_f, v, cmap=plt.cm.Reds)
    
    cbar = plt.colorbar(format = '%.0e', cmap=plt.cm.Reds, 
                        ticks=np.linspace(0, high, 5))
    
    cbar.set_label(label = 'rmse value (a. u.)', fontsize=16)
    cbar.ax.tick_params(direction='in')
    
    plt.show()
    
    return best_Rs, best_Rsh

# Calculate FF with higher degree of accuracy
def ff(n, Voc, Isc, Rs, Rsh, T=tT):
    # n: ideality factor [1,2]
    # Voc (V): Open circuit voltage
    # Isc (A.cm2): short circuit current == Iph (A): photogenerated current source
    # Rs (Ohm): Series resistance
    # Rsh (Ohm): Shunt resistance
    # T (K): temperature at which the diode is operated
    
    # Calculate normalized open circuit voltage
    Vth = Vthermal(T=T)
    voc=Voc/(n*Vth)
    
    # Calculate normalized resistances
    rs = Rs/(Voc/Isc)
    rsh = Rsh/(Voc/Isc)
    
    # Calculate ideal FF (Rs=0, Rsh= infinite ∞)
    FF0 = (voc-np.log(voc+0.72))/(voc+1)
    
    # Calculate FF for a variable Rs 
    FFs = FF0*(1-1.1*rs)+rs**2/5.4
    
    # Calculate FF for a variable Rsh
    FFsh = FF0*(1-((voc+0.7)/voc)*(FF0/rsh))
    
    # Calculate FF for a variable Rs and Rsh
    FF = FFs*(1-((voc+0.7)/voc)*(FFs/rsh))
    
    return FF


dire = "C:/Users/ruiui/Desktop/sun simulator new/Results_2024-08-02/"

file_path = "C:/Users/ruiui/Desktop/sun simulator new/Results_2024-08-02/C1_median.csv" # cell 1
LT_curve = "C:/Users/ruiui/Desktop/mod_corr/c1/F/IVF.csv"
n, Is, Rs, Rsh, title = (1.196, 5.844e-12, 2.06, 999, 'C1')

file_path = "C:/Users/ruiui/Desktop/sun simulator new/Results_2024-08-02/C2_median.csv" # cell 2
LT_curve = "C:/Users/ruiui/Desktop/mod_corr/c2/F/IVF.csv"
n, Is, Rs, Rsh, title = (1.148, 2.821e-12, 1.40, 874, 'C2')

file_path = "C:/Users/ruiui/Desktop/sun simulator new/Results_2024-08-02/C3_median.csv" # cell 3
LT_curve = "C:/Users/ruiui/Desktop/mod_corr/c3/F/IVF.csv"
n, Is, Rs, Rsh, title = (1.125, 2.001e-12, 1.23, 1731, 'C3')

file_path = "C:/Users/ruiui/Desktop/sun simulator new/Results_2024-08-02/C4_median.csv" # cell 4
LT_curve = "C:/Users/ruiui/Desktop/mod_corr/c4/F/IVF.csv"
n, Is, Rs, Rsh, title = (1.202, 8.046e-12, 1.12, 1590, 'C4')

file_path = "C:/Users/ruiui/Desktop/sun simulator new/Results_2024-08-02/C5_median.csv" # cell 5
LT_curve = "C:/Users/ruiui/Desktop/mod_corr/c5/F/IVF.csv"
n, Is, Rs, Rsh, title = (1.193, 7.353e-12, 1.03, 1381, 'C5')

file_path = "C:/Users/ruiui/Desktop/sun simulator new/Results_2024-08-02/C6_median.csv" # cell 6
LT_curve = "C:/Users/ruiui/Desktop/mod_corr/c6/F/IVF.csv"
n, Is, Rs, Rsh, title = (1.170, 4.629e-12, 1.17, 1734, 'C6')

file_path = "C:/Users/ruiui/Desktop/sun simulator new/Results_2024-08-02/C7_median.csv" # cell 7
LT_curve = "C:/Users/ruiui/Desktop/mod_corr/c7/F/IVF.csv"
n, Is, Rs, Rsh, title = (1.212, 9.550e-12, 1.03, 1448, 'C7')

file_path = "C:/Users/ruiui/Desktop/sun simulator new/Results_2024-08-02/C8_median.csv" # cell 8
LT_curve = "C:/Users/ruiui/Desktop/mod_corr/c8/F/IVF.csv"
n, Is, Rs, Rsh, title = (1.236, 1.600e-11, 0.99, 1349, 'C8')

file_path = "C:/Users/ruiui/Desktop/sun simulator new/Results_2024-08-02/C9_median.csv" # cell 9
LT_curve = "C:/Users/ruiui/Desktop/mod_corr/c9/F/IVF.csv"
n, Is, Rs, Rsh, title = (1.184, 6.765e-12, 0.98, 1638, 'C9')


dfi=pd.read_csv("C:/Users/ruiui/Desktop/sun simulator new/New modules testing/C27PCB.csv")
# n, Is, Rs, Rsh, title = (1.212, 9.550e-12, 1.03, 1448, 'C7')

df_filtered = dfi[(dfi['Voltage'] > 0) & (dfi['Current'] > 0)]

# Extract the filtered Voltage and Current as numpy arrays
V = df_filtered['Voltage'].values
I = df_filtered['Current'].values
P = V * I

# Interpolate the data to have the same voltage points for all curves
interp_func = interp1d(V, I, kind='linear')
voltage_axis = np.linspace(V.min(), V.max(), num=100)  # Choose the number of desired voltage points
interpolated_current = interp_func(voltage_axis)

interp_func = interp1d(V, P, kind='linear')
interpolated_power = interp_func(voltage_axis)

# Create a new DataFrame 'df' with means for 'Voltage (V)', 'Current (A)', and 'Power (W)'
data = {
    'Voltage (V)': voltage_axis,
    'Current (A)': interpolated_current,
    'Power (W)': interpolated_power
}
    
df = pd.DataFrame(data)

# Recalculate data
df['Voltage (V)'] = df['Voltage (V)'] / 6  # Module has 5 cells connected in series
df['Current (A)'] = df['Current (A)'] / (156.25 / 6)  # Calculate current density
df['Power (W)'] = df['Voltage (V)'] * df['Current (A)']  # Calculate photogenerated power


# # Read the CSV file into a DataFrame
# df = pd.read_csv(file_path)

# # Extract the voltage and current columns
# voltage = df['Voltage (V)'].tolist()
# current = df['Current (A)'].tolist()

# # Calculate power as P = V * I
# power = [v * i for v, i in zip(voltage, current)]

# # Create a new DataFrame 'df' with means for 'Voltage (V)', 'Current (A)', and 'Power (W)'
# data = {
#     'Voltage (V)': voltage,
#     'Current (A)': current,
#     'Power (W)': power
# }

# df = pd.DataFrame(data)

# # Recalculate data
# df['Voltage (V)'] = df['Voltage (V)'] / 5  # Module has 5 cells connected in series
# df['Current (A)'] = df['Current (A)'] / (169 / 5)  # Calculate current density
# df['Power (W)'] = df['Voltage (V)'] * df['Current (A)']  # Calculate photogenerated power

# Extract main parameters
Voc = np.max(df['Voltage (V)']) # Open-circuit voltage
Isc = np.max(df['Current (A)']) # Short-circuit current density
Pmax = np.max(df['Power (W)']) # Maximum power point
Vmp = np.array(df['Voltage (V)'].iloc[np.where(df['Power (W)'] == Pmax)])[0] # Voltage at maximum power (MP)
Imp = np.array(df['Current (A)'].iloc[np.where(df['Power (W)'] == Pmax)])[0] # Current density at MP

# Determine ideality factor (n) and Reverse saturation current (Is) from the experimental value of Voc
n, Is = find_n_Is(Voc=Voc)

# Determine series and shunt resistance by solving I-V using W-function
Rs, Rsh = findResistance(df)

# Generate modeled I-V curve
Voc_sim, Vmp_sim, Isc_sim, Imp_sim, Pmax_sim, V, I, P = iv(Isc=Isc, Is=Is, Voc=None, Rs=Rs, Rsh=Rsh)


def extract_parameters(voltage, current, W):
    try:
        # Calculate power P = V * I
        power = [v * i for v, i in zip(voltage, current)]
        
        # Voc: Voltage where the current is approximately 0 (open-circuit voltage)
        Voc = next(v for v, i in zip(voltage, current) if i < 0.0001)
        
        # Isc: Current where the voltage is 0 (short-circuit current)
        Isc = max(current)
        
        # MPP: Maximum Power Point, where power P = V * I is maximized
        MPP = max(power)
        MPP_index = power.index(MPP)
        
        # Vmp: Voltage at Maximum Power Point
        Vmp = voltage[MPP_index]
        
        # Imp: Current at Maximum Power Point
        Imp = current[MPP_index]
        
        # FF: Fill Factor
        FF = (MPP / (Isc * Voc)) * 100
        
        # Ef: Efficiency, given by (MPP / W) * 100
        Ef = (MPP / 0.1)
        
        return Voc, Isc, Vmp, Imp, MPP, FF, Ef
    except Exception as e:
        print(f"Error extracting parameters: {e}")
        return None, None, None, None, None, None, None

W = 12.5*12.5* 9 * 0.1

csv_voltage = []
csv_current = []
csv_data = pd.read_csv(LT_curve)

# Extend the lists with the CSV data
csv_voltage.extend(csv_data['Voltage (V)'])
csv_current.extend(csv_data['Current (I)'])

# Convert lists to numpy arrays for element-wise operations
csv_voltage = np.array(csv_voltage)
csv_current = np.array(csv_current)

# Calculate voltage and current density
csv_voltage = csv_voltage / 5  # Module has 5 cells connected in series
csv_current = (csv_current / (169 / 5))  # Calculate current density

# Calculate photogenerated power
csv_power = csv_voltage * csv_current

Vocl, Iscl, Vmpl, Impl, Pmaxl, FFl, PCEl = extract_parameters(csv_voltage, csv_current, W)

# Plot experimental and modeled data
# plt.style.use('seaborn-notebook')
fig, ax1 = plt.subplots(figsize=(10,6))
plt.title(title, fontsize=18)
ax2 = ax1.twinx()
ax1.set_xlabel('Voltage (V)', fontsize=16)
ax1.set_ylabel('Current density (mA/cm$^2$)', c = 'C0', fontsize=16)
ax2.set_ylabel('Power density (mW/cm$^2$)', c = 'C1', fontsize=16)
ax1.tick_params(axis='both', which='major', labelsize=16)
ax2.tick_params(axis='both', which='major', labelsize=16)
ax1.tick_params(direction='in', left = True, top = True)
ax2.tick_params(direction='in', right = True, top = True)

ax1.plot(df['Voltage (V)'],df['Current (A)']*1000, lw = 1, c='C0')
ax2.plot(df['Voltage (V)'],df['Power (W)']*1000, lw = 1, c='C1')
ax1.plot(V,I*1000, lw = 1.25, c='C0', ls='-.' )
ax2.plot(V,P*1000, lw = 1.25, c='C1', ls='-.')

# Plot CSV data
ax1.plot(csv_voltage, csv_current*1000, lw=1.5, c='C2', ls=':')  # Convert A to mA/cm²
ax2.plot(csv_voltage, csv_power*1000, lw=1.5, c='C3', ls=':')    # Convert W to mW/cm²

ax1.set_ylim(0, Isc*1100)
ax2.set_ylim(0, Pmax*1100)
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
plt.xlim(0, Voc*1.05)

# Calculate RMSE for current using filtered data
filtered_VLT = V[V <= df['Voltage (V)'].max()]
filtered_ILT = I[V <= df['Voltage (V)'].max()]
filtered_PLT = P[V <= df['Voltage (V)'].max()]
rmse_current = np.sqrt(mean_squared_error(filtered_VLT, df['Current (A)'][:len(filtered_VLT)]*1000, squared=False))
rmse_power = np.sqrt(mean_squared_error(filtered_VLT, df['Power (W)'][:len(filtered_VLT)]*1000, squared=False))

# Just for legend purposes
ax1.plot(-1,-1, lw = 1.25, c='grey', label='Experimental')
ax1.plot(-1,-1, lw = 1.25, c='grey', ls='-.', label='Model data')
ax1.plot(-1,-1, lw = 1.25, c='grey', ls=':', label='L')
ax1.legend(loc='best', bbox_to_anchor=(0, 0.35, 0.3, 0.5), fontsize=12)    
plt.annotate(
            f'n = {n:.3f}\nIs = {Is:.3e} A/cm$^2$\nRs = {Rs:.2f} Ω/cm$^2$\nRsh = {Rsh} Ω/cm$^2$\nrmse (IV): {rmse_current:.2f}\nrmse (PV): {rmse_power:.2f}',
            (0.025, 5),
            ha='left',
            va='baseline',
            fontsize=12,
            )

# Extract parameters for comparison and create summary table
FF = (Vmp*Imp)/(Voc*Isc)*100
FF_sim = (Vmp_sim*Imp_sim)/(Voc_sim*Isc_sim)*100
#FF = ff(n, Voc, Isc, Rs, Rsh, T=300)*100
#FF_sim = ff(n, Voc_sim, Isc_sim, Rs, Rsh, T=300)*100

PCE = Vmp*Imp/0.1
PCE_sim = Vmp_sim*Imp_sim/0.1

col_labels=['Exp.','P','ΔP','LTS','ΔL']
row_labels=['Voc (V)','Vmp (V)','Jsc (mA/cm$^2$)', 'Jmp (mA/cm$^2$)', 'Pmax (mW/cm$^2$)', 'FF (%)', 'η (%)']
table_vals=[
    [f'{Voc:.3f}',f'{Voc_sim:.3f}',f'{(Voc_sim-Voc)/Voc*100:.1f}%',f'{Vocl:.3f}',f'{(Vocl-Voc)/Voc*100:.1f}%'],
    [f'{Vmp:.3f}',f'{Vmp_sim:.3f}',f'{(Vmp_sim-Vmp)/Vmp*100:.1f}%',f'{Vmpl:.3f}',f'{(Vmpl-Vmp)/Vmp*100:.1f}%'],
    [f'{Isc*1000:.3f}',f'{Isc_sim*1000:.3f}',f'{(Isc_sim-Isc)/Isc*100:.1f}%',f'{Iscl*1000:.3f}',f'{(Iscl-Isc)/Isc*100:.1f}%'],
    [f'{Imp*1000:.3f}',f'{Imp_sim*1000:.3f}',f'{(Imp_sim-Imp)/Imp*100:.1f}%',f'{Impl*1000:.3f}',f'{(Impl-Imp)/Imp*100:.1f}%'],
    [f'{Pmax*1000:.3f}',f'{Pmax_sim*1000:.3f}',f'{(Pmax_sim-Pmax)/Pmax*100:.1f}%',f'{Pmaxl:.3f}',f'{(Pmaxl-Pmax)/Pmax*100:.1f}%'],
    [f'{FF:.2f}',f'{FF_sim:.2f}',f'{(FF_sim-FF)/FF*100:.1f}%',f'{FFl:.2f}',f'{(FFl-FF)/FF*100:.1f}%'],
    [f'{PCE*100:.2f}',f'{PCE_sim*100:.2f}',f'{(PCE_sim-PCE)/PCE*100:.1f}%',f'{PCEl*100:.2f}',f'{(PCEl-PCE)/PCE*100:.1f}%']]
    
# Create a white rectangle patch to cover the graph
rect = Rectangle((0, 0), 1, 1, fill=True, color='white', alpha=1)
ax1.add_patch(rect)

# Add the table
table = plt.table(cellText=table_vals,
                  colWidths=[0.14, 0.14, 0.1, 0.14, 0.1],
                  rowLabels=row_labels,
                  colLabels=col_labels,
                  loc='best',
                  bbox=(0.483, 0.01, 0.425, 0.375),
                  alpha=1)

# Adjust the font size for the cells
table.auto_set_font_size(False)
table.set_fontsize(12)  # You can adjust this value as needed


# Remove the grid from the plot
ax1.grid(False)
ax2.grid(False)

# plt.savefig(f'{title[xi]}.svg', dpi=300, bbox_inches='tight')
# plt.show()

# # Append the results to the DataFrame
# new_row = pd.DataFrame({
#         'file': [txt],
#         'n': [n],
#         'Is': [Is],
#         'Rs': [Rs],
#         'Rsh': [Rsh],
#         'Voc': [Voc_sim],
#         'Vmp': [Vmp_sim],
#         'Jsc': [Isc_sim*1000],
#         'Jmp': [Imp_sim*1000],
#         'Pmax': [Pmax_sim*1000],
#         'FF': [FF_sim],
#         'η': [PCE_sim*100]
#         })

# results_df = pd.concat([results_df, new_row], ignore_index=True)
    
# headers = ['file', 'n', 'Is', 'Rs', 'Rsh', 'Voc', 'Vmp', 'Jsc', 'Jmp', 'Pmax', 'FF', 'η']
# print(tabulate(results_df, headers=headers, tablefmt="grid"))

# # Write the DataFrame to an Excel file
# results_df.to_csv('resultscelsconts_new.csv', index=False)

