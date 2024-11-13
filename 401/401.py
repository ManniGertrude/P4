import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.odr as odr
from scipy.optimize import curve_fit
from scipy.integrate import quad
import numpy as np
import csv
import os
import pandas as pd
from sklearn.metrics import r2_score


input_dir = os.path.dirname(os.path.abspath(__file__))

def read_csv_input(test):
    data = pd.read_csv(f'{input_dir}\\{test}', sep="\t", header=0).astype(np.float64)
    return data


 
def gaus(x,a, o,m,a1,o1,m1,a2,o2,m2,a3,o3,m3,a4,o4,m4,a5,o5,m5,a6,o6,m6, d, n):
    g1= a/(o*(2*np.pi)**(1/2))*np.exp(-(x-m)**2/(2*o)**2)
    g2= a1/(o1*(2*np.pi)**(1/2))*np.exp(-(x-m1)**2/(2*o1)**2) 
    g3= a2/(o2*(2*np.pi)**(1/2))*np.exp(-(x-m2)**2/(2*o2)**2)
    g4= a3/(o3*(2*np.pi)**(1/2))*np.exp(-(x-m3)**2/(2*o3)**2)
    g5= a4/(o4*(2*np.pi)**(1/2))*np.exp(-(x-m4)**2/(2*o4)**2)
    g6= a5/(o5*(2*np.pi)**(1/2))*np.exp(-(x-m5)**2/(2*o5)**2)
    g7= a6/(o6*(2*np.pi)**(1/2))*np.exp(-(x-m6)**2/(2*o6)**2)
    lin = d*x + n
    return g1 +g2 +g3+ g4 + g5 +g6+g7 + lin

def fit_multi_gaus(data, Name):
    xdata = np.asarray(data['U_B'], dtype=np.float64)
    ydata = np.asarray(data['U_A'], dtype=np.float64)
    y_err = [0.1]*len(ydata) # [0.1*data['U_A']]
    yerr = np.asarray(y_err)
    plt.grid()
    plt.errorbar(xdata, ydata, color='cornflowerblue', yerr= yerr, marker='o', markersize = 2, linestyle='none', label = 'Datenpunkte')
    p0 = np.asarray([10, 1, 6, 10, 1, 12, 10, 1,17 ,10, 1, 21,10,1,26, 10,1, 31,10, 1,36, 0 ,-1])
    popt, pcov = curve_fit(gaus, xdata, ydata,  p0=p0, sigma=yerr,absolute_sigma=True, maxfev=int(1e7))
    perr = np.sqrt(np.diag(pcov))
    print (popt,perr)
    ydatapre = gaus(xdata, *popt)
    chisq = np.sum(((ydata-ydatapre)**2)/yerr**2)
    chi_ndf = chisq/(len(xdata)-len(p0))
    print ('\chi^2', chisq, 'chi/ndf', chi_ndf)
    plt.plot(xdata, gaus(xdata, *popt), color ='midnightblue', linestyle = '-', label='Anpassungsfunktion')
    bottom, top = plt.ylim()
    plt.ylim(bottom, top*1.2)
    plt.xlabel('Beschleunigungsspannung $U_{B}$ /V')
    plt.ylabel('Anodenstrom $I_{A}$ /A')
    plt.title ('Anodenstrom bei $T=165$°C, $U_G=2,5$V mit $\chi^2$/ndf= %.2f'%(chi_ndf))
    plt.legend()
    plt.savefig(f'{input_dir}\\gausmulti2{Name}.png')
    plt.show


   
def gaus_sep(x, a,o,m):
    return a/(o*(2*np.pi)**(1/2))*np.exp(-(x-m)**2/(2*o)**2)

def individual_gaus(data, Name):
    print(data)
    xdata = np.asarray(data['U_B'], dtype=np.float64)
    ydata = np.asarray(data['U_A'], dtype=np.float64)
    y_err = [0.1]*len(ydata) # [0.1*value for value in data['U_A']]
    yerr = np.asarray(y_err)
    #print(xdata, ydata)
    plt.grid()
    max = []
    #1.Bereich
    xdata_filtered = xdata[np.where((xdata > 10) & (xdata < 14.5))]
    ydata_filtered = ydata[np.where((xdata > 10) & (xdata < 14.5))]
    yerr_filtered = yerr[np.where((xdata > 10) & (xdata < 14.5))]
    p0=([10, 1, 11])
    popt, pcov = curve_fit(gaus_sep, xdata_filtered, ydata_filtered, p0=p0,
                                              sigma=yerr_filtered, absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))
    print (popt, perr)
    rsquaredfil = r2_score(ydata_filtered,gaus_sep(xdata_filtered, *popt))
    print ('$R^2$ :', rsquaredfil)
    plt.plot(xdata_filtered, gaus_sep(xdata_filtered, *popt), color = 'midnightblue',
            label="Anpassungsfunktion")
    max_x = opt.fmin(lambda x: -gaus_sep(x, *popt), 0)
    if max_x > 0:
        max.append(max_x[0])
    #2.bereich
    xdata_filtered = xdata[np.where((xdata > 14) & (xdata < 19))]
    ydata_filtered = ydata[np.where((xdata > 14) & (xdata < 19))]
    yerr_filtered = yerr[np.where((xdata > 14) & (xdata < 19))]
    p0=([10, 1, 16])
    popt, pcov = curve_fit(gaus_sep, xdata_filtered, ydata_filtered,  p0=p0,
                                              sigma=yerr_filtered, absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))
    print (popt, perr)
    rsquaredfil = r2_score(ydata_filtered,gaus_sep(xdata_filtered, *popt))
    print ('$R^2$ :', rsquaredfil)
    plt.plot(xdata_filtered, gaus_sep(xdata_filtered, *popt),color = 'midnightblue')
    max_x = opt.fmin(lambda x: -gaus_sep(x, *popt), 0)
    if max_x > 0:
        max.append(max_x[0])
    #3.bereich
    xdata_filtered = xdata[np.where((xdata > 18.5) & (xdata < 24))]
    ydata_filtered = ydata[np.where((xdata > 18.5) & (xdata < 24))]
    yerr_filtered = yerr[np.where((xdata > 18.5) & (xdata < 24))]
    p0=([10, 1, 21])
    popt, pcov = curve_fit(gaus_sep, xdata_filtered, ydata_filtered,  p0=p0,
                                              sigma=yerr_filtered, absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))
    print (popt, perr)
    rsquaredfil = r2_score(ydata_filtered,gaus_sep(xdata_filtered, *popt))
    print ('$R^2$ :', rsquaredfil)
    plt.plot(xdata_filtered, gaus_sep(xdata_filtered, *popt),color = 'midnightblue')
    max_x = opt.fmin(lambda x: -gaus_sep(x, *popt), 0)
    if max_x > 0:
        max.append(max_x[0])
    #4.Bereich
    xdata_filtered = xdata[np.where((xdata > 23.5) & (xdata < 29))]
    ydata_filtered = ydata[np.where((xdata > 23.5) & (xdata < 29))]
    yerr_filtered = yerr[np.where((xdata > 23.5) & (xdata < 29))]
    p0=([10, 1, 26])
    popt, pcov = curve_fit(gaus_sep, xdata_filtered, ydata_filtered,  p0=p0,
                                              sigma=yerr_filtered, absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))
    rsquaredfil = r2_score(ydata_filtered,gaus_sep(xdata_filtered, *popt))
    print ('$R^2$ :', rsquaredfil)
    print (popt, perr)
    plt.plot(xdata_filtered, gaus_sep(xdata_filtered, *popt),color = 'midnightblue')
    max_x = opt.fmin(lambda x: -gaus_sep(x, *popt), 0)
    if max_x > 0:
        max.append(max_x[0])
    #5.bereich
    xdata_filtered = xdata[np.where((xdata > 28.5) & (xdata < 34))]
    ydata_filtered = ydata[np.where((xdata > 28.5) & (xdata < 34))]
    yerr_filtered = yerr[np.where((xdata > 28.5) & (xdata < 34))]
    p0=([10, 1, 31])
    popt, pcov = curve_fit(gaus_sep, xdata_filtered, ydata_filtered,  p0=p0,
                                              sigma=yerr_filtered, absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))
    print (popt, perr)
    rsquaredfil = r2_score(ydata_filtered,gaus_sep(xdata_filtered, *popt))
    print ('$R^2$ :', rsquaredfil)
    plt.plot(xdata_filtered, gaus_sep(xdata_filtered, *popt),color = 'midnightblue')
    max_x = opt.fmin(lambda x: -gaus_sep(x, *popt), 0)
    if max_x > 0:
        max.append(max_x[0])
    #6.bereich
    xdata_filtered = xdata[np.where((xdata > 33.5) & (xdata < 38.5))]
    ydata_filtered = ydata[np.where((xdata > 33.5) & (xdata < 38.5))]
    yerr_filtered = yerr[np.where((xdata > 33.5) & (xdata < 38.5))]
    p0=([10, 1, 36])
    popt, pcov = curve_fit(gaus_sep, xdata_filtered, ydata_filtered,  p0=p0,
                                              sigma=yerr_filtered, absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))
    print (popt, perr)
    rsquaredfil = r2_score(ydata_filtered,gaus_sep(xdata_filtered, *popt))
    print ('$R^2$ :', rsquaredfil)
    plt.plot(xdata_filtered, gaus_sep(xdata_filtered, *popt),color = 'midnightblue')
    max_x = opt.fmin(lambda x: -gaus_sep(x, *popt), 0)
    if max_x > 0:
        max.append(max_x[0])
    #7.bereich
    plt.errorbar(xdata, ydata, yerr= yerr, color='cornflowerblue', marker='o',markersize =1, linestyle='none')
    bottom, top = plt.ylim()
    plt.ylim(bottom, top*1.2)
    plt.xlabel('Beschleunigungsspannung $U_{B}$ /V')
    plt.ylabel('Anodenstrom $I_{A}$ /A')
    plt.legend()
    plt.savefig(f'{input_dir}\\gausindividual{Name}.png')
    plt.show
    
    maxx = np.array(max)
    print (maxx)
    abs= []
    maximum = np.zeros(len(maxx))
    for i in range (1, len(maxx)):
        maximum[i] = maxx[i]-maxx[i-1]
        abs.append(maximum[i])
    abstand =np.array(abs)
    print (abstand)
    

f = ["Data\\Tconst_165\\U2_2_0V.csv", 
     "Data\\Tconst_165\\U2_2_7V.csv", 
     "Data\\Tconst_165\\U2_3_4V.csv", 
     "Data\\Tconst_165\\U2-4_0V.csv", 
     "Data\\Uconst_2.7\\T_170.csv", 
     "Data\\Uconst_2.7\\T_175.csv", 
     "Data\\Uconst_2.7\\T_180.csv", 
     "Data\\Uconst_2.7\\U2_2_7V_t_165.csv"]

NameList = ["U2_2_0V", "U2_2_7V", "U2_3_4V", "U2_4_0V", "T_170", "T_175", "T_180", "T_165"]

for i in range(len(f)):
    data = read_csv_input(f[i])
    fit_multi_gaus(data, NameList[i])
    individual_gaus(data, NameList[i])



