"""
Written by Enrico Ciraci 10/11/2018

Extract the contribution of ENSO/IOD using mulit-linear regression method (MLR)
from the selected monthly time series.

The input time series is decomposed  employing the following model finction:
X(t) = beat0 + beta1*t + beta2*cos(2pi*t) + beta3*sin(2pi*t) +
       beta4*cos(4pi*t) + beta5*sin(2pi*t) + beta6*Nino3.4 + 
       beta7*H{Nino3.4} + beta8*IOD + beta9*H{IOD}

Where:

- Nino3.4 is the monthly Nino 3.4 ENSO index;
- H{Nino3.4} is the monthly Nino 3.4 ENSO index Hilber transfrom;
- IOD is the Indian Ocean Dipole Index;
- H{IOD} is the Indian Ocean Dipole Index Hilber transfrom;

More info about the signal decomposition can be found in:
Forootan et al. 2016:
"Quantifying the impacts of ENSO and IOD on rain gauge and remotely 
sensed precipitation products over Australia"
https://www.researchgate.net/publication/283722412_
    Quantifying_the_impacts_of_ENSO_and_IOD_on_rain_gauge_and_
    remotely_sensed_precipitation_products_over_Australia

Download IOD index from:
- NOTE: You need to chose a specific version of the index
http://www.jamstec.go.jp/frsgc/research/d1/iod/e/iod/dipole_mode_index.html

Download NINO 3.4 index from:
http://www.cpc.ncep.noaa.gov/data/indices/
"""
# - python dependencies
from __future__ import print_function
import os
import numpy as np
import scipy as sp
from scipy.fftpack import hilbert
from scipy import signal
import matplotlib.pyplot as plt



def digit_date(year, month):

    # -- create output date variable
    t_date = np.zeros((1))
    
    # -- Vector containing the number of days for a leap and a standard year
    dpm_leap = np.array([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=np.float)
    dpm_stnd = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=np.float)
    
    # -- I need this matrix in order to calculate the total number of day passed from
    # -- The begin of the year and to pass to the date digital format
    mon_mat = np.tri(12, 12, -1)
    
    # -- month index
    mon_ind = np.int(month-1)
    
    # -- Verifying if we are considering a std or a leap year
    m4 = (year % 4)
    m100 = (year % 100)
    m400 = (year % 400)
    leap = np.squeeze(np.nonzero((m4 == 0) & (m100 != 0) | (m400 == 0)))
    stnd = np.squeeze(np.nonzero((m4 != 0) & (m100 == 0) | (m400 != 0)))
    
    # -- leap years
    t_date[leap] = (year + (np.dot(mon_mat[mon_ind, :], dpm_leap)/np.sum(dpm_leap)) + ((dpm_leap[mon_ind]/2.)/np.sum(dpm_leap)))
                   
    # -- std years
    t_date[stnd] = (year + (np.dot(mon_mat[mon_ind, :], dpm_stnd)/np.sum(dpm_stnd)) + ((dpm_stnd[mon_ind]/2.)/np.sum(dpm_stnd)))
    t_date = np.float(t_date)

    return t_date




def main():
    # - read nino 3.4 time series
    input_data_path = os.path.join('.', 'data', 'nino_3.4.txt')
    d_input = np.loadtxt(input_data_path, skiprows=1)
    year = d_input[:, 0]
    month = d_input[:, 1]
    nino34 = d_input[:, 4]
    d_time = np.zeros(len(year))
    
    # - Calculate digital time axis
    for t in range(len(year)):
        d_time[t] = digit_date(year[t], month[t])
    # - find time period after 1980
    ind = np.where((d_time >= 1980) & (d_time <= 2014))
    d_time = d_time[ind]
    nino34 = nino34[ind]
    # - calculate Hilbert transform through fast fourier transform (FFT)
    # - of the time series
    nino34_hilb = hilbert(nino34)


    # - read IOD-DMI index
    input_data_path = os.path.join('.', 'data', 'dmi.monthly.txt')
    d_input_dmi = np.loadtxt(input_data_path, skiprows=0)
    year_dmi = d_input_dmi[:, 0]
    month_dmi = d_input_dmi[:, 1]
    dmi = d_input_dmi[:, 2]
    d_time_dmi = np.zeros(len(year_dmi))
    # - Calculate digital time axis
    for t in range(len(year_dmi)):
        d_time_dmi[t] = digit_date(year_dmi[t], month_dmi[t])
    # - find time period after 1980
    ind = np.where((d_time_dmi >= 1980) & (d_time_dmi <= 2014))
    d_time_dmi = d_time_dmi[ind]
    dmi = dmi[ind]
    # - calculate Hilbert transform through fast fourier transform (FFT)
    # - of the time series
    dmi_hilb = hilbert(dmi)


    # - Reproduce figure 3 in Forotan et al. 2015
    plt.figure(figsize=(9, 2), dpi=200)
    plt.title('Compare with Fig. 2 in Forootan et al. 2016')
    plt.fill_between(d_time, -nino34, facecolor='#0099ff', 
                     label='-Nino3.4', zorder=0, edgecolor='b', alpha=0.7)
    plt.plot(d_time, nino34_hilb, color='#0099ff', label='-H(Nino3.4)', 
             zorder=2, ls='-.', lw=1, alpha=0.7)
    plt.fill_between(d_time_dmi, -dmi, facecolor='#ff4d4d', label='-DMI', 
                     zorder=1, edgecolor='#CC0000')
    plt.plot(d_time_dmi, dmi_hilb, color='#990000', label='-H(DMI)', 
             zorder=2, ls='-.', lw=1)
    plt.ylim([-3, 3])
    plt.ylabel('Standard Index')
    plt.xlabel('year')
    plt.xlim([1980, 2015])
    plt.grid()
    plt.legend(loc=1, prop={'size': 6})
    plt.savefig(os.path.join('.', 'output', 'Enso_ts_Forotan_et_al_2016.png'), dpi=200)
    plt.close()

    # - load smaple GRACE time series - for this example, use Upper Indus River basin time series
    tws_input = np.loadtxt(os.path.join('.', 'data', 'Indus_UpperBasin_grace-gia_[Gt].ts.txt'), skiprows=1)
    tws_d_time = tws_input[:, 1]
    tws_mass = tws_input[:, 2]

    # - crop indexes  across the same GRACE time period
    last_month = np.min(np.array([d_time_dmi[-1], d_time[-1]]))
    index = np.where((d_time >= tws_d_time[0]) & (d_time <= last_month))
    d_time = d_time[index]
    nino34_hilb = nino34_hilb[index]
    nino34 = nino34[index]

    index = np.where((d_time_dmi >= tws_d_time[0]) & (d_time_dmi <= last_month))
    d_time_dmi = d_time_dmi[index]
    dmi_hilb = dmi_hilb[index]
    dmi = dmi[index]

    # - remove last tws months
    index = np.where(tws_d_time <= d_time[-1])
    tws_d_time = tws_d_time[index]
    tws_mass = tws_mass[index]

    # - interpolate GRACE missin months 
    tws_mass_interp = np.zeros(len(nino34))
    for tt in range(len(nino34)):
        tws_mass_interp[tt] = np.interp(d_time[tt], tws_d_time, tws_mass)

    tws_mass -= np.mean(tws_mass)
    tws_mass_interp -= np.mean(tws_mass_interp)
    # - NOTE: The TWS time series should probably detrended at this point

    # - verify the result of the interpolation
    plt.figure()
    plt.title('Interpolation Result')
    plt.plot(tws_d_time, tws_mass, color='r', 
             label='TWS Original', zorder=0)
    plt.plot(d_time, tws_mass_interp, color='b', ls='-.', 
             label='TWS Interp.', zorder=1, alpha=0.7)
    plt.grid()
    plt.legend(loc=1, prop={'size': 6})
    plt.savefig(os.path.join('.', 'output', 'TWS_ts.png'), dpi=200)
    plt.close()


    #-- CREATING DESIGN MATRIX FOR REGRESSION
    P_x0 = np.ones((len(d_time)))              #-- Constant Term
    P_x1 = (d_time-np.mean(d_time))          #-- Linear Term
    #--- Annual term = 2*pi*t*harmonic, Semi-Annual = 4*pi*t*harmonic
    P_asin = np.sin(2*np.pi*d_time)
    P_acos = np.cos(2*np.pi*d_time)
    P_ssin = np.sin(4*np.pi*d_time)
    P_scos = np.cos(4*np.pi*d_time)
    tmat = np.array([P_x0, P_x1, P_asin, P_acos, P_ssin, P_scos, nino34, nino34_hilb, dmi, dmi_hilb]).T

    # - Normalize TWS anomaly before applying the Least Squares Fit
    tws_mass_interp_norm = np.copy(tws_mass_interp)/np.std(tws_mass_interp)
    
    # - Apply linear regression
    beta_mat = np.linalg.lstsq(tmat, tws_mass_interp_norm)[0]
    # -
    beta_0 = beta_mat[0]
    beta_1 = beta_mat[1]
    beta_2 = beta_mat[2]
    beta_3 = beta_mat[3]
    beta_4 = beta_mat[4]
    beta_5 = beta_mat[5]
    beta_6 = beta_mat[6]
    beta_7 = beta_mat[7]
    beta_8 = beta_mat[8]
    beta_9 = beta_mat[9]
    
    # - Annual Cycle Amplitude and pahse
    annual_amp = np.sqrt((beta_2**2) + (beta_3**2))
    annual_phase = (180/np.pi) * np.arctan(beta_3/beta_2)
    # - Semi-Annual Cycle Amplitude and pahse
    semi_annual_amp = np.sqrt((beta_4**2) + (beta_5**2))
    annual_phase = (180/np.pi) * np.arctan(beta_5/beta_4)
    # - Enso Contribution Amplitude and pahse
    enso_crontib_amp = np.sqrt((beta_6**2) + (beta_7**2))
    enso_phase = (180/np.pi) * np.arctan(beta_7/beta_6)
    # - IOD Contribution Amplitude abd pahse
    iod_crontib_amp = np.sqrt((beta_8**2) + (beta_9**2))
    iod_phase = (180/np.pi) * np.arctan(beta_9/beta_8)
    
    # - ENSO/IOD mode form the MLR techiniques
    x_enso_iod = (beta_6*nino34) + (beta_7*nino34_hilb) + (beta_8*dmi) + (beta_8*dmi_hilb)
    # - Seasonal Mode
    x_enso_seasonal = np.copy(tws_mass_interp_norm) - x_enso_iod
    
    # - verify the result of the interpolation
    plt.figure()
    plt.plot(d_time, tws_mass_interp_norm - np.mean(tws_mass_interp_norm), color='r', 
             label='TWS Normalized', zorder=0)
    
    plt.plot(d_time, x_enso_iod - np.mean(x_enso_iod), color='b', ls='-.', 
             label='ENSO/IOD mode', zorder=1, alpha=0.7)
    
    plt.plot(d_time, x_enso_seasonal - np.mean(x_enso_seasonal), color='g', ls='-.', 
             label='ENSO/Seasonal mode', zorder=1, alpha=0.7)
    plt.grid()
    plt.legend(loc=1, prop={'size': 6})
    plt.savefig(os.path.join('.', 'output', 'ENSO_IOD_ts.png'), dpi=200)
    plt.close()


# -- run main program
if __name__ == '__main__':
    main()
