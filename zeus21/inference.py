"""

Code to compute binned observed power spectra and likelihood for inference
"""

import numpy as np
from . import cosmology
from . import correlations
from . import sfrd
from . import inputs

class Power_Spectra_Binned:
    def __init__(self, Power_Spectra, T21_coefficients, k_range,z_range):
    
        self.z_center_value = np.ndarray(len(z_range)-1)
        self.k_center_value = np.ndarray(len(k_range)-1)


        klist = Power_Spectra.klist_PS
        zlist = T21_coefficients.zintegral
        self.z_bins = z_range
        self.k_bins =  k_range


        for i in range(len(self.z_center_value)):
            self.z_center_value[i] = (self.z_bins[i]-self.z_bins[i+1])/2+self.z_bins[i+1]

        for i in range(len(self.k_center_value)):
            self.k_center_value[i] = (self.k_bins[i]-self.k_bins[i+1])/2+self.k_bins[i+1]

        
        iz_list = []
        ik_list = []

        for j in range(len(self.z_center_value)):
            iz_list.append(min(range(len(zlist)), key=lambda i: np.abs(zlist[i]-self.z_center_value[j])))
        for j in range(len(self.k_center_value)):
            ik_list.append(min(range(len(klist)), key=lambda i: np.abs(klist[i]-self.k_center_value[j])))


        self.PS21 = np.ndarray((len(iz_list), len(ik_list)))

        for i in range(len(ik_list)):
            for j in range(len(iz_list)):


                self.PS21[j,i] = Power_Spectra.Deltasq_T21[iz_list[j],ik_list[i]]


class Inference:
    def __init__(self, data, noise, k_range, z_range, h_peak_range, k_peak_range, 
                 hmf_dict, CosmoParams, ClassCosmo, Correlations, 
                 h_peak, k_peak, epsstar = 0.1, alphastar = 0.5, L40_xray = 3.0, E0_xray = 500, z=10, RSD=1, Flag_Only_ETHOS = True):
        
        if Flag_Only_ETHOS == True:
            self.logPrior = log_prior_ETHOS_only(h_peak, k_peak, h_peak_range, k_peak_range)
        #else:
            #self.logPrior = log_prior(h_peak, k_peak,h_peak_range, k_peak_range, epsstar_range, alphastar_range, L40_xray_range, E0_xray_range, epsstar = 0.1, alphastar = 0.5, L40_xray = 3.0, E0_xray = 500,Flag_Only_ETHOS = False)
        self.logLikelihood = log_likelihood(data, noise, k_range,z_range, hmf_dict, CosmoParams, ClassCosmo, Correlations, h_peak, k_peak, epsstar, alphastar, L40_xray, E0_xray, z, RSD, Flag_Only_ETHOS)
        self.logPosterior = self.logPrior + self.logLikelihood


def log_posterior(theta, data, noise, k_range, z_range, h_peak_range, k_peak_range, 
                 hmf_dict, CosmoParams, ClassCosmo, Correlations, 
                 epsstar = 0.1, alphastar = 0.5, L40_xray = 3.0, E0_xray = 2.7, z=10, RSD=1, Flag_Only_ETHOS = True):
    

    if Flag_Only_ETHOS == True:

        h_peak, k_peak = theta

        k_peak = 10**k_peak
        E0_xray = 100**E0_xray
        logPrior = log_prior_ETHOS_only(h_peak, k_peak, h_peak_range, k_peak_range)
        #else:
            #logPrior = log_prior(h_peak, k_peak,h_peak_range, k_peak_range, epsstar_range, alphastar_range, L40_xray_range, E0_xray_range, epsstar = 0.1, alphastar = 0.5, L40_xray = 3.0, E0_xray = 500,Flag_Only_ETHOS = False)
    logLikelihood = log_likelihood(data, noise, k_range,z_range, hmf_dict, CosmoParams, ClassCosmo, Correlations, h_peak, k_peak, epsstar, alphastar, L40_xray, E0_xray, z, RSD, Flag_Only_ETHOS)
    
    return logPrior + logLikelihood       

def log_prior(h_peak, k_peak,h_peak_range, k_peak_range, epsstar_range, alphastar_range, L40_xray_range, E0_xray_range,
              epsstar, alphastar, L40_xray, E0_xray):
        

    h_peak_min = np.min(h_peak_range)
    h_peak_max = np.max(h_peak_range)
    k_peak_min = np.min(k_peak_range)
    k_peak_max = np.max(k_peak_range) 
    epsstar_min = np.min(epsstar_range)
    epsstar_max = np.max(epsstar_range)
    alphastar_min = np.min(alphastar_range)
    alphastar_max = np.max(alphastar_range)
    L40_xray_min = np.min(L40_xray_range)
    L40_xray_max = np.max(L40_xray_range)
    E0_xray_min = np.min(E0_xray_range)
    E0_xray_max = np.max(E0_xray_range) 


    if h_peak_min <= h_peak <= h_peak_max:
        p_h = 1
    else:
        p_h = 0
    if k_peak_min < k_peak < k_peak_max:
        p_k = 1#k_peak**-1
    else:
        p_k     = 0
    if epsstar_min < epsstar < epsstar_max:
            p_epsstar = 1#epsstar**-1 
    else:
        p_epsstar = 0
    if alphastar_min < alphastar < alphastar_max:
        p_alphastar = 1
    else:
        p_alphastar = 0
    if L40_xray_min < L40_xray < L40_xray_max:
        p_L40_xray = 1#L40_xray**-1 
    else:
        p_L40_xray = 0
    if E0_xray_min < E0_xray < E0_xray_max:
        p_E0_xray = 1#E0_xray**-1
    else:
        p_E0_xray = 0

    return np.log(p_h*p_k*p_epsstar*p_alphastar*p_L40_xray*p_E0_xray)


def log_likelihood(data, noise, k_range,z_range, hmf_dict, CosmoParams, ClassCosmo, Correlations, h_peak, k_peak, epsstar = 0.1, alphastar = 0.5, L40_xray = 3.0, E0_xray = 500, z=10, RSD=1, Flag_Only_ETHOS = True):

  

    if Flag_Only_ETHOS == True:
        k_peak = 10**k_peak 
        #theta = h_peak, k_peak
        #hmf = hmfdict[h_peak, k_peak]
        hmf = read_hmf(h_peak, k_peak, hmf_dict)
        AstroParams = inputs.Astro_Parameters(Cosmo_Parameters=CosmoParams)

    else:

        k_peak = 10**k_peak
        epsstar = 10**epsstar
        L40_xray = 10**L40_xray
        E0_xray = 10**E0_xray

        #theta = h_peak, k_peak, epsstar, alphastar, L40_xray, E0_xray
        #hmf = hmfdict[h_peak, k_peak]
        hmf = read_hmf(h_peak, k_peak, hmf_dict)
        AstroParams = inputs.Astro_Parameters(Cosmo_Parameters=CosmoParams, epsstar=epsstar, alphastar=alphastar, L40_xray=L40_xray, E0_xray=E0_xray)    
    

    T21_coeff = sfrd.get_T21_coefficients(CosmoParams, ClassCosmo, AstroParams, hmf, zmin=z) 
    powerspec21_model = correlations.Power_Spectra(CosmoParams, ClassCosmo, Correlations, T21_coeff, RSD_MODE=RSD)
        
    #self.klist = powerspec21_model.klist_PS
    #self.zlist = T21_coeff.zintegral

    obsPS = Power_Spectra_Binned(powerspec21_model, T21_coeff, k_range,z_range)

    lh = -1/2*(np.log(2*np.pi*noise**2)+((data - obsPS.PS21)/ noise)**2)
    log_lh = np.sum(lh)

    return log_lh

def log_prior_ETHOS_only(h_peak, k_peak, h_peak_range, k_peak_range):



    h_peak_min = np.min(h_peak_range)
    h_peak_max = np.max(h_peak_range)
    k_peak_min = np.min(k_peak_range)
    k_peak_max = np.max(k_peak_range) 


        
    if h_peak_min <= h_peak <= h_peak_max:
        p_h = 1
    else:
        p_h = 0
    if k_peak_min < k_peak < k_peak_max:
        p_k = 1 #k_peak**-1
    else:
        p_k = 0
    return np.log(p_h*p_k)



def read_hmf(h_peak, k_peak, hmf_dict): 

    #read from nearest dictionary keys to input h_peak and k_peak
    
    h_peak_input = h_peak
    k_peak_input = k_peak
    hmf_dict = hmf_dict

    input = (h_peak_input, k_peak_input)
    hmf_keys = list(hmf_dict.keys())

    closest_keys = min(hmf_keys, key=lambda x: ((x[0] - input[0])**2 + (x[1] - input[1])**2)**0.5)

    return(hmf_dict[closest_keys])
 
