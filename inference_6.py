
import numpy as np
import zeus21
#set up the CLASS cosmology
from classy import Class
ClassCosmo = Class()
ClassCosmo.compute()
import emcee
import corner
import pickle
from datetime import datetime

#Cosmology
CosmoParams_input = zeus21.Cosmo_Parameters_Input()
ClassCosmo = zeus21.runclass(CosmoParams_input)
CosmoParams = zeus21.Cosmo_Parameters(CosmoParams_input,ClassCosmo)
print('CLASS has run')

#Astronomy
Correlations = zeus21.Correlations(CosmoParams,ClassCosmo)
AstroParams = zeus21.Astro_Parameters(CosmoParams)

#Load dictionary of HMF
with open('hmf_dict.pickle', 'rb') as f:
    hmf_dict= pickle.load(f)
print('Loaded dictionary')

#Observed Power Spectra
z_range = np.array([27.4, 23.4828, 20.5152, 18.1892, 16.3171, 14.7778, 13.4898, 312.3962, 11.4561, 10.6393, 9.92308]) #bins in z
k_range = np.array([0.1, 0.11, 0.121, 0.1331, 0.14641, 0.161051, 0.177156, 0.194872, 0.214359, 0.235795, 0.259374, 
                   0.285312, 0.313843, 0.345227, 0.37975, 0.417725, 0.459497, 0.505447, 0.555992, 0.611591, 0.67275, 
                   0.740025, 0.814027, 0.89543, 0.984973, 1.08347, 1.19182, 1.311, 1.4421]) #bins in k

T21_coeff = zeus21.get_T21_coefficients(CosmoParams, ClassCosmo, AstroParams,hmf_dict[0.5,100], zmin=10) #input to fiducail PS
powerspec21 = zeus21.Power_Spectra(CosmoParams, ClassCosmo, Correlations, T21_coeff, RSD_MODE=1) #continuous PS
obs_PS21 = zeus21.Power_Spectra_Binned(powerspec21, T21_coeff, k_range,z_range) #Binned PS
print('Created power spectra')

#Ranges for Priors
k_peak_range = np.array([0,2.5]) #log
h_peak_range = np.array([0.2,0.4,0.5,0.6,0.7,0.8,0.9,1.])
epsstar_range = np.array([-5,0]) #log
alphastar_range = np.array([0,3])
L40_xray_range = np.array([-3,3]) #log
E0_xray_range = np.array([2,3]) #log

#Input for MCMC
data = obs_PS21.PS21 #Observed data
noise = np.sqrt((data/10.)**2+4**2) #Obserced noise on data

n_walkers = 25
n_steps = 500
n_params = 6
args = [data, noise, k_range, z_range, h_peak_range, k_peak_range, epsstar_range, alphastar_range, L40_xray_range, E0_xray_range, hmf_dict, CosmoParams, ClassCosmo, Correlations]

#Initial Guesses
h_peak_random = np.random.rand(n_walkers)*(np.max(h_peak_range) - np.min(h_peak_range)) + np.min(h_peak_range)
k_peak_random = np.random.rand(n_walkers)*(np.max(k_peak_range) - np.min(k_peak_range)) + np.min(k_peak_range)
eps_random = np.random.rand(n_walkers)*(np.max(epsstar_range) - np.min(epsstar_range)) + np.min(epsstar_range)
alpha_random = np.random.rand(n_walkers)*(np.max(alphastar_range) - np.min(alphastar_range)) + np.min(alphastar_range)
L40_random = np.random.rand(n_walkers)*(np.max(L40_xray_range) - np.min(L40_xray_range)) + np.min(L40_xray_range)
E0_random= np.random.rand(n_walkers)*(np.max(E0_xray_range) - np.min(E0_xray_range)) + np.min(E0_xray_range)

initial_guesses = np.array([h_peak_random, k_peak_random, eps_random, alpha_random, L40_random, E0_random]).T

current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
print('Ready to run MCMC')

#Run MCMC
sampler = emcee.EnsembleSampler(n_walkers, n_params, zeus21.log_posterior, args=args) 
sampler.run_mcmc(initial_guesses, n_steps, progress=True);

#Save sampler 

np.save(f"samples_{current_time}.npy", sampler.get_chain())

print('File saved')