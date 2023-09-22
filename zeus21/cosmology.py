"""

Cosmology helper functions and other tools

Author: Julian B. Muñoz
UT Austin and Harvard CfA - January 2023

"""

import numpy as np
from classy import Class
import scipy
from scipy.interpolate import RegularGridInterpolator
from scipy.special import erfc
from scipy.interpolate import interp1d,InterpolatedUnivariateSpline
from scipy.integrate import quad, simps
from . import constants
from . import correlations
import mcfit
import os
base_path = os.path.abspath(os.path.dirname(__file__))



T_ETHOS_params = {'b':[-2.1,-3.7,4.1],
                  'd':[1.8,-6.7,2.5],
                  'tau':[0.03, 2.6, 0.27],
                  'sig':0.2}

h2_params = {'B':[1., 3., -0.32, -1.],
             'C':[0.54, 8., 0.7, 1.],
             'A':[0.8, 0.53,0.08],
             'A_skew':[0.17, 0.7, 0.2, -3]}











def runclass(CosmologyIn):
    "Set up CLASS cosmology. Takes CosmologyIn class input and returns CLASS Cosmology object"
    ClassCosmo = Class()
    ClassCosmo.set({'omega_b': CosmologyIn.omegab,'omega_cdm': CosmologyIn.omegac,
                    'h': CosmologyIn.h_fid,'A_s': CosmologyIn.As,'n_s': CosmologyIn.ns,'tau_reio': CosmologyIn.tau_fid})
    ClassCosmo.set({'output':'mPk','lensing':'no','P_k_max_1/Mpc':CosmologyIn.kmax_CLASS, 'z_max_pk': CosmologyIn.zmax_CLASS})
    #hfid = ClassCosmo.h() # get reduced Hubble for conversions to 1/Mpc

    # and run it (see warmup for their doc)
    ClassCosmo.compute()

    return ClassCosmo

def Hub(Cosmo_Parameters, z):
#Hubble(z) in km/s/Mpc
    return Cosmo_Parameters.h_fid * 100 * np.sqrt(Cosmo_Parameters.OmegaM * pow(1+z,3.)+Cosmo_Parameters.OmegaR * pow(1+z,4.)+Cosmo_Parameters.OmegaL)

def HubinvMpc(Cosmo_Parameters, z):
#H(z) in 1/Mpc
    return Hub(Cosmo_Parameters,z)/constants.c_kms

def Hubinvyr(Cosmo_Parameters,z):
#H(z) in 1/yr
    return Hub(Cosmo_Parameters,z)*constants.KmToMpc*constants.yrTos

def rho_baryon(Cosmo_Parameters,z):
#\rho_baryon in Msun/Mpc^3 as a function of z
    return Cosmo_Parameters.OmegaB * Cosmo_Parameters.rhocrit * pow(1+z,3.0)

def n_baryon(Cosmo_Parameters, z):
#density of baryons in 1/cm^3
    return rho_baryon(Cosmo_Parameters, z) / Cosmo_Parameters.mu_baryon_Msun / (constants.Mpctocm**3.0)



def Tcmb(ClassCosmo, z):
    T0CMB = ClassCosmo.T_cmb()
    return T0CMB*(1+z)

def Tadiabatic(CosmoParams, z):
    "Returns T_adiabatic as a function of z from thermodynamics in CLASS"
    return CosmoParams.Tadiabaticint(z)
def xefid(CosmoParams, z):
    "Returns fiducial x_e(z) w/o any sources. Uses thermodynamics in CLASS for z>15, and fixed below to avoid the tanh approx."
    _zcutCLASSxe = 15.
    _xecutCLASSxe = CosmoParams.xetanhint(_zcutCLASSxe)
    return CosmoParams.xetanhint(z) * np.heaviside(z - _zcutCLASSxe, 0.5) + _xecutCLASSxe * np.heaviside(_zcutCLASSxe - z, 0.5)

def adiabatic_index(z):
    "Returns adiabatic index (delta_Tad/delta) as a function of z. Fit from 1506.04152. to ~3% on z = 6 − 50)."
    return 0.58 - 0.005*(z-10.)


def MhofRad(Cosmo_Parameters,R):
    #convert input Radius in Mpc comoving to Mass in Msun
    return Cosmo_Parameters.constRM *pow(R, 3.0)

def RadofMh(Cosmo_Parameters,M):
    #convert input M halo in Msun radius in cMpc
    return pow(M/Cosmo_Parameters.constRM, 1/3.0)



def ST_HMF(Cosmo_Parameters, Mass, sigmaM, dsigmadM):
    A_ST = Cosmo_Parameters.Amp_ST
    a_ST = Cosmo_Parameters.a_ST
    p_ST = Cosmo_Parameters.p_ST
    delta_crit_ST = Cosmo_Parameters.delta_crit_ST

    nutilde = np.sqrt(a_ST) * delta_crit_ST/sigmaM

    return -A_ST * np.sqrt(2./np.pi) * nutilde * (1. + nutilde**(-2.0*p_ST)) * np.exp(-nutilde**2/2.0) * (Cosmo_Parameters.rho_M0 / (Mass*sigmaM)) * dsigmadM
    #return -A_ST * np.sqrt(2./np.pi) * nutilde * (1. + nutilde**(-2.0*p_ST)) * np.exp(-nutilde**2/2.0) * (Cosmo_Parameters.rho_M0 / (Mass * sigmaM)) * dsigmadM

def PS_HMF_unnorm(Cosmo_Parameters, Mass, nu, dlogSdM):
    'Returns the Press-Schechter HMF (unnormalized since we will take ratios), given a halo Mass [Msun], nu = delta_tilde/S_tilde, with delta_tilde = delta_crit - delta_R, and variance S = sigma(M)^2 - sigma(R)^2. Used for 21cmFAST mode.'

    return nu * np.exp(-Cosmo_Parameters.a_corr_EPS*nu**2/2.0) * dlogSdM* (1.0 / Mass)
    #written so that dsigmasq/dM appears directly, since that is not modified by EPS, whereas sigma_tot^2 = sigma^2(M) - sigma^2(R). The sigma in denominator will be sigma_tot





def T(k, alpha, b, c):
    """
    General non-CDM transfer function following Muriga+17

    Args:
        alpha: measure of cutoff scale length
        b: shape of cutoff
        c: shape of cutoff
    """
    return (1. + (alpha*k)**b)**c



def T_ETHOS(k, k_peak, h_peak, c=-20):
    """
    Bohr+2020 parameterisation

    Args:
        k: array
        k_peak:
        h_peak: must be in h_model list
        c: cutoff = gamma, all cases - large negative gamma, doesn't make a difference
    """

    b = f_exp(h_peak, *T_ETHOS_params['b'])
    d = f_exp(h_peak, *T_ETHOS_params['d'])
    tau = f_exp(h_peak, *T_ETHOS_params['tau'])
    sig = T_ETHOS_params['sig']

    h2 = h2_model(h_peak,k_peak)

    alpha = d/k_peak * (1./np.sqrt(2)**(1./c)-1)**(1./b)

    peak2_ratio = 1.805
    x_peak1 = (k - k_peak)/k_peak
    x_peak2 = (k - peak2_ratio*k_peak)/k_peak

    parameterisation = np.abs(T(k, alpha, b, c) \
                       - np.sqrt(h_peak) * np.exp(-0.5*(x_peak1/sig)**2) \
                       + np.sqrt(h2)/4. * erfc(x_peak2/tau - 2) \
                              * erfc(- x_peak2/sig - 2) * np.cos(1.1083*np.pi*k/k_peak))

    return parameterisation


def f_exp(h_peak, a, b, c):
    """
    exponential function
    """
    return a*np.exp(b*h_peak) + c


def f_tanh(h_peak, a=0.6, b=3.3, c=0.6, d=1):
    """
    tanh function
    """
    return a*(np.tanh(b*(h_peak-c))+d)


def f_skewgauss(h_peak, peak=0.16, h0=0.53, sig=0.08, alpha=0):
    """
    gaussian function
    """
    X = (h_peak - h0)/sig
    return peak/np.sqrt(2.*np.pi)/sig * np.exp(-0.5*X**2.) * (1 + scipy.special.erf(alpha*X/np.sqrt(2.)))


def h2_model(h_peak, k_peak):
    """
    h2(h_peak, k_peak)
    """
    A = f_skewgauss(h_peak, *h2_params['A_skew']) - f_skewgauss(0., *h2_params['A_skew'])
    B = f_tanh(h_peak, *h2_params['B'])
    C = f_tanh(h_peak, *h2_params['C']) - f_tanh(0., *h2_params['C'])
    return A*np.exp(B*k_peak) + C


def T_ETHOS_smooth(k, k_peak, h_peak, c=-20):
    """
    Bohr+2020 parameterisation

    Args:
        k: array
        k_peak:
        h_peak: must be in h_model list
        c: cutoff = gamma, all cases - large negative gamma, doesn't make a difference
    """

    b   = f_exp(h_peak, *T_ETHOS_params['b'])
    d   = f_exp(h_peak, *T_ETHOS_params['d'])
    tau = f_exp(h_peak, *T_ETHOS_params['tau'])
    sig = T_ETHOS_params['sig']

    h2 = h2_model(h_peak, k_peak)

    alpha = d/k_peak * (1./np.sqrt(2)**(1./c)-1)**(1./b)

    peak2_ratio = 1.805
    x_peak1 = (k - k_peak)/k_peak
    x_peak2 = (k - peak2_ratio*k_peak)/k_peak

    parameterisation = np.abs(T(k, alpha, b, c) \
                       - np.sqrt(h_peak) * np.exp(-0.5*(x_peak1/sig)**2) \
                       + np.sqrt(h2)/4. * erfc(x_peak2/tau - 2) \
                              * erfc(- x_peak2/sig - 2) * np.cos(1.1083*np.pi*k/k_peak))
    
    return parameterisation

def W_Smooth(k, R, beta=3.6, c=3.6):
    x = k*R/c
    return 1./(1+x**beta)

def dW_dR_Smooth(k, R, beta=3.6, c=3.6):
    return -beta/R * W_Smooth(k, R, beta, c)**2. * (k*R/c)**beta

    

class HMF_interpolator:
    "Class that builds an interpolator of the HMF. Returns an interpolator"

    def __init__(self, Cosmo_Parameters, ClassCosmo):
    #def __init__(self, Cosmo_Parameters, ClassCosmo, cosmo=None, h_peak=0., k_peak=100., 
	    #	     f_params='Schneider18', window_function='Smooth',
	    	#     logk=False, smooth_Tk=False, N_k=10000, LCDM=True):
        self._Mhmin = 1e5
        self._Mhmax = 1e14
        self._NMhs = np.floor(35*constants.precisionboost).astype(int)#35
        self.Mhtab = np.logspace(np.log10(self._Mhmin),np.log10(self._Mhmax),self._NMhs) # Halo mases in Msun
        self.RMhtab = RadofMh(Cosmo_Parameters, self.Mhtab)


        self.logtabMh = np.log(self.Mhtab)



        #ETHOS

        self.Flag_ETHOS = Cosmo_Parameters.Flag_ETHOS
        self.window_function = Cosmo_Parameters.window_function
        self.k_peak = Cosmo_Parameters.k_peak
        self.h_peak = Cosmo_Parameters.h_peak
        self.logk = Cosmo_Parameters.logk
        self.N_k = Cosmo_Parameters.N_k

        self.growth_facs = {}
        self._zmin=Cosmo_Parameters.zmin_CLASS
        self._zmax = Cosmo_Parameters.zmax_CLASS
        self._Nzs=np.floor(100*constants.precisionboost).astype(int)
        self.zHMFtab = np.linspace(self._zmin,self._zmax,self._Nzs)
        self.cosmo = Cosmo_Parameters.cosmo
        self.rhocrit_h2 = self.cosmo.critical_density0.to(u.Msun/u.Mpc**3.).value/self.cosmo.h**2.
        self.rho_mean = self.cosmo.Om0 * self.rhocrit_h2
        self.delta_crit = 1.686
        self.Pk = correlations.Correlations(Cosmo_Parameters, ClassCosmo)._PklinCF
        self.klist_Pk = correlations.Correlations(Cosmo_Parameters, ClassCosmo)._klistCF 
        self.klist = np.logspace(np.log10(1.1e-5), np.log10(500),len(self.klist_Pk))
        self.CLASS_Pk = np.zeros_like(self.klist) # P(k) in 1/Mpc^3
        for ik, kk in enumerate(self.klist):
            self.CLASS_Pk[ik] = ClassCosmo.pk(kk, 0.0)#ClassCosmo.pk(self.klist,0.0)
        self.load_Pk_LCDM()
        self.sigma2_M = np.vectorize(self.sigma2_M)
        self.dsigma2_dM = np.vectorize(self.dsigma2_dM)
        self.growth_facs = {}
       
              
        #check resolution
        if (Cosmo_Parameters.kmax_CLASS < 1.0/self.RMhtab[0]):
            print('Warning! kmax_CLASS may be too small! Run CLASS with higher kmax')


        if(Cosmo_Parameters.Flag_ETHOS==False):
            self.sigmaMhtab = np.array([[ClassCosmo.sigma(RR,zz) for zz in self.zHMFtab] for RR in self.RMhtab]) 
            self._depsM=0.01 #for derivatives, relative to M
            self.dsigmadMMhtab = np.array([[(ClassCosmo.sigma(RadofMh(Cosmo_Parameters, MM*(1+self._depsM)),zz)-ClassCosmo.sigma(RadofMh(Cosmo_Parameters, MM*(1-self._depsM)),zz))/(MM*2.0*self._depsM) for zz in self.zHMFtab] for MM in self.Mhtab])

        if(Cosmo_Parameters.Flag_ETHOS==True):

            self.sigmaMhtab = np.zeros(shape=(len(self.Mhtab),len(self.zHMFtab)))
            self.dsigmadMMhtab = np.zeros(shape=(len(self.Mhtab),len(self.zHMFtab)))
            self.gftab = np.array([self.growth_fac(zz) for zz in self.zHMFtab])
            self.sigmaMtab = np.array([self.sigma2_M(MM) for MM in self.Mhtab])
            self.dsigmadMtab = np.array([self.dsigma2_dM(MM) for MM in self.Mhtab])


            for iM, MM in enumerate(self.Mhtab):                
                for iz, zz in enumerate(self.zHMFtab):
                    sigmaM = np.sqrt(self.sigmaMtab[iM])*self.gftab[iz]
                    dsigmadM = self.dsigmadMtab[iM]*self.growth_fac_func2(zz)/2/np.sqrt(self.sigmaMtab[iM])
                    self.sigmaMhtab[iM,iz] = sigmaM
                    self.dsigmadMMhtab[iM,iz] = dsigmadM

                        

        if(Cosmo_Parameters.Flag_emulate_21cmfast==True):
            #ADJUST BY HAND adjust sigmas to match theirs, since the CLASS TF they use is at a fixed cosmology from 21cmvFAST but the input cosmology is different
            self.sigmaMhtab*=np.sqrt(0.975)
            self.dsigmadMMhtab*=np.sqrt(0.975)

            #this correction is because 21cmFAST uses the dicke() function to compute growth, which is ~0.5% offset at high z. This offset makes our growth the same as dicke() for a Planck2018 cosmology. Has to be added separately to the growth(z) correction above since they come in different places
            _offsetgrowthdicke21cmFAST = 1-0.000248*(self.zHMFtab-5.)
            self.sigmaMhtab*=_offsetgrowthdicke21cmFAST
            self.dsigmadMMhtab*=_offsetgrowthdicke21cmFAST
            #Note that these two changes may be different if away from Planck2018



        self.HMFtab = np.zeros_like(self.sigmaMhtab)
        #self.sigmatab = np.zeros_like(self.sigmaMhtab)
        #self.dsigmatab = np.zeros_like(self.sigmaMhtab)


        #self.gftab = np.array([self.growth_fac(zz) for zz in self.zHMFtab])
        #self.sigmaMtab = np.array([self.sigma2_M(MM) for MM in self.Mhtab])
        #self.dsigmadMtab = np.array([self.dsigma2_dM(MM) for MM in self.Mhtab])
       
        for iM, MM in enumerate(self.Mhtab):
            for iz, zz in enumerate(self.zHMFtab):
                sigmaM = self.sigmaMhtab[iM,iz]
                dsigmadM = self.dsigmadMMhtab[iM,iz]
                self.HMFtab[iM,iz] = ST_HMF(Cosmo_Parameters, MM, sigmaM, dsigmadM)

        #for iM, MM in enumerate(self.Mhtab):
         #   for iz, zz in enumerate(self.zHMFtab):
          #      if self.Flag_ETHOS == False:
           #         sigmaM = self.sigmaMhtab[iM,iz]
            #        dsigmadM =self.dsigmadMMhtab[iM,iz]
             #       self.HMFtab[iM,iz] = ST_HMF(Cosmo_Parameters, MM, sigmaM, dsigmadM)           
              #  else:
               #     sigmaM = np.sqrt(self.sigmaMtab[iM])*self.gftab[iz]
                #    dsigmadM = self.dsigmadMtab[iM]*self.growth_fac_func2(zz)/2/np.sqrt(self.sigmaMtab[iM])
                 #   self.HMFtab[iM,iz] = ST_HMF(Cosmo_Parameters, MM, sigmaM, dsigmadM)
                  #  self.sigmatab[iM,iz] = sigmaM
                   # self.dsigmatab[iM,iz] = dsigmadM




        _HMFMIN = np.exp(-300.) #min HMF to avoid overflowing
        logHMF_ST_trim = self.HMFtab
        logHMF_ST_trim[np.array(logHMF_ST_trim <= 0.)] = _HMFMIN
        logHMF_ST_trim = np.log(logHMF_ST_trim)


        self.fitMztab = [np.log(self.Mhtab), self.zHMFtab]
        self.logHMFint = RegularGridInterpolator(self.fitMztab, logHMF_ST_trim)

        self.sigmaintlog = RegularGridInterpolator(self.fitMztab, self.sigmaMhtab)# no need to log since it doesnt vary dramatically

        self.dsigmadMintlog = RegularGridInterpolator(self.fitMztab, self.dsigmadMMhtab)


        #also build an interpolator for sigma(R) of the R we integrate over (for CD and EoR). These R >> Rhalo typically, so need new table.
        self.sigmaofRtab = np.array([[ClassCosmo.sigma(RR,zz) for zz in self.zHMFtab] for RR in Cosmo_Parameters._Rtabsmoo])
        self.fitRztab = [np.log(Cosmo_Parameters._Rtabsmoo), self.zHMFtab]
        self.sigmaRintlog = RegularGridInterpolator(self.fitRztab, self.sigmaofRtab) #no need to log either

    def load_Pk_LCDM(self):
        
        self.Pk_CLASS_LCDM = scipy.interpolate.interp1d(self.klist,self.CLASS_Pk)
        #Pk_LCDM_file = base_path+"/../newLy-a_cdm_sim_model_matterpower.dat"
        #kcdm, P  = np.loadtxt(Pk_LCDM_file).T
        #tf = Transfer()
        #tf.transfer_model = 'EH' # default CAMB is bad at high k
        #kcdm, P = tf.k, tf.power
        
       # self.Pk_CLASS_LCDM  = scipy.interpolate.interp1d(kcdm, P)
        
        return

    def Pk_ETHOS(self, h_peak, k_peak, c=-20):
        #if h_peak < 0.2:
            #return lambda k : T_WDM_Sebastian(k,k_peak)**2. * self.Pk_CLASS_LCDM(k)*self.cosmo.h
        #elif 0.0 > h_peak > 0.2:
        #    return lambda k: T_WDM_First_Peak(k, k_peak, h_peak, mu)**2. * self.Pk_CLASS_LCDM(k)*self.cosmo.h 
        #else:
            return lambda k : T_ETHOS(k, k_peak, h_peak, c)**2. * self.Pk_CLASS_LCDM(k)*self.cosmo.h


    def sigma2_M(self, M, beta=3.6, c=3.6, vb=False):

	    """
	    Mass variance of smoothed density field, smoothed on scale R

	    Args:
	    	M: mass in units Msun

	    Returns:
	    	sigma^2(M)
	    """
	    if vb:
	    	print(f'sigma2: using beta={beta}, c={c}')

	    R = (3*M/(4*np.pi*self.rho_mean))**(1/3)
	    k_peak = self.k_peak
	    h_peak = self.h_peak

	    def sigma2_integrand(k, R):

	    	Pk = self.Pk_ETHOS(h_peak=h_peak, k_peak=k_peak)(k)
	    	#if vb:
	    	#    print(f'sigma2: ETHOS h_peak={h_peak}, k_peak={k_peak}')

	    	#if vb:
	    	#    print(f'sigma2: Using {self.window_function} window function')
	    	if h_peak == 0: #self.window_function == 'Smooth':
	    	    W  = W_Smooth(k, R, beta=50., c=2.8)
	    	else:
	    	    W = W_Smooth(k, R, beta, c)#print('Invalid window function')

	    	return k**2. * Pk * W**2.

	    kmin, kmax = np.min(self.klist)*1.1, np.max(self.klist)*0.999#1.1e-5,2000
	    if self.logk:
	    	k = np.logspace(np.log10(kmin), np.log10(kmax), num=self.N_k)
	    else:
	    	k = np.linspace(kmin, kmax, num=self.N_k)

	    integral = InterpolatedUnivariateSpline(k, sigma2_integrand(k, R)).integral(kmin, kmax)

	    return integral/2./np.pi**2.

    def dsigma2_dM(self, M, beta=3.6, c=3.6):
	    """
	    Derivative of mass variance of smoothed density field, smoothed on scale R
	    with respect to M

	    Args:
	    	M: mass in units Msun

	    Returns:
	    	dsigma^2(M)/dM
	    """
	    R = (3*M/(4*np.pi*self.rho_mean))**(1/3)
	    k_peak = self.k_peak
	    h_peak = self.h_peak

	    def dsigma2_dM_integrand(k, R):

	    	Pk = self.Pk_ETHOS(h_peak=h_peak, k_peak=k_peak)(k)

	    	if h_peak == 0.0: #self.window_function == 'Smooth':
	    	    W  = W_Smooth(k, R, beta=50., c=2.8)
	    	    dW_dR = dW_dR_Smooth(k, R, beta=50., c=2.8)
	    	else:
	    	    W  = W_Smooth(k, R, beta, c)
	    	    dW_dR = dW_dR_Smooth(k, R, beta, c)#print('Invalid window function')

	    	return k**2. * Pk * W * dW_dR

	    kmin, kmax = np.min(self.klist)*1.1, np.max(self.klist)*0.999
	    if self.logk:
	    	k = np.logspace(np.log10(kmin), np.log10(kmax), num=self.N_k)
	    else:
	    	k = np.linspace(kmin, kmax, num=self.N_k)

	    integral = InterpolatedUnivariateSpline(k, dsigma2_dM_integrand(k, R)).integral(kmin, kmax)

	    return 2*R/3./M * integral /2./np.pi**2.




    def growth_fac(self, z):
	    try:
	    	return self.growth_facs[z]
	    except KeyError:

	    	self.growth_facs[z] = self.growth_fac_func2(z)/self.growth_fac_func2(z=0.)
	    	return self.growth_facs[z]

    def growth_fac_func2(self, z):
	    a_z = 1./(1+z)

	    def E(a):
	    	return np.sqrt(self.cosmo.Om0/a**3. + (1-self.cosmo.Om0))

	    def integrand_g(x):
	    	return (x*E(x))**(-3.)

	    int_part = quad(integrand_g,0,a_z)[0]

	    return 5*self.cosmo.Om0/2 * E(a_z) * int_part



    def HMF_int(self, Mh, z):
        "Interpolator to find HMF(M,z), designed to take a single z but an array of Mh in Msun"
        _logMh = np.log(Mh)

        logMhvec = np.asarray([_logMh]) if np.isscalar(_logMh) else np.asarray(_logMh)
        inarray = np.array([[LM,z] for LM in logMhvec])

        return np.exp(self.logHMFint(inarray) )



    def sigma_int(self,Mh,z):
        "Interpolator to find sigma(M,z), designed to take a single z but an array of Mh in Msun"
        _logMh = np.log(Mh)
        logMhvec = np.asarray([_logMh]) if np.isscalar(_logMh) else np.asarray(_logMh)
        inarray = np.array([[LM,z] for LM in logMhvec])
        return self.sigmaintlog(inarray)

    def sigmaR_int(self,RR,z):
        "Interpolator to find sigma(RR,z), designed to take a single z but an array of RR in cMpc"
        _logRR = np.log(RR)
        logRRvec = np.asarray([_logRR]) if np.isscalar(_logRR) else np.asarray(_logRR)
        inarray = np.array([[LR,z] for LR in logRRvec])
        return self.sigmaRintlog(inarray)


    def dsigmadM_int(self,Mh,z):
        "Interpolator to find dsigma/dM(M,z), designed to take a single z but an array of Mh in Msun. Used in 21cmFAST mode"
        _logMh = np.log(Mh)
        logMhvec = np.asarray([_logMh]) if np.isscalar(_logMh) else np.asarray(_logMh)
        inarray = np.array([[LM,z] for LM in logMhvec])
        return self.dsigmadMintlog(inarray)


def growth(Cosmo_Parameters, z):
    "Scale-independent growth factor, interpolated from CLASS"
    zlist = np.asarray([z]) if np.isscalar(z) else np.asarray(z)
    if (Cosmo_Parameters.Flag_emulate_21cmfast==True):
        _offsetgrowthdicke21cmFAST = 1-0.000248*(zlist-5.) #as in HMF, to fix growth. have to do it independently since it depends on z.
        return Cosmo_Parameters.growthint(zlist) * _offsetgrowthdicke21cmFAST
    else:
        return Cosmo_Parameters.growthint(zlist)


def dgrowth_dz(CosmoParams, z):
    "Derivative of growth factor growth() w.r.t. z"
    zlist = np.asarray([z]) if np.isscalar(z) else np.asarray(z)
    dzlist = zlist*0.001
    return (growth(CosmoParams, z+dzlist)-growth(CosmoParams, z-dzlist))/(2.0*dzlist)


def redshift_of_chi(CosmoParams, z):
    "Returns z(chi) for any input comoving distance from today chi in Mpc"
    return CosmoParams.zfofRint(z)



def T021(Cosmo_Parameters, z):
    "Prefactor in mK to T21 that only depends on cosmological parameters and z. Eg Eq.(21) in 2110.13919"
    return 34 * pow((1+z)/16.,0.5) * (Cosmo_Parameters.omegab/0.022) * pow(Cosmo_Parameters.omegam/0.14,-0.5)


#UNUSED:
# def interp2Dlinear_only_y(arrayxy, arrayz, x, y):
#     "2D interpolator where the x axis is assumed to be an array identical to the trained x. That is, an array of 1D linear interpolators. arrayxy is [x,y]. arrayz is result. x is the x input (=arrayxy[0]),  and y the y input. Returns z result (array)"
#     if((x != arrayxy[0]).all()):
#         print('ERROR on interp2Dlinear_only_y, x need be the same in interp and input')
#         return -1
#     Ny = len(arrayxy[1])
#     ymin, ymax = arrayxy[1][[0,-1]]
#     if((y > ymax or y<ymin).all()):
#         print('EXTRAPOLATION on interp2Dlinear_only_y on y axis. max={} curr={}'.format(ymax, y))
#
#     ystep = (ymax-ymin)/(Ny-1.)
#     ysign = np.sign(ystep).astype(int)
#
#     iy = np.floor((y-ymin)/ystep).astype(int)
#     iy=np.fmin(iy,Ny-1-(ysign+1)//2) #if positive ysign stop 1 lower
#     iy=np.fmax(iy,0-(ysign-1)//2) #if negative ysign stop 1 higher
#
#     y1 = ymin + iy * ystep
#     y2 = y1 + ystep
#
#     fy1 = arrayz[:,iy]
#     fy2 = arrayz[:,iy+ysign]
#
#     fy = fy1 + (fy2-fy1)/ystep * (y - y1)
#     return fy
