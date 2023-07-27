import numpy as np
import pandas as pd
import warnings
import astropy.units as u
import astropy.constants as c
from astropy.coordinates import SkyCoord
from scipy.stats import norm
import os

from .planetABC import PlanetABC
from .planetModel import PlanetModel
from .retriever import Retriever

warnings.simplefilter(action='ignore',category=UserWarning)

class PlanetFromLit(PlanetABC):
    
    """
    TO DO: 
    - make sure default params have the right units
    - save system to a file of some kind
    - summarize results
    """
    
    def __init__(self,st_name, pl_letter, obs, configs, input_df=None,  
                 det_type=None,
                 plx=np.nan, plx_err=np.nan,
                 dist=np.nan, dist_err=np.nan, 
                 st_sptype=np.nan, 
                 st_Vmag=np.nan, st_Vmag_err=np.nan, 
                 st_mass=np.nan, st_mass_err=np.nan,
                 st_loc=np.nan,
                 per=np.nan, per_err=np.nan,
                 sma=np.nan, sma_err=np.nan,
                 inc=np.nan, inc_err=np.nan,
                 ecc=np.nan, ecc_err=np.nan,
                 longnode=np.nan, longnode_err=np.nan,
                 tperi=np.nan, 
                 argperi=np.nan, argperi_err= [36,36] * u.deg, 
                 pl_mass=np.nan, pl_mass_err=np.nan,
                 pl_msini=np.nan, pl_msini_err=np.nan,
                 pl_rad=np.nan, pl_rad_err=np.nan,
                 albedo_wavelens=np.nan,
                 albedo_vals=[np.nan], albedo_vals_err=[np.nan],
                 self_lum=np.nan,
                 rv_data=None,
                 ecc_prior_uniform=False,
                 param_list=None,
                 astrom_uncertainty_df=None
                ):
        
        PlanetABC.__init__(self, 
                                       st_name, pl_letter, 
                                       configs, input_df,
                                       det_type,
                                       plx, plx_err,
                                       dist, dist_err,       
                                       st_sptype, 
                                       st_Vmag, st_Vmag_err, 
                                       st_mass, st_mass_err,
                                       st_loc,
                                       per, per_err,
                                       sma, sma_err,
                                       inc, inc_err,
                                       ecc, ecc_err,
                                       longnode, longnode_err,
                                       tperi, 
                                       argperi, argperi_err, 
                                       pl_mass, pl_mass_err,
                                       pl_msini, pl_msini_err,
                                       pl_rad, pl_rad_err,
                                       albedo_wavelens,
                                       albedo_vals, albedo_vals_err,
                                       self_lum, rv_data,
                                       ecc_prior_uniform
                                      )
        self.obs = obs
        self.planet_list = []
        self.param_list = param_list
        self.detection_prob = None
        self.observation_epochs = None
        self.astrom_uncertainty_df = astrom_uncertainty_df
        self.ret = Retriever(self,astrom_uncertainty_df=self.astrom_uncertainty_df)
        
    ########################################################################################################################
        
    def add_planet(self,inc):
        """
        TO DO: 
        - Calculate period from Kepler's law for DI planets (as in exoscene Planet class)
        """
        planet = PlanetModel(self.st_name, self.pl_letter,  
                             self.obs,
                             self.configs, 
                             det_type=self.det_type,
                             plx=self.plx, plx_err=self.plx_err,
                             dist=self.dist, dist_err=self.dist_err, 
                             st_sptype=self.st_sptype, 
                             st_Vmag=self.st_Vmag, st_Vmag_err=self.st_Vmag_err, 
                             st_mass=self.st_mass, st_mass_err=self.st_mass_err,
                             st_loc=self.st_loc,
                             per=self.per, per_err=self.per_err,
                             sma=self.sma, sma_err=self.sma_err,
                             inc=inc, inc_err=self.inc_err, inc_prior=self.inc,
                             ecc=self.ecc, ecc_err=self.ecc_err,
                             longnode=self.longnode, longnode_err=self.longnode_err,
                             tperi=self.tperi, 
                             argperi=self.argperi, argperi_err=self.argperi_err, 
                             pl_mass=self.pl_mass, pl_mass_err=self.pl_mass_err,
                             pl_msini=self.pl_msini, pl_msini_err=self.pl_msini_err,
                             pl_rad=self.pl_rad, pl_rad_err=self.pl_rad_err,
                             albedo_wavelens=self.albedo_wavelens,
                             albedo_vals=self.albedo_vals, albedo_vals_err=self.albedo_vals_err,
                             self_lum=self.self_lum,
                             rv_data=self.rv_data,
                             ecc_prior_uniform=self.ecc_prior_uniform)
        
        planet.ret = self.ret
        
        self.planet_list.append(planet)
        
    ########################################################################################################################
        
    def build_planets(self,inc_list=None): 
        """
        TO DO: Add functionality to explore other parameters later
        """
        
        if self.param_list == None:
            if np.isnan(self.inc):
                raise ValueError("Inclination must be provided if param_list==None")
            if np.isnan(self.ecc):
                raise ValueError("Eccentricity must be provided if param_list==None")
            self.add_planet(inc=self.inc)
        
        elif self.param_list == ['inc']:
            
            if not np.isnan(self.inc):
                
                inc_err_temp = [18 if np.isnan(self.inc_err[i]) else self.inc_err[i].value for i in range(2)] * u.deg
                
                inc_list = [self.inc-2*inc_err_temp[1], self.inc-inc_err_temp[1], self.inc, self.inc+inc_err_temp[0], self.inc+2*inc_err_temp[0]]
                
            elif inc_list == None:
                
                inc_list = self.configs['inc_list'] * u.deg
                if np.any(inc_list > 90 * u.deg):
                    warnings.warn('target with no inclination priors has planet model(s) with inclination > 90 deg, detection probability calculation is not configured for this.')
            
              
            for inc in inc_list:
                if inc < 0 * u.deg:
                    inc += 180 * u.deg
                elif inc > 180 * u.deg:
                    inc -= 180 * u.deg
                    
                self.add_planet(inc=inc)
        
        else:
            raise ValueError("param_list only configured for ['inc'] or None.")       
            
    ########################################################################################################################
            
    def compute_detection_prob(self):
        
        if len(self.planet_list) < 1:
            
            raise ValueError("No planets in .planet_list attribute. Run .build_planets() method first.")
        
        epochs = np.array(self.planet_list[0].ephem_pm.index)
        
        if self.param_list is None: # Return 1 where detectable, 0 where undetectable
            
            if len(self.planet_list) > 1: # Assume all are equally likely
                pl_detectability_data = np.array([np.where(planet.ephem_detectable[0][1].detectable,1,0) for planet in self.planet_list])
                
                detect_prob = np.sum(pl_detectability_data,axis=0)
                
                detect_prob_df = pd.DataFrame.from_dict({"t_yr" : epochs,
                                                         "detect_prob" : detect_prob})

                detect_prob_df.set_index('t_yr',inplace=True)

                self.detection_prob = detect_prob_df
                
            else:
                ephem_df = self.planet_list[0].ephem_detectable[0][1]

                detect_prob = np.where(ephem_df['detectable'],1,0)

                detect_prob_df = pd.DataFrame.from_dict({"t_yr" : epochs,
                                                        "detect_prob" : detect_prob})

                detect_prob_df.set_index('t_yr',inplace=True)

                self.detection_prob = detect_prob_df
            
        elif self.param_list == ['inc']:
            """
            Compute detection probability for inclination spread of planets
            """
            
            pl_detectability_data = {np.round(planet.inc.value,1) : np.where(planet.ephem_detectable[0][1].detectable,1,0) for planet in self.planet_list}
            

            incs = np.array((list(pl_detectability_data.keys())))
            incs.sort()
            
            int_edges = np.array(((incs[1:]+incs[:-1])/2.)) 
            
            if np.isnan(self.inc): # For targets with unconstrained inclination
                
                if np.any(incs>90):
                    raise ValueError("Cannot currently determine probabilities for inclinations > 90 deg.")
            
                edges_rad = np.deg2rad([0.,*int_edges,90.])
                inc_probs = np.cos(edges_rad[:-1]) - np.cos(edges_rad[1:])            
                
            else: # For targets with constrained inclination, assuming 1 sigma uncertainty is reported
                
            
                edges = [0.,*int_edges,180.]
                
                inc_err = [18 if np.isnan(self.inc_err[i]) else self.inc_err[i].value for i in range(2)] 
                
                rv_upper = norm(loc=self.inc,scale=inc_err[0])
                rv_lower = norm(loc=self.inc,scale=inc_err[1])
                
                bins = np.transpose([edges[:-1],edges[1:]])
                
                inc_probs = []
                
                for inc_bin in bins:
                    
                    if np.all(inc_bin<= self.inc.value):
                        inc_probs.append(rv_lower.cdf(inc_bin[1]) - rv_lower.cdf(inc_bin[0]))
                        
                    elif np.all(inc_bin>= self.inc.value):
                        inc_probs.append(rv_upper.cdf(inc_bin[1]) - rv_upper.cdf(inc_bin[0]))
                        
                    else:
                        inc_probs.append((rv_lower.cdf(self.inc.value) - rv_lower.cdf(inc_bin[0])) + (rv_upper.cdf(inc_bin[1]) - rv_upper.cdf(self.inc.value)))
                
            inc_probs_sum = np.sum(inc_probs)
            
            # Normalize inclination probabilities
            if np.round(inc_probs_sum,3) != 1.000:
                inc_probs = np.array(inc_probs) / inc_probs_sum
                
            prob_dict = {np.round(x[0],1) : x[1] for x in np.transpose([incs,inc_probs])}
            
            detect_prob = np.sum([prob_dict[np.round(inc,1)] * pl_detectability_data[np.round(inc,1)] for inc in incs],axis=0)
            
            detect_prob_df = pd.DataFrame.from_dict({"t_yr" : epochs,
                                                     "detect_prob" : detect_prob})

            detect_prob_df.set_index('t_yr',inplace=True)

            self.detection_prob = detect_prob_df
        
        else:
            raise ValueError("planet.param_list is not one of (None or ['inc'])")
            
    ############################################################    
        
    def choose_observation_eps(self,prob_min=None):
        
        """
        TO DO: 
        - actually find lowest inc planet in planet_list rather than assuming it's first.
        - find out why HD 160691 c returns two of the same epochs rather than using the np.unique() bandaid
        """
               
        def best_eps_by_time():  ## Choose epochs based on 12 bins over time
            
            best_epochs = []
            
            def find_first_ep(bin_num): # Find the first good epoch in a given epoch bin
                
                min_ep= epoch_hist_edges[bin_num]  
                
                ephem_df_temp = ephem_df[ephem_df.t_yr>min_ep]
                
                return np.array(ephem_df_temp.t_yr)[0]
            
            ephem_df.reset_index(inplace=True)
                
            epoch_hist,epoch_hist_edges = np.histogram(epochs,bins=12,range=(epochs[0],epochs[-1]))

            for ep_bin in range(12):

                if epoch_hist[ep_bin] > 0: # If there are epochs in the bin, get the epoch range of that bin and find the first good epoch in that range
                    best_epochs.append(find_first_ep(ep_bin))     


            # If this returns less than the goal # of observations, it's because all good epochs
            # fall within few bins. In this case, just choose evenly spaced
            # epochs over the available ones

            n_eps = len(best_epochs)
            
            if n_eps < self.configs['goal_epochs']:
                
                skip = len(epochs) // (self.configs['goal_epochs'] - 1)

                indices = list(range(0,len(epochs),skip))
                
                if len(indices) < self.configs['goal_epochs']:
                    indices.append(len(epochs)-1)

                best_epochs = [epochs[ind] for ind in indices]

            elif n_eps > self.configs['goal_epochs']:
                
                skip = n_eps // (self.configs['goal_epochs'] - 1)

                indices = list(range(0,n_eps,skip))

                if len(indices) < self.configs['goal_epochs']:
                    indices.append(len(best_epochs)-1)

                best_epochs = [best_epochs[ind] for ind in indices]

            
            best_epochs.sort() # Sort the results
            
            return best_epochs

        def best_eps_by_pa():
            
            best_epochs = []
            
            def find_first_pa_ep(bin_num): # Find the first good epoch in a given epoch bin
                
                pa_low = pa_bin_edges[bin_num]
        
                pa_high = pa_bin_edges[bin_num+1]
        
                ephem_df_temp = ephem_df[(ephem_df_planet.pa_deg>pa_low) & (ephem_df_planet.pa_deg<pa_high)].copy()
                
                if len(ephem_df_temp) > 0:
                
                    return np.array(ephem_df_temp.index)[0]
                
                return None
            
            ephem_df_planet = self.planet_list[0].ephem_pm # Ephemeris info for lowest inclination planet
        
            pa_bin_edges = range(0,370,30) # Bin 12 regions of PA
            
            for pa_bin in range(12):

                # Find the first good epoch in each bin
                ep = find_first_pa_ep(pa_bin)
                
                if not ep is None:
                    
                    best_epochs.append(ep)
                    
            n_eps = len(best_epochs)
            
            if n_eps < self.configs['goal_epochs']:
                
                best_epochs = best_eps_by_time()
                
            elif n_eps > self.configs['goal_epochs']:
                
                skip = n_eps//(self.configs['goal_epochs'] - 1)

                indices = list(range(0,n_eps,skip))
                
                if len(indices) < self.configs['goal_epochs']:
                    indices.append(n_eps-1)

                best_epochs = [best_epochs[ind] for ind in indices]
            
            return best_epochs 
        
        def propagate_epochs():
            
            for planet in self.planet_list: 

                planet.observation_epochs = self.observation_epochs
        
        final_eps = []
        
        # Make sure detection probability has been calculated
        
        if self.detection_prob is None:
            
            raise ValueError("Target detection probability has not been computed. Run .compute_detection_prob() method first.")
            
        # Calculate maximum possible probability of observation to use as minimum
        
        prob_max = np.max(self.detection_prob.values)
        
        print(f'\tMax detection probability over primary mission: {np.round(prob_max,2)}')
        
        if prob_min is None and prob_max != 0.0:
            
            prob_min = prob_max
            
        if not prob_min is None:
                
            ephem_df = self.detection_prob.copy()

            assert self.configs['goal_epochs'] in [*range(1,13),'all'], "Epoch choosing mechanism is only configured for 1-12 epochs or all epochs!"

            if self.configs['goal_epochs'] == ['all']:

                final_eps = np.array(ephem_df.index)

            else:
                
                ephem_df = ephem_df[ephem_df.detect_prob >= prob_min].copy()

                epochs = np.array(ephem_df.index)


                if len(epochs) < 1:
                    
                    pass
                
                elif len(epochs) < self.configs['goal_epochs']:

                    final_eps = np.array(ephem_df.index)

                else:

                    obs_window = (epochs[-1] - epochs[0]) * u.year

                    if self.per > obs_window: # Bin over time for long period planets

                        final_eps = best_eps_by_time()

                    else: # Bin over position angle for short period planets

                        final_eps = best_eps_by_pa()

        self.observation_epochs = np.unique(final_eps)
        
        propagate_epochs()

        
