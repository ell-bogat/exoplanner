import numpy as np
import pandas as pd
import warnings
import os
import datetime
import csv

import astropy
import astropy.units as u
import astropy.constants as c

import exoscene
from exoscene.planet import Planet

from orbitize.system import radec2seppa, seppa2radec

from . import planetABC

mass_max = (0.075 * c.M_sun).to(u.M_jup) # Hydrogen fusion mass limit according to Auddy et. al. 2016 https://www.hindawi.com/journals/aa/2016/5743272/

class PlanetModel(planetABC.PlanetABC):
    """
    TO DO: 
    - make sure default params have the right units
    - add method to check for missing data
    - add method to calculate and save ephemeris
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
                 inc=np.nan, inc_err=np.nan, inc_prior=np.nan,
                 ecc=np.nan, ecc_err=np.nan,
                 longnode=np.nan, longnode_err=np.nan,
                 tperi=np.nan, 
                 argperi=np.nan, argperi_err=np.nan, 
                 pl_mass=np.nan, pl_mass_err=np.nan,
                 pl_msini=np.nan, pl_msini_err=np.nan,
                 pl_rad=np.nan, pl_rad_err=np.nan,
                 albedo_wavelens=np.nan,
                 albedo_vals=[np.nan], albedo_vals_err=[np.nan],
                 self_lum=np.nan,
                 rv_data=None,
                 ecc_prior_uniform=False
                ):
        
        # Initialize like PlanetABC
        
        planetABC.PlanetABC.__init__(self, st_name, pl_letter, 
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
                           self_lum,rv_data,
                           ecc_prior_uniform
                          )
        
        # Observatory info
        self.obs = obs
        
        # Copy of target priors for posterior info
        self.post = planetABC.PlanetABC(st_name, pl_letter, 
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
                                       inc_prior, inc_err,
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
        
        # Model specific output dir:
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H.%M.%S.%f")
        self.planet_dir = os.path.join(self.target_dir,"".join(["inc=",str(np.round(self.inc.value,2)),"_T=",self.timestamp]))
        if not os.path.exists(self.planet_dir):
            os.mkdir(self.planet_dir)
            
        # Fill in missing model values as needed
        
        # Planet mass
        if np.isnan(self.pl_mass):
            mass = self.pl_msini / np.sin(np.deg2rad(self.inc))
            if mass > mass_max:
                warnings.warn(f'{self.sysname} at inclination {self.inc} is a star ({mass} M_Jup). Skipping this inclination.')
                return None
            else:
                self.pl_mass = mass
                
        # Planet radius, based on Bashi et. al. 2016: Two empirical regimes of the planetary mass-radius relation
        if np.isnan(self.pl_rad):
                if self.pl_mass < 0.39 * u.M_jup:
                    self.pl_rad = 1.8128 * (self.pl_mass.value ** 0.55) * u.R_jup
                else:
                    self.pl_rad = 1.0902 * (self.pl_mass.value ** 0.01) * u.R_jup
        
        # Orbital period
        if np.isnan(self.per):
                self.per = np.sqrt(4 * np.pi**2 / (c.G * (self.st_mass + self.pl_mass)) * self.sma**3).to(u.year)  
        
        # Time of periastron
        if np.isnan(self.tperi):
                self.tperi = astropy.time.Time('2026-11-01T00:00:00',format='isot',scale='utc').decimalyear*u.year
                
        # Argument of periastron
        if np.isnan(self.argperi):
            self.argperi = 90 * u.deg 
        
        elif self.det_type == 'Radial Velocity':
            self.argperi += (180 * u.deg) 
            if self.argperi > 360 * u.deg:
                self.argperi -= 360 * u.deg
        
        if np.any(np.isnan(self.argperi_err)):
            self.argperi_err = [36,36] * u.deg
            
        if np.isnan(self.longnode):
            self.longnode = 320 * u.deg
        
        if np.any(np.isnan(self.longnode_err)):
            self.longnode_err = [90,90] * u.deg
        
        if np.isnan(self.albedo_wavelens):
            self.albedo_wavelens = [575] * u.nm
        
        if np.isnan(self.albedo_vals):
            if pl_mass > 0.1 * u.M_jup:
                self.albedo_vals = [0.5]
            else:
                self.albedo_vals = [0.3]
        
        if np.any(np.isnan(self.albedo_vals_err)):
            self.albedo_vals_err = [[0.1,0.1]]

        if np.any(np.isnan(self.plx_err)):
            self.plx_err = 0.1 * np.array([self.plx.value,self.plx.value]) * u.mas

        if np.any(np.isnan(self.dist_err)):
            self.dist_err = 0.1 * np.array([self.dist.value,self.dist.value]) * u.pc

        if np.any(np.isnan(self.st_Vmag_err)):
            self.st_Vmag_err = 0.1 * np.array([self.st_Vmag,self.st_Vmag])

        if np.any(np.isnan(self.st_mass_err)):
            self.st_mass_err = 0.1 * np.array([self.st_mass.value,self.st_mass.value]) * c.M_sun

        if np.any(np.isnan(self.per_err)):
            self.per_err = 0.1 * np.array([self.per.value,self.per.value]) * u.year

        if np.any(np.isnan(self.sma_err)):
            self.sma_err = 0.1 * np.array([self.sma.value,self.sma.value]) * u.AU

        if np.any(np.isnan(self.inc_err)):
            self.inc_err = [18,18] * u.deg

        if np.any(np.isnan(self.ecc_err)):
            self.ecc_err = [0.1,0.1]

        if np.any(np.isnan(self.pl_mass_err)):
            self.pl_mass_err = 0.1 * np.array([self.pl_mass.value,self.pl_mass.value]) * u.M_jup

        if np.any(np.isnan(self.pl_msini_err)):
            self.pl_msini_err = 0.1 * np.array([self.pl_msini.value,self.pl_msini.value]) * u.M_jup

        if np.any(np.isnan(self.pl_rad_err)):
            self.pl_rad_err = 0.1 * np.array([self.pl_rad.value,self.pl_rad.value]) * u.R_jup
            
        # Attributes to hold data later
        
        self.model = None # For the exoscene planet object
        self.ephem_1per = None
        self.ephem_pm = None
        self.ephem_detectable = None
        self.observation_epochs = None
        self.observation_hdul = None
        self.int_times = []
        
    ################################################################################
        
    def check_missing_data(self):
        pass
        
    ################################################################################
        
    def calc_ephem(self,
                   t_series=None,
                   t0= None,
                   t_span=1*u.year,
                   t_step=5*u.day
                  ):
        """
        TO DO: write to HDF5, fname f"{self.sysname}_ephem_from_tseries.hdf5"
        """
        
        # Create exoscene planet object
        if self.det_type == 'Radial Velocity':
            a_in = None
            P_in = self.per
        else:
            a_in = self.sma
            P_in = None
       
        self.model = Planet(self.sysname, 
                             dist = self.dist, 
                             a = a_in, 
                             P = P_in,
                             ecc = self.ecc, 
                             inc = self.inc, 
                             longnode = self.longnode,
                             argperi = self.argperi, 
                             tperi = self.tperi, 
                             mplan = self.pl_mass,
                             radius = self.pl_rad,
                             albedo_wavelens = self.albedo_wavelens, 
                             albedo_vals = self.albedo_vals)

        if t0 is None:
            t0 = self.obs.t0_pm

        if t_series != None:
            tseries, delx, dely, beta, phasefunc, orad = self.model.compute_ephem(tarray = t_series)
            fluxratio = phasefunc * self.model.albedo_vals[0] * (self.model.radius.to(u.AU) / orad)**2

            table_fname = os.path.join(self.planet_dir, f"{self.sysname}_ephem_from_tseries.csv")
            table_fname_temp = os.path.join(self.planet_dir,"temp.csv")

            exoscene.planet.write_ephem_table(self.model, tarray = t_series, table_fname = table_fname_temp)    

            with open(table_fname_temp, 'r') as f:
                reader = csv.reader(f, delimiter=',')
                headers = next(reader)
                data = list(reader)    
            
            headers = ['t_yr','deltax_mas','deltay_mas','phase','r_AU','fluxratio_575']
            
            ephem_table = pd.DataFrame(data,columns=headers)
            ephem_table = ephem_table.astype({'t_yr': 'float64',
                                                'deltax_mas': 'float64',
                                                'deltay_mas': 'float64',
                                                'phase': 'float64',
                                                'r_AU': 'float64',
                                                'fluxratio_575': 'float64',
                                              },
                                              errors='raise'
                                             )
            ephem_table.set_index('t_yr',inplace=True)
            ephem_table['ra_mas'] = -ephem_table['deltax_mas']
            ephem_table['dec_mas'] = ephem_table['deltay_mas']
            ephem_table['fluxratio_575'] = ephem_table['fluxratio_575'] + self.self_lum
            seppa = radec2seppa(np.array(ephem_table['ra_mas']),np.array(ephem_table['dec_mas']))
            ephem_table['sep_mas'] = seppa[0]
            ephem_table['pa_deg'] = seppa[1]
            ephem_table['n_day'] = ephem_table.index%1 * 365
            ephem_table.to_csv(table_fname,index=True)

        else:
            # High cadence ephemeris
            #global tstep_high
            
            table_fname = os.path.join(self.planet_dir, f'{self.sysname}_ephem_{np.round(t_span.value,2)}yrs.csv')
            table_fname_temp = os.path.join(self.planet_dir,"temp.csv")
            
            exoscene.planet.write_ephem_table(self.model, tbeg = t0, tend = t0 + t_span,
                                              tstep = t_step, table_fname = table_fname_temp)

            with open(table_fname_temp, 'r') as f:
                reader = csv.reader(f, delimiter=',')
                headers = next(reader)
                data = list(reader)    
            
            headers = ['t_yr','deltax_mas','deltay_mas','phase','r_AU','fluxratio_575']
            
            ephem_table = pd.DataFrame(data,columns=headers)
            ephem_table = ephem_table.astype({'t_yr': 'float64',
                                                'deltax_mas': 'float64',
                                                'deltay_mas': 'float64',
                                                'phase': 'float64',
                                                'r_AU': 'float64',
                                                'fluxratio_575': 'float64',
                                              },
                                              errors='raise'
                                             )
            ephem_table.set_index('t_yr',inplace=True)
            ephem_table['ra_mas'] = -ephem_table['deltax_mas']
            ephem_table['dec_mas'] = ephem_table['deltay_mas']
            ephem_table['fluxratio_575'] = ephem_table['fluxratio_575'] + self.self_lum
            seppa = radec2seppa(np.array(ephem_table['ra_mas']),np.array(ephem_table['dec_mas']))
            ephem_table['sep_mas'] = seppa[0]
            ephem_table['pa_deg'] = seppa[1]
            ephem_table['n_day'] = ephem_table.index%1 * 365
            
            if np.round(t_span,2) == np.round(self.per,2):
                self.ephem_1per = ephem_table
            
            elif np.round(t_span,2) == np.round(self.obs.t_span_pm,2):
                self.ephem_pm = ephem_table
                    
            ephem_table.to_csv(table_fname,index=True)            
            
            
    ################################################################################
        
    def get_detectable_epochs(self,
                              avoid_GB_obs=True,
                              avoid_sun=True,
                             ):
        
        def sep_rad(lambda1, phi1, lambda2, phi2):
            """
            Compute the angular separation from (lambda1, phi1)
            in radians to (lambda2, phi2) in radians.
            lambda1, phi1 : arrays or scalars
            lambda2, phi2 : scalars
            """
            def haversine(theta):
                return np.sin(theta / 2.0) ** 2

            hav_d_over_r = haversine(phi2 - phi1) + \
                np.cos(phi1) * np.cos(phi2) * haversine(lambda2 - lambda1)
            central_angle_rad = 2 * np.arcsin(np.sqrt(hav_d_over_r))

            return central_angle_rad
        
        
        if self.ephem_detectable is None:
            self.ephem_detectable = []
        
        # Load ephemeris table
        ephem_df = self.ephem_pm.copy()
        ephem_df['fluxratio_575'] = ephem_df['fluxratio_575'].astype('float64')
        ephem_df['t_yr_copy'] = np.array(ephem_df.index)
        ephem_df.sort_index(inplace=True)
        
        #print(f"\tTotal epochs: {len(ephem_df)}")
        
        
        ephem_df['detectable_by_obs'] = np.logical_and(np.greater(ephem_df['fluxratio_575'],self.obs.contrast_min),
                                                       np.logical_and(np.greater(ephem_df['sep_mas'],self.obs.iwa.value),
                                                                      np.less(ephem_df['sep_mas'],self.obs.owa.value)))
        
        ephem_df['detectable'] = ephem_df['detectable_by_obs'] 
        
        #print(f"\tDetectable epochs: {np.count_nonzero(ephem_df['detectable'])}")
        
        if avoid_sun:
            
            sun_avoidance_r_rad = self.obs.sun_avoidance_r.to(u.radian)
            
            # Calculate longitude of sun 
            sun_skycoord = astropy.coordinates.get_sun(astropy.time.Time(ephem_df['t_yr_copy'], format='decimalyear', scale='utc'))

            ephem_df['sun_lon_deg'] = sun_skycoord.geocentrictrueecliptic.lon
            ephem_df['sun_lat_deg'] = sun_skycoord.geocentrictrueecliptic.lat

            ephem_df['st_lon_deg'] = self.st_loc_ec.lon
            ephem_df['st_lat_deg'] = self.st_loc_ec.lat
    
            ephem_df['sun_lon_rad'], ephem_df['sun_lat_rad'], ephem_df['st_lon_rad'], ephem_df['st_lat_rad'] = np.deg2rad([ephem_df['sun_lon_deg'],ephem_df['sun_lat_deg'],ephem_df['st_lon_deg'],ephem_df['st_lat_deg']])
        
            ephem_df['sun_sep_rad'] = sep_rad(ephem_df['sun_lon_rad'],ephem_df['sun_lat_rad'],ephem_df['st_lon_rad'],ephem_df['st_lat_rad']) * u.radian
            
            ephem_df['antisun_sep_rad'] = np.pi - ephem_df['sun_sep_rad']
            
            ephem_df['outside_sun_cones'] = np.logical_and(np.greater(ephem_df['sun_sep_rad'], sun_avoidance_r_rad), np.greater(ephem_df['antisun_sep_rad'], sun_avoidance_r_rad))
            
            ephem_df['detectable'] = np.logical_and(ephem_df['detectable'],ephem_df['outside_sun_cones'])
        
        if avoid_GB_obs:
            
            ephem_df['n_day'] = ephem_df.index % 1 * 365
            
            outside_windows = np.full(ephem_df['n_day'].shape,True)
            
            for window in self.obs.galactic_obs_windows:
                outside_window = np.logical_not((np.logical_and(np.greater(ephem_df['n_day'],window[0]), np.less(ephem_df['n_day'],window[1]))))
                outside_windows = np.logical_and(outside_window,outside_windows)
                
            
            ephem_df['outside_galactic_obs_window'] = outside_windows
            
            ephem_df['detectable'] = np.logical_and(ephem_df['detectable'],ephem_df['outside_galactic_obs_window'])
            
        detectable_df = pd.DataFrame.from_dict({"t_yr" : ephem_df.index,
                                                "detectable" : ephem_df['detectable']})
        
        detectable_df.set_index('t_yr',inplace=True)
            
        self.ephem_detectable.append(
            [{'avoid_sun' : avoid_sun,
              'avoid_GB_obs' : avoid_GB_obs},
             detectable_df])
        
    ##########################################################################################################################

    def simulate_observations(self,showplots=None):
        
        if showplots is None:
            showplots = self.configs['showplots']
        
        self.obs.simulate_observations(self,showplots=showplots)
        
    ##########################################################################################################################
    
    def retrieve_astrom(self,showplots=None):
        
        if showplots is None:
            showplots = self.configs['showplots']
        
        self.ret.retrieve_astrom(self,showplots=showplots)
        
    def retrieve_orbit(self):
        
        self.ret.retrieve_orbit(self)