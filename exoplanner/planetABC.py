import numpy as np
import pandas as pd
import astropy
import astropy.units as u
import astropy.constants as c
import os

class PlanetABC:
    """
    TO DO: 
    - make sure default params have the right units
    - Use None as default values and convert nans in input df?
    - Add error to periastron time? (would need to propagate through child classes)
    """
    # Class attributes go here:
    
    # Init method:
    def __init__(self, st_name, pl_letter, configs, input_df=None,  
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
                 argperi=np.nan, argperi_err=np.nan, 
                 pl_mass=np.nan, pl_mass_err=np.nan,
                 pl_msini=np.nan, pl_msini_err=np.nan,
                 pl_rad=np.nan, pl_rad_err=np.nan,
                 albedo_wavelens=np.nan,
                 albedo_vals=[np.nan], albedo_vals_err=[np.nan],
                 self_lum=np.nan,
                 rv_data=None,
                 ecc_prior_uniform=None
                ):
        
        # Names
        self.sysname = f"{st_name} {pl_letter}"
        self.st_name = st_name
        self.pl_letter = pl_letter
        
        # Directory setup
        self.results_dir = os.path.join(*configs['results_dir'])
        if not os.path.exists(self.results_dir):
            os.mkdir(self.results_dir)
        
        if not os.path.exists(os.path.join(self.results_dir,'targets')):
            os.mkdir(os.path.join(self.results_dir,'targets'))
        
        self.target_dir = os.path.join(self.results_dir,'targets',self.sysname)
        if not os.path.exists(self.target_dir):
            os.mkdir(self.target_dir)
        
        # Configs
        self.configs = configs
        
        # Info that's not in the input database
        self.argperi_err = argperi_err
        self.longnode = longnode
        self.longnode_err = longnode_err
        self.albedo_wavelens = albedo_wavelens
        self.albedo_vals = albedo_vals
        self.albedo_vals_err = albedo_vals_err
        self.self_lum = self_lum
        self.rv_data = rv_data
        self.ecc_prior_uniform = ecc_prior_uniform
        
        if self.ecc_prior_uniform is None:
            self.ecc_prior_uniform = True if self.sysname in self.configs['ecc_prior_uniform'] else False
        
        if not input_df is None:
            row = input_df.loc[(st_name,pl_letter),:]
            
            self.det_type = row['discoverymethod'] # 'Radial Velocity' or 'Direct Imaging'
            
            self.st_sptype = row['SP_TYPE_EXOSCENE']
            
            if not np.isnan(row['FLUX_V']):
                self.st_Vmag = row['FLUX_V'] 
                self.st_Vmag_err = [row['FLUX_ERROR_V'], np.absolute(row['FLUX_ERROR_V'])]
            else:
                self.st_Vmag = row['sy_vmag']
                self.st_Vmag_err = [row['sy_vmagerr1'], np.absolute(row['sy_vmagerr2'])]

            if not np.isnan(row['PLX_VALUE']):
                self.plx = row['PLX_VALUE'] * u.mas
                self.plx_err = [row['PLX_ERROR'], np.absolute(row['PLX_ERROR'])] * u.mas
            else:
                self.plx = row['sy_plx']  * u.mas
                self.plx_err = [row['sy_plxerr1'], np.absolute(row['sy_plxerr2'])] * u.mas

            # TO DO: Make system distance dependent on whether parallax is provided
            self.dist = row['sy_dist'] * u.pc
            self.dist_err = [row['sy_disterr1'], np.absolute(row['sy_disterr2'])] * u.pc

            self.st_mass = row['st_mass'] * c.M_sun
            self.st_mass_err = [row['st_masserr1'], np.absolute(row['st_masserr2'])] * c.M_sun
            self.st_loc = [row['RA'],row['DEC']] # RA: hms, Dec: dms
            
            self.sma = row['pl_orbsmax'] * u.au
            self.sma_err = [row['pl_orbsmaxerr1'], np.absolute(row['pl_orbsmaxerr2'])] * u.au
            self.per = (row['pl_orbper'] * u.day).to(u.year)
            self.per_err = ([row['pl_orbpererr1'], np.absolute(row['pl_orbpererr2'])] * u.day).to(u.year)
            self.ecc = row['pl_orbeccen']
            self.ecc_err = [row['pl_orbeccenerr1'], np.absolute(row['pl_orbeccenerr2'])]
            self.inc = row['pl_orbincl'] * u.deg
            self.inc_err = [row['pl_orbinclerr1'], np.absolute(row['pl_orbinclerr2'])] * u.deg
            self.argperi = row['pl_orblper'] * u.deg
            self.tperi = astropy.time.Time(row['pl_orbtper'] ,format='jd',scale='utc').decimalyear*u.year if not np.isnan(row['pl_orbtper']) else np.nan
            self.pl_mass = row['pl_massj'] * u.M_jup
            self.pl_mass_err = [row['pl_massjerr1'], np.absolute(row['pl_massjerr2'])] * u.M_jup
            self.pl_msini = row['pl_msinij'] * u.M_jup
            self.pl_msini_err = [row['pl_msinijerr1'], np.absolute(row['pl_msinijerr2'])] * u.M_jup
            self.pl_rad = row['pl_radj'] * u.R_jup
            self.pl_rad_err = [row['pl_radjerr1'], np.absolute(row['pl_radjerr2'])] * u.R_jup
        
        else:
            self.det_type = det_type
            
            self.st_sptype = st_sptype
            self.st_Vmag = st_Vmag
            self.st_Vmag_err = st_Vmag_err
            self.plx = plx
            self.plx_err = plx_err
            self.dist = dist
            self.dist_err = dist_err
            self.st_mass = st_mass
            self.st_mass_err = st_mass_err
            self.st_loc = st_loc
            
            self.sma = sma
            self.sma_err = sma_err
            self.per = per
            self.per_err = per_err
            self.ecc = ecc
            self.ecc_err = ecc_err
            self.inc = inc
            self.inc_err = inc_err
            self.argperi = argperi
            self.tperi = tperi
            self.pl_mass = pl_mass
            self.pl_mass_err = pl_mass_err
            self.pl_msini = pl_msini
            self.pl_msini_err = pl_msini_err
            self.pl_rad = pl_rad
            self.pl_rad_err = pl_rad_err
          
        # Pull self-luminosity data from config file
        if self.self_lum == 0.0 and (self.sysname in self.configs['self_luminosity']):
            self.self_lum = self.configs['self_luminosity'][self.sysname]
        else:
            self.self_lum = 0
            
        # Pull RV data from config file
        if self.configs['fit_rv']:
            if self.rv_data is None:
                rv_input_fname = '_'.join(self.st_name.split(' ')) + '_rv.csv'
                rv_path = os.path.join('RV',rv_input_fname)
                if st_name in self.configs['rv_dict']:
                    if os.path.exists(rv_path):
                        self.rv_path = rv_path
                        self.rv_data = self.configs['rv_dict'][self.st_name]
                    else:
                        raise OSError(f"{self.st_name} is listed in the RV dictionary but directory {rv_path} could not be found.")
                    
        # Calculate ecliptic star coordinates from ICRS coords:
        self.st_loc_ec = astropy.coordinates.SkyCoord(' '.join(self.st_loc), unit=(u.hourangle, u.deg)).transform_to('geocentrictrueecliptic')
        
                    
        if np.isnan(self.per) and np.isnan(self.sma):
            raise ValueError("Either period or sma must be defined for the planet.")
            
        if np.isnan(self.pl_mass) and np.isnan(self.pl_msini):
            raise ValueError("Either mass or msini must be defined for the planet.")