import numpy as np
import astropy.units as u
import pkg_resources
import os
import exoscene
import exoscene.image
import exoscene.star
import scipy
import astropy.io.fits as fits
import matplotlib.pyplot as plt

class Observatory:
    """
    Describes the observatory & mission parameters/restrictions
    """
    # Class attributes go here:
    
    # Init method:
    def __init__(self, name='Observatory',
                 configs=None,
                 iwa=150*u.mas, owa=400*u.mas, 
                 contrast_min=1e-9,
                 sun_avoidance_r = 56 * u.deg,
                 p_noise=True,
                 snr_goal=15,
                 n_epochs_goal=3,
                 t_span_pm=5*u.year, # Duration of primary mission in years
                 t0_pm=2026*u.year,
                 galactic_obs_windows = []
                ):
        
        self.configs = configs
        
        if not self.configs is None:
            
            if self.configs['iwa_mas'] is None:
                self.iwa = (self.configs['iwa_lod'] * self.configs['wl'] * 1e-09 / self.configs['d'] * u.radian).to(u.mas)
            else:
                self.iwa = self.configs['iwa_mas'] * u.mas
            
            if self.configs['owa_mas'] is None:
                self.owa = (self.configs['owa_lod'] * self.configs['wl'] * 1e-09 / self.configs['d'] * u.radian).to(u.mas)
            else:
                self.owa = self.configs['owa_mas'] * u.mas
            
            self.name = self.configs['obs_name']
            self.contrast_min = float(self.configs['contrast_floor'])
            self.p_noise = self.configs['p_noise']
            self.snr_goal = self.configs['snr']
            self.galactic_obs_windows = self.configs['galactic_obs_windows']
            self.sun_avoidance_r = self.configs['sun_avoidance_r'] * u.deg
            self.n_epochs_goal = self.configs['goal_epochs']
            self.t_span_pm = self.configs['t_span_pm'] * u.year
            self.t0_pm = self.configs['t0_pm'] * u.year
            self.data_pixscale_as = (self.configs['pixscale_mas'] * u.milliarcsecond).to(u.arcsec) # Pixel scale of CGI science camera

        
            
        else:
        
            self.name = name
            self.iwa = iwa.to(u.mas)
            self.owa = owa.to(u.mas)
            self.contrast_min = contrast_min
            self.sun_avoidance_r = sun_avoidance_r.to(u.deg)
            self.p_noise = p_noise
            self.snr_goal = snr_goal
            self.galactic_obs_windows = galactic_obs_windows
            self.n_epochs_goal = n_epochs_goal
            self.t_span_pm = t_span_pm.to(u.year)
            self.t0_pm = t0_pm.to(u.year)
            
    def test_psf(self,target):
    
        ## Load CGI PSF model
        hlc_psf_path =  pkg_resources.resource_filename('exoscene', 'data/cgi_hlc_psf')
        
        psf_cube_fname = os.path.join(hlc_psf_path, 'os9_psfs.fits')
        
        psf_r_fname = os.path.join(hlc_psf_path, 'os9_psfs_r_offset.fits')
        
        psf_angle_fname = os.path.join(hlc_psf_path, 'os9_psfs_angle_offset.fits')

        psf_cube = fits.getdata(psf_cube_fname)
        
        psf_hdr = fits.getheader(psf_cube_fname)
        
        #print(f"\tShape of PSF model: {psf_cube.shape}")

        self.hires_pixscale_as = psf_hdr['PIX_AS'] * u.arcsec
        
        hires_pixscale_LoD = psf_hdr['PIXSCALE']

        data_scale_fac = self.hires_pixscale_as.value / self.data_pixscale_as.value
        
        self.data_pixscale_LoD = hires_pixscale_LoD / data_scale_fac
        
        #print("\tCCD pixel to model pixel scale factor = {:.3f}".format(data_scale_fac))

        r_offsets_LoD = fits.getdata(psf_r_fname)[0, :]
        
        self.r_offsets_as = r_offsets_LoD * self.hires_pixscale_as / hires_pixscale_LoD
        
        self.angles = fits.getdata(psf_angle_fname)[:, 0]

        Np_psf_hires = psf_cube.shape[-1]

        # Show radial offsets and position angles present in the PSF model cube
        
        #print("\n\tPSF model radial offsets (lam/D):\n\t\t{:}".format(r_offsets_LoD))
        #print("\tPSF model position angles (deg):\n\t\t{:}".format(self.angles))

        # Apply half-pixel shift to place center at 99.5, 99.5
        
        self.offset_psfs = scipy.ndimage.interpolation.shift(psf_cube, (0, 0, -0.5, -0.5),
                                                        order = 1, prefilter=False,
                                                        mode = 'constant', cval=0)
        Np = self.offset_psfs.shape[-1]
        
        self.cx = Np // 2 - 0.5 # New array center in zero-based indices

        #print("\tDimensions of PSF model cube: {:}".format(self.offset_psfs.shape))
        #print("\tNew array center along both axes: {:.1f}".format(self.cx))    

        ## Set detector downsample parameters and test an example

        self.npad = 8 # pad before and after array edge before binning

        self.max_detect_width = 1.1 * u.arcsec
        test_hires_psf = exoscene.image.get_hires_psf_at_xy_os9(
                self.offset_psfs,
                self.r_offsets_as.value, self.angles,
                self.hires_pixscale_as.value,
                delx_as = (-0.1 * u.arcsec).value,
                dely_as = (0.2 * u.arcsec).value,
                cx = self.cx)

        # Get scene array dimensions and data mask
        padded_test_hires_psf = np.pad(test_hires_psf,
                                   ((self.npad, self.npad), (self.npad,self.npad)),
                                   mode='constant')

        cx_padded = padded_test_hires_psf.shape[0] // 2 - 0.5

        self.xcoord_psf = ((np.arange(padded_test_hires_psf.shape[0]) - cx_padded)
                      * self.hires_pixscale_as)
        
        self.ycoord_psf = self.xcoord_psf.copy()

        binned_test_psf, det_xcoord, det_ycoord = exoscene.image.resample_image_array(
                padded_test_hires_psf, self.hires_pixscale_as,
                img_xcoord = self.xcoord_psf, img_ycoord = self.xcoord_psf,
                det_pixscale = self.data_pixscale_as,
                det_width = self.max_detect_width,
                binfac = 10, conserve = 'sum')

        #print(f"\t{padded_test_hires_psf.shape}")
        #print(f"\t{binned_test_psf.shape}")
        
        self.imwidth = binned_test_psf.shape[-1]
        
        #print(f"\t{np.max(test_hires_psf), np.max(binned_test_psf),np.max(binned_test_psf) / np.max(test_hires_psf)}")

        # Check conservation of flux
        #print(f"\t{np.sum(test_hires_psf), np.sum(binned_test_psf)}")
        np.testing.assert_allclose(
                np.sum(test_hires_psf),
                np.sum(binned_test_psf), rtol=1e-4)
        
        # Coordinate center
        np.testing.assert_almost_equal(0, det_xcoord[det_xcoord.shape[0]//2].value)

        '''
        # Plot the high-res & binned test
        plt.figure(figsize=(14,5))
        plt.subplot(121)
        plt.imshow(test_hires_psf, origin='lower')
        plt.colorbar()
        plt.subplot(122)
        plt.imshow(binned_test_psf, origin='lower')
        plt.colorbar()   
        plt.close()
        '''

        ## Estimate unocculted star count rate in peak pixel of PSF
        (peak_ang_ind, peak_sep_ind, _, _) = np.unravel_index(
            np.argmax(self.offset_psfs), self.offset_psfs.shape)
        
        #print("\n\tPeak angle index = {:d}, peak separation index = {:d}".format(peak_ang_ind, peak_sep_ind))

        padded_peak_psf = np.pad(self.offset_psfs[peak_ang_ind, peak_sep_ind],
                                 ((self.npad, self.npad), (self.npad,self.npad)),
                                 mode='constant')

        binned_peak_psf, _, _ = exoscene.image.resample_image_array(
                padded_peak_psf, self.hires_pixscale_as,
                img_xcoord = self.xcoord_psf, img_ycoord = self.xcoord_psf,
                det_pixscale = self.data_pixscale_as,
                det_width = self.max_detect_width,
                binfac = 10, conserve = 'sum')

        ## Compute star PSF peak countrate based on collecting area and throughput
        minlam = (psf_hdr['minlam'] * u.micron).to(u.nanometer)
        maxlam = (psf_hdr['maxlam'] * u.micron).to(u.nanometer)
        
        #print(f"Min lambda: {minlam}")
        #print(f"Max lambda: {maxlam}")

        self.star_flux = exoscene.star.bpgs_spectype_to_photonrate(spectype = target.planet_list[0].st_sptype,
                                                              Vmag = target.planet_list[0].st_Vmag,
                                                              minlam = minlam.value,
                                                              maxlam = maxlam.value)

        
        #print("\n\tBand-integrated irradiance of host star: {:.3E}".format(self.star_flux))

        self.A_eff = 3.5786 * u.m**2
        
        self.non_coron_optical_losses = 0.404 # Band 1 CBE at end-of-life

        #print(self.star_flux,binned_peak_psf,self.A_eff,self.non_coron_optical_losses)
        
        unocc_star_countrate_img = self.star_flux * binned_peak_psf * self.A_eff * self.non_coron_optical_losses
        
        self.unocc_star_countrate_peakpix = np.max(unocc_star_countrate_img)
        
        #print('\n\tStellar PSF peak count rate = {:.2E}'.format(self.unocc_star_countrate_peakpix))
            
        """
        cw = 19
        fig = plt.figure(figsize=(8,6))
        plt.imshow(unocc_star_countrate_img.value)
        plt.axis('off')
        plt.colorbar()
        plt.show()
        plt.close()
        """

        ## Define an approximation to the HLC field stop to mask out light from large angular separations
        fieldstop_rad = 9.0 # CGI HLC field stop radius in lam/D
        xs_p = np.arange(self.imwidth) - self.imwidth // 2
        ys_p = xs_p.copy()
        xxs_p, yys_p = np.meshgrid(xs_p, ys_p)
        rrs_p = np.sqrt(xxs_p**2 + yys_p**2)
        datamask_nan_ind = np.nonzero(rrs_p >= fieldstop_rad / self.data_pixscale_LoD)
        
        self.datamask_nan = np.where(~(rrs_p >= fieldstop_rad / self.data_pixscale_LoD), 1, np.nan)
        
        """
        plt.figure()
        plt.imshow(self.datamask_nan)
        plt.colorbar()
        plt.show()
        plt.close()
        """
        
    def simulate_observations(self,planet,showplots=None):
        """
        TO DO:
        - Return noiseless scenes in units of photons, not photons/sec
        - Add info to headers: configs, integration time, etc.
        """
        
        if showplots is None:
            
            showplots = planet.configs['showplots']
        
        def make_time_series_images(t_series,p_noise=self.p_noise,int_ts=[1 * u.hour]):

            if len(int_ts) == 1:
                int_ts = [int_ts[0]] * len(t_series)

            assert len(t_series) == len(int_ts)
            Nt = len(t_series)

            ## Loop through time steps and create the series cube

            # Init noiseless & noisy series cubes
            planet_scene_series = np.zeros((Nt, self.imwidth, self.imwidth)) * u.photon / u.second
            noisy_scene_series = np.zeros((Nt, self.imwidth, self.imwidth)) * u.photon
    
            for tt, time in enumerate(t_series):
                t_ephem = np.argmin(np.abs(ephem_tseries.value - time))
                
                if not planet.ephem_detectable[0][1].iloc[t_ephem][0]:
                    
                    continue
                
                deltax_as = ephem_df['ra_mas'].iloc[t_ephem] / 1000
                deltay_as = ephem_df['dec_mas'].iloc[t_ephem] / 1000
                flux_ratio = ephem_df['fluxratio_575'].iloc[t_ephem]

                # Get noiseless planet PSF at epochs, units: flux ratio
                planet_psf = exoscene.image.get_hires_psf_at_xy_os9(
                         self.offset_psfs, self.r_offsets_as.value, self.angles,
                         self.hires_pixscale_as.value, deltax_as, deltay_as, self.cx)

                padded_planet_psf = np.pad(planet_psf, ((self.npad, self.npad), (self.npad,self.npad)), mode='constant')

                # Get noiseless planet PSF at epochs, at detector resolution, units: flux ratio
                binned_planet_psf, _, _ = exoscene.image.resample_image_array(
                        img = padded_planet_psf, 
                        img_pixscale = self.hires_pixscale_as, 
                        img_xcoord = self.xcoord_psf, 
                        img_ycoord = self.xcoord_psf,
                        det_pixscale = self.data_pixscale_as,
                        det_width = self.max_detect_width,
                        binfac = 10, conserve = 'sum')
                
                noiseless_scene = (binned_planet_psf * self.star_flux * flux_ratio
                                            * self.A_eff * self.non_coron_optical_losses)

                planet_scene_series[tt] += noiseless_scene

                # Add poisson noise 
                if p_noise:
                    #read_noise_electrons = 100 * u.electron
                    #print("Noiseless_scene:")
                    #ICD.display(noiseless_scene)

                    flatnoise = np.random.normal(loc=1,scale=0.05,size=noiseless_scene.shape)

                    flatnoise_scene = noiseless_scene * flatnoise

                    noisy_scene,_ = exoscene.image.get_detector_exposure(
                        countrate_map = noiseless_scene/2, # /2 added to simulate photon - electron efficiency
                        total_inttime = int_ts[tt], 
                        read_inttime = int_ts[tt],
                        dark_cur = 0 * u.photon/u.second, 
                        read_noise = 0 * u.photon, 
                        return_read_cube=False)

                    noisy_scene_series[tt] += noisy_scene

                    noisy_scene_series[tt] *= self.datamask_nan


            if not p_noise:
                
                return planet_scene_series
            
            return noisy_scene_series

        ephem_df = planet.ephem_pm

        ephem_tseries = np.array(ephem_df.index) * u.year

        ## Display the planet PSF scene at several time samples, in flux ratio units, and save to fits

        # Generate noiseless tseries images with 10 hours of integration time
        planet_scene_series_10_hr = make_time_series_images(planet.observation_epochs,int_ts=[10*u.hour],p_noise=False) #

        ### Calculate int time needed to get 200 photons in 3x3 grid around peak

        int_times = []

        for tt,ep in enumerate(planet.observation_epochs):
            
            scene = planet_scene_series_10_hr[tt]

            # Locate peak
            img_width = scene.shape[0]
            
            XXs, YYs = np.meshgrid(np.arange(img_width), np.arange(img_width))

            peak_row = np.nanargmax(np.ravel(scene)) // img_width
            
            peak_col = np.nanargmax(np.ravel(scene)) % img_width
            
            peak_val = scene[peak_row, peak_col]
            
            # Get a cutout of the source
            cut_width = 3
            src_cutout = (scene)[
                    peak_row - cut_width//2 : peak_row + cut_width // 2 + 1,
                    peak_col - cut_width//2 : peak_col + cut_width // 2 + 1]
            num_photons = np.nansum(src_cutout.flatten())
            
            # Calculate the new integration time, capped at 1000 hours
            #if num_photons <= 0:
            #    raise Warning("<0 photons detected in a 3x3 box around the source!")
            
            new_int_time = (10 * (self.snr_goal**2 * u.photon)/ num_photons).to(u.hour)  if num_photons > 0 else (1000 * u.hour)
            new_int_time = np.min([new_int_time.value,1000]) * u.hour

            if new_int_time.value > 1000:
                raise ValueError("Integration time > 1000 hours!")
            
            #new_int_t = (10 * (self.snr_goal**2 * u.photon)/ num_photons).to(u.hour)
            #new_int_time = np.min([new_int_t.value, 1000] ) * u.hour

            int_times.append(new_int_time)

        planet_scene_series = make_time_series_images(planet.observation_epochs,int_ts=int_times)
        
        planet.int_times = int_times

        datacube = []
        
        for tt,time_show in enumerate(planet.observation_epochs):
            
            tt_show = np.argmin(np.abs(time_show - planet.observation_epochs))
            
            crop = 4

                
            data = planet_scene_series[tt_show, crop:-crop, crop:-crop] #/ unocc_star_countrate_peakpix
                
            datacube.append(data)
            
            if showplots:
            
                plt.figure(figsize = (8,6))
                plt.imshow(data,extent=(42,0,0,42)) # ,extent=(99,0,0,99)

                xtick_locs = (np.arange(1, -1, -0.2) / self.data_pixscale_as.value
                              + (self.imwidth - 2 * crop) // 2)
                xtick_labels = ['{:+.1f}'.format(loc) for loc in np.arange(-1, 1, 0.2)]
                ytick_labels = ['{:+.1f}'.format(loc) for loc in np.arange(1, -1, -0.2)]
                plt.xticks(xtick_locs, xtick_labels, size=14)
                plt.xlim([0, self.imwidth - 2 * crop - 1])
                plt.yticks(xtick_locs, ytick_labels, size=14)
                plt.ylim([0, self.imwidth - 2*crop - 1])
                plt.tick_params('both', length=8, width=1.5, which='major', top=True, right=True,
                                direction='in')
                plt.xlabel('Offset from star (arcsec)')

                plt.title(f'{planet.sysname}, t={np.round(time_show,2)}',size='medium')
                plt.colorbar(shrink=0.8)
                plt.tight_layout()

                plt.show()

                plt.close()

        datacube_fpath = os.path.join(planet.planet_dir,f'simulated_observations.fits')

        # Write data cube to a fits file
        hdu = fits.PrimaryHDU(datacube)
        hdul = fits.HDUList([hdu])
        hdr = hdul[0].header
        
        for i,ep in enumerate(planet.observation_epochs):
            hdr[f'epoch{i}'] = ep
        
        hdul.writeto(datacube_fpath)
        
        planet.observation_hdul = hdul
        