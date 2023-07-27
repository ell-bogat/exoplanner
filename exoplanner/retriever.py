import os
import scipy
import skimage
import warnings
import pkg_resources
import exoscene
import astropy
import orbitize
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as c
import astropy.io.fits as fits
import multiprocessing as mp

from scipy.stats import norm
from astropy.time import Time
from skimage import registration
from orbitize.system import radec2seppa, seppa2radec
from orbitize import sampler

warnings.simplefilter(action='ignore',category=UserWarning)

class Retriever:
    
    """
    TO DO:
    - see how much is repeated between Observatory.test_psf and the psf model generation here?
    """
    
    def __init__(self, target,
                 astrom_uncertainty_df=None
                ):
        
        self.configs = target.configs
        
        self.astrom_uncertainty = astrom_uncertainty_df
        
        self.obs = target.obs
        
        self.results_dir = target.results_dir
        
        self.cal_data_dir = os.path.join(*self.configs['cal_dir'])
        
        self.psf_model_fname = os.path.join(self.results_dir, 'hlc_centered_psf.fits')
        
        # Generate centered PSF model if it does not exist
            
        if not os.path.exists(self.psf_model_fname):
            
            # Load PSF model
            offset_psfs_fname = os.path.join(self.cal_data_dir, 'OS6_HLC_PSFs_offset.fits')
            offset_psfs_offsets_fname = os.path.join(self.cal_data_dir, 'OS6_HLC_PSFs_offset_list.fits')

            cx_os6 = 100 # True center of the high-res OS6 array,
                         # derived from checking the peak-offset correspondence in the off-axis PSF model. 
                
            pixscale_as = (fits.getheader(offset_psfs_fname))['PIX_AS']
            pixscale_LoD = (fits.getheader(offset_psfs_fname))['PIXSCALE']

            offset_psfs_os6 = fits.getdata(offset_psfs_fname)
            offsets_LoD = fits.getdata(offset_psfs_offsets_fname)
            offsets_as = offsets_LoD * pixscale_as / pixscale_LoD

            # Apply half-pixel shift to OS6 arrays to place center at (99.5, 99.5)
            offset_psfs = scipy.ndimage.interpolation.shift(offset_psfs_os6, (0, -0.5, -0.5),
                                                            order = 1, prefilter=False,
                                                            mode = 'constant', cval=0)
            
            r_p = 60 # offset PSF position with relatively little distortion from FPM or field stop
            
            theta = 0 # destination theta
            r_as = r_p * pixscale_as
            oi = np.argmin(np.abs(r_as - offsets_as)) # index of closest radial offset in Krist model array
            dr_as = r_as - offsets_as[oi] # radial shift needed in arcseconds
            dr_p = dr_as / pixscale_as # radial shift needed in pixels

            shift_psf = scipy.ndimage.interpolation.shift(offset_psfs[oi], (0, dr_p),
                                                          order = 1, prefilter=False)

            cent_shift_psf = np.roll(shift_psf, -r_p, axis=1)
            
            # Show the model
            
            if self.configs['showplots']:
                plt.figure(figsize=(10, 8))
                plt.imshow(cent_shift_psf)
                plt.colorbar()
                plt.show()
                plt.close()

            fits.writeto(self.psf_model_fname, cent_shift_psf)
            
    #####################################################################
            
    def retrieve_astrom(self,planet,showplots=None):

        """
        TO DO:
        - Find source of pixscale_ratio
        - Move PSF peak map loading to __init__ fn
        """

        def correct_sep(measured_sep):

                x = measured_sep

                p = [ 2.83859299e-14 ,
                     -5.45080635e-11  ,
                     4.39691652e-08 ,
                     -1.92508226e-05 ,
                     4.91988178e-03 ,
                     -7.29782266e-01 ,
                     5.77003863e+01 ,
                     -1.85256752e+03]

                x7,x6,x5,x4,x3,x2,x1,x0 = p

                return measured_sep - (x7 * x ** 7 + x6 * x ** 6 + x5 * x ** 5 + x4 * x ** 4 + x3 * x ** 3 + x2 * x ** 2 + x1 * x + x0)



        def psffit_costfunc(params, hires_psf, data_array, hires_pixscale_as, data_pixscale_as, amp_normfac):

            # PSD fit cost function- returns sum of square diff b/n model & data

            # shift and scale hires model

            xshift = params[0] / hires_pixscale_as.value
            yshift = params[1] / hires_pixscale_as.value

            scalefac = params[2] * amp_normfac

            shifted_hires_psf = scipy.ndimage.shift(hires_psf, (yshift, xshift))

            xcoord_hires = ((np.arange(hires_psf.shape[0]) - hires_psf.shape[0]//2)
                            * hires_pixscale_as)

            binned_psf_model, _, _ = exoscene.image.resample_image_array(
                scalefac * shifted_hires_psf, hires_pixscale_as,
                img_xcoord = xcoord_hires, img_ycoord = xcoord_hires,
                det_pixscale = data_pixscale_as,
                det_width = (data_array.shape[0] + 1) * data_pixscale_as,
                binfac = 10, conserve = 'sum')


            return np.nansum((binned_psf_model - data_array)**2)

        def eval_psf(params, hires_psf, data_array, hires_pixscale_as, data_pixscale_as, amp_normfac):

            #fn to evaluate PSF fit

            xshift = params[0] / hires_pixscale_as.value
            yshift = params[1] / hires_pixscale_as.value

            scalefac = params[2] * amp_normfac

            shifted_hires_psf = scipy.ndimage.shift(hires_psf, (yshift, xshift))

            xcoord_hires = ((np.arange(hires_psf.shape[0]) - hires_psf.shape[0]//2)
                            * hires_pixscale_as)

            binned_psf_model, _, _ = exoscene.image.resample_image_array(
                scalefac * shifted_hires_psf, hires_pixscale_as,
                img_xcoord = xcoord_hires, img_ycoord = xcoord_hires,
                det_pixscale = data_pixscale_as,
                det_width = (data_array.shape[0] + 1) * data_pixscale_as,
                binfac = 10, conserve = 'sum')

            return binned_psf_model, np.sum((binned_psf_model - data_array)**2)    

        if showplots is None:
            showplots = self.configs['showplots']

        star_cal_fname = os.path.join(self.cal_data_dir, "HLC_scistar_unocc_PSF_model.fits")
        psf_peak_map_fname = os.path.join(self.cal_data_dir, 'OS6_HLC_PSF_peak_map.fits')

        datacube = planet.observation_hdul[0].data

        img_width = datacube.shape[-1]
        
        if img_width == 0:
            warnings.warn(f'Image data for {planet.sysname} is empty. Returning np.nan for all epochs.')
            planet.post.astrometry = np.nan
            
            return None

        # Set constants and HLC sys settings

        pixscale_as = planet.obs.data_pixscale_as.value
        pixscale_mas = pixscale_as * 1000 * u.mas
        pixscale_ld = planet.obs.data_pixscale_LoD

        xtick_locs = np.arange(-1, 1, 0.2) / pixscale_as + img_width // 2
        xtick_labels = ['{:+.1f}'.format(loc) for loc in np.arange(-1, 1, 0.2)]

        crop = 2    

        # Load peak maps

        psf_peak_map = fits.getdata(psf_peak_map_fname)
        psf_peak_map_hdr = fits.getheader(psf_peak_map_fname)
        pixscale_ratio = 4.2000023908478346 # Where does this number come from??
        peak_map_width = psf_peak_map.shape[0]

        peak_map_xs = ((np.arange(peak_map_width) - peak_map_width // 2) 
                       * pixscale_mas.value / pixscale_ratio)

        peak_map_interp_func = scipy.interpolate.RegularGridInterpolator(
                (peak_map_xs, peak_map_xs), psf_peak_map)

        data_xs = ((np.arange(img_width) - img_width // 2)
                   * pixscale_mas.value)

        data_YYs, data_XXs = np.meshgrid(data_xs, data_xs, indexing='ij')
        data_interp_grid = (data_YYs.ravel(), data_XXs.ravel())

        peak_map_interp_result = peak_map_interp_func(data_interp_grid)
        peak_map_interp = peak_map_interp_result.reshape(data_XXs.shape)
        peak_map_interp_norm = peak_map_interp / np.max(peak_map_interp)

        if self.configs['verbose']:
            print(f'Peak map interp shape: {peak_map_interp.shape}')
            print(f'PSF peak map max:{psf_peak_map.max()}, peak map interp max:{peak_map_interp.max()}')

        # Plot PSF peak map and peak map w/ normal interpolation

        if showplots:
            plt.figure(figsize=(12,5))
            plt.subplot(121)
            plt.imshow(psf_peak_map)
            plt.colorbar()
            plt.subplot(122)
            plt.imshow(peak_map_interp_norm)
            plt.colorbar()
            plt.show()
            plt.close()

        # Load PSF model

        psf_model = fits.getdata(self.psf_model_fname)
        psf_model_shift = scipy.ndimage.shift(psf_model, [0.5, 0.5])
        psf_model_crop = 11
        psf_model_crop = psf_model_shift[psf_model.shape[0] // 2 - psf_model_crop//2:psf_model.shape[0] // 2 + psf_model_crop//2 + 1,
                                         psf_model.shape[0] // 2 - psf_model_crop//2:psf_model.shape[0] // 2 + psf_model_crop//2 + 1]

        if showplots:
            plt.figure(figsize=(8,8))
            plt.imshow(psf_model_crop[:,:])
            plt.show()
            plt.close()

        astrom_table = []

        # Iterate through data epochs
        for epoch_ind in range(len(datacube)):

            epoch = epoch_ind + 1
            data_img = datacube[epoch_ind]
            
            
            if showplots:
                print(f'\n\tEpoch {epoch}\n')
                plt.figure(figsize=(8,6))
                plt.imshow(data_img)
                plt.xticks(xtick_locs, xtick_labels, size=14)
                plt.xlim([0 + crop, img_width - 1 - crop])
                plt.yticks(xtick_locs, xtick_labels, size=14)
                plt.ylim([0 + crop, img_width - 1 - crop])
                plt.xlabel('Offset from star (arcsec)')
                plt.colorbar()
                plt.show()
                #plt.savefig(f"results/planetc_ep{epoch}.png", dpi=200)
                plt.close()

            # Find the position of max value in the image

            img_width = data_img.shape[0]
            XXs, YYs = np.meshgrid(np.arange(img_width), np.arange(img_width))

            peak_row = np.nanargmax(np.ravel(data_img)) // img_width
            peak_col = np.nanargmax(np.ravel(data_img)) % img_width
            peak_val = data_img[peak_row, peak_col]

            if self.configs['verbose']:
                
                print("Peak col = {:d}, peak row = {:d}".format(peak_col, peak_row))
                print(f"Peak value = {peak_val}")

            # Calculate attenuation due to coronagraph mask

            src_peak_map_col = (peak_map_width // 2 
                                + int(np.round((peak_col - img_width // 2)
                                * pixscale_ratio)))
            src_peak_map_row = (peak_map_width // 2
                                + int(np.round((peak_row - img_width // 2)
                                * pixscale_ratio)))

            if self.configs['verbose']:
                
                print("Source position in peak map: {:}, {:}".format(src_peak_map_col, src_peak_map_row))

            psf_atten = psf_peak_map[src_peak_map_row, src_peak_map_col] / np.max(psf_peak_map)

            if self.configs['verbose']:
                
                print("Relative PSF attenuation: {:.2f}".format(psf_atten))

            # Get astrometry estimate by cross-correlating w/ PSF model

            # Get a cutout of the source
            cut_width = 5

            src_cutout = (data_img)[
                    peak_row - cut_width//2 : peak_row + cut_width // 2 + 1,
                    peak_col - cut_width//2 : peak_col + cut_width // 2 + 1]

            # Rescale
            scale_fac = pixscale_ld / 0.1

            try:
                src_cutout_rescaled = skimage.transform.rescale(src_cutout, scale = scale_fac)
            
            except:
                
                print(f"\tSource rescale did not converge - skipping epoch {epoch_ind}.")
                
                continue
            
            if self.configs['verbose']:
                print(src_cutout_rescaled.shape)

            if showplots:
                plt.figure(figsize=(12,5))
                plt.subplot(121)
                plt.imshow(src_cutout)
                plt.colorbar()
                plt.subplot(122)
                plt.imshow(src_cutout_rescaled)
                plt.colorbar()
                plt.show()
                plt.close()

            # Crop
            src_cutout_rescaled_cropped = src_cutout_rescaled[
                    src_cutout_rescaled.shape[0] // 2 - psf_model_crop.shape[0] // 2:src_cutout_rescaled.shape[0] // 2 + psf_model_crop.shape[0] // 2 + 1,
                    src_cutout_rescaled.shape[0] // 2 - psf_model_crop.shape[0] // 2:src_cutout_rescaled.shape[0] // 2 + psf_model_crop.shape[0] // 2 + 1]

            # Do cross correlation
            cross_corr_result = registration.phase_cross_correlation(reference_image = psf_model_crop,
                                                                             moving_image = src_cutout_rescaled_cropped,
                                                                             upsample_factor = 10,
                                                                             reference_mask=np.array(~np.isnan(psf_model_crop)),
                                                                             moving_mask= np.array(~np.isnan(src_cutout_rescaled_cropped)))

            if self.configs['verbose']:
                print('Cross correlation result: ',cross_corr_result)

            # Convert result to offset in arcsec
            src_y = peak_row - cross_corr_result[0] / scale_fac
            src_x = peak_col - cross_corr_result[1] / scale_fac

            src_y_offset = (src_y - img_width // 2) * pixscale_mas
            src_x_offset = (src_x - img_width // 2) * pixscale_mas

            ymas = src_y_offset.to(u.mas).value
            xmas = src_x_offset.to(u.mas).value

            # ESTIMATE FROM CROSS-CORRELATION

            ra_mas = xmas
            dec_mas = ymas        
            sep_mas,pa_deg = radec2seppa(ra_mas,dec_mas)

            #sep_mas_corrected = correct_sep(sep_mas)        
            #ra_mas_corrected, dec_mas_corrected = seppa2radec(sep_mas_corrected,pa_deg)

            if self.configs['do_PSF_fit']:

                # Make new source cutout
                src_cx = data_img.shape[-1] // 2
                x_offset_src_image = src_x_offset.to(u.arcsec).value / planet.obs.data_pixscale_as.value
                y_offset_src_image = src_y_offset.to(u.arcsec).value / planet.obs.data_pixscale_as.value

                x_cent_src_image = int(np.round(src_cx + x_offset_src_image))
                y_cent_src_image = int(np.round(src_cx + y_offset_src_image))

                src_boxsize = 5

                if x_cent_src_image < src_boxsize//2:
                    x_cent_src_image = src_boxsize//2
                elif x_cent_src_image > data_img.shape[1] - 1 - src_boxsize//2:
                    x_cent_src_image = data_img.shape[1] - 1 - src_boxsize//2

                if y_cent_src_image < src_boxsize//2:
                    y_cent_src_image = src_boxsize//2
                elif y_cent_src_image > data_img.shape[0] - 1 - src_boxsize//2:
                    y_cent_src_image = data_img.shape[0] - 1 - src_boxsize//2

                # Store the precise offset of the source "data" cutout array
                src_cutout_xoffset_as = (x_cent_src_image - src_cx) * planet.obs.data_pixscale_as
                src_cutout_yoffset_as = (y_cent_src_image - src_cx) * planet.obs.data_pixscale_as            

                src_cutout = data_img[y_cent_src_image - src_boxsize//2:y_cent_src_image + src_boxsize//2 + 1,
                                       x_cent_src_image - src_boxsize//2:x_cent_src_image + src_boxsize//2 + 1]


                # Get PSF data
                hlc_psf_path =  pkg_resources.resource_filename('exoscene', 'data/cgi_hlc_psf')

                psf_cube_fname = os.path.join(hlc_psf_path, 'os9_psfs.fits')
                psf_r_fname = os.path.join(hlc_psf_path, 'os9_psfs_r_offset.fits')
                psf_angle_fname = os.path.join(hlc_psf_path, 'os9_psfs_angle_offset.fits')

                psf_cube = fits.getdata(psf_cube_fname)
                psf_hdr = fits.getheader(psf_cube_fname)

                #print("Shape of PSF model: {:}".format(psf_cube.shape))

                hires_pixscale_as = psf_hdr['PIX_AS'] * u.arcsec
                hires_pixscale_LoD = psf_hdr['PIXSCALE']

                data_scale_fac = hires_pixscale_as.value / planet.obs.data_pixscale_as.value
                data_pixscale_LoD = hires_pixscale_LoD / data_scale_fac

                if self.configs['verbose']:
                    print("CCD pixel to model pixel scale factor = {:.3f}".format(data_scale_fac))

                r_offsets_LoD = fits.getdata(psf_r_fname)[0, :]
                r_offsets_as = r_offsets_LoD * hires_pixscale_as / hires_pixscale_LoD
                angles = fits.getdata(psf_angle_fname)[:, 0]

                Np_psf_hires = psf_cube.shape[-1]

                offset_psfs = scipy.ndimage.interpolation.shift(psf_cube, (0, 0, -0.5, -0.5),
                                                                order = 1, prefilter=False,
                                                                mode = 'constant', cval=0)

                Np = offset_psfs.shape[-1]
                cx = Np // 2 - 0.5 # New array center in zero-based indices

                if self.configs['verbose']:
                    print("Dimensions of PSF model cube: {:}".format(offset_psfs.shape))
                    print("New array center along both axes: {:.1f}".format(cx))

                # Get hi-res PSF model at position estimated from cross-correlation
                hires_psf = exoscene.image.get_hires_psf_at_xy_os9(
                        offset_psfs,
                        r_offsets_as.value, angles,
                        hires_pixscale_as.value,
                        delx_as = src_x_offset.to(u.arcsec).value,
                        dely_as = src_y_offset.to(u.arcsec).value,
                        cx = cx)


                # Test binning PSF cutout
                x_offset_hirespix = src_x_offset.to(u.arcsec).value / hires_pixscale_as.value
                y_offset_hirespix = src_y_offset.to(u.arcsec).value / hires_pixscale_as.value

                # Get the integer indices of the pixels closest to the model PSF center
                x_cent_hirespix = int(np.round(cx + (x_cent_src_image - src_cx) * planet.obs.data_pixscale_as.value / hires_pixscale_as.value))
                y_cent_hirespix = int(np.round(cx + (y_cent_src_image - src_cx) * planet.obs.data_pixscale_as.value / hires_pixscale_as.value))

                # Store the precise offset of the hi-res PSF model cutout array
                hires_model_xoffset_as = (x_cent_hirespix - cx) * hires_pixscale_as
                hires_model_yoffset_as = (y_cent_hirespix - cx) * hires_pixscale_as

                # Store the offset difference between the data and model arrays
                cutout_xdelta_as = src_cutout_xoffset_as - hires_model_xoffset_as
                cutout_ydelta_as = src_cutout_yoffset_as - hires_model_yoffset_as        

                hires_boxsize = 26
                hires_cutout = hires_psf[y_cent_hirespix - hires_boxsize//2:y_cent_hirespix + hires_boxsize//2 + 1,
                                         x_cent_hirespix - hires_boxsize//2:x_cent_hirespix + hires_boxsize//2 + 1]

                xcoord_cutout = ((np.arange(hires_cutout.shape[0]) - hires_boxsize//2)
                              * hires_pixscale_as)

                data_boxsize = 5 

                binned_test_cutout, det_xcoord, det_ycoord = exoscene.image.resample_image_array(
                        hires_cutout, hires_pixscale_as,
                        img_xcoord = xcoord_cutout, img_ycoord = xcoord_cutout,
                        det_pixscale = planet.obs.data_pixscale_as,
                        det_width = (data_boxsize + 1) * planet.obs.data_pixscale_as,
                        binfac = 10, conserve = 'sum')

                amp_normfac = 1 / np.max(binned_test_cutout)
                
                if self.configs['verbose']:
                    print("Binned test cutout Amplitude normalization factor:",amp_normfac)

                guess_params = (0., 0., np.nanmax(src_cutout))
                param_bounds = [(-planet.obs.data_pixscale_as.value * (data_boxsize // 2),
                                  planet.obs.data_pixscale_as.value * (data_boxsize // 2)),
                                (-planet.obs.data_pixscale_as.value * (data_boxsize // 2),
                                  planet.obs.data_pixscale_as.value * (data_boxsize // 2)),
                                (0, 10 * np.nanmax(src_cutout))]

                psffit_result = scipy.optimize.minimize(
                        psffit_costfunc,
                        x0 = guess_params,
                        args = (hires_cutout, src_cutout,
                                hires_pixscale_as, planet.obs.data_pixscale_as,
                                amp_normfac),
                        method='Nelder-Mead',
                        bounds = param_bounds
                        )

                if self.configs['verbose']:
                    print("PSF Fit result:\n",psffit_result)

                best_fit_psf, sumofsq = eval_psf(psffit_result.x, hires_cutout, src_cutout,
                                         hires_pixscale_as, planet.obs.data_pixscale_as,
                                         amp_normfac)

                if showplots:
                    plt.figure(figsize=(16,4))
                    plt.subplot(141)
                    plt.title("Binned initial model")
                    plt.imshow(binned_test_cutout)
                    plt.colorbar()
                    plt.subplot(142)
                    plt.title("Best fit PSF")
                    plt.imshow(best_fit_psf)
                    plt.colorbar()
                    plt.subplot(143)
                    plt.title("Data")
                    plt.imshow(src_cutout)
                    plt.colorbar()
                    plt.subplot(144)
                    plt.title("Residuals")
                    plt.imshow(best_fit_psf - src_cutout)
                    plt.colorbar()
                    plt.tight_layout()
                    plt.show()
                    plt.close()

                src_x_offset_fit = src_x_offset.to(u.mas) + psffit_result.x[0] * u.arcsec + cutout_xdelta_as
                src_y_offset_fit = src_y_offset.to(u.mas) + psffit_result.x[1] * u.arcsec + cutout_ydelta_as

                sep_mas,pa_deg = radec2seppa(src_x_offset_fit.to(u.mas).value,src_y_offset_fit.to(u.mas).value)

                amp = psffit_result.x[2]

                if self.configs['verbose']:
                    print(f'\nEpoch {epoch_ind+1} astrometry result: ')

                    print("\tCross corr:",src_x_offset.to(u.mas), src_y_offset.to(u.mas))
                    print("\tPSF Fit:",src_x_offset_fit, src_y_offset_fit,f"\n\t\t\tamp={np.round(amp,2)} electrons")

                src_x_offset = src_x_offset_fit
                src_y_offset = src_y_offset_fit

                #sigma = (1 / pixscale_ld) / peak_to_bg_SNR * pixscale_mas
                #print(f'Sigma: {sigma}')

            else:
                amp = np.nan

            astrom_table.append([planet.observation_hdul[0].header['epoch'+str(epoch_ind)], src_x_offset_fit.value, src_y_offset_fit.value, sep_mas, pa_deg, self.configs['snr']])

        astrom_df = pd.DataFrame(astrom_table,columns=['t_yr', 'ra_mas','dec_mas','sep_mas','pa_deg','snr'])

        seps = np.round(astrom_df.sep_mas,-1)

        astrom_df['uncertainty_mas'] = [self.astrom_uncertainty.loc[sep,'meds'] + self.astrom_uncertainty.loc[sep,'stds'] for sep in seps ]
        
        astrom_df.set_index('t_yr',inplace=True)

        astrom_data_fname = f'retrieved_astrometry.csv'
        
        astrom_df.to_csv(os.path.join(planet.planet_dir,astrom_data_fname))

        planet.post.astrometry = astrom_df
        
    #################################################################################################

    def retrieve_orbit(self,planet):
        
        num_planets = 1 # Default number of planets that orbitize will fit
        
        num_target = 1  # Which planet is the target planet we are trying to image
        
        # Get astrometry data:    
        astrom_df = planet.post.astrometry
        
        if not type(astrom_df) == pd.core.frame.DataFrame:
            
            if np.isnan(astrom_df):
                return None
            
            else:
                raise ValueError('Astrometry has not been retrieved for this planet. Please run planet.retrieve_astrom() first.')
                
        elif len(astrom_df) < 1:
            
            return None

        
        # Get RV data
        rv_available = False # Initialize boolean, then search for rv data
        if self.configs['fit_rv']:
            rv_input_fname = '_'.join(st_name.split(' ')) + '_rv.csv'
            rv_path = os.path.join('RV',rv_input_fname)

            if os.path.exists(rv_path):
                print(f'\tInput RV data located for {st_name}...') 
                rv_data = pd.read_csv(rv_path) #, usecols = ('time','rv','err')
                rv_available = True        

                # Load RV data info
                try:
                    num_planets = rvdict[st_name]['n_rv_planets']
                    num_target = rvdict[st_name]['n_target_pl']
                    num_walkers = 24 # + (4 * num_planets)
                    num_steps = num_steps * num_planets
                    rv_planets = rvdict[st_name]['pl_list']
                except KeyError:
                    raise ValueError('RV data available for {%s} but rvdict is not properly configured for it.'.format(st_name))

        # Make input data table according to parameters above
        print()
        print('\tLoading input astrometry data...')

        # Format epochs as Modified Julian Date
        epochs = Time(np.array(astrom_df.index),format='decimalyear')
        epochs.format = 'mjd'

        # Start an empty table
        astrom_orbitize_table = astropy.table.Table()

        # Create columns
        epoch_col = astropy.table.Table.Column(name = 'epoch', data = epochs.value, dtype = float)
        object_col = astropy.table.Table.Column(name = 'object', data = [num_target] * len(epochs), dtype = int)
        sep_col = astropy.table.Table.Column(name = 'quant1', data = np.array(astrom_df['sep_mas']), dtype = float)
        sep_err_col = astropy.table.Table.Column(name = 'quant1_err', data = np.array(astrom_df['uncertainty_mas']), dtype = float)
        pa_col = astropy.table.Table.Column(name = 'quant2', data = np.array(astrom_df['pa_deg']), dtype = float)
        pa_err_col = astropy.table.Table.Column(name = 'quant2_err', data = np.rad2deg(np.arctan(astrom_df['uncertainty_mas'] / np.array(astrom_df['sep_mas']))), dtype = float)
        quant_type_col = astropy.table.Table.Column(name = 'quant_type', data = ['seppa'] * len(epochs), dtype = str)

        # Add each column to the table
        astrom_orbitize_table.add_column(epoch_col, index = 0)
        astrom_orbitize_table.add_column(object_col, index = 1)
        astrom_orbitize_table.add_column(sep_col, index = 2)
        astrom_orbitize_table.add_column(sep_err_col, index = 3)
        astrom_orbitize_table.add_column(pa_col, index = 4)
        astrom_orbitize_table.add_column(pa_err_col, index = 5)
        astrom_orbitize_table.add_column(quant_type_col, index = 6)


        if self.configs['fit_rv']:
            
            rv_input_fname = '_'.join(planet.st_name.split(' ')) + '_rv.csv'
            rv_path = os.path.join('RV',rv_input_fname)
            
            if os.path.exists(rv_path):
                
                print(f'\tLoading input RV data for {planet.st_name}...') 

                rv_data = pd.read_csv(rv_path) #, usecols = ('time','rv','err')

                # Assign arbirtrary low error value so the software doesn't encounter a value of 0.0
                rv_data['err'] = 0.1

                # Convert m/s to km/s
                rv_data['rv'] = rv_data['rv'] * 0.001
                rv_data['err'] = rv_data['err'] * 0.001

                rv_vels = np.array(rv_data['rv'].values)
                rv_errs = rv_data['err'].values
                rv_epochs_jd = rv_data['time'].values
                rv_epochs_mjd = [astropy.time.Time(jd_epoch, format='jd').mjd for jd_epoch in rv_epochs_jd]

                # Format RV data
                rv_orbitize_table = astropy.table.Table()

                rv_epoch_col = astropy.table.Table.Column(name = 'epoch', data = rv_epochs_mjd, dtype = float)
                rv_object_col = astropy.table.Table.Column(name = 'object', data = np.zeros([len(rv_epochs_mjd)]), dtype = int)
                rv_vel_col = astropy.table.Table.Column(name = 'quant1', data = rv_vels, dtype = float)
                rv_err_col = astropy.table.Table.Column(name = 'quant1_err', data = rv_errs, dtype = float)
                rv_quant2_col = astropy.table.Table.Column(name = 'quant2', data = np.nan * np.ones([len(rv_epochs_mjd)]), dtype = float)
                rv_quant2_err_col = astropy.table.Table.Column(name = 'quant2_err', data = np.nan * np.ones([len(rv_epochs_mjd)]),
                                                               dtype = float)
                rv_quant_type_col = astropy.table.Table.Column(name = 'quant_type', data = ['rv'] * len(rv_epochs_mjd), dtype = str)

                rv_orbitize_table.add_column(rv_epoch_col, index = 0)
                rv_orbitize_table.add_column(rv_object_col, index = 1)
                rv_orbitize_table.add_column(rv_vel_col, index = 2)
                rv_orbitize_table.add_column(rv_err_col, index = 3)
                rv_orbitize_table.add_column(rv_quant2_col, index = 4)
                rv_orbitize_table.add_column(rv_quant2_err_col, index = 5)
                rv_orbitize_table.add_column(rv_quant_type_col, index = 6)

                if fit_rel_astrom:
                    # Combine with astrometry data
                    orbitize_input_table = astropy.table.vstack([astrom_orbitize_table,rv_orbitize_table])
                else:
                    orbitize_input_table = rv_orbitize_table

            else:
                raise ValueError(f'\t\t\tNo RV data available for {st_name}.')
                #orbitize_input_table = astrom_orbitize_table
        else:
            orbitize_input_table = astrom_orbitize_table

        # Write the input table to a file:
        orbitize_input_table_filename = os.path.join(planet.planet_dir,'orbitize_input.csv')
        
        orbitize.read_input.write_orbitize_input(orbitize_input_table,
                                                 orbitize_input_table_filename,
                                                 file_type = 'csv'
                                                 )
        

        # Run the MCMC algorithm using the settings above
        t_0 = datetime.datetime.now() # Record time at beginning 
        print(f"\t\tStart time: {t_0}")
        
        print('\n\tCollecting MCMC input parameters...')

        # Fill out uncertainties
        
        if np.any(np.isnan(planet.post.st_mass_err)):
            planet.post.st_mass_err = [0.01,0.01] * planet.post.st_mass
            
        if np.any(np.isnan(planet.post.ecc_err)):
            planet.post.ecc_err = [0.1,0.1]
        
        if np.any(np.isnan(planet.post.per_err)):
            planet.post.per_err = [0.1,0.1] * planet.post.per
        
        if np.any(np.isnan(planet.post.sma_err)):
            planet.post.sma_err = [0.1,0.1] * planet.post.sma
        
        # Define star/system parameters
        plx = planet.post.plx.value
        plx_err = np.mean([planet.post.plx_err[0].value,planet.post.plx_err[1].value])      
        system_mass = ((planet.post.st_mass)).to(u.solMass).value
        mass_err = (np.mean([planet.post.st_mass_err[0].value,planet.post.st_mass_err[1].value])*u.kg).to(u.solMass).value
        
        # Read data table
        data_table = orbitize.read_input.read_file(orbitize_input_table_filename)

        # Initialize the system
        print('\n\tInitializing MCMC system...')
        sys = orbitize.system.System(num_planets,
                                     data_table,
                                     system_mass,
                                     plx,
                                     mass_err=mass_err,
                                     plx_err=plx_err,
                                     fit_secondary_mass=self.configs['fit_rv']
                                    )

        planet.post.sys = sys

        ## Configure priors
        if self.configs['use_priors']:   
            print('\tApplying RV priors...')

            for i in range(1,num_planets+1):

                # Constrain parameters for target planet
                if i == num_target:

                    # Constrain by period if discovered by radial velocity, otherwise use sma
                    if planet.post.det_type == 'Radial Velocity':

                        P_mean = planet.post.per
                        P_sig = np.mean([planet.post.per_err[0].value,planet.post.per_err[1].value]) * u.day

                        sma_min = (((P_mean-P_sig) ** 2 * c.G * (system_mass-mass_err) * u.M_sun / (4 * np.pi ** 2)) ** (1./3)).to(u.AU)
                        sma_max = (((P_mean+P_sig) ** 2 * c.G * (system_mass+mass_err) * u.M_sun / (4 * np.pi ** 2)) ** (1./3)).to(u.AU)

                        sma_mean = np.mean([sma_min.value,sma_max.value]) * u.AU
                        sma_sig = ((sma_max - sma_min)/2)

                    else:
                        sma_mean = planet.post.sma
                        sma_sig = np.mean([planet.post.sma_err[0].value,planet.post.sma_err[1].value]) * u.AU

                    print(f'\t\tsma{num_target} mean: {sma_mean}')
                    print(f'\t\tsma{num_target} sigma: {sma_sig}')

                    sys.sys_priors[sys.param_idx['sma'+str(i)]] = orbitize.priors.GaussianPrior(sma_mean.value, sma_sig.value)

                    if not planet.ecc_prior_uniform:
                        # Exoplanet archives constrain the eccentricity.  
                        ecc_mean = planet.post.ecc
                        ecc_sig = np.mean([planet.post.ecc_err[0],planet.post.ecc_err[1]])

                        print(f'\t\tecc{num_target} mean: {ecc_mean}')
                        print(f'\t\tecc{num_target} sigma: {ecc_sig}')

                        sys.sys_priors[sys.param_idx['ecc'+str(i)]] = orbitize.priors.GaussianPrior(ecc_mean, ecc_sig,
                                                                                                        no_negatives=True)
                    else:
                        sys.sys_priors[sys.param_idx['ecc'+str(i)]] = orbitize.priors.UniformPrior(0.001, 0.999)

                    aop_mean = np.deg2rad(planet.post.argperi.value)
                    aop_sig = np.deg2rad(np.mean([planet.post.argperi_err[0].value,planet.post.argperi_err[1].value]))
                    
                    sys.sys_priors[sys.param_idx['aop'+str(i)]] = orbitize.priors.GaussianPrior(aop_mean, aop_sig)

                    # The position angle of the ascending node is constrained by comparing the sign of the RV signal
                    # at the time of the planet imaging detection, and the projected position of the planet.
                    # Crudely, we can restrict the ascending node position angle to the range 180 and 360 deg.
                    
                    pan_min = np.deg2rad(180.0)
                    pan_max = np.deg2rad(360.0)  
                    
                    sys.sys_priors[sys.param_idx['pan'+str(i)]] = orbitize.priors.UniformPrior(pan_min, pan_max)   

                    if self.configs['fit_rv']:
                        if np.isnan(planet.post.mass):
                            mass_min = (planet.post.msini - planet.post.msini_err[0]).value / (c.M_sun.to(u.M_jup).value)
                            mass_max = (0.075 * c.M_sun).to(u.M_jup).value / (c.M_sun.to(u.M_jup).value)

                            sys.sys_priors[sys.param_idx['m'+str(i)]] = orbitize.priors.UniformPrior(mass_min, mass_max)

                        else:
                            mass_mean = (planet.post.mass).value / (c.M_sun.to(u.M_jup).value)
                            mass_sig = (np.mean([planet.post.mass_err[0].value,planet.post.mass_err[1].value])) / (c.M_sun.to(u.M_jup).value)

                            sys.sys_priors[sys.param_idx['m'+str(i)]] = orbitize.priors.GaussianPrior(mass_mean, mass_sig)

                # Fix parameters for all non-target planets
                else:
                    
                    raise ValueError("multiplanet system RV fitting not configured")
                    
                    sma = rv_planets[i]['a'].value
                    sys.sys_priors[sys.param_idx['sma'+str(i)]] = sma 

                    ecc = rv_planets[i]['ecc']
                    sys.sys_priors[sys.param_idx['ecc'+str(i)]] = ecc                 

                    inc = np.deg2rad(sys_d['planet']['inc'].value)
                    sys.sys_priors[sys.param_idx['inc'+str(i)]] = inc                 

                    aop = np.deg2rad(rv_planets[i]['w'].value)
                    sys.sys_priors[sys.param_idx['aop'+str(i)]] = aop                 

                    tau = orbitize.basis.t0_to_tau(rv_planets[i]['tp'].value, sys.tau_ref_epoch, rv_planets[i]['per'].value)
                    sys.sys_priors[sys.param_idx['tau'+str(i)]] = tau                 

                    pan = np.deg2rad(320)
                    sys.sys_priors[sys.param_idx['pan'+str(i)]] = pan                 

                    pl_mass = rv_planets[i]['msini'].value / (np.sin(np.deg2rad(sys_d['planet']['inc'].value)))
                    sys.sys_priors[sys.param_idx['m'+str(i)]] = pl_mass



        ## Initialize sampler
        
        sampler_func = getattr(sampler, "MCMC")
        
        num_threads = mp.cpu_count()
        
        mcmc_sampler = sampler_func(sys, self.configs['num_temps'],self.configs['num_walkers'],num_threads)
        
        planet.post.mcmc_sampler = mcmc_sampler
        
        # Print priors and fixed parameters
        print('\tSys param priors:')
        for lab in sys.labels:
            try:
                print(f'\t\t{lab}:', vars(sys.sys_priors[sys.param_idx[lab]]))
            except:
                print(f'\t\t{lab}:', sys.sys_priors[sys.param_idx[lab]])

        # Run MCMC!
        print(f'\n\tRunning MCMC for {self.configs["num_steps"]} steps...')
        
        mcmc_sampler.run_sampler(total_orbits=self.configs['num_walkers'] * self.configs['num_steps'], burn_steps=0, thin=self.configs['thin'])

        results = mcmc_sampler.results
        
        planet.post.orbitize_results = results
        
        t_f = datetime.datetime.now() # Record time at end to determine runtime
        print(f"\t\tEnd time: {t_f}")
        print(f"\t\tRuntime: {t_f-t_0}")

        # Summarize and save the data in a dataframe
        print('\tExploring Results...')

        # First analyze all steps, then chop the first 1000 for burn-in (divide by 2 b/c of thinning)
        for b_in in [0,self.configs['num_steps']//4]:

            mcmc_sampler.chop_chains(b_in,trim=0)

            results = mcmc_sampler.results

            # Make an empty dataframe:
            q = [0.0225, 0.16, 0.5, 0.84, 0.9775]
            tuples = ((planet.sysname,q[0]),(planet.sysname,q[1]),(planet.sysname,q[2]),(planet.sysname,q[3]),(planet.sysname,q[4]))
            params = ['sma1','ecc1','inc1','aop1','pan1','tau1','per1','m1']
            if num_planets > 1:
                for n in range(2,num_planets+1):
                    params.extend(['sma'+str(n),'ecc'+str(n),'inc'+str(n),'aop'+str(n),'pan'+str(n),'tau'+str(n),'per'+str(n),'m'+str(n)])
            params.extend(['plx','mtot','gamma','sigma','m0'])
            m_index = pd.MultiIndex.from_tuples(tuples,names=['run','quantile'])
            df_run = pd.DataFrame(np.full((5,len(params)),np.nan),
                                  columns=params,
                                  index=m_index)

            # Iterate through parameter list
            print('\tPlotting histograms...')
            for param, indx in sys.param_idx.items():
                measurements = [x[indx] for x in mcmc_sampler.results.post]
                quants = np.quantile(measurements, [0.0225, 0.16, 0.5, 0.84, 0.9775])
                df_run[param] = quants

            # Calculate period for each planet for RV priors and RV none runs
            print('\tCalculating period and plotting corner figures...')

            plt.style.use('bmh')

            # Calculate period w/ uncertainty
            for planet_num in range(1,num_planets+1):
                quants = [0.0225, 0.16, 0.5, 0.84, 0.9775]
                for quant in quants:    
                    m_quant = np.round(1-quant,4)
                    try:
                        mtot = df_run['m0'][(planet.sysname,m_quant)] * u.M_sun
                    except:
                        pass
                    if np.isnan(mtot):
                        mtot = df_run['mtot'][(planet.sysname,m_quant)] * u.M_sun

                    sma = np.median(df_run['sma'+ str(planet_num)][(planet.sysname,quant)]) * u.AU
                    per = np.sqrt( 4 * np.pi**2 / (c.G * mtot) * sma**3 ).to(u.day) 

                    df_run['per'+str(planet_num)][(planet.sysname,quant)] = per.value

            # Convert inc to deg
            df_run.inc1 = np.rad2deg(df_run.inc1)

            # Save results
            print('\tSaving results object and DF summary')
            fpath=os.path.join(planet.planet_dir,f'orbitize_results_{b_in}BurnIn.hdf5')
            results.save_results(fpath)
            fpath=os.path.join(planet.planet_dir,f'orbitize_results_{b_in}BurnIn.csv')
            df_run.to_csv(fpath)


        planet.post.results_df = df_run