import numpy as np
import os
import pkg_resources
import pandas as pd
pd.options.mode.chained_assignment = None
import datetime
import multiprocessing as mp
import copy
dcopy = copy.deepcopy 

import skimage
import skimage.transform
import skimage.registration

import scipy.signal
import scipy.ndimage
import scipy.interpolate

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation
plt.style.use('bmh')
matplotlib.rcParams['image.cmap'] = 'inferno'
matplotlib.rcParams['font.size'] = 16
matplotlib.rcParams['image.origin'] = 'lower'

import astropy.units as u
import astropy.constants as c
import astropy.io.fits as fits
import astropy.table
from astropy.utils.exceptions import AstropyWarning
Time = astropy.time.Time
import warnings
warnings.simplefilter('ignore', category=AstropyWarning)

import orbitize
from orbitize import driver
from orbitize.system import radec2seppa, seppa2radec
import orbitize.kepler as kepler

import exoscene
from exoscene.planet import Planet

seafoam = '#dbfeff'

# Create HLC PSF model
def create_psf_model(results_dir,cal_data_dir,show_plots=False,verbose=False):
    if verbose:
        print("Generating HLC centered PSF model using OS6...")
    psf_model_fname = os.path.join(results_dir, 'hlc_centered_psf.fits')
    if not os.path.exists(psf_model_fname):

        # Load PSF model
        offset_psfs_fname = os.path.join(cal_data_dir, 'OS6_HLC_PSFs_offset.fits')
        offset_psfs_offsets_fname = os.path.join(cal_data_dir, 'OS6_HLC_PSFs_offset_list.fits')

        cx_os6 = 100 # True center of the high-res OS6 array,
                     # derived from checking the peak-offset correspondence in the off-axis PSF model. 
        pixscale_as = (fits.getheader(offset_psfs_fname))['PIX_AS']
        pixscale_LoD = (fits.getheader(offset_psfs_fname))['PIXSCALE']
        
        if verbose:
            print(f"PSF Model pixel scale (arcsec): {pixscale_as}")
            print(f"PSF Model pixel scale (LoD): {pixscale_LoD}")

        offset_psfs_os6 = fits.getdata(offset_psfs_fname)
        offsets_LoD = fits.getdata(offset_psfs_offsets_fname)
        offsets_as = offsets_LoD * pixscale_as / pixscale_LoD

        if verbose:
            print('offset_psfs_os6.shape',offset_psfs_os6.shape)
            print('offsets_LoD.shape',offsets_LoD.shape)
            print("offsets_LoD",offsets_LoD)

        # Apply half-pixel shift to OS6 arrays to place center at (99.5, 99.5)
        offset_psfs = scipy.ndimage.interpolation.shift(offset_psfs_os6, (0, -0.5, -0.5),
                                                        order = 1, prefilter=False,
                                                        mode = 'constant', cval=0)
        if verbose:
            print("offset_psfs.shape",offset_psfs.shape)

        r_p = 60 # offset PSF position with relatively little distortion from FPM or field stop
        if verbose:
            print("Using PSF model with angular separation {:.2f} lam/D".format(r_p * pixscale_LoD))
        theta = 0 # destination theta
        r_as = r_p * pixscale_as
        oi = np.argmin(np.abs(r_as - offsets_as)) # index of closest radial offset in Krist model array
        dr_as = r_as - offsets_as[oi] # radial shift needed in arcseconds
        dr_p = dr_as / pixscale_as # radial shift needed in pixels

        shift_psf = scipy.ndimage.interpolation.shift(offset_psfs[oi], (0, dr_p),
                                                      order = 1, prefilter=False)

        cent_shift_psf = np.roll(shift_psf, -r_p, axis=1)
        if verbose:
            print("cent_shift_psf.shape",cent_shift_psf.shape)

        # Show the model
        plt.figure(figsize=(10, 8))
        plt.imshow(cent_shift_psf)
        plt.colorbar()
        if show_plots:
            plt.show()
        plt.close()

        fits.writeto(psf_model_fname, cent_shift_psf)
    
    return psf_model_fname


def retrieve_astrom(sys_d,version,pl_list,system_name,sys_dir,cal_data_dir,SNR,psf_model_fname,astrom_uncertainty,PSF_fit=True,show_plots=True,poisson_noise=True,verbose=False):
    print("Retrieving astrometry...")
    # Define separation error correction
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
                #[ 2.40534659e-14, -4.74381767e-11, 3.91348305e-08 ,-1.74555652e05 ,4.52904114e-03 ,-6.79945104e-01,5.42569313e+01 ,-1.75331160e+03]
                
            x7,x6,x5,x4,x3,x2,x1,x0 = p
            return measured_sep - (x7 * x ** 7 + x6 * x ** 6 + x5 * x ** 5 + x4 * x ** 4 + x3 * x ** 3 + x2 * x ** 2 + x1 * x + x0)
        
    data_fname = os.path.join(sys_dir,f"{system_name}_PoissonNoise{poisson_noise}_v{version}.fits")
    
    if not os.path.exists(data_fname):
        print(f"Searched for file: {data_fname}")
        print('Data not found.')
        return np.nan
    
    # Define PSD fit cost function- returns sum of square diff b/n model & data
    def psffit_costfunc(params, hires_psf, data_array, hires_pixscale_as, data_pixscale_as, amp_normfac):
        #shift and scale hires model
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

        #return sum of squares
        return np.nansum((binned_psf_model - data_array)**2)

    
    # Define fn to evaluate PSF fit
    def eval_psf(params, hires_psf, data_array, hires_pixscale_as, data_pixscale_as, amp_normfac):
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
    
    
    star_cal_fname = os.path.join(cal_data_dir, "HLC_scistar_unocc_PSF_model.fits")
    psf_peak_map_fname = os.path.join(cal_data_dir, 'OS6_HLC_PSF_peak_map.fits')
    
    # Load data
    print(f"Loading data from {data_fname}")
    with fits.open(data_fname) as hdul:
        hdr = hdul[0].header
        datacube = hdul[0].data
    img_width = datacube.shape[-1]
    if verbose:
        print(f'Data cube shape:{datacube.shape}')
    # Set constants and HLC sys settings

    pix_scale = 21.08 * u.milliarcsecond
    pixscale_as = pix_scale.to(u.arcsec).value   
    pix_as = pixscale_as
    pix_ld = 0.4200002390847835
    xtick_locs = np.arange(-1, 1, 0.2) / pixscale_as + img_width // 2
    xtick_labels = ['{:+.1f}'.format(loc) for loc in np.arange(-1, 1, 0.2)]
    crop = 2    
    
    # Load peak maps
    
    psf_peak_map = fits.getdata(psf_peak_map_fname)
    psf_peak_map_hdr = fits.getheader(psf_peak_map_fname)
    pixscale_ratio = 4.2000023908478346
    peak_map_width = psf_peak_map.shape[0]

    peak_map_xs = ((np.arange(peak_map_width) - peak_map_width // 2) 
                   * pix_scale.value / pixscale_ratio)

    peak_map_interp_func = scipy.interpolate.RegularGridInterpolator(
            (peak_map_xs, peak_map_xs), psf_peak_map)

    data_xs = ((np.arange(img_width) - img_width // 2)
               * pix_scale.value)

    data_YYs, data_XXs = np.meshgrid(data_xs, data_xs, indexing='ij')
    data_interp_grid = (data_YYs.ravel(), data_XXs.ravel())

    peak_map_interp_result = peak_map_interp_func(data_interp_grid)
    peak_map_interp = peak_map_interp_result.reshape(data_XXs.shape)
    peak_map_interp_norm = peak_map_interp / np.max(peak_map_interp)

    if verbose:
        print(f'Peak map interp shape: {peak_map_interp.shape}')
        print(f'PSF peak map max:{psf_peak_map.max()}, peak map interp max:{peak_map_interp.max()}')

    # Plot PSF peak map and peak map w/ normal interpolation

    if show_plots:
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

    psf_model = fits.getdata(psf_model_fname)
    psf_model_shift = scipy.ndimage.shift(psf_model, [0.5, 0.5])
    psf_model_crop = 11
    psf_model_crop = psf_model_shift[psf_model.shape[0] // 2 - psf_model_crop//2:psf_model.shape[0] // 2 + psf_model_crop//2 + 1,
                                     psf_model.shape[0] // 2 - psf_model_crop//2:psf_model.shape[0] // 2 + psf_model_crop//2 + 1]
    
    if show_plots:
        plt.figure(figsize=(8,8))
        plt.imshow(psf_model_crop[:,:])
        plt.show()
        plt.close()

    astrom_table = []

    # Iterate through data epochs
    for epoch_ind in range(len(datacube)):

        epoch = epoch_ind + 1
        data_img = datacube[epoch_ind]
        print(f'\nEpoch {epoch}\n')

        # Plot data image
        plt.figure(figsize=(8,6))
        plt.imshow(data_img)
        plt.xticks(xtick_locs, xtick_labels, size=14)
        plt.xlim([0 + crop, img_width - 1 - crop])
        plt.yticks(xtick_locs, xtick_labels, size=14)
        plt.ylim([0 + crop, img_width - 1 - crop])
        plt.xlabel('Offset from star (arcsec)')
        plt.colorbar()
        if show_plots:
            plt.show()
        #plt.savefig(f"results/planetc_ep{epoch}.png", dpi=200)
        plt.close()

        # Find the position of max value in the image

        img_width = data_img.shape[0]
        XXs, YYs = np.meshgrid(np.arange(img_width), np.arange(img_width))

        peak_row = np.nanargmax(np.ravel(data_img)) // img_width
        peak_col = np.nanargmax(np.ravel(data_img)) % img_width
        peak_val = data_img[peak_row, peak_col]
        if verbose:
            print("Peak col = {:d}, peak row = {:d}".format(peak_col, peak_row))
            print(f"Peak value = {peak_val}")

        # Plot a box around the peak source
        if show_plots:
    
            plt.figure(figsize=(10, 4))

            plt.subplot(121)
            phot_box_width = 3
            plot_box_width = 7
            plt.imshow(data_img[
                                peak_row - plot_box_width//2 : peak_row + plot_box_width // 2 + 1,
                                peak_col - plot_box_width//2 : peak_col + plot_box_width // 2 + 1])
            plt.colorbar()

            plt.subplot(122)
            plt.imshow(data_img[
                                peak_row - phot_box_width//2 : peak_row + phot_box_width // 2 + 1,
                                peak_col - phot_box_width//2 : peak_col + phot_box_width // 2 + 1])
            plt.colorbar()
            plt.show()
            plt.close()

        # Calculate attenuation due to coronagraph mask

        src_peak_map_col = (peak_map_width // 2 
                            + int(np.round((peak_col - img_width // 2)
                            * pixscale_ratio)))
        src_peak_map_row = (peak_map_width // 2
                            + int(np.round((peak_row - img_width // 2)
                            * pixscale_ratio)))

        if verbose:
            print("Source position in peak map: {:}, {:}".format(src_peak_map_col, src_peak_map_row))

        psf_atten = psf_peak_map[src_peak_map_row, src_peak_map_col] / np.max(psf_peak_map)
        if verbose:
            print("Relative PSF attenuation: {:.2f}".format(psf_atten))

        # Get astrometry estimate by cross-correlating w/ PSF model

        # Get a cutout of the source
        cut_width = 5
        src_cutout = (data_img)[
                peak_row - cut_width//2 : peak_row + cut_width // 2 + 1,
                peak_col - cut_width//2 : peak_col + cut_width // 2 + 1]

        # Rescale
        scale_fac = pix_ld / 0.1
        try:
            src_cutout_rescaled = skimage.transform.rescale(src_cutout, scale = scale_fac)
        except:
            print(f"\tSource rescale did not converge- skipping epoch {epoch_ind}.")
            continue
        
        if verbose:
            print(src_cutout_rescaled.shape)

        if show_plots:
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

        if show_plots:
            plt.figure()
            plt.imshow(src_cutout_rescaled_cropped)
            plt.show()
            plt.close()


        # Do cross correlation
        cross_corr_result = skimage.registration.phase_cross_correlation(reference_image = psf_model_crop,
                                                                         moving_image = src_cutout_rescaled_cropped,
                                                                         upsample_factor = 10,
                                                                         reference_mask=np.array(~np.isnan(psf_model_crop)),
                                                                         moving_mask= np.array(~np.isnan(src_cutout_rescaled_cropped)))

        print('Cross correlation result: ',cross_corr_result)
        
        # Convert result to offset in arcsec
        src_y = peak_row - cross_corr_result[0] / scale_fac
        src_x = peak_col - cross_corr_result[1] / scale_fac
        #print(src_y, src_x)

        src_y_offset = (src_y - img_width // 2) * pix_scale
        src_x_offset = (src_x - img_width // 2) * pix_scale

        ymas = src_y_offset.to(u.mas).value
        xmas = src_x_offset.to(u.mas).value
        
        # ESTIMATE FROM CROSS-CORRELATION
        ra_mas = xmas
        dec_mas = ymas        
        sep_mas,pa_deg = radec2seppa(ra_mas,dec_mas)
        #sep_mas_corrected = correct_sep(sep_mas)        
        #ra_mas_corrected, dec_mas_corrected = seppa2radec(sep_mas_corrected,pa_deg)
        
        if PSF_fit:
            # Get fit from OS9 off-axis PSF (Neil's code)
            data_pixscale_as = (21.08 * u.milliarcsecond).to(u.arcsec) # Pixel scale of CGI science camera
            
            # Make new source cutout
            src_cx = data_img.shape[-1] // 2
            x_offset_src_image = src_x_offset.to(u.arcsec).value / data_pixscale_as.value
            y_offset_src_image = src_y_offset.to(u.arcsec).value / data_pixscale_as.value

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
            src_cutout_xoffset_as = (x_cent_src_image - src_cx) * data_pixscale_as
            src_cutout_yoffset_as = (y_cent_src_image - src_cx) * data_pixscale_as            

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

            data_scale_fac = hires_pixscale_as.value / data_pixscale_as.value
            data_pixscale_LoD = hires_pixscale_LoD / data_scale_fac
            if verbose:
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
            if verbose:
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
            x_cent_hirespix = int(np.round(cx + (x_cent_src_image - src_cx) * data_pixscale_as.value / hires_pixscale_as.value))
            y_cent_hirespix = int(np.round(cx + (y_cent_src_image - src_cx) * data_pixscale_as.value / hires_pixscale_as.value))
            
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
                    det_pixscale = data_pixscale_as,
                    det_width = (data_boxsize + 1) * data_pixscale_as,
                    binfac = 10, conserve = 'sum')

            amp_normfac = 1 / np.max(binned_test_cutout)
            if verbose:
                print("Binned test cutout Amplitude normalization factor:",amp_normfac)

            guess_params = (0., 0., np.nanmax(src_cutout))
            param_bounds = [(-data_pixscale_as.value * (data_boxsize // 2),
                              data_pixscale_as.value * (data_boxsize // 2)),
                            (-data_pixscale_as.value * (data_boxsize // 2),
                              data_pixscale_as.value * (data_boxsize // 2)),
                            (0, 10 * np.nanmax(src_cutout))]

            psffit_result = scipy.optimize.minimize(
                    psffit_costfunc,
                    x0 = guess_params,
                    args = (hires_cutout, src_cutout,
                            hires_pixscale_as, data_pixscale_as,
                            amp_normfac),
                    method='Nelder-Mead',
                    bounds = param_bounds
                    )

            if verbose:
                print("PSF Fit result:\n",psffit_result)
            
            best_fit_psf, sumofsq = eval_psf(psffit_result.x, hires_cutout, src_cutout,
                                     hires_pixscale_as, data_pixscale_as,
                                     amp_normfac)
            if show_plots:
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
            print(f'\nEpoch {epoch_ind+1} astrometry result: ')

            print("\tCross corr:",src_x_offset.to(u.mas), src_y_offset.to(u.mas))
            print("\tPSF Fit:",src_x_offset_fit, src_y_offset_fit,f"\n\t\t\tamp={np.round(amp,2)} electrons")

            src_x_offset = src_x_offset_fit
            src_y_offset = src_y_offset_fit
            #sigma = (1 / pix_ld) / peak_to_bg_SNR * pix_scale
            #print(f'Sigma: {sigma}')
        
        else:
            amp = np.nan
            
        astrom_table.append([hdr['epoch'+str(epoch_ind)], src_x_offset_fit.value, src_y_offset_fit.value, sep_mas, pa_deg, SNR, version])
        
    astrom_df = pd.DataFrame(astrom_table,columns=['t_yr', 'ra_mas','dec_mas','sep_mas','pa_deg','SNR','version'])
    
    seps = np.round(astrom_df.sep_mas,-1)
    astrom_df['uncertainty_mas'] = [astrom_uncertainty.loc[sep,'meds'] + astrom_uncertainty.loc[sep,'stds'] for sep in seps ]
    
    astrom_data_fname = system_name + f'_retrieved_astrometry_PoissonNoise{poisson_noise}_SNR{SNR}_v{version}.csv'
    astrom_df.to_csv(os.path.join(sys_dir,astrom_data_fname))
    
    
    ### Explore Results ###
    
    
    # Use the exoscene Planet class to define a 1-planet system
    if sys_d['det_type'] == 'Radial Velocity':
        a_in = None
        P_in = sys_d['planet']['period']
    else:
        a_in = sys_d['planet']['sma_au']
        P_in = None
    
    t_peri = (astropy.time.Time(sys_d['planet']['tperi'], format='jd', scale='utc').decimalyear) * u.year
    
    planet = Planet(system_name, 
                     dist = sys_d['planet']['dist'], 
                     a = a_in, 
                     P = P_in,
                     ecc = sys_d['planet']['ecc'], 
                     inc = sys_d['planet']['inc'], 
                     longnode = sys_d['planet']['long_node'],
                     argperi = sys_d['planet']['argperi'], 
                     tperi = t_peri, 
                     mplan = sys_d['planet']['mass'],
                     radius = sys_d['planet']['radius'],
                     albedo_wavelens = sys_d['planet']['albedo_wavelens'], 
                     albedo_vals = sys_d['planet']['albedo_vals'])
    
    planetlist = [planet]
    
    # Get "true" astrometry data
    astrom_truth_path = os.path.join(sys_dir,f"ephem_{system_name}_best_epochs_v{version}.csv") # This will be the filename even if < 3 good epochs were found
    if os.path.exists(astrom_truth_path):
        astrom_truth_df = pd.read_csv(astrom_truth_path)
    else:
        print("ASTROMETRY TRUTH FILE IS MISSING!")
        return np.nan
    
    
    # Get returned astrometry data
    astrom_returned_path = os.path.join(sys_dir,system_name + f'_retrieved_astrometry_PoissonNoise{poisson_noise}_SNR{SNR}_v{version}.csv') 
    astrom_returned_df = pd.read_csv(astrom_returned_path,index_col=[0])
    astrom_returned_df['sys_inc'] = [system_name] * (len(astrom_returned_df))
    astrom_returned_df['ep_ind'] = range(len(astrom_returned_df))
    
    # Re-order columns
    cols = astrom_returned_df.columns.tolist()
    astrom_returned_df = astrom_returned_df[cols]
    
    # Calculate error
    if len(astrom_returned_df) != len(astrom_truth_df):
        print('Error:',system_name, "- NaN returned")
        return np.nan
    
    astrom_returned_df['ra_mas_true'] = -astrom_truth_df['delta x (mas)']
    astrom_returned_df['dec_mas_true'] = astrom_truth_df['delta y (mas)']
    sep_mas_true , pa_deg_true = radec2seppa(astrom_returned_df['ra_mas_true'],astrom_returned_df['dec_mas_true'])
    astrom_returned_df['sep_mas_true'] = sep_mas_true
    astrom_returned_df['pa_deg_true'] = pa_deg_true
                        
    err_mas = np.sqrt((astrom_returned_df.ra_mas - astrom_returned_df.ra_mas_true) ** 2 + (astrom_returned_df.dec_mas - astrom_returned_df.dec_mas_true) ** 2)
    astrom_returned_df['astrom_err_mas'] = err_mas
    
    cols = ['sys_inc', 'ep_ind','SNR','version', 't_yr', 'ra_mas','ra_mas_true', 'dec_mas', 'dec_mas_true', 'sep_mas', 'sep_mas_true', 'pa_deg', 'pa_deg_true', 'astrom_err_mas','uncertainty_mas']

    astrom_returned_df = astrom_returned_df[cols]
    
    data_arr = astrom_returned_df.to_numpy()
    
    pl_results_df = pd.DataFrame(data_arr,columns=cols)
    pl_results_df.set_index(['sys_inc','ep_ind','SNR','version'],inplace=True)
    pl_results_df.to_csv(os.path.join(sys_dir,f"planet_astrometry_results_PoissonNoise{poisson_noise}_SNR{SNR}_v{version}.csv"))
    
    return [pl_results_df,astrom_returned_path]
        
    
def retrieve_orbit(sys_d,version,pl_list,system_name,sys_dir,rvdict,SNR,poisson_noise=True,joint_RV_fit=False,apply_priors=True,fit_rel_astrom=True,show_plots=True,verbose=False):

    st_name = sys_d['star']['name']
    
    # Assign variables:
    ecc_prior = sys_d['ecc_prior']
    num_temps = 20
    num_walkers = 24
    num_threads = mp.cpu_count()
    num_steps = sys_d['num_steps']
    burn_steps = 0
    burn_steps_post = num_steps//4
    thin = 2
    num_planets = 1 # Default number of planets that orbitize will fit
    num_target = 1 # Which planet is the target planet we are trying to image

    def make_input_table():
        print()
        print('\tLoading input astrometry data...')

        # Format epochs as Modified Julian Date
        epochs_yrs = np.array(astrom_df.t_yr)
        global epochs
        epochs = astropy.time.Time(epochs_yrs,format='decimalyear')
        epochs.format = 'mjd'

        # Compute sep, pa from ra, dec
        sep = astrom_df.sep_mas
        pa = astrom_df.pa_deg
        err = np.array(astrom_df.uncertainty_mas)

        # Start an empty table
        astrom_orbitize_table = astropy.table.Table()

        # Create columns
        epoch_col = astropy.table.Table.Column(name = 'epoch', data = epochs.value, dtype = float)
        object_col = astropy.table.Table.Column(name = 'object', data = [num_target] * len(epochs), dtype = int)
        sep_col = astropy.table.Table.Column(name = 'quant1', data = np.array(sep), dtype = float)
        sep_err_col = astropy.table.Table.Column(name = 'quant1_err', data = err, dtype = float)
        pa_col = astropy.table.Table.Column(name = 'quant2', data = np.array(pa), dtype = float)
        pa_err_col = astropy.table.Table.Column(name = 'quant2_err', data = np.rad2deg(np.arctan(err / np.array(sep))), dtype = float)
        quant_type_col = astropy.table.Table.Column(name = 'quant_type', data = ['seppa'] * len(epochs), dtype = str)

        # Add each column to the table
        astrom_orbitize_table.add_column(epoch_col, index = 0)
        astrom_orbitize_table.add_column(object_col, index = 1)
        astrom_orbitize_table.add_column(sep_col, index = 2)
        astrom_orbitize_table.add_column(sep_err_col, index = 3)
        astrom_orbitize_table.add_column(pa_col, index = 4)
        astrom_orbitize_table.add_column(pa_err_col, index = 5)
        astrom_orbitize_table.add_column(quant_type_col, index = 6)

        rv_input_fname = '_'.join(st_name.split(' ')) + '_rv.csv'
        rv_path = os.path.join('RV',rv_input_fname)

        if joint_RV_fit:
            if os.path.exists(rv_path):
                print(f'\tLoading input RV data for {st_name}...') 

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

        print('\tOrbitize input table compiled.')    

        # Write the input table to a file:
        global orbitize_input_table_filename
        orbitize_input_table_filename = os.path.join(sys_dir,
                                                 f"orbitize-input_Astrom{fit_rel_astrom}_RV{joint_RV_fit}_PoissonNoise{poisson_noise}_SNR{SNR}_v{version}.csv")
        print(f'\tInput table filename: {orbitize_input_table_filename}')
        orbitize.read_input.write_orbitize_input(orbitize_input_table,
                                                 orbitize_input_table_filename,
                                                 file_type = 'csv'
                                                 )
        return orbitize_input_table

    def run_MCMC(rvdict):
        print('\n\tCollecting MCMC input parameters...')

        # Define star/system parameters
        plx = sys_d['star']['plx'].value
        plx_err = np.mean([sys_d['star']['plx_err'][0].value,sys_d['star']['plx_err'][1].value])      
        system_mass = ((sys_d['star']['mass'])).to(u.solMass).value
        mass_err = (np.mean([sys_d['star']['mass_err'][0].value,sys_d['star']['mass_err'][1].value])*u.kg).to(u.solMass).value
        det_type = sys_d['det_type']

        # Read data table
        data_table = orbitize.read_input.read_file(orbitize_input_table_filename)

        # Initialize the system
        print('\n\tInitializing MCMC system...')
        global sys 
        sys = orbitize.system.System(num_planets,
                                     data_table,
                                     system_mass,
                                     plx,
                                     mass_err=mass_err,
                                     plx_err=plx_err,
                                     fit_secondary_mass=joint_RV_fit
                                    )


        ## Configure priors
        if apply_priors:   
            print('\tApplying RV priors...')

            for i in range(1,num_planets+1):

                # Constrain parameters for target planet
                if i == num_target:

                    # Constrain by period if discovered by radial velocity, otherwise use sma
                    if det_type == 'Radial Velocity':

                        P_mean = sys_d['planet']['period']
                        P_sig = np.mean([sys_d['planet']['period_err'][0].value,sys_d['planet']['period_err'][1].value]) * u.day

                        sma_min = (((P_mean-P_sig) ** 2 * c.G * (system_mass-mass_err) * u.M_sun / (4 * np.pi ** 2)) ** (1./3)).to(u.AU)
                        sma_max = (((P_mean+P_sig) ** 2 * c.G * (system_mass+mass_err) * u.M_sun / (4 * np.pi ** 2)) ** (1./3)).to(u.AU)

                        sma_mean = np.mean([sma_min.value,sma_max.value]) * u.AU
                        sma_sig = ((sma_max - sma_min)/2)

                    else:
                        sma_mean = sys_d['planet']['sma_au'] * u.AU
                        sma_sig = np.mean([sys_d['planet']['sma_au_err'][0].value,sys_d['planet']['sma_au_err'][1].value]) * u.AU

                    print(f'\t\tsma{num_target} mean: {sma_mean}')
                    print(f'\t\tsma{num_target} sigma: {sma_sig}')

                    sys.sys_priors[sys.param_idx['sma'+str(i)]] = orbitize.priors.GaussianPrior(sma_mean.value, sma_sig.value)

                    if ecc_prior == 'gaussian':
                        # Exoplanet archives constrain the eccentricity.  
                        ecc_mean = sys_d['planet']['ecc']
                        ecc_sig = np.mean([sys_d['planet']['ecc_err'][0],sys_d['planet']['ecc_err'][1]])

                        print(f'\t\tecc{num_target} mean: {ecc_mean}')
                        print(f'\t\tecc{num_target} sigma: {ecc_sig}')

                        sys.sys_priors[sys.param_idx['ecc'+str(i)]] = orbitize.priors.GaussianPrior(ecc_mean, ecc_sig,
                                                                                                        no_negatives=True)
                    else:
                        sys.sys_priors[sys.param_idx['ecc'+str(i)]] = orbitize.priors.UniformPrior(0.001, 0.999)

                        aop_mean = np.deg2rad((sys_d['planet']['argperi']).value)
                        aop_sig = np.deg2rad(np.mean([sys_d['planet']['argperi_err'][0].value,sys_d['planet']['argperi_err'][1].value]))
                        sys.sys_priors[sys.param_idx['aop'+str(i)]] = orbitize.priors.GaussianPrior(aop_mean, aop_sig)

                    # The position angle of the ascending node is constrained by comparing the sign of the RV signal
                    # at the time of the planet imaging detection, and the projected position of the planet.
                    # Crudely, we can restrict the ascending node position angle to the range 180 and 360 deg.
                    pan_min = np.deg2rad(180.0)
                    pan_max = np.deg2rad(360.0)  
                    sys.sys_priors[sys.param_idx['pan'+str(i)]] = orbitize.priors.UniformPrior(pan_min, pan_max)   

                    if joint_RV_fit:
                        if not np.isnan(sys_d['planet']['msini']):
                            mass_min = (sys_d['planet']['msini'] - sys_d['planet']['mass_err'][0]).value / (c.M_sun.to(u.M_jup).value)
                            mass_max = (0.075 * c.M_sun).to(u.M_jup).value / (c.M_sun.to(u.M_jup).value)

                            sys.sys_priors[sys.param_idx['m'+str(i)]] = orbitize.priors.UniformPrior(mass_min, mass_max)

                        else:
                            mass_mean = (sys_d['planet']['msini']).value / (c.M_sun.to(u.M_jup).value)
                            mass_sig = (np.mean([sys_d['planet']['mass_err'][0].value,sys_d['planet']['mass_err'][1].value])) / (c.M_sun.to(u.M_jup).value)

                            sys.sys_priors[sys.param_idx['m'+str(i)]] = orbitize.priors.GaussianPrior(mass_mean, mass_sig)

                # Fix parameters for all non-target planets
                else:
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
        sampler_func = getattr(orbitize.sampler, "MCMC")
        global mcmc_sampler
        mcmc_sampler = sampler_func(sys,num_temps,num_walkers,num_threads)

        # Print priors and fixed parameters
        print('\tSys param priors:')
        for lab in sys.labels:
            try:
                print(f'\t\t{lab}:', vars(sys.sys_priors[sys.param_idx[lab]]))
            except:
                print(f'\t\t{lab}:', sys.sys_priors[sys.param_idx[lab]])

        # Run MCMC!
        print(f'\n\tRunning MCMC for {num_steps} steps...')
        mcmc_sampler.run_sampler(total_orbits=num_walkers * num_steps, burn_steps=0, thin=thin)

        global results
        results = mcmc_sampler.results
        
    def make_results_df():
        print('\tExploring Results...')

        # First analyze all steps, then chop the first 1000 for burn-in (divide by 2 b/c of thinning)
        for b_in in [0,burn_steps_post]:
            filetag = f'_NumSteps{num_steps}_BurnIn{b_in}_RVFit{joint_RV_fit}_RelAstromFit{fit_rel_astrom}_PNoise{poisson_noise}_SNR{SNR}'

            mcmc_sampler.chop_chains(b_in,trim=0)
            results = mcmc_sampler.results

            # Make an empty dataframe:
            q = [0.0225, 0.16, 0.5, 0.84, 0.9775]
            tuples = ((system_name,q[0]),(system_name,q[1]),(system_name,q[2]),(system_name,q[3]),(system_name,q[4]))
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
                        mtot = df_run['m0'][(system_name,m_quant)] * u.M_sun
                    except:
                        pass
                    if np.isnan(mtot):
                        mtot = df_run['mtot'][(system_name,m_quant)] * u.M_sun

                    sma = np.median(df_run['sma'+ str(planet_num)][(system_name,quant)]) * u.AU
                    per = np.sqrt( 4 * np.pi**2 / (c.G * mtot) * sma**3 ).to(u.day) 

                    df_run['per'+str(planet_num)][(system_name,quant)] = per.value

            # Convert inc to deg
            df_run.inc1 = np.rad2deg(df_run.inc1)
                
            df_run['version'] = version

            # Save results
            print('\tSaving results object and DF summary')
            fpath=os.path.join(sys_dir,f'orbitize_results{filetag}_v{version}.hdf5')
            results.save_results(fpath)
            fpath=os.path.join(sys_dir,f'orbitize_results{filetag}_v{version}.csv')
            df_run.to_csv(fpath)


        return df_run
    
    
    # Get astrometry data:    
    astrom_data_pname = os.path.join(sys_dir,system_name + f'_retrieved_astrometry_PoissonNoise{poisson_noise}_SNR{SNR}_v{version}.csv')
    if os.path.exists(astrom_data_pname):
        astrom_df = pd.read_csv(astrom_data_pname,index_col=0)
    else:
        print(astrom_data_pname, 'NOT FOUND')

    # Get RV data
    rv_available = False # Initialize boolean, then search for rv data
    if joint_RV_fit:
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
    orbitize_input_table = make_input_table()

    # Run the MCMC algorithm using the settings above
    t_0 = datetime.datetime.now() # Record time at beginning 
    print(f"\t\tStart time: {t_0}")
    run_MCMC(rvdict)

    t_f = datetime.datetime.now() # Record time at end to determine runtime
    print(f"\t\tEnd time: {t_f}")
    print(f"\t\tRuntime: {t_f-t_0}")

    # Summarize and save the data in a dataframe
    results_df = make_results_df()

    return [results_df,results,sys]

def explore_orbitize_results(sys_d, version,pl_list, system_name, sys_dir, rvdict, SNR, sys=None, results=None, poisson_noise=True, PSF_fit=True,joint_RV_fit=False, apply_priors=True, fit_rel_astrom=True, show_plots=True,eps=None,verbose=False):
 
    def init_sampler():
        print('\n\tCollecting MCMC input parameters...')

        # Initialize system parameters
        star_name = st_name
        plx = sys_d['star']['plx'].value
        plx_err = sys_d['star']['plx_err'][0].value
        system_mass = sys_d['star']['mass'].to(u.M_sun).value # [Msol]
        mass_err = sys_d['star']['mass_err'][0].to(u.M_sun).value # [Msol]

        # Read input table
        data_table = orbitize.read_input.read_file(orbitize_input_table_filename)

        # Set up system
        global sys 
        sys = orbitize.system.System(num_planets,
                                     data_table,
                                     system_mass,
                                     plx,
                                     mass_err=mass_err,
                                     plx_err=plx_err)

        # Configure MCMC
        total_orbits = num_walkers * num_steps # number of steps x number of walkers (at lowest temperature)

        global my_driver
        global mcmc_sampler

        print('\tInitializing MCMC Sampler')
        if joint_RV_fit:
            num_secondary_bodies = num_planets # There are 2 planets in the RV signal
            my_driver = driver.Driver(
                    orbitize_input_table_filename, 'MCMC', num_secondary_bodies, 
                    system_mass, plx, mass_err=mass_err, plx_err=plx_err,
                    system_kwargs = {'fit_secondary_mass':True}, #, 'tau_ref_epoch':0},
                    mcmc_kwargs={'num_temps': num_temps, 'num_walkers': num_walkers, 'num_threads': num_threads})
            mcmc_sampler = my_driver.sampler
            sys = mcmc_sampler.system

        else:
            mcmc_sampler = orbitize.sampler.MCMC(sys, num_temps, num_walkers, num_threads)
            
        return [mcmc_sampler, sys]

    def plot_sky(): # Show the astrometry data & orbit fit on sky
        print('\n\tPlotting astrometry...')

        # Start a plot
        fig,ax = plt.subplots(figsize=(8,8)) 

        ## Plot astrometry

        # Plot the star
        ax.scatter([0], [0], marker='*', s=50)

        # Iterate through planets
        for pp,planet in enumerate(planets):

            # Get the planet num for the results dataframe:
            pl_num = sys_d[planet]

            # Get the input data for one planet:
            pl_df = orbitize_input_df.loc[pp+1,:].copy()

            # Calculate planet ra/dec from the sep/pa
            ra_true = np.array(pl_df.quant1) * np.sin(np.radians(pl_df.quant2))
            dc_true = np.array(pl_df.quant1) * np.cos(np.radians(pl_df.quant2))

            ax.scatter(ra_true, dc_true, marker='x', s=50)

            ## Plot median orbit

            quant = [0.5000]

            # Define orbital parameters
            per = float(results_df.loc[quant,'per'+str(pl_num)])
            sma = float(results_df.loc[quant,'sma'+str(pl_num)])
            ecc = float(results_df.loc[quant,'ecc'+str(pl_num)])
            inc = float(results_df.loc[quant,'inc'+str(pl_num)])
            aop = float(results_df.loc[quant,'aop'+str(pl_num)])
            pan = float(results_df.loc[quant,'pan'+str(pl_num)])
            tau = float(results_df.loc[quant,'tau'+str(pl_num)])
            plx = float(results_df.loc[quant,'plx'])
            mtot = float(results_df.loc[quant,'m0']) #+ results_df.loc[quant,'m'+str(pl_num)])
            mplanet = float(results_df.loc[quant,'m'+str(pl_num)])

            # Number of epochs at which to solve the orbits
            epochs = np.linspace(0, per, 200)

            # Calculate orbit
            ra, dec, rv = orbitize.kepler.calc_orbit(epochs, sma, ecc, np.radians(inc),
                                            np.radians(aop), np.radians(pan), tau, plx, mtot, 
                                            mass_for_Kamp=mplanet, tau_warning=False)

            # Create line collection to plot
            ra0 = ra
            dec0 = dec
            points = np.array([ra, dec]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, color='k')

            # Plot the true orbital track
            ax.add_collection(lc)

        ## Modify plot

        ax.set_aspect('equal') 
        max_abs = np.max([np.nanmax(np.absolute(ra_true)),np.nanmax(np.absolute(dc_true))])
        buff = 200
        plt.xlim(max_abs+buff, -(max_abs+buff))
        plt.ylim(-(max_abs+buff), max_abs+buff)
        plt.xlabel('RA offset')
        plt.ylabel('Dec offset')


        # Show and save the plot
        if show_plots:
            plt.show()
        fpath = os.path.join(sys_dir,f'returned_astrometry_PNoise{poisson_noise}_v{version}.png')
        plt.savefig(fpath)
        plt.close()

    def plot_RVs(results_obj, object_to_plot=1,
                num_orbits_to_plot=100, num_epochs_to_plot=100):

        if object_to_plot > results_obj.num_secondary_bodies:
            raise ValueError("Only {0} secondary bodies being fit. Requested to plot body {1} which is out of range".format(results_obj.num_secondary_bodies, object_to_plot))

        fig,ax1 = plt.subplots(1,num=f'RV',figsize=(8,4),dpi=300)
        ax1.set_ylabel('RV [km/s]')
        ax1.set_xlabel('Epoch')

        # Plot true RV data
        rv_df = orbitize_input_df.loc[0].copy()
        rv_df.reset_index(inplace=True)
        rv_df.columns = ['epoch_mjd','rv','err','q2','q2err','qtype']
        rv_df.loc[:,'epoch_yr'] = Time(rv_df.epoch_mjd,format='mjd').decimalyear
        ax1.scatter(rv_df.epoch_yr,rv_df.rv,marker='x',c='k',label='rv data',zorder = 5,linewidths=1)
        #ax1.errorbar(rv_df.epoch_yr,rv_df.rv,yerr=rv_df.err,zorder=5,color='k',label='rv data',fmt='none',elinewitdh=1)
        dict_of_indices = {
            'sma': 0,
            'ecc': 1,
            'inc': 2,
            'aop': 3,
            'pan': 4,
            'tau': 5,
            'plx': 6 * results_obj.num_secondary_bodies,
        }
        epochs_seppa = np.zeros((num_orbits_to_plot, num_epochs_to_plot))

        start_index = (object_to_plot - 1) * 6

        sma = results_obj.post[:, start_index + dict_of_indices['sma']]
        ecc = results_obj.post[:, start_index + dict_of_indices['ecc']]
        inc = results_obj.post[:, start_index + dict_of_indices['inc']]
        aop = results_obj.post[:, start_index + dict_of_indices['aop']]
        pan = results_obj.post[:, start_index + dict_of_indices['pan']]
        tau = results_obj.post[:, start_index + dict_of_indices['tau']]
        plx = results_obj.post[:, dict_of_indices['plx']]

        # Then, get the other parameters
        if 'mtot' in results_obj.labels:
            mtot = results_obj.post[:, -1]
        elif 'm0' in results_obj.labels:
            m0 = results_obj.post[:, -1]
            m1 = results_obj.post[:, -(results_obj.num_secondary_bodies+1) + (object_to_plot-1)]
            mtot = m0 + m1
        if 'gamma' in results_obj.labels:
            dict_of_indices['gamma'] = 6 * results_obj.num_secondary_bodies + 1
            dict_of_indices['sigma'] = 6 * results_obj.num_secondary_bodies + 2
            gamma = results_obj.post[:, dict_of_indices['gamma']]

        # Select random indices for plotted orbit
        if num_orbits_to_plot > len(sma):
            num_orbits_to_plot = len(sma)
        choose = np.random.randint(0, high=len(sma), size=num_orbits_to_plot)

        raoff = np.zeros((num_orbits_to_plot, num_epochs_to_plot))
        deoff = np.zeros((num_orbits_to_plot, num_epochs_to_plot))
        vz_star = np.zeros((num_orbits_to_plot, num_epochs_to_plot))
        epochs = np.zeros((num_orbits_to_plot, num_epochs_to_plot))

        start_mjd = rv_df.epoch_mjd.iloc[0]
        if verbose:
            print('start_mjd:',start_mjd)
        sep_pa_end_year = rv_df.epoch_yr.iloc[-1]
        if verbose:
            print('sep_pa_end_year:',sep_pa_end_year)

        for i in np.arange(num_orbits_to_plot):

            orb_ind = choose[i]

            if ecc[orb_ind] > 1:
                continue

            epochs_seppa[i, :] = np.linspace(
                start_mjd,
                Time(sep_pa_end_year, format='decimalyear').mjd,
                num_epochs_to_plot
            )
            raoff0, deoff0, vzoff0 = kepler.calc_orbit(
                epochs_seppa[i, :], sma[orb_ind], ecc[orb_ind], inc[orb_ind], aop[orb_ind], pan[orb_ind],
                tau[orb_ind], plx[orb_ind], mtot[orb_ind], tau_ref_epoch=results_obj.tau_ref_epoch,
                mass_for_Kamp=m0[orb_ind],tau_warning=False
            )

            raoff[i, :] = raoff0
            deoff[i, :] = deoff0
            vz_star[i, :] = vzoff0*-(m1[orb_ind]/m0[orb_ind]) + gamma[orb_ind]

            yr_epochs = Time(epochs_seppa[i, :], format='mjd').decimalyear
            plot_epochs = np.where(yr_epochs <= sep_pa_end_year)[0]
            yr_epochs = yr_epochs[plot_epochs]
            if i==1:
                ax1.plot(yr_epochs, vz_star[i, :], color='purple', alpha=0.3,zorder=3, label='orbitize model') # 
            else:
                ax1.plot(yr_epochs, vz_star[i, :], color='purple', alpha=0.3,zorder=3) # , label='orbitize model'


        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        max_rv = np.nanmax(rv_df.rv)
        if verbose:
            print('Max RV:',max_rv)
        #plt.tight_layout()
        ax1.set_ylim(-max_rv*3,max_rv*3)
        if show_plots:
            plt.show()
        v=0
        fpath = os.path.join(sys_dir,f'RV_plot_PNoise{poisson_noise}_v{version}.png')
        plt.savefig(fpath) 
        plt.close()

    def explore_results(sys,rvdict,orbit=True,chains=True,eps=eps,burn_in=0):
        print('\tExploring Results...')

        # Make an empty dataframe:
        q = ["truth",0.0225, 0.16, 0.5, 0.84, 0.9775]
        res_dict = {'quantile': q}
        
        if eps==None or sys==None:
            print('No epochs or sys found in check at beginning of explore_results routine.')
            for param in ['sma1','ecc1','inc1','aop1','pan1','tau1','plx','m1','m0','run_params']:
                res_dict[param] = [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]
            run_df = pd.DataFrame.from_dict(res_dict)
            run_df.set_index('quantile',inplace=True)
            
            # Get true value 
            if 'inc' in param:
                true_val = np.deg2rad(sys_d['planet']['inc'].value)
            elif param == 'm0' or param == 'mtot':
                true_val = sys_d['star']['mass'].to(u.M_sun).value
            elif param == 'plx':
                true_val = sys_d['star']['plx'].value
            elif param[-1] == str(num_target):
                if param[0] == 's':
                    true_val = sys_d['planet']['sma_au'].value
                elif param[0] == 'e':
                    true_val = sys_d['planet']['ecc']
                elif param[0] == 'm':
                    true_val = sys_d['planet']['mass'].value / (c.M_sun.to(u.M_jup).value)
                elif param[0] == 'a':
                    true_val = np.deg2rad(sys_d['planet']['argperi'].value)
                elif param[0] == 't':
                    true_val = orbitize.basis.t0_to_tau(sys_d['planet']['tperi']-2400000.5, 58849, sys_d['planet']['period'].value/365.25)
                elif param[0] == 'p':
                    true_val = np.deg2rad(sys_d['planet']['long_node'].value)
            else:
                true_val = np.nan
                pl_num = param[-1]
                if param[0] == 's':
                    true_val = rvdict[st_name]['pl_list'][pl_num]['sma'].value
                elif param[0] == 'e':
                    true_val = rvdict[st_name]['pl_list'][pl_num]['ecc']
                elif param[0] == 'm':
                    true_val = rvdict[st_name]['pl_list'][pl_num]['m'].value
                
            run_df.loc["truth",param] = true_val
            
        else:
            print("sys & results exist.")
            for param, indx in sys.param_idx.items():
                measurements = [x[indx] for x in mcmc_sampler.results.post]
                quants = np.quantile(measurements, [0.0225, 0.16, 0.5, 0.84, 0.9775])
                res_dict[param] = [np.nan, *quants]

            run_df = pd.DataFrame.from_dict(res_dict)
            run_df.set_index('quantile',inplace=True)

            # Examine chains

            print('\tExamining chains...')

            params = []
            i = 1
            while i <= num_planets:
                params.extend(['sma'+str(i),'ecc'+str(i),'inc'+str(i),'aop'+str(i),'tau'+str(i)])
                if joint_RV_fit:
                    params.append('m'+str(i))
                i += 1
            if joint_RV_fit:
                params.append('m0')
            else:
                params.append('mtot')

            print('\t\tParam list:',params)

            for param in params:

                # Get true value 
                if 'inc' in param:
                    true_val_hist = sys_d['planet']['inc'].value
                    true_val = np.deg2rad(sys_d['planet']['inc'].value)
                elif param == 'm0' or param == 'mtot':
                    true_val = sys_d['star']['mass'].to(u.M_sun).value
                    true_val_hist = true_val
                elif param == 'plx':
                    true_val = sys_d['star']['plx'].value
                    true_val_hist = true_val
                elif param[-1] == str(num_target):
                    if param[0] == 's':
                        true_val = sys_d['planet']['sma_au'].value
                        true_val_hist = true_val
                    elif param[0] == 'e':
                        true_val = sys_d['planet']['ecc']
                        true_val_hist = true_val
                    elif param[0] == 'm':
                        true_val = sys_d['planet']['mass'].value / (c.M_sun.to(u.M_jup).value)
                        true_val_hist = sys_d['planet']['mass'].value
                    elif param[0] == 'a':
                        true_val = np.deg2rad(sys_d['planet']['argperi'].value)
                        true_val_hist = sys_d['planet']['argperi'].value
                    elif param[0] == 't':
                        true_val = orbitize.basis.t0_to_tau(sys_d['planet']['tperi']-2400000.5, 58849, sys_d['planet']['period'].value/365.25)
                        true_val_hist = true_val
                    elif param[0] == 'p':
                        true_val = np.deg2rad(sys_d['planet']['long_node'].value)
                        true_val_hist = true_val
                else:
                    pl_num = param[-1]
                    if param[0] == 's':
                        true_val = rvdict[st_name]['pl_list'][pl_num]['sma'].value
                        true_val_hist = true_val
                    elif param[0] == 'e':
                        true_val = rvdict[st_name]['pl_list'][pl_num]['ecc']
                        true_val_hist = true_val
                    elif param[0] == 'm':
                        true_val = rvdict[st_name]['pl_list'][pl_num]['m'].value
                        true_val_hist = true_val

                run_df.loc["truth",param] = true_val

                if chains:
                    chains_plots = mcmc_sampler.examine_chains(param_list=[param], n_walkers=20)
                    chains_plot = chains_plots[0]
                    chains_axes = chains_plot.axes
                    chains_ax = chains_axes[0]
                    plot_steps = num_steps//2 - b_in

                    label_val = np.round(true_val,6) if not (param[0]=='m' and param[-1]!='0' and param[-1]!='t') else np.round(true_val * 1047.57,6)
                    chains_ax.plot([true_val]*int(plot_steps),label=f"True Value = {label_val}",linestyle='-',zorder=6)


                    low_val = run_df[param][0.1600]
                    label_val = np.round(low_val,6) if not (param[0]=='m' and param[-1]!='0' and param[-1]!='t') else np.round(low_val,6) * 1047.57
                    chains_ax.plot([low_val]*int(plot_steps),label=f"Quantile 0.16 = {np.round(label_val,6)}",linestyle=':')

                    mean_val = run_df[param][0.5000]
                    label_val = np.round(mean_val,6) if not (param[0]=='m' and param[-1]!='0' and param[-1]!='t') else np.round(mean_val,6) * 1047.57
                    chains_ax.plot([mean_val]*int(plot_steps),label=f"Median Value = {np.round(label_val,6)}",linestyle='--')

                    high_val = run_df[param][0.8400]
                    label_val = np.round(high_val,6) if not (param[0]=='m' and param[-1]!='0' and param[-1]!='t') else np.round(high_val,6) * 1047.57
                    chains_ax.plot([high_val]*int(plot_steps),label=f"Quantile 0.84 = {np.round(label_val,6)}",linestyle=':')        

                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
                    chains_ax.set_title(f"Chains for {param}:") # \n{num_steps}s {thin}th {b_in}b {trim}tr.        

                    if show_plots:
                        plt.show()
                    plt.close()
                    
                    single_hist = results.plot_corner(param_list=[param],truths=[true_val_hist])
                    if show_plots:
                        plt.show()
                    fpath = os.path.join(sys_dir,f"{param}_hist_{filetag}_v{version}.png")
                    plt.savefig(fpath)
                    plt.close()
                    
            # Plot corner figure    
            print('\tPlotting corner figures...')

            for pp,planet in enumerate(planets):
                planet_num = pp+1

                '''# Calculate period
                quants = [0.0225, 0.16, 0.5, 0.84, 0.9775]
                for quant in quants:    
                    m_quant = 1-quant
                    try:
                        mtot = df_run['m0'][(run_name,m_quant)] * u.M_sun
                    except:
                        pass
                    if np.isnan(mtot):
                        mtot = df_run['mtot'][(run_name,m_quant)] * u.M_sun
                    sma = np.median(df_run['sma'+ str(planet_num)][(run_name,quant)]) * u.AU
                    per = np.sqrt( 4 * np.pi**2 / (c.G * mtot) * sma**3 ).to(u.day)
                    df_run['per'+str(planet_num)][(run_name,quant)] = per.value'''

                # Plot corner plot
                plt.style.use('bmh')
                if joint_RV_fit:
                    params = ['sma'+str(planet_num),
                              'ecc'+str(planet_num),
                              'inc'+str(planet_num),
                              'm'+str(planet_num)]
                    truths = [sys_d['planet']['sma_au'].value,
                              sys_d['planet']['ecc'],
                              sys_d['planet']['inc'].value,
                              sys_d['planet']['mass'].value
                             ]
                    """params = ['sma'+str(planet_num),
                              'ecc'+str(planet_num),
                              'inc'+str(planet_num),
                              'aop'+str(planet_num),
                              'pan'+str(planet_num),
                              'tau'+str(planet_num),
                              'm'+str(planet_num),
                              'plx']
                    truths = [sys_d['planet']['sma_au'].value,
                              sys_d['planet']['ecc'],
                              sys_d['planet']['inc'].value,
                              sys_d['planet']['argperi'].value,
                              sys_d['planet']['long_node'].value,
                              orbitize.basis.t0_to_tau(sys_d['planet']['tperi']-2400000.5, 58849, sys_d['planet']['period'].value/365.25),
                              sys_d['planet']['mass'].value,
                              sys_d['star']['plx'].value
                             ]"""
                    if not fix_m0:
                        params.append('m0')
                        truths.append(sys_d['star']['mass'].to(u.M_sun).value)
                else: 
                    params = ['sma'+str(planet_num),
                              'ecc'+str(planet_num),
                              'inc'+str(planet_num),
                              'mtot']
                    truths = [sys_d['planet']['sma_au'].value,
                              sys_d['planet']['ecc'],
                              sys_d['planet']['inc'].value,
                              sys_d['star']['mass'].to(u.M_sun).value
                             ]

                #median_vals = np.median(results.post,axis=0) # Median of each parameter
                #range_vals = np.ones_like(median_vals)*0.95
                corner_figure = results.plot_corner(param_list=params,truths=truths,quiet=True)

                if show_plots:
                    plt.show()
                    
                fpath = os.path.join(sys_dir,f"corner_{filetag}_v{version}.png")
                plt.savefig(fpath) 
                plt.close()


            # Plot orbit results on-sky
            if orbit:
                print('\tPlotting orbit results on sky...')
                orbit_fig = results.plot_orbits(rv_time_series=False,start_mjd=61041.,
                        num_orbits_to_plot=100, num_epochs_to_plot=100,sep_pa_end_year=2036.0)
                ax_orb, ax_cbar, ax_sep, ax_pa  = orbit_fig.axes

                # Plot the star
                ax_orb.scatter([0], [0], marker='*', s=100,c='k')

                if fit_rel_astrom:
                    # Iterate through planets
                    print('Planets:',planets)

                    for pp in planets:

                        # Get the input data for one planet:
                        pl_df = orbitize_input_df.loc[pp,:].copy()
                        pl_df.reset_index(inplace=True)
 
                        # Calculate planet ra/dec from the sep/pa
                        ra_true = np.array(pl_df.quant1) * np.sin(np.radians(pl_df.quant2))
                        dc_true = np.array(pl_df.quant1) * np.cos(np.radians(pl_df.quant2))
                        yrs = Time(pl_df.epoch,format='mjd').decimalyear

                        ax_orb.scatter(ra_true, dc_true, marker='X', s=50,zorder=6,c='w',edgecolors='k')
                        ax_sep.scatter(yrs, pl_df.quant1, marker='X', s=50,zorder=6,c='w',edgecolors='k')
                        ax_pa.scatter(yrs, pl_df.quant2, marker='X', s=50,zorder=6,c='w',edgecolors='k')

                if show_plots:
                    plt.show()
                    
                v=0
                fpath = os.path.join(sys_dir,f"orbit_{filetag}_v{version}.png")
                plt.savefig(fpath)
                plt.close()

                if joint_RV_fit:
                    # Iterate through planets and plot RV 
                    for planet in planets: 
                        plot_RVs(results,object_to_plot=planet) 
            
            data_df = pd.read_csv(orbitize_input_table_filename)
            data_df = (data_df[data_df['object']==1])
            data_df['tau'] = (data_df['epoch'] - data_df['epoch'].iloc[0]) / sys_d['planet']['period'].to(u.day).value % 1
            taus = np.sort(np.array(data_df['tau']))
            coverage = 1 - np.max([*taus[1:],1] - taus)
            run_df['num_epochs'] = len(eps)
            run_df['coverage'] = coverage
            run_df['min_sep'] = np.min(data_df['quant1'])
            run_df['max_sep'] = np.max(data_df['quant1'])
            fpath = os.path.join(sys_dir,f'final_orbitize_results{filetag}_v{version}.hdf5')
            print(f'\tSaving results object to {fpath}')     
            results.save_results(fpath)

        # Convert units as needed
        for col in run_df.columns:
            if 'inc' in col:
                run_df[col] = np.rad2deg(run_df[col])
            elif 'aop' in col:
                run_df[col] = np.rad2deg(run_df[col])
            elif 'pan' in col:
                run_df[col] = np.rad2deg(run_df[col])
            elif col[0] == 'm' and col[-1] != '0':
                run_df[col] = run_df[col] * (c.M_sun.to(u.M_jup).value)

            
        run_df['star'] = st_name
        run_df['pl_letter'] = sys_d['sysname'][-1]
        run_df['inc'] = [run_df['inc'+str(num_target)]['truth'] ] * 6
        run_df['run_params'] = np.nan
        if "mtot" in run_df.columns:
            run_df['m0'] = run_df.mtot
            run_df.drop(columns=['mtot'],inplace=True)
        run_params = mcmc_dict
        run_params['burn_steps'] = burn_in
        for key,item in run_params.items():
            run_df[key] = np.nan
            if key == "planets":
                item = np.array(item)
            for q in run_df.index :
                    run_df[key][q] = item
                    
        run_df['version'] = version

        run_df.reset_index(inplace=True)
        run_df.set_index(['star','pl_letter','inc','version','quantile'],inplace=True)

        # Save results
        #filetag_temp = f'_NumSteps{num_steps}_BurnIn{b_in}_RVFit{joint_RV_fit}_RelAstromFit{fit_rel_astrom}_PNoise{Poisson_Noise}_MassPriorGaussian'
            
        fpath = os.path.join(sys_dir,f'final_orbitize_results_df{filetag}_v{version}.csv')   
        print('\tSaving run summary to',fpath)
        
        run_df.to_csv(fpath)

        return run_df

    #####################
    
    st_name = sys_d['star']['name']

    # Assign variables:
    mcmc_dict = {
        'SNR' : SNR,
        'Poisson_noise' : poisson_noise,
        'PSF_fit' : PSF_fit,
        'RV_fit' : joint_RV_fit,
        'RV_priors' : apply_priors,
        'astrom_fit' : fit_rel_astrom,
        'planets' : [1],
        'ecc_prior' : sys_d['ecc_prior'],
        'num_temps' : 20,
        'num_walkers' : 24,
        'num_threads' : mp.cpu_count(),
        'num_steps' : sys_d['num_steps'],
        'burn_steps' : 0,
        'burn_steps_post' : sys_d['num_steps'] // 4,
        'thin' : 2,
        'num_planets' : 1,
        'num_target' : 1,
        'fix_m0' : False
    }   
    
    planets = mcmc_dict['planets']
    ecc_prior = mcmc_dict['ecc_prior']
    num_temps = mcmc_dict['num_temps']
    num_walkers = mcmc_dict['num_walkers']
    num_threads = mcmc_dict['num_threads']
    num_steps = mcmc_dict['num_steps']
    burn_steps = mcmc_dict['burn_steps']
    burn_steps_post = mcmc_dict['burn_steps_post']
    thin = mcmc_dict['thin']
    num_planets = mcmc_dict['num_planets'] # Default number of planets that orbitize will fit
    num_target = mcmc_dict['num_target'] # Which planet is the target planet we are trying to image
    fix_m0 = mcmc_dict['fix_m0']
    
    Poisson_Noise = mcmc_dict['Poisson_noise']
    joint_RV_fit = mcmc_dict['RV_fit']
    apply_RV_priors = mcmc_dict['RV_priors']
    fit_rel_astrom = mcmc_dict['astrom_fit']

    filetag = f'_NumSteps{num_steps}_BurnIn{burn_steps}_RVFit{joint_RV_fit}_RelAstromFit{fit_rel_astrom}_PNoise{Poisson_Noise}_SNR{SNR}'

    # Get astrometry data:
    
    astrom_data_pname = os.path.join(sys_dir,system_name + f'_retrieved_astrometry_PoissonNoise{Poisson_Noise}_SNR{SNR}_v{version}.csv')
    #print(f"Looking for astrom data at {astrom_data_pname}")
    if os.path.exists(astrom_data_pname):
        astrom_df = pd.read_csv(astrom_data_pname,index_col=0)
    else:
        print(astrom_data_pname, 'NOT FOUND')

    # Get RV data
    rv_available = False # Initialize boolean, then search for rv data
    if joint_RV_fit:
        rv_input_fname = '_'.join(st_name.split(' ')) + '_rv.csv'
        rv_path = os.path.join('RV',rv_input_fname)
        #print(f"Looking for rv data at {rv_path}")
        
        if os.path.exists(rv_path):
            print(f'\tLoading input RV data for {st_name}...') 
            rv_data = pd.read_csv(rv_path) #, usecols = ('time','rv','err')
            rv_available = True        

            # Load RV data info
            try:
                num_planets = rvdict[st_name]['n_rv_planets']
                planets = list(range(1,num_planets+1))
                num_target = rvdict[st_name]['n_target_pl']
                num_walkers = 24 # + (4 * num_planets)
                num_steps = num_steps * num_planets
                rv_planets = rvdict[st_name]['pl_list']
            except KeyError:
                raise ValueError('RV data available for {%s} but rvdict is not properly configured for it.'.format(st_name))

        if not rv_available:
            raise ValueError('No RV data available for {%s}.'.format(st_name))
    
    # Load input data table according to parameters above
    orbitize_input_table_filename = os.path.join(sys_dir,
                                                 f"orbitize-input_Astrom{fit_rel_astrom}_RV{joint_RV_fit}_PoissonNoise{Poisson_Noise}_SNR{SNR}_v{version}.csv")
    
    #print(f"Looking for orbitize input file at {orbitize_input_table_filename}")
    if os.path.exists(orbitize_input_table_filename):
        orbitize_input_df = pd.read_csv(orbitize_input_table_filename)    
        orbitize_input_df.set_index(['object','epoch'],inplace=True)

        # Load results dataframe
        results_df = pd.read_csv(os.path.join(sys_dir,f'orbitize_results{filetag}_v{version}.csv'))
        results_df.set_index('quantile',inplace=True)

        # Load orbitize results object
        results = orbitize.results.Results()
        fname = os.path.join(sys_dir,f'orbitize_results{filetag}_v{version}.hdf5')
        results.load_results(filename=fname)

        print(f'Loaded results from {fname}')
        
        # Initialize an MCMC sampler and add results to it
        mcmc_sampler, sys = init_sampler()
        mcmc_sampler.results = results

        # Save a copy of the sampler
        sys_copy = dcopy(sys)
        sampler_copy = dcopy(mcmc_sampler)
        results_copy = sampler_copy.results

        # Reset system, sampler, & results objects, and remove chains from the beginning/end
        sys = dcopy(sys_copy)
        mcmc_sampler = dcopy(sampler_copy)
        results = mcmc_sampler.results

        # First analyze all steps, then chop the first half for burn-in (divide by 2 b/c of thinning)
        for b_in in [0,burn_steps_post]:
            filetag = f'_NumSteps{num_steps}_BurnIn{b_in}_RVFit{joint_RV_fit}_RelAstromFit{fit_rel_astrom}_PNoise{Poisson_Noise}_SNR{SNR}'

            mcmc_sampler.chop_chains(b_in,trim=0)
            results = mcmc_sampler.results

            print("sys:",sys)
            print("mcmc_sampler:",mcmc_sampler)
            print("results:",results)
            print("eps:",eps)
            
            # Explore results- look at orbit on sky, corner plots, chains, etc.
            df = explore_results(sys,rvdict,chains=True,orbit=True,eps=eps,burn_in=b_in)
    else:
        print(f"\tAstromerty input file not found at {orbitize_input_table_filename}")
        df = explore_results(rvdict,eps=eps)

    return df
