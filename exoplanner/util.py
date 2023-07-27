import numpy as np
import os
import pandas as pd
pd.options.mode.chained_assignment = None
import exoscene

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('bmh')
mpl.rcParams['image.cmap'] = 'inferno'
mpl.rcParams['font.size'] = 16
mpl.rcParams['image.origin'] = 'lower'
seafoam = '#dbfeff'

import astropy.units as u


def plot_phase_track(planet_list, astrom_eps=None, returned_astrom_eps=None,showplots=None,contrast=False,show_pm=True,title=None):
    
    """
    TO DO: 
    """
    
    def mas2pix(pt_mas):
        pt_pix = (pt_mas / pixscale_mas.value + ((imwidth-1)/2)) 
        return pt_pix
    
    if len(planet_list) < 1:
        print('No planets in planet_list')
        return None
    
    if showplots is None:
        showplots = planet_list[0].configs['showplots']
        
    obs = planet_list[0].obs
    
    # Image parameters
    imwidth = 100
    center = (imwidth-1)/2
    pixscale = (0.2 * u.AU) * (planet_list[0].dist / (17.9323 *u.pc))
    au2mas_scale = 1000 / planet_list[0].dist.value
    pixscale_mas = pixscale.value / planet_list[0].dist.value * 1000 * u.mas
    numpts = 500

    time_series = obs.t0_pm.value + np.linspace(0, obs.t_span_pm.value, numpts) # years
    
    phase_track_img = np.zeros((imwidth, imwidth))
    
    label_flag = True
    
    # Start plot
    lp_fig,ax = plt.subplots(1,figsize=(10,6),dpi=300) 
    
    cs = ['red','orange','yellow','green','cyan','blue','purple','magenta']
    
    ax.add_artist(plt.Circle((center,center),obs.iwa.value / pixscale_mas.value,fill=False,edgecolor='w',linestyle='--')) 
    ax.add_artist(plt.Circle((center,center),obs.owa.value / pixscale_mas.value,fill=False,edgecolor='w',linestyle='--'))
    
    data_tag = 'contrastlist' if contrast else 'phasefunclist'
               
    for planet in planet_list:
    
        ephem_df = planet.ephem_pm

        for tt, time in enumerate(time_series):

            res = exoscene.planet.planet_cube(imwidth, pixscale, [planet.model], epoch = time*u.year)
            #display(res)
            
            if planet.self_lum > 0:
                res['phasefunclist']= [1.]
                res['contrastlist'] = [planet.self_lum]

            # Prevent error from trying to plot points outside the image.
            if (res['coordlist'][0][0]<imwidth) and (res['coordlist'][0][1]<imwidth) and (res['coordlist'][0][0]>=0) and (res['coordlist'][0][1]>=0):                   
                if (phase_track_img[ res['coordlist'][0] ] == 0) or (res[data_tag][0] > phase_track_img[ res['coordlist'][0]]):
                    phase_track_img[ res['coordlist'][0] ] = res[data_tag][0]

            else:
                continue
            
        
        # Plot astrometry of best epochs
        if not (astrom_eps is None and returned_astrom_eps is None):

            fig_fname = f'phase_track_pm_with_astrom.pdf'

            for i,ep in enumerate(astrom_eps):

                loc_pix = mas2pix(ephem_df.loc[ep,['ra_mas','dec_mas']])
                label = f'Observation @ {np.round(ep,2)}' if label_flag else None
                ax.scatter(*loc_pix,label=label,marker='o',edgecolors= cs[i],facecolors='none',s=100,zorder=2,linewidths=2)

        # Plot returned astrometry
        if not returned_astrom_eps is None:

            fig_fname = f'phase_track_pm_with_returned_astrom.pdf'
            astrom_df = planet.post.astrometry
            
            for i,ep in enumerate(returned_astrom_eps):

                try:
                    loc_pix = mas2pix(astrom_df.loc[ep,['ra_mas','dec_mas']])
                    
                except KeyError:
                    loc_pix = [np.nan,np.nan]
                
                label = f'Returned Astrom. @ {np.round(ep,2)}' if label_flag else None
                
                ax.scatter(*loc_pix,label=label,marker='+',facecolors= cs[i],s=100,zorder=2,linewidths=2)
            
        # Plot first and last epochs in primary mission
        if show_pm:
            ep_first = planet.ephem_pm.index[0]
            loc_first = mas2pix(planet.ephem_pm.loc[ep_first,['ra_mas','dec_mas']])
            label = f'First epoch: {np.round(ep_first,2)}' if label_flag else None
            plt.scatter(*loc_first,label=label,marker='P',s=100,c='w',edgecolor='k',zorder=1.9)

            ep_last = planet.ephem_pm.index[-1]
            loc_last = mas2pix(planet.ephem_pm.loc[ep_last,['ra_mas','dec_mas']])
            label = f'Last epoch: {np.round(ep_last,2)}' if label_flag else None
            plt.scatter(*loc_last,label=label,marker='P',s=100,c='k',edgecolor='w',zorder=1.9)

        label_flag = False

    vmax = np.nanmax(phase_track_img) if contrast else 1
    plt.imshow(phase_track_img, origin='lower', interpolation='none',extent=(99, 0, 0, 99),vmin=0,vmax=vmax)
    cb_label = "Flux Contrast" if contrast else 'Relative Phase'
    plt.colorbar(label=cb_label,shrink=0.8)
    
    if title is None:
        inc_range = f'{np.round(planet_list[0].inc.value,2)}' if len(planet_list) < 2 else f'{np.round(planet_list[0].inc.value,2)} - {np.round(planet_list[-1].inc,2)}'
        title = f'Position and Lambert phase of {planet.sysname} (inc={inc_range})\nover {obs.name} primary mission'
    
    plt.title(title,fontsize='small')
    
    fig_fname = f'phase_track_pm.pdf'
    
    # Star location
    plt.scatter(center,center,label=planet.st_name,marker='*',c='w',edgecolor='k',s=100) 
    
    # Plot formatting
    plt.legend(fontsize='x-small',loc='upper left',bbox_to_anchor=(1.25,1.0))
    
    xtick_locs = (np.arange(-1000, 1000, 200) / pixscale_mas.value + (imwidth-1) / 2)
    xtick_labels = ['{:+.0f}'.format(loc) for loc in np.arange(-1000, 1000, 200)]
    plt.xticks(xtick_locs, xtick_labels, size=14) #,color=seafoam
    plt.xlim([imwidth-1,0])
    plt.xlabel('Offset from star (mas)',fontsize='x-small')
    
    plt.yticks(xtick_locs, xtick_labels, size=14) #,color=seafoam
    plt.ylim([0, imwidth-1])
    plt.tick_params('both', length=8, width=1, which='major', top=True, right=True,
                    direction='in', color=seafoam)
    
    ax.patch.set_facecolor('k')
    
    plt.tight_layout()
    
    fpath = os.path.join(planet_list[0].planet_dir,fig_fname) if len(planet_list) < 2 else os.path.join(planet_list[0].target_dir,fig_fname)
    
    plt.savefig(fpath)
    
    if showplots:
        plt.show()
    
    plt.close()

def plot_observability(targ_list,results_dir=os.path.join('.','results'),hfig=8,wfig=12):

    ts = np.array(targ_list[0].detection_prob.index)
    
    planets, prob_array = np.transpose(np.array([[targ.sysname,targ.detection_prob.detect_prob.values] for targ in targ_list],dtype='object'))
    prob_array = np.stack(prob_array)
    #display(prob_array)
    
    fig, ax = plt.subplots(1,figsize=(wfig,hfig))
    
    plt.imshow(prob_array,aspect='auto',interpolation='none',origin='upper')
    
    plt.yticks(ticks=list(range(len(prob_array))),labels=planets,fontsize='x-small')
    
    xticks = [[],[]]
    years = list(np.floor(ts))
    
    for i,ep in enumerate(years):
        if not (ep in xticks[1]):
            xticks[0].append(i)
            xticks[1].append(int(ep))
    
    plt.xticks(ticks=xticks[0],labels=xticks[1],fontsize='x-small')
    plt.colorbar(label='Detection Probability')
    
    plt.show()
    plt.close()
    
    
def check_missing_data (systems_dict):# Check again for missing data
    for planet, p_dict in systems_dict.items():
        for param, value in p_dict['star'].items():
            try:
                if np.any(np.isnan(value)):
                    print(f'Missing data: {planet}, {param}')
            except:
                pass
        for param, value in p_dict['planet'].items():
            try:
                if np.any(np.isnan(value)):
                    print(f'Missing data: {planet}, {param}')
            except:
                pass

def summarize_fits(res_dir,verbose=False):
    f_list = []
    for subdir, dirs, files in os.walk(os.path.join(res_dir,'pl_systems_incs_outputs')):
        if not ".ipynb_checkpoints" in subdir:
            #print(subdir)
            for filename in files:
                filepath = subdir + os.sep + filename

                if filename.startswith("final_orbitize_results") and filepath.endswith(".csv"):
                    f_list.append(filepath)

    f_list.sort()

    init_flag = True
    count = 0
    #print(f_list)
    
    for csv in f_list:

        df = pd.read_csv(csv)
        df['run_ID'] = count
        df.set_index(['run_ID','star','pl_letter','inc','quantile'],inplace=True) #,'version'
        count += 1

        if init_flag:
            master_df = df.copy()
            init_flag = False
        else:
            master_df = master_df.append(df)
    return master_df

