# Python Standard Library packages:
import os
from copy import deepcopy

# Other main packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.ticker import AutoMinorLocator
plt.rc('xtick', direction='in', top='on')
#plt.rc('xtick.minor', visible=True)
plt.rc('ytick', direction='in', right='on')

# Astro-packages
import astropy.units as u
from astropy.coordinates import SkyCoord

# Gaia
from astroquery.gaia import Gaia
Gaia.MAIN_GAIA_TABLE = "gaiaedr3.gaia_source" # Select Data Release 3
Gaia.ROW_LIMIT = -1 # Set the number of output raw limit to infinite

# Lightkurve
import lightkurve as lk


# https://docs.lightkurve.org/
# https://docs.lightkurve.org/tutorials/
# https://docs.lightkurve.org/reference/api/lightkurve.KeplerTargetPixelFile.html

# Define the working directory
def working_dir(maindir=None):

    if maindir is None:
        return os.getcwd()
    else:
        return maindir

maindir = working_dir()


def query_lc(ID, method='simple', mission=(), author='any', cadence=None,
    sec=None, cutout_size=None, quarter=None, campaign=None):

    '''
    Function to get the lightcurve of a target source by using the lightkurve
    package.

    Parameters
    ----------
    ID : str
        ID of the source to query.

    method : str
        Method used to query the lightcurve.
        'simple' uses the lk.search_lightcurve method to find available observations. (Default)
        'tpf' uses lk.search_targetpixelfile method.
        'tesscut' uses lk.search_tesscut method.

    mission : str/tuple of str, optional
        'Kepler', 'K2', or 'TESS'. By default, all will be returned.

    author : str/tuple of str, optional
        Author of the data product ('provenance_name' in the MAST API).
        Official Kepler, K2, and TESS pipeline products have author names
        'Kepler', 'K2', and 'SPOC'. By default, all will be returned.

    cadence : 'long'/'short'/'fast'/int/float, optional
        Synonym for 'exptime':
        - 'long' selects 10-min and 30-min cadence products.
        - 'short' selects 1-min and 2-min products.
        - 'fast' selects 20-sec products.
        Alternatively, you can pass the exact exposure in seconds.
        This keyword will likely be deprecated in the future.

    sec : int/list of ints, optional
        TESS Sector number. By default, all will be returned.

    cutout_size : int/float/tuple, optional
        Side length of cutout in pixels. Tuples should have dimensions (y, x).
        Default size is (5, 5).

    quarter : int/list of ints, optional
        Kepler Quarter. By default, all will be returned.

    campaign : int/list of ints, optional
        K2 Campaign. By default, all will be returned.

    Returns
    -------
    Lightkurve object of the queried target.
    '''

    ID = ID.strip()

    if mission == None:
        if ID.startswith(('TIC','tic')):
            mission = 'TESS'
            quarter = campaign = None
        elif ID.startswith(('KIC','kic','KPLR','kplr','KTWO','ktwo','K2','k2')):
            mission = ('Kepler','K2')
            sec = None
        else:
            mission = ()

    while method not in ['simple','tpf','tesscut']:
        method = input('Input method is not available, please type a valid one: ')

    if method == 'simple':
        lc = lk.search_lightcurve(ID, mission=mission, author=author, cadence=cadence,
            sector=sec, quarter=quarter, campaign=campaign)

    if method == 'tpf':
        lc = lk.search_targetpixelfile(ID, mission=mission, author=author, cadence=cadence,
            sector=sec, quarter=quarter, campaign=campaign)

    if method == 'tesscut':
        mission = author = 'TESS'
        lc = lk.search_tesscut(ID, sector=sec)

    if len(lc) == 0:
        print('No data-product found for this query.\n')
        return None

    print(lc)

    select = input('Please select which observation you want to download (#,:): ')

    if method in ['simple','tpf'] and cutout_size is not None:
        print("WARNING: Input cutout_size value is not used for 'simple/tpf' query methods and will be ignored.\n" )
        cutout_size = None

    if select == ':':
        lc =  lc.download_all(cutout_size=cutout_size, download_dir=maindir)
    else:
        lc = lc[int(select)]
        lc = lc.download(cutout_size=cutout_size, download_dir=maindir) # NOT FULLY WORKING, FIX

    lc.targetid = ID.replace(' ','')

    if not os.path.isdir(maindir+'/DATA/'+ID):
        os.mkdir(maindir+'/DATA/'+ID)
        os.mkdir(maindir+'/DATA/'+ID+'/plots/')
        os.mkdir(maindir+'/DATA/'+ID+'/lightcurve/')
        print ("Directory tree created in %s " % (maindir+'/DATA/'+ID))

    else:
        if not os.path.isdir(maindir+'/DATA/'+ID+'/plots/'):
            os.mkdir(maindir+'/DATA/'+ID+'/plots/')
        if not os.path.isdir(maindir+'/DATA/'+ID+'/lightcurve/'):
            os.mkdir(maindir+'/DATA/'+ID+'/lightcurve/')

    return lc


def change_aperture(tpf, ini_mask='pipeline', method='threshold', star_cut=8, sky_cut=0.01,
    ref_pixel='center'):

    '''
    Function to visually change the tpf mask.

    Parameters
    ----------
    tpf : lk.targetpixelfile
        The input target pixel file from either TESS or Kepler.

    ini_mask : 'pipeline'/'new'/np.ndarray, optional
        Initial mask for the aperture:
        - 'pipeline' takes the default pipeline mask. (Default)
        - 'new' takes the tpf.mask_new if created before.
        - Manual input array of values for the initial mask.

    method : 'threshold'/'basic', optional
        Method used to select the aperture mask:
        - 'threshold' uses a threshold value to cut the pixels. (Default)
        - 'basic' manually change the True/False values of the mask.

    star_cut : int/float, optional
        If 'threshold' method is selected, input cut value to select the star.
        Default is 8.

    sky_cut : int/float, optional
        If 'threshold' method is selected, input cut value to select the background.
        Default is 0.01.

    ref_pixel : (int,int) tuple/'center'/None
        (col,row) pixel coordinate closest to the desired region. For example, use
        `reference_pixel=(0,0)` to select the region closest to the bottom left corner
        of the target pixel file. If 'center' (default) then the region closest to the
        center pixel will be selected. If `None` then all regions will be selected.

    Returns
    -------
    New mask for the tpf.
    '''

    if not (type(tpf) == lk.targetpixelfile.KeplerTargetPixelFile \
        or  type(tpf) == lk.targetpixelfile.TessTargetPixelFile):
        print('Input tpf is not recognised as such. Exiting...\n')
        return None

    if ini_mask in ['pipeline','pipe']:
        mask_new = tpf.pipeline_mask
    elif ini_mask in ['new','mask_new']:
        mask_new = tpf.mask_new
    elif type(ini_mask) == np.ndarray:
        mask_new = ini_mask
    else:
        print('Input ini_mask is not valid. Exiting...\n')
        return None

    change = 'y'
    while change == 'y':

        if 'fig_ap' in locals():
            plt.close(fig_ap)

        print('Showing current mask...')

        if method == 'basic':
            print(*[[1 if i == True else 0 for i in j] for j in mask_new.tolist()], sep=',\n')

        elif method == 'threshold':
            # Aperture mask defined by a threshold method using a sigma-above-background
            # value, assuming the star is located in the center (should be).
            mask_new = tpf.create_threshold_mask(threshold=star_cut, reference_pixel=ref_pixel)

        # Define "sky" background mask (assuming threshold = 0.01)
        mask_background = ~tpf.create_threshold_mask(threshold=sky_cut, reference_pixel=None)

        fig_ap, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,4))
        tpf.plot(ax=ax2, aperture_mask=mask_background, mask_color='w')
        tpf.plot(ax=ax1, aperture_mask=mask_new, mask_color='r')

        _,nrows,ncols = tpf.shape

        ax1.set_yticks([tpf.row+i for i in range(nrows)])
        ax1.set_xticks([tpf.column+i for i in range(ncols)])
        ax1.set_yticklabels([i for i in range(nrows)])
        ax1.set_xticklabels([i for i in range(ncols)])

        ax2.set_yticks([tpf.row+i for i in range(nrows)])
        ax2.set_xticks([tpf.column+i for i in range(ncols)])
        ax2.set_yticklabels([i for i in range(nrows)])
        ax2.set_xticklabels([i for i in range(ncols)])

        ax1.set_title('Current mask')
        ax2.set_title('Background mask')
        fig_ap.tight_layout()
        #fig_ap.show()
        plt.show(block=False)

        change = '-'
        while change not in ['y','n']:
            change = input('Do you want to change it? [n/y]: ')

            if change == 'y':

                done = False
                while done == False:

                    if method == 'basic':
                        raw,col,val = input('From bottom left, use raw,column,0/1: ').split(',')
                        if raw.isnumeric() and col.isnumeric() and val.isnumeric():
                            if val == '1':
                                val = True
                            elif val == '0':
                                val = False
                            mask_new[int(raw)][int(col)] = val
                            done = True
                        else:
                            done = False

                    else:
                        star_cut = input('Enter new threshold value (current value is %d): ' % star_cut)
                        try:
                            star_cut = float(star_cut)
                            done = True
                        except:
                            print('Input threshold value is not a floar or an integer.')
                            pass

    fig_ap.savefig(maindir+'/DATA/'+tpf.targetid+"/plots/"+tpf.targetid+'_mask.png', dpi=300, bbox_inches='tight')

    #print('Showing final mask...')
    #tpf.plot(aperture_mask=mask_new, title='Final mask')

    tpf.mask_new = mask_new
    tpf.mask_background = mask_background

    return tpf


def contaminants(tpf, mask='pipeline', dmag=5, dist_cont=100):

    '''
    Function to visually locate potential contaminants from Gaia EDR3 and export the
    closest ones (see dist_cont) to an output txt file.

    Parameters
    ----------
    tpf : lk.targetpixelfile
        The input target pixel file from either TESS or Kepler.

    mask : 'pipeline'/'new'/np.ndarray, optional
        Initial mask used to plot the aperture:
        - 'pipeline' takes the default pipeline mask. (Default)
        - 'new' takes the tpf.mask_new if created before.
        - None if no mask is used (default).

    dmag : int/float, optional
        Gaia G magnitude magnitude difference used to limit the contaminants.
        Default is 5.

    dist_cont : int/float, optional
        Maximum distance of the contaminant sources to be exported to a txt file.

    Returns
    -------
    Nothing but the plot of the contamminant is created.
    '''

    if not (type(tpf) == lk.targetpixelfile.KeplerTargetPixelFile \
        or  type(tpf) == lk.targetpixelfile.TessTargetPixelFile):
        print('Input tpf is not recognised as such. Exiting...\n')
        return None

    if mask in ['pipeline','pipe']:
        mask = tpf.pipeline_mask
    elif mask in ['new','mask_new']:
        mask = tpf.mask_new
    else:
        mask = None

    ra_0 = tpf.wcs.wcs.crval[0]
    dec_0 = tpf.wcs.wcs.crval[1]
    RADEC = SkyCoord(ra_0, dec_0, unit=(u.degree, u.degree), frame=tpf.wcs.wcs.radesys.lower())

    height = u.Quantity(tpf.wcs.wcs.crpix[0]*tpf.shape[1], u.arcsec)
    width  = u.Quantity(tpf.wcs.wcs.crpix[1]*tpf.shape[2], u.arcsec)

    query = Gaia.cone_search_async(RADEC, radius=np.sqrt(width**2+height**2))
    query = query.get_results()

    if len(query) == 0:
        print('Gaia query failed for object', tpf.targetid)
        return None

    dmin = query['phot_g_mean_mag'].min() + dmag
    query = query[query['phot_g_mean_mag'] <= dmin]

    nlim = 40
    if len(query) > nlim:
        query.sort('phot_g_mean_mag')
        query = query[:nlim]
        print('Query returned more than %i sources, getting the brightest ones.' % nlim)
        star_cut = query[-1]['phot_g_mean_mag']

    fig_ga, axg = plt.subplots(figsize=(6,4))
    tpf.plot(ax=axg)

    if mask is not None:
        [axg.add_patch(patches.Rectangle((j-.5+tpf.column, i-.5+tpf.row), 1, 1, color='r', alpha=.4)) \
            for i in range(mask.shape[0]) for j in range(mask.shape[1]) if mask[i, j]]

    axg.set_ylim(axg.get_ylim())
    axg.set_xlim(axg.get_xlim())

    s_fac = np.exp(query['phot_g_mean_mag'].min())*500

    for star in query:
        ra_pix,dec_pix=tpf.wcs.world_to_pixel_values(star['ra'],star['dec'])

        axg.scatter(tpf.column+ra_pix, tpf.row+dec_pix, s=s_fac/np.exp(star['phot_g_mean_mag']),
            fc='orange', ec='k', alpha=0.8, lw=.5, zorder=1)
        axg.text(tpf.column+ra_pix+.2, tpf.row+dec_pix+.2, round(star['phot_g_mean_mag'],1),
            fontsize=7, c='w', zorder=2).set_clip_on(True)

    axg.set_title('Gaia sources with Gmag < %.1f' % dmin)
    fig_ga.tight_layout()
    #fig_ga.show()
    plt.show(block=False)

    fig_ga.savefig(maindir+'/DATA/'+tpf.targetid+"/plots/"+tpf.targetid+'_Gaia.png', dpi=300, bbox_inches='tight')

    query['dist'] = query['dist']*3600
    query = query[query['dist'] < dist_cont]['source_id','ra','dec','pm','pmra','pmdec','ruwe','phot_g_mean_mag','dist']
    query.write(maindir+'/DATA/'+tpf.targetid+'/contaminants.txt', format='ascii.fixed_width_two_line', overwrite=True)

    return None


def tpf_to_lc(tpf, mask='pipeline', flux_err_cut=0):

    '''
    Function to convert a tpf object to its lightcurve.

    Parameters
    ----------
    tpf : lk.targetpixelfile
        The input target pixel file from either TESS or Kepler.

    mask : 'pipeline'/'new'/np.ndarray, optional
        Mask for the aperture:
        - 'pipeline' takes the default pipeline mask. (Default)
        - 'new' takes the tpf.mask_new if created before.
        - Manual input array of values for the initial mask.

    flux_err_cut : int/float, optional [NOT IN USE]
        Threshold value for the flux error to remove bad data. Default is 0.

    Returns
    -------
    Lightkurve object from the tpf object.
    '''

    if not (type(tpf) == lk.targetpixelfile.KeplerTargetPixelFile \
        or  type(tpf) == lk.targetpixelfile.TessTargetPixelFile):
        print('Input tpf is not recognised as such. Exiting...\n')
        return None

    if mask == 'pipeline':
        mask = tpf.pipeline_mask
    elif mask in ['new','mask_new']:
        mask = tpf.mask_new

    lc_raw = tpf.to_lightcurve(aperture_mask=mask)
    #lc_raw = lc_raw[lc_raw.flux_err > flux_err_cut] # Need to keep the np.nan for detrending
    lc_raw.targetid = tpf.targetid

    return lc_raw


def detrended_tpf_to_lc(lc, tpf, mask_background, npcs=20):

    '''
    Function to perform the detrending by PCA given an input lightcurve.

    Parameters
    ----------
    lc : lk.lightcurve
        The input lightcurve object from either TESS or Kepler.

    tpf : lk.targetpixelfile
        The input target pixel file from either TESS or Kepler.

    mask_background : np.ndarray
        Mask used to consider the background. Usually tpf.mask_background.

    npcs : int, optional
        Define the initial number of principal components to inspect. Default is 20.

    Returns
    -------
    Lightkurve object from the detrended tpf object.
    '''

    if not (type(lc) == lk.lightcurve.KeplerLightCurve \
        or  type(lc) == lk.lightcurve.TessLightCurve):
        print('Input lightcurve is not recognised as such. Exiting...\n')
        return None

    if not (type(tpf) == lk.targetpixelfile.KeplerTargetPixelFile \
        or  type(tpf) == lk.targetpixelfile.TessTargetPixelFile):
        print('Input tpf is not recognised as such. Exiting...\n')
        return None

    mask_nan = [True if not np.isnan(i) else False for i in lc.flux]

    # Define Regressors to perform PCA and remove systematics
    regressors = tpf.flux[mask_nan][:][:,mask_background]

    while npcs != '':
        try:
            npcs = int(npcs)
        except:
            print('Input value for npcs must be an integer. Exiting...\n')
            return None

        if 'fig_pca' in locals():
            plt.close(fig_pca)

        # Design regressor matrix
        dm = lk.DesignMatrix(regressors, name='regressors').pca(npcs).append_constant()

        # Plot first npcs components to inspect
        fig_pca, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
        ax.plot(tpf[mask_nan].time.value, dm.values[:,:-1] + np.arange(npcs)*0.2, '.', color='k', ms=2)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_title('The first principal component is at the bottom')

        fig_pca.tight_layout()
        #fig_pca.show()
        plt.show(block=False)

        npcs = input('Value of npcs is %d. Hit return to accept and continue, or type another value: ' % npcs)

    fig_pca.savefig(maindir+'/DATA/'+tpf.targetid+'/plots/'+tpf.targetid+'_pca_regressors.png', dpi=300, bbox_inches='tight')

    # Apply the detrending and get the detrended light curve

    rc = lk.RegressionCorrector(lc.remove_nans())
    lc = rc.correct(dm)

    # Plot a simple diagnostic plot
    rc.diagnose()
    plt.savefig(maindir+'/DATA/'+tpf.targetid+'/plots/'+tpf.targetid+'_raw_light_curve.png', dpi=300, bbox_inches='tight')
    plt.show(block=False)

    lc.targetid = tpf.targetid

    return lc


def sig_clip_lc(lc, sigma=6):

    '''
    Function to perform a sigma clipping to the input lightcurve.

    Parameters
    ----------
    lc : lk.lightcurve
        The input lightcurve object from either TESS or Kepler.

    sigma : int/float, optional
        Sigma clipping factor applied to the lightcurve. Default is 6.

    Returns
    -------
    Clipped lightkurve object from the original lightkurve object.
    '''

    if not (type(lc) == lk.lightcurve.KeplerLightCurve \
        or  type(lc) == lk.lightcurve.TessLightCurve):
        print('Input lightcurve is not recognised as such. Exiting...\n')
        return None

    tmp_lc = deepcopy(lc)

    change = 'y'
    while change == 'y':

        if 'fig_sig' in locals():
            plt.close(fig_sig)

        # Apply sigma-clipping
        lc_clean, mask_outliers = tmp_lc.remove_outliers(sigma=sigma, return_mask=True)

        # Plot diagnostic light-curve figures (before and after)
        fig_sig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(10,8))

        ax1.plot(lc.time.value[mask_outliers], lc.flux.value[mask_outliers],
            marker='.', ls='None', color='red', label='Outliers')
        lc.plot(ax=ax1, marker='.', ls='None')
        ax1.legend(loc='best')
        ax1.set_title('Light curve')

        lc_clean.plot(ax=ax2, marker='.', ls='None')
        ax2.set_title('Light curve, outliers removed (scatter plot)')

        lc_clean.plot(ax=ax3)
        ax3.set_title('Light curve, outliers removed (line plot)')

        fig_sig.tight_layout()
        fig_sig.subplots_adjust(hspace=0.5)
        #fig_sig.show()
        plt.show(block=False)

        change = input('Do you want to change this value? [n/y]: ')
        if change == 'y':
            sigma = input('Enter new sigma value (current value is %d): ' % sigma)
            sigma = float(sigma)

        elif change not in ['y','n']:
            print('Not a valid input.')
            change = 'y'

    lc.remove_outliers(sigma=sigma, return_mask=True)

    fig_sig.savefig(maindir+'/DATA/'+lc.targetid+'/plots/'+lc.targetid+'_detrended_light_curve.png', dpi=300, bbox_inches='tight')

    return lc


def get_mag_lc(lc):

    '''
    Function to calculate the magnitude from the flux and add it to the
    lightkurve object.

    Parameters
    ----------
    lc : lk.lightcurve
        The input lightcurve object from either TESS or Kepler.

    Returns
    -------
    Same lightkurve object with the magnitude as lc.mag included.
    '''

    if not (type(lc) == lk.lightcurve.KeplerLightCurve \
        or  type(lc) == lk.lightcurve.TessLightCurve):
        print('Input lightcurve is not recognised as such. Exiting...\n')
        return None

    if hasattr(lc, 'remove_nans'):
        flux = lc.remove_nans().flux.value
    else:
        flux = lc.flux.value

    mag = -2.5 * np.log10(flux)
    mag -= np.median(mag)
    lc.magnitude = mag

    return lc


def export_lc(lc, output_path='default', append=''):

    '''
    Function to export the lightcurve from a lightkurve object.

    Parameters
    ----------
    lc : lk.lightcurve
        The input lightcurve object from either TESS or Kepler.

    output_path : str, optional
        Path where the lightcurve will be saved.
        Default is maindir/ID/lightcurve/

    append : str, optional
        Append suffix after the ID and before the extensio. Default is ''.

    Returns
    -------
    Nothing but the lightcurve is exported.
    '''

    if not (type(lc) == lk.lightcurve.KeplerLightCurve \
        or  type(lc) == lk.lightcurve.TessLightCurve):
        print('Input lightcurve is not recognised as such. Exiting...\n')
        return None

    if hasattr(lc, 'magnitude'):
        master_flux = lc.magnitude
    else:
        lc = get_mag_lc(lc)
        master_flux = lc.magnitude

    master_time = lc.time.value

    # Check sorting
    #master_index_sort = np.argsort(master_time, axis = 0)
    #master_time = master_time[master_index_sort]
    #master_flux = master_flux[master_index_sort]

    # Remove NaNs
    master_time = master_time[~np.isnan(master_flux)]
    master_flux = master_flux[~np.isnan(master_flux)]

    # Remove the median
    master_flux = master_flux - np.median(master_flux)

    if output_path in ['def','default']:
        output_path = maindir+'/DATA/'+lc.targetid+'/lightcurve/'

    np.savetxt(output_path+lc.targetid+append+'.txt', np.array([master_time, master_flux]).T,
        header='time, magnitude', fmt='%.10f', delimiter=', ', comments='')

    return None


def lc_to_perid(lc, oversample_factor=1, output_path='default'):

    '''
    Function to obtain the preidiogram and associated plots for a given lightcurve.

    Parameters
    ----------
    lc : lk.lightcurve
        The input lightcurve object from either TESS or Kepler.

    Returns
    -------
    The peridiogram object is returned.
    '''

    if not (type(lc) == lk.lightcurve.KeplerLightCurve \
        or  type(lc) == lk.lightcurve.TessLightCurve):
        print('Input lightcurve is not recognised as such. Exiting...\n')
        return None

    pg = lc.to_periodogram(method='lombscargle', oversample_factor=oversample_factor)
    pg.targetid = lc.targetid

    pg.magnitude = 1e6*((pg.power - np.median(pg.power)) / np.median(pg.power))

    fig_pg, ax_pg = plt.subplots(figsize=(5,2))
    ax_pg.set_title('Peridiogram')
    ax_pg.set_xlabel(r'Freq. [d$^{-1}$]')
    ax_pg.set_ylabel('Amp. (mmag)')
    ax_pg.plot(pg.frequency, pg.magnitude, lw=.5)

    fig_pg.savefig(maindir+'/DATA/'+lc.targetid+'/plots/'+lc.targetid+'_peridiogram.png', dpi=300, bbox_inches='tight')

    return pg


def export_pg(pg, output_path='default', append=''):

    '''
    Function to export the peridiogram from a peridiogram object.

    Parameters
    ----------
    pg : lk.periodogram.LombScarglePeriodogram
        The input peridiogram object from either TESS or Kepler.

    output_path : str, optional
        Path where the peridiogram will be saved.
        Default is maindir/ID/lightcurve/

    append : str, optional
        Append suffix after the ID and before the extensio. Default is ''.

    Returns
    -------
    Nothing but the lightcurve is exported.
    '''

    if not type(pg) == lk.periodogram.LombScarglePeriodogram:
        print('Input peridiogram is not recognised as such. Exiting...\n')
        return None

    if not hasattr(pg, 'magnitude'):
        pg.magnitude = 1e6*((pg.power - np.median(pg.power)) / np.median(pg.power))

    if output_path in ['def','default']:
        output_path = maindir+'/DATA/'+pg.targetid+'/lightcurve/'

    np.savetxt(output_path+pg.targetid+'_pg_'+append+'.txt', np.array([pg.frequency, pg.magnitude]).T,
        header='frequency, magnitude', fmt='%.10f', delimiter=', ', comments='')

    return None
