# Python Standard Library packages:
import re
import os
import sys
import time
import platform
import warnings

# Other main packages
import numpy as np
import progressbar as pb

# Astro-packages
import astropy.units as u
from astropy.io import fits,ascii
from astropy.table import Table, join, setdiff, vstack, hstack
from astropy.coordinates import SkyCoord
#   Vizier:
from astroquery.vizier import Vizier # Only used to query in Gaia DR2
#   Simbad
from astroquery.simbad import Simbad
Simbad.add_votable_fields('flux(B)','flux(V)','sptype')
#   Gaia
from astroquery.gaia import Gaia
Gaia.MAIN_GAIA_TABLE = "gaiaedr3.gaia_source" # Select early Data Release 3
Gaia.ROW_LIMIT = -1 # Set the number of output raw limit to infinite
sys.path.append(os.path.expanduser('~') + '/MEGA/PhD/programs/python/edr3_zp')
import zpt; zpt.load_tables()


# Load the working paths:
dir = ''
while not os.path.isfile(dir + 'paths.txt'):
    print('File paths.txt not found...')
    dir = input('Please provide the full path to the file now: ')

with open(dir + 'paths.txt', 'r') as paths:
    paths = [i.split('=') for i in paths.read().splitlines() if not i == '' and not i.startswith('#')]
    paths = [[i[0],i[1]] if i[1].endswith('/') == True else [i[0],i[1]+'/'] for i in paths]

dirs = {}
for i in paths:
    dirs[i[0]] = i[1]

maindir = dirs['main']
datadir = dirs['data']
ibdir   = dirs['ib']
mauidir = dirs['maui']
mistdir = dirs['mist']
tessdir = dirs['tess']

def search(myfile, path):

    '''
    Function to search a file within a directory.

    Parameters
    ----------
    myfile : str
        Name of the file to search.

    path : str
        Path where to search for the file.

    Returns
    -------
    Path to the searched file.
    '''

    f_dir = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file == myfile:
                f_dir = os.path.join(root, file)

    if f_dir == []:
        print('File %s not found.\n' % myfile)
        return None

    return f_dir


def findstar(spectra=None, SNR=None):

    '''
    Function to get the paths of the searched spectra allowing to limitate the
    results by a minimum SNR if provided in the header, or the one with best SNR.

    Parameters
    ----------
    spectra : str, optional
        Enter the input spectra, either name(s) of the star(s), the fits files
        separated by coma, a .txt/.lst file containing the (file)names, or '*'
        if you want to select all the fits files inside the working folder.

    SNR : str/int, optional
        If 'best' as input, it finds only the best SNR spectrum for each star.
        If 'bestMF' same as 'best' but prioritizing spectra from HERMES/FEROS.
        If specified, it returns all the spectra above the chosen SNR.

    Returns
    -------
    Paths to the files found.
    '''

    if spectra == None:
        while spectra == '':
            print('No file/files were selected.\n')
            spectra = input('Enter the input spectra (name(s), *, *.fits, *.txt/.lst): ')

    if spectra == '*':
        list_spectra = ['*']

    elif '.lst' in spectra or '.txt' in spectra:
        print('\nSearching file in %s... \n' % (maindir))

        list_dir = search(spectra, maindir)
        list_spectra = []

        with open(list_dir, 'r') as spectra:
            list_spectra = spectra.read().splitlines()

        if len(list_spectra) == 0:
            print('No spectra in list found.\nExiting...')
            return None

        list_spectra = [spectrum.split()[0] for spectrum in list_spectra \
                       if not spectrum.startswith('#') and not spectrum == '']

    elif 'fits' in spectra:
        list_spectra = spectra.split(',')

    else:
        list_spectra = spectra.replace(' ', '').split(',') # This catches 'HDXXXX, HDYYYY'

    dir_spectra = []
    for spectrum in list_spectra:
        match = 0
        for root, dirs, files in os.walk(datadir):
            for file in files:

                if file.startswith('._'): continue

                elif spectrum == '*' and file.endswith('.fits'):
                    dir_spectra.append(os.path.join(root, file))
                    match = 1

                elif file == spectrum:
                    dir_spectra.append(os.path.join(root, file))
                    match = 1

                elif spectrum + '_' in file and file.endswith('.fits'):
                    dir_spectra.append(os.path.join(root, file))
                    match = 1

        if not match == 1:
            print('File/source %s not found.\n' % spectrum)

    if len(dir_spectra) == 0:
        return None #quit()

    # Spectra selection based on selected SNR.
    if any(['ascii' in spectrum for spectrum in list_spectra]) and SNR != None:
        print('SNR for ascii files is not yet implemented, ignoring SNR keyword.')

    else:
        if SNR == 'best':
            dir_spectra = snr(dir_spectra)

        elif SNR == 'bestMF':
            dir_spectra = snr(dir_spectra, get_MF=True)

        elif type(SNR) == int:
            dir_spectra = snr(dir_spectra, snrcut=SNR)

    # Order all spectra from a single target by date.
    if len(list_spectra) == 1:
        dir_spectra = sorted(dir_spectra, key = lambda path: \
        path.split(os.sep)[-1].split('_')[1] + path.split(os.sep)[-1].split('_')[2])

    return dir_spectra


def searchlines(line, tol=1):

    '''
    IN DEVELOPMENT - this is to easily find spectral lines in a future included list of
    spectral lines developed within the IACOB team.

    Parameters
    ----------
    line : float
        Approximate wavelenght of the line to search

    tol : int/float, optional
        Tolerance in the search of the line

    Returns
    -------
    List of lines found for the input line.
    '''

    line = float(line.replace('?', ''))
    linesdb = findlines('synt_lines_OB.lst')
    indexes = [linesdb[0].index(i) for i in linesdb[0] if i > line-tol and i < line+tol]
    print('Nearest lines are:')
    [print(linesdb[0][idx],linesdb[1][idx],linesdb[2][idx]) for idx in indexes]
    line = input('Choose a line from the list: ')

    return line


def findlines(list):

    '''
    Function to extract atomic lines from a list containing their information or
    provide output format for a given wavelenght.

    Parameters
    ----------
    list : str/float
        Enter the list of lines to fit in .txt/.lst, or coma-separated string
        with wavelenghts, or single float/int wavelenght.

    Returns
    -------
    List of wavelenghts, element names, and loggf.
    '''

    path = maindir+'lists/lines'

    lines = []

    # Single input in float or int format without quotation marks:
    if type(list) == float or type(list) == int:
        lines.append(float(list))
        elements = loggf = [None]

    # Lines in a lst/txt file with more information on each line:
    elif '.lst' in list or '.txt' in list:
        list_dir = search(list, path)
        with open(list_dir, 'r') as file_lines:
            lines = file_lines.read().splitlines()

        rows = [line.split(',') for line in lines if not line.startswith('#') and not line == '']
        lines = [float(line[0]) for line in rows]

        try:
            elements = [line[1].strip() for line in rows]
        except:
            elements = [None]*len(lines)

        try:
            loggf = [float(line[2].strip()) for line in rows]
        except:
            loggf = [None]*len(lines)

    # String of lines separated by coma:
    else:
        lines = list.split(',')
        elements = loggf = [None]*len(lines)

        lines_f = []
        for n in range(len(lines)):
            if '?' in lines[n]:
                lines_f.append(searchlines(lines[n]))
            else:
                lines_f.append(lines[n])

        lines = [float(line) for line in lines_f]

    return lines, elements, loggf


def findlist(list):

    '''
    Function to extract items in a txt/lst file.

    Parameters
    ----------
    list : str
        Enter the list of items in .txt/.lst.

    Returns
    -------
    Items contained in the input list.
    '''

    path = maindir+'lists'

    # To catch wrong int/float inputs:
    if type(list) == float or type(list) == int:
        print('Input cannot be int or float format. \n Exiting...')
        return None

    # Lines in a lst/txt file with more information on each line:
    elif '.lst' in list or '.txt' in list:
        list_dir = search(list,path)
        with open(list_dir, 'r') as file_list:
            list = file_list.read().splitlines()

        items = [row.split(',')[0] for row in list if not row.startswith('#') and not row == '']

    else:
        items = list.replace(' ', '').split(',') # This catches 'HDXXXX, HDYYYY'

    return items


def findtable(table, path=None, format=None, delimiter=' ', header_start=None,
    fits_strip_end=True, fix_missing=False):

    '''
    Function to get the data from a table with different formats.
    See https://docs.astropy.org/en/stable/io/ascii/read.html for more details
    of 'format', 'delimiter' and 'header_start' keywords.

    Parameters
    ----------
    table : str
        Enter the fits table containing the data.

    path : str, optional
        Path where to search for the file.

    format : str, optional
        Format of the input table. See ascii.read() and Table.read() documentation.
        Default is None.

    delimiter : str, optional
        The string used to separate values. Default is a whitespace.

    header_start : int, optional
        Index of the row which contains the header. Default is None.

    fits_strip_end : boolean, optional
        If 'True' it strips all strings within the data of a fits table.

    fix_missing : boolean, optional
        If 'True' it will replace missing strings by '' abd missing floats by float 'nan'.

    Returns
    -------
    Data in table, in table format.
    '''

    if path == None:
        path = maindir + 'tables'

    table_dir = search(table, path)

    if table_dir == None:
        return None

    if 'csv' in table.split('.') or format == 'csv':
        delimiter = ','
        format = 'csv'

    if '.fits' in table:
        #try:
        #    with fits.open(table_dir,mode='readonly') as hdu_list:
        #        data = hdu_list[1].data
        #except: data = Table.read(table_dir,format='fits')

        data = Table.read(table_dir, format='fits')
        if fits_strip_end == True:
            tostrip = [i for i in data.colnames if data.dtype[i].type in [np.str_,np.bytes_]]
            #tostrip = [data.colnames[i] for i in range(len(data.dtype)) if data.dtype[i].char == 'S'] # old kept just in case
            for col in tostrip:
                 data[col] = [i.strip() if np.ma.is_masked(i) == False else i for i in data[col]]

    elif header_start != None:
        data = ascii.read(table_dir, header_start=header_start, format=format, delimiter=delimiter)

    else:
        data = ascii.read(table_dir, format=format, delimiter=delimiter)
        #    data = Table.read(table_dir, format='ascii', delimiter=delimiter, fill_values=None)
        # header_start is not included as a keyword as it slows down the process when is None

    if fix_missing == True:
        for j in [i for i in data.colnames if data.dtype[i].type in [np.str_,np.bytes_]]:
            data[j].fill_value = ''
        for j in [i for i in data.colnames if data.dtype[i].type in [np.float64]]:
            data[j].fill_value = np.nan
        data = data.filled()

    return data


def xmatch_table(table1, table2, match_col=None, output_cols=None,
    output_name='xmatch_table', format='fits'):

    '''
    Function to generate xmatched tables with selected output columns and format.

    Parameters
    ----------
    table1 : str
        Name of the main table.

    table2 : str
        Name of the second table to cross-match with the first.

    match_col : str, optional
        Name of the column to be used as anchor for the cross-match.
        If empty, the user is asked among the colums that are in both tables.

    output_cols : str, optional
        Coma separated string with the list of column names to be included in
        the output file. If empty, all the columns are selected.

    output_name : str, optional
        Name of the output file. Default is 'xmatch_table'.
        Note: Do not include the extension.

    format : str, optional
        Enter the output format for the table: 'fits' (default), 'ascii' or 'csv'.

    Returns
    -------
    Nothing but the output table with cross-match is generated in the
    same path where the two input tables are located.
    '''

    table1 = findtable(table1)
    table2 = findtable(table2)

    # Cross-match column:
    same_cols = [i for i in table1.colnames if i in table2.colnames]
    if match_col == None:

        if same_cols == []:
            print('No columns found with the same name in both tables. Exiting... \n')
            return None

        else:
            print(same_cols)
            match_col = input('Select one of the above column names for the cross-match: ')

    # Remove extra columns:
    if len(same_cols) > 1:
        print('WARNING: More than one column has the same name.')
        print('Output table will contain *_1, *_2 for them.')

        rm = input('Remove repeated columns from second table? [y/n]: ')
        if rm == 'y':
            table2.remove_columns(
            [i for i in table2.colnames if i in table1.colnames and not i == match_col])

    # Cross-match:
    if table1[match_col].dtype.char == 'S':
        table1[match_col] = [i.strip() for i in table1[match_col]]
        table2[match_col] = [i.strip() for i in table2[match_col]]

    table = join(table1, table2, keys=match_col)
    if len(table) == 0:
        print('No matches for the joined tables. Exiting... \n')
        return None

    # Output columns:
    if output_cols != None:
        output_cols = output_cols.split(',')
        table.remove_columns([i for i in table.colnames if not i in output_cols])

    # Saving the file:
    full_path = maindir + 'tables/' + output_name + '.' + format
    if format == 'ascii':
        format += '.fixed_width_two_line'
    table.write(full_path, format=format, overwrite=True)


def snr(spectra, snrcut=None, get_MF=None):

    '''
    Function to provide the spectrum with best signal to noise ratio, or all the
    spectra above a given value of signal to noise ratio, taken from the header.

    Parameters
    ----------
    spectra : list
        List of paths of the spectra to filter.

    snrcut : int, optional
        If established, it returns all the spectra above the chosen SNR.

    get_MF : Boolean, optional
        If True, it returns available spectra from either Mercator or Feros with
        SNR within 15% less than the best SNR spectra taken with FIES.

    Returns
    -------
    Paths to filtered spectrum/spectra.
    '''

    names_stars = []
    for spectrum in spectra:
        id_star = spectrum.split(os.sep)[-1].split('_')[0]
        if id_star not in names_stars:
            names_stars.append(id_star)

    best_spectra = []
    for star in names_stars:
        SNR_best = 0
        SNR_best_MF = 0
        for spectrum in spectra:
            filename = spectrum.split(os.sep)[-1].split('_')
            id_star = filename[0]
            date = int(filename[1]+filename[2])
            instr = filename[3]

            if star != id_star:
                continue

            else:
                # Retrieve the key values fron the fits header
                hdu = fits.open(spectrum)# Open the fits image file
                hdu0 = hdu[0]            # Load the header list of primary header
                header = hdu0.header    # Read the values of the headers
                SNR = float(header['I-SNR'])  # Estimated Signal to Noise Ratio

                if snrcut == None:
                    # Date is used for spectra with same SNR choosing the newest one.
                    # Instr is used for when get_MF is enabled.
                    if SNR > SNR_best or (SNR == SNR_best and date > best_spec_date):
                        SNR_best = SNR; best_spec = spectrum
                        best_spec_date = date; best_spec_inst = instr

                    if get_MF == True and instr in ['M','F'] and SNR > SNR_best_MF:
                        SNR_best_MF = SNR; best_spec_MF = spectrum

                elif SNR > int(snrcut):
                    best_spectra.append(spectrum)

        if snrcut == None:
            if get_MF == True and best_spec_inst=='N' and (0.85*SNR_best<SNR_best_MF \
                or (0.70*SNR_best<SNR_best_MF and SNR_best_MF>100)):
                 best_spec = best_spec_MF
            best_spectra.append(best_spec)

        elif len(best_spectra) == 0:
            print('No spectra found with SNR higher than %s.' % snrcut)

    return best_spectra


def table_db(list, db, coords=None, limdist=None, lim_lb=None, spt=None, lc=None,
    snrcut=None, spccode=False, bmag=None, vmag=None, gaia=False, radius=1, skip=None):

    '''
    Function to generate a FITS table with information about sources coming from
    IACOB/FEROS database, a list of names or coordinates, allowing to limitate
    the results by B/V magnitudes, SpT, LC, distance, and also providing Gaia data.

    Parameters
    ----------
    list : str
        Enter the input list, either name(s)/FITS of the source(s) separated by coma,
        or a .txt/.lst file containing the source names or coordinates.
        (Coordinates must be provided as h:m:s +-d:m:s or d +-d without comas)
        (If "db" is "IACOB", type '*' to select all the available FITS)

    db : str
        Enter the input database: IACOB/Simbad

    coords : str, optional
        Enter 'header' to take the coordinates from header. Otherwise it
        takes the coordinates from Simbad, quering the fits filename.

    limdist : list, optional
        Enter the RADEC coordinates [hms] or [deg] of the origin point from
        where to measure the distance and the distance [deg] where to find stars.
        e.g. ['12:30:30.2 +40:20:10.3', 3] or ['35.2368 +57.6622',4.5]

    lim_lb : list, optional
        Enter the galactic l(min,max), b(min,max) coordinates in degrees if you want to
        filter the sources which lie outside the input boudaries. e.g. [100,150,-15,15]

    spt : str, optional
        Enter a desired spectral types to search, separated by coma e.g. 'O,B1'.

    lc : str, optional
        Enter a desired luminosity classes to search, separated by coma e.g. 'I,V'.

    snrcut : int, optional
        If specified, it returns all the spectra above the chosen SNR.

    spccode : boolean, optional
        If True, it will create separate columns with SpT and LC numbers.
        Default is False.

    bmag : str, optional
        Enter a desired bmag to cut the results e.g. '<6'.

    vmag : str, optional
        Enter a desired vmag to cut the results e.g. '>8.5'.

    gaia : str, optional
        If 'DR2'/'EDR3', it will create separate columns with Gaia DR2/EDR3 data.
        Default is False.

    radius : int,float
        Enter the search radius in arcsec for the Gaia query. Default is 1.

    skip : str, optional
        Enter a coma separated list of targets to exclude in the table.

    Returns
    -------
    Nothing, but saves the generated table in the */table/ folder.
    '''

    if db == 'IACOB':
        lst_sources_all = findstar(spectra=list, SNR=snrcut)
        # For each source, the best available SNR is picked
        lst_sources_f = snr(lst_sources_all, get_MF=True)
        type_list = 'names'

    elif db == 'Simbad':
        lst_sources_f = findlist(list)
        type_list = input('The list contains names or coordinates? [names/coords]: ')
        if not type_list in ['names','coords']:
             print('Input answer is not valid. Exiting... \n')
             return None

    else:
        print('Database not recognised. Exiting... \n')
        return None

    # Tuning the format of the input variables
    if limdist != None:
        RADEC = limdist[0]
        dist = float(limdist[1])

    if spt != None: spt = re.split(' |,', spt)

    if lc != None: lc = re.split(' |,', lc)

    #===========================================================================
    #=============================== Query Gaia ================================
    if gaia == 'DR2':
        gaia_columns = ['BPmag','e_BPmag','+Gmag','e_Gmag','RPmag','e_RPmag',\
                   'pmRA','e_pmRA','pmDE','e_pmDE','Plx','e_Plx']
        gaia_columns.extend(['astrometric_n_good_obs_al','astrometric_chi2_al'])
        table_u0 = findtable('table_u0_g_col.txt', delimiter=',')
        offset = input('Apply +0.03 mas offset to parallax? [y/n]: ')

        v = Vizier(columns=gaia_columns)
        v.ROW_LIMIT = 1

    elif gaia == 'DR3':
        v = Vizier()
        v.ROW_LIMIT = 1

    #===========================================================================
    #============================== Progress Bar ===============================
    bar = pb.ProgressBar(maxval=len(lst_sources_f),
                         widgets=[pb.Bar('=', '[', ']'), ' ', pb.Percentage()])
    bar.start()

    #===========================================================================
    #================================= Sources =================================
    for source,i in zip(lst_sources_f, range(len(lst_sources_f))):
        row = Table()
        #=======================================================================
        #============= Retrieve the key values fron the fits header ============
        OBJRA = OBJDEC = None
        if db == 'IACOB':
            hdu = fits.open(source)  # Open the fits image file
            hdu0 = hdu.verify('fix') # Fix header keywords
            hdu0 = hdu[0]            # Load the header list of primary header
            header = hdu0.header     # Read the values of the headers

            row['ID'] = [source.split(os.sep)[-1].split('_')[0]]

            # Official name (not implemented yet)
            try:
                row['Name'] = header['I-Name']
            except:
                row['Name'] = row['ID'] # '-'

            # Reference spectrum (added at the end)
            filename = source.split(os.sep)[-1]

            # Gather the coordinates:
            if '_M_' in filename:
                try:
                    OBJRA = header['OBJ_RA']
                    OBJDEC = header['OBJ_DEC']
                except:
                    pass # Only new fits include it

            elif '_N_' in filename:
                OBJRA = header['OBJRA']*360/24
                OBJDEC = header['OBJDEC']

            elif '_F_' in filename:
                OBJRA = header['RA']
                OBJDEC = header['DEC']

            source = row['ID'][0]

        elif db == 'Simbad':
            row['ID'] = [source]

        #=======================================================================
        #=========================== Skip bad sources ==========================
        if db == 'IACOB' and any(bad in source[:3] for bad in ['DO2']): continue

        if skip != None and any(bad in source for bad in skip.split(',')): continue

        #=======================================================================
        #=============== Simbad query by object name/coordinates ===============
        if type_list == 'names':
            simbad = query_Simbad(source, OBJRA, OBJDEC)
            if simbad == None: continue

        elif type_list == 'coords':
            if ':' in source:
                c = SkyCoord(source, unit=(u.hour,u.deg))
            else:
                c = SkyCoord(source, unit=u.deg)

            try: # The actual query
                simbad = Simbad.query_region(c, radius=radius*u.arcsec)
            except: # Retry after 2 seconds
                time.sleep(2)
                print('Trying with a larger radius (%i arcsec)' % 2*radius)
                simbad = Simbad.query_region(c, radius=2*radius*u.arcsec)

            try: # For more than one result, takes the first one
                if len(simbad) > 1: simbad = Table(simbad[0])
            except:
                print('Source %s not found in Simbad' % source)
                continue

            source = simbad['MAIN_ID'][0]

            row['ID'] = row['Name'] = [source]

        #=======================================================================
        #========================= Get the coordinates =========================
        if coords == 'header':
            RADEC_0 = SkyCoord(ra=header['RA'], dec=header['DEC'], unit=(u.deg)) # Why no [0] here?
        else:
            RADEC_0 = SkyCoord(ra=simbad['RA'], dec=simbad['DEC'], unit=(u.hour,u.deg))[0]

        row['RA_J2000']  = RADEC_0.ra.to_string(unit=u.hour, sep=':', pad=True, alwayssign=True)
        row['DEC_J2000'] = RADEC_0.dec.to_string(unit=u.deg, sep=':', pad=True, alwayssign=True)
        row['RAdeg_J2000']  = round(RADEC_0.ra.deg, 6)
        row['RAdeg_J2000'].unit = u.deg
        row['DECdeg_J2000'] = round(RADEC_0.dec.deg, 6)
        row['DECdeg_J2000'].unit = u.deg

        #=======================================================================
        #========================= Limit by distance ===========================
        if limdist != None:
            if any([j in RADEC for j in [':','h']]):
                RADEC_f = SkyCoord(RADEC, unit=(u.hour,u.deg))
            else:
                RADEC_f = SkyCoord(RADEC, unit=u.deg)

            if RADEC_0.separation(RADEC_f).deg > dist: continue

        #=======================================================================
        #============================ Limit by l,b =============================
        if lim_lb != None:
            l_0 = RADEC_0.galactic.l.deg
            b_0 = RADEC_0.galactic.b.deg
            lim_lb=[float(i) for i in lim_lb]
            if not (lim_lb[0] < l_0 and lim_lb[1] > l_0 and lim_lb[2] < b_0 and lim_lb[3] > b_0):
                continue

        #=======================================================================
        #======================= Get the spectral class ========================
        if db == 'IACOB':

            SpC_0 = header['I-SPC']
            if (not type(SpC_0) == str or SpC_0.strip() == '-'):
                row['SpC'] = SpC_0 = simbad['SP_TYPE'][0].replace(' ','')
                row['SpC_ref'] = 'SIMBAD'

            else:
                row['SpC'] = SpC_0.replace(' ','')
                try: row['SpC_ref'] = header['I-SPCREF']
                except: row['SpC_ref'] = '???'

        else:
            row['SpC'] = SpC_0 = simbad['SP_TYPE'][0].replace(' ','')
            row['SpC_ref'] = 'SIMBAD'

        #=======================================================================
        #========================= Limit by SpT or LC ==========================
        spt_0 = []; lc_0 = []
        if spt != None or lc != None:

            if 'I' in SpC_0:
                spt_0 = SpC_0[:SpC_0.index('I')]
                lc_0 = SpC_0[SpC_0.index('I'):]

            elif 'V' in SpC_0:
                spt_0 = SpC_0[:SpC_0.index('V')]
                lc_0 = SpC_0[SpC_0.index('V'):]

            elif len(re.split(':|p|n|f|\\(|\\)',SpC_0.strip())[0]) <= 4: # e.g. O9.5:npe 4+:npe
                spt_0 = SpC_0.strip()

            else:
                spt_0 = SpC_0

            if spt != None and not any([j in spt_0 for j in spt]): continue

            match = False
            if lc != None and lc_0 != []:
                if 'IV' in lc and 'IV' in lc_0: match = True
                lc_0 = lc_0.replace('IV', '')
                if 'V' in lc and 'V' in lc_0: match = True
                if 'III' in lc and 'III' in lc_0: match = True
                lc_0 = lc_0.replace('III', '')
                if 'II' in lc and 'II' in lc_0: match = True
                lc_0 = lc_0.replace('II', '')
                if 'I' in lc and 'I' in lc_0: match = True
                if match == False: continue

        #=======================================================================
        #===================== Get the spectral class code =====================
        if spccode == True and SpC_0 != '':
            row['SpT_code'], row['LC_code'] = spc_code(SpC_0)
        elif spccode == True and SpC_0 == '':
            row['SpT_code'] = row['LC_code'] = np.nan

        #=======================================================================
        #========================= Limit by magnitude ==========================
        bmag_0 = simbad['FLUX_B'][0]
        if str(bmag_0) == '--':
            bmag_0 = np.nan

        if bmag != None and ~np.isnan(bmag_0):
            if bmag[0] == '<' and float(bmag[1:]) < bmag_0: continue
            elif bmag[0] == '>' and float(bmag[1:]) > bmag_0: continue

        row['mag_B'] = round(bmag_0, 2)
        row['mag_B'].unit = u.mag

        vmag_0 = simbad['FLUX_V'][0]
        if str(vmag_0) == '--':
            vmag_0 = np.nan

        if vmag != None and ~np.isnan(vmag_0):
            if vmag[0] == '<' and float(vmag[1:]) < vmag_0: continue
            elif vmag[0] == '>' and float(vmag[1:]) > vmag_0: continue

        row['mag_V'] = round(vmag_0, 2)
        row['mag_V'].unit = u.mag

        #=======================================================================
        #============== Add extra columns from header or database ==============
        # Counting FIES / HERMES / FEROS spectra
        if db == 'IACOB':
            FIES = 0
            HERMES = 0
            FEROS = 0
            for j in lst_sources_all:
                if source + '_' in j and j.endswith('.fits'):

                    if '_N_' in j:
                        FIES = FIES + 1

                    elif '_M_' in j:
                        HERMES = HERMES + 1

                    elif '_F_' in j:
                        FEROS = FEROS + 1

            row['FIES'] = FIES
            row['HERMES'] = HERMES
            row['FEROS'] = FEROS

            row['Ref_file'] = filename

            # SNR of the best spectra (prioritizing for M or F)
            row['SNR_best'] = int(header['I-SNR'])

        # SB feature of the star (added at the end)
        try:
            row['SB'] = header['I-SB']
        except:
            row['SB'] = '----'

        # Comments in the header (added at the end)
        try:
            row['Comments'] = header['I-commen']
        except:
            row['Comments'] = '----'

        #=======================================================================
        #======================== Get Gaia DR2/3 data ==========================
        if gaia in ['DR2','DR3']:

            gFlag = True
            if gaia == 'DR2':
                catalog = "I/345/gaia2"

            elif gaia == 'DR3':
                catalog = "I/350/gaiaedr3"

            if vmag_0 != '' and float(vmag_0) < 5:
                print('\nWARNING: %s is too bright (Vmag = %r)' % (source, vmag_0))

            try:
                gaiaq = v.query_object(source,catalog=catalog,radius=radius*u.arcsec)[0]

                if gaia == 'DR2':
                    # Correct Gaia photometry:
                    if str(gaiaq['Gmag'][0]) != '--' or str(gaiaq['e_Gmag'][0]) != '--':
                        if gaiaq['e_Gmag'] < 8e-3:  gaiaq['e_Gmag'] = 8e-3
                        if (6 <= gaiaq['Gmag'] <= 16):
                            gaiaq['Gmag'] = gaiaq['Gmag'] - 0.0032*(gaiaq['Gmag']-6)
                        if gaiaq['Gmag'] < 6:
                            gaiaq['Gmag'] = gaiaq['Gmag'] + 0.0271*(6-gaiaq['Gmag'])
                        gaiaq['Gmag'].unit = u.mag
                    else:
                        gFlag = False; print(str(source),'has missing Gaia G photometry.')

                    if str(gaiaq['e_BPmag'][0]) != '--':
                        if gaiaq['e_BPmag'] < 9e-3:
                            gaiaq['e_BPmag'] = 9e-3
                    else:
                        print(str(source),'has missing Gaia B photometry.')
                        gFlag = False

                    if str(gaiaq['e_RPmag'][0]) != '--':
                        if gaiaq['e_RPmag'] < 10e-3:
                            gaiaq['e_RPmag'] = 10e-3
                    else:
                        print(str(source),'has missing Gaia R photometry.')
                        gFlag = False

                    # Correct Gaia astrometry:
                    try:
                        if offset == 'y':
                            gaiaq['Plx'] = round(gaiaq['Plx'] + 0.03, 6)
                    except:
                        print(str(source),'has missing Gaia parallax.')

                    # Add RUWE column
                    if gFlag == True:
                        UWE = np.sqrt(float(gaiaq['chi2AL'])/(float(gaiaq['NgAL']) - 5))

                        diff = abs(gaiaq['Gmag'][0] - table_u0['g_mag']) \
                            + abs(gaiaq['BPmag'][0] - gaiaq['RPmag'][0] - table_u0['bp_rp'])

                        diff = diff.tolist()

                        gaiaq['RUWE'] = round(UWE/table_u0['u0'][diff.index(min(diff))],4)

                        gaiaq.remove_columns(['NgAL','chi2AL'])

                elif gaia == 'DR3':

                    gaiaq = gaiaq[[i for i in gaiaq.columns if not \
                        any([j in i for j in ['dr2','scan_','_corr','_obs_','ipd_','transits']])]]

                row = hstack([row,gaiaq])

            except:
                print('\n%s could not be queried in Gaia.' % source)

        try:
            table = vstack([table,row])
        except:
            table = row

        bar.update(i)

    bar.finish()

    try:
        print('Table has been successfully created with length %i' % len(table))
    except:
        print('Table is empty, no sources were found.')

    # Some final formatting:
    table['mag_B'] = table['mag_B'].astype('float32')
    table['mag_V'] = table['mag_V'].astype('float32')
    table['SB'] = table['SB'].astype('<U8')
    table['Comments'] = table['Comments'].astype('<U32')

    # Export table
    table.write(maindir + 'tables/tablestars.fits', format='fits', overwrite=True)

    return None


def spc_code(spc):

    '''
    Function to translate input SpC (Spectral Class) into SpT and LC codes.

    Parameters
    ----------
    spc : str
        Enter the spectral class to turn into SpT and LC codes.

    Returns
    -------
    SpT and LC codes.
    '''

    spt_dic = {'O': 1, 'B': 2, 'A': 3, 'F': 4, 'G': 5, 'K': 6, 'M': 7}
    lc_dic = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5}

    spc_c = spc.split('+')[0].split('/')[0].replace(':', '')

    if spc_c in ['~'] or spc_c.startswith(('WC','WN','WR','C')):
        spt_c = lc_c = np.nan

    else:
        # Spectral type
        spc_c = spc_c.replace('Mn','')

        if len(re.findall('Ve+[0-9]',spc_c)) != 0:
            spc_c = spc_c.replace(re.findall('Ve+[0-9]',spc_c)[0], '')

        spt_c = re.findall('[O,B,A,F,G,K,M]', spc_c)
        num = len(spt_c)
        spt_c_lst = [spt_dic[i] for i in spt_c]

        spt_c = re.findall('[0-9.]+', spc_c)
        spt_c_lst += [float(i)/10 for i in spt_c]

        try:
            # This is for B1-2 in which there is B(1) and 1-2(2)
            if num < len(spt_c_lst)-num:
                spt_c = (sum(spt_c_lst[:num])/num+sum(spt_c_lst[num:])/len(spt_c_lst[num:]))
            else:
                spt_c = sum(spt_c_lst)/num
        except:
            print(spc,spt_c_lst,num)

        # Luminosity class
        lc_c = re.findall('[I,V]+', spc_c)
        lc_c_lst = [lc_dic[i] for i in lc_c]

        if len(lc_c_lst) == 0:
            lc_c = np.nan
        else:
            lc_c = np.asarray(lc_c_lst).mean()

    return spt_c,lc_c


def query_Simbad(name=None, ra=None, dec=None, radius='5s', otypes=False):

    '''
    Function to query an object in Simbad database.

    Parameters
    ----------
    name : str, optional
        Enter the name of the source to query.

    ra : float, optional
        Enter the right ascension of the source, in degrees.

    dec : float, optional
        Enter the declination of the source, in degrees.

    radius : str, optional
        Enter a string with the radius for the sky search.

    otypes : boolean, optional
        If True, the object types will be also queried. Default is False.
        Note that sometimes this issues some errors.

    Returns
    -------
    Queried object in Table format.
    '''

    if otypes is True:
        Simbad.add_votable_fields('otypes')

    if name is not None:
        # For some reason sometimes adding a whitespace fix some querying issues
        for name_i in [name, name+' ', ' '+name]:
            try:
                simbad = Simbad.query_object(name_i)
                time.sleep(0.15)
                break
            except:
                simbad = None

        while simbad is None: # type(simbad) == type(None)

            print('Provide alternative name for %s in Simbad.' % name)
            print('In some cases try replacing "HD" by "HD " or vice versa.')

            if ra != None and dec != None:
                print('Type "sky" to query %s around input ra/dec (if given).' % radius)

            print('Hit return to skip this source.')

            check = input('Alternative: ')
            if check == '':
                print('Skipping source: %s\n' % name)
                break

            elif check == 'sky' and ra != None and dec != None:
                simbad = Simbad.query_region(SkyCoord(ra, dec, unit='deg'), radius=radius)
                if simbad is None: # type(simbad) == type(None)
                    print('No objects found.')

            else:
                try:
                    simbad = Simbad.query_object(check)
                except:
                    simbad = None

            if simbad is not None and len(simbad) > 1:
                print('More than one Simbad result, choosing the brigtest source...')
                simbad.sort('FLUX_V')
                simbad = Table(simbad[0])

        return simbad

    elif ra != None and dec != None:
        simbad = Simbad.query_region(SkyCoord(ra, dec, unit='deg'), radius=radius)

        if simbad is None: # type(simbad) == type(None)
            print('No objects found.')

        elif len(simbad) > 1:
            print('More than one Simbad result, choosing the brigtest source...')
            simbad.sort('FLUX_V')
            simbad = Table(simbad[0])

        return simbad

    else:
        print('\nERROR: Name or RA+DEC is required for the query...')
        return None


def query_Gaia(gaia='edr3', name=None, ra=None, dec=None, radec=None, radius=5, get_zp=False):

    '''
    Function to query an object in Gaia EDR3 database.

    Parameters
    ----------
    gaia : str, optional
        Enter the Gaia data you want to query, either 'dr2' or 'edr3' (default)

    name : str, optional
        Enter the name (ID) of the source to query.

    ra : float, optional
        Enter the right ascension of the source, in degrees.

    dec : float, optional
        Enter the declination of the source, in degrees.

    radec : str, optional
        Enter the string with the coordinates as hh(:)mm(:)ss.sss(:) +-dd(:)mm(:)ss(:)

    radius : int, optional
        Enter an integer with the radius for the sky search in arcseconds. Default is 5.

    Returns
    -------
    Queried object in Table format.
    '''

    if gaia not in ['dr2','DR2','edr3','EDR3']:
        print('Input Gaia catalog not selected bewteen dr2 and edr3. Exiting...\n')
        return None
    elif gaia in ['dr2','DR2']:
        Gaia.MAIN_GAIA_TABLE = "gaiadr2.gaia_source"
    elif gaia in ['edr3','EDR3']:
        Gaia.MAIN_GAIA_TABLE = "gaiaedr3.gaia_source"

    if name is not None:

        RADEC = SkyCoord.from_name(name)

        while RADEC is None: # type(simbad) == type(None)

            print('Provide alternative name for %s in Gaia.' % name)
            print('In some cases try replacing "HD" by "HD " or vice versa.')
            print('Hit return to skip this source.')

            check = input('Alternative: ')
            if check == '':
                print('Skipping source: %s\n' % name)
                break

            else:
                try:
                    RADEC = SkyCoord.from_name(check)
                except:
                    RADEC = None

    elif ra != None and dec != None:

        RADEC = SkyCoord(ra=ra, dec=dec, unit=(u.deg,u.deg), frame='icrs')

        if RADEC is None: # type(simbad) == type(None)
            print('No objects found.')

    elif radec != None:

        RADEC = SkyCoord(radec, unit=(u.hour,u.deg), frame='icrs')

    else:
        print('\nERROR: Name or RA+DEC is required for the query...')
        return None

    #width  = u.Quantity(radius, u.arcsec)
    #height = u.Quantity(radius, u.arcsec)
    #query = Gaia.query_object_async(coordinate=RADEC, width=width, height=height)

    radius = u.Quantity(radius, u.arcsec)
    query = Gaia.cone_search_async(coordinate=RADEC, radius=radius)

    if len(query) == 0:
        if name is not None:
            print('Gaia query failed for object',name)
        else:
            print('Gaia query failed for object with RA DEC =',RADEC.ra,RADEC.dec)
        return None

    if len(query) > 1:
        print('More than one Gaia result, choosing the brigtest source...')
        query.sort('phot_g_mean_mag')
        query = Table(query[0])

    if get_zp == True and gaia in ['edr3','EDR3']:
        if query['astrometric_params_solved'] <= 3:
            print('Astrometric parameters solved is <3. Skipping source_id =',str(query['source_id'][0]))
            zp_offset = 0.0

        else:
            query['zp_offset'] = zpt.get_zpt(
                query['phot_g_mean_mag'], query['nu_eff_used_in_astrometry'], \
                query['pseudocolour'], query['ecl_lat'], query['astrometric_params_solved'])

    if name is None:
        name = ['--']

    query['ID'] = name

    query = query[['ID'] + [i for i in query.colnames][:-1]]
    if 'DESIGNATION' in query.colnames:
        query.remove_column('DESIGNATION')

    return query


def checknames(list, max_dist=90):

    '''
    Function to detect errors in the filenames/headers.

    Parameters
    ----------

    list : str
        Enter the input list, either name(s)/FITS of the source(s) separated by coma,
        or a .txt/.lst file containing the source names or files.

    max_dist : int/float
        Enter the threshold distance to consider the spectra to point to a different
        source and not caused by telescope pointing offset.

    Returns
    -------
    Nothing, but a file with the errors found is generated.
    '''

    dir_spectra = findstar(list)

    bar = pb.ProgressBar(maxval=len(dir_spectra),
                         widgets=[pb.Bar('=', '[', ']'), ' ', pb.Percentage()])
    bar.start()

    type_errors = {'Simbad':[],'filename':[],'duplicate':[],'baddate':[],'radec0':[],'distance':[]}

    IDs = {'F': [], 'M':[], 'N':[]}

    date_0_long = []; i = 0
    for spectrum in dir_spectra:

        bar.update(i + dir_spectra.index(spectrum))
        #time.sleep(0.1)

        filename = spectrum.split(os.sep)[-1]
        id_star = spectrum.split(os.sep)[-1].split('_')[0]

        # Problems quering in Simbad
        if not id_star in type_errors['Simbad']:
            simbad = query_Simbad(id_star)
            if simbad == None:
                type_errors['Simbad'].append(id_star)

        # Retrieve the key values fron the fits header
        hdu = fits.open(spectrum) # Open the fits image file
        hdu0 = hdu.verify('fix')  # Fix header keywords
        hdu0 = hdu[0]             # Load the header list of primary header
        header = hdu0.header      # Read the values of the headers

        if '_F_' in filename: IDs['F'].append(header['ARCFILE'])
        elif '_M_' in filename: IDs['M'].append(str(header['UNSEQ']))
        elif '_N_' in filename: IDs['N'].append(header['FILENAME'])

        # Catch files with wrong object name comparing filename and header
        if id_star != header['OBJECT'].replace(' ','').upper():
            type_errors['filename'].append(filename)

        # Get the date and time from the header
        if '_N_' in filename or '_M_' in filename:
            date_0 = header['DATE-AVG'] #; print date_0
            date_0_short = str(date_0[0:4]) + str(date_0[5:7]) + str(date_0[8:10])
        elif '_F_' in filename:
            date_0 = header['ARCFILE']
            date_0_short = str(date_0[6:10]) + str(date_0[11:13]) + str(date_0[14:16])

        # Get the date from the filename
        date_filename = spectrum.split(os.sep)[-1].split('_')[1]

        # Catch two+ files with same date in header but different filenames
        # e.g. HD111111 and HDE111111
        if date_0 not in date_0_long:
            date_0_long.append(date_0)
        else:
            type_errors['duplicate'].append(filename)

        # Catch wrong dates in the filenames when comparing with the date in the header
        if not date_filename == date_0_short:
            type_errors['baddate'].append(filename)

        # Get the coordinates from the header
        RA_0 = header['RA']
        DEC_0 = header['DEC']
        RADEC_0 = str(RA_0) + ' ' + str(DEC_0)

        # Catch missing RA/DEC:
        if '0.0000' in str(RA_0) or '0.0000' in str(DEC_0):
            type_errors['radec0'].append(filename)

        # Catch spectra of different object than the object queried in Simbad
        if simbad != None and (not '0.0000' in str(RA_0) or not '0.0000' in str(DEC_0)):
            RADEC = str(simbad['RA'][0]).replace(' ', ':') \
                + ' ' + str(simbad['DEC'][0]).replace(' ', ':')

            c1 = SkyCoord(RADEC_0, unit=u.deg)
            c2 = SkyCoord(RADEC, unit=(u.hour,u.deg))

            difcoord = round(c1.separation(c2).arcsec, 3)

            if difcoord > max_dist:
                type_errors['distance'].append(filename+' -> '+str(difcoord))

    bar.finish()

    # Wrtiting the errors to the file:
    error_descriptions = [
    '# Problem querying these objects in Simbad. Correct the name and re-run to check object coordinates.\n',
    '# File names which has a wrong name for the object when compared to the header.\n',
    '# Duplicated files with same full date but different name in the file name.\n',
    '# Files with different date in the header and the file name.\n',
    '# Spectra with no RA or DEC values in the header. Correct the header and re-run to check object coordinates.\n',
    '# Distance from Simbad query is above the threshold for these spectra.\n']

    errorsDB = open(maindir+'lists/Errors_DB.txt', 'w')
    for description,error in zip(error_descriptions,type_errors):
        if len(type_errors[error]) == 0:
            continue
        errorsDB.write(description)
        [errorsDB.write(i+'\n') for i in type_errors[error]]
    errorsDB.close()

    # Writing the file unique identificators to a file
    IDs_spectra = open(maindir+'lists/IDs_spectra.txt', 'w') # TEMPORAL
    for description,tel in zip(['# FEROS files\n','# HERMES files\n','# FIES files\n'],IDs):
        if len(IDs[tel]) == 0:
            continue
        IDs_spectra.write(description)
        [IDs_spectra.write(i+'\n') for i in IDs[tel]]
    IDs_spectra.close()

    return None


def checkfits(list, coords_Simbad=True, radius=60, savepath=''):

    '''
    Function to detect errors in the filenames/headers.

    Parameters
    ----------

    list : str
        Enter the input list, either name(s)/FITS of the source(s) separated by coma,
        or a .txt/.lst file containing the source names or files.

    coords_Simbad : boolean, optional
        XX

    radius : int
        Search radius where to find the source by the coordinates, in arcsec.
        Default is 60 arcseg.

    savepath : str, optional
        If set, updated fits files will be saved in that location (full path must be set).

    Returns
    -------
    Nothing, but a file with the errors found is generated.
    '''

    dir_spectra = findstar(list)

    for spectrum in dir_spectra:

        filename = spectrum.split(os.sep)[-1]
        id_star = spectrum.split(os.sep)[-1].split('_')[0]

        # Retrieve the key values fron the fits header
        hdu = fits.open(spectrum) # Open the fits image file
        hdu0 = hdu.verify('fix')  # Fix header keywords
        hdu0 = hdu[0]             # Load the header list of primary header
        header = hdu0.header      # Read the values of the headers

        print('\nName (filename) / Name (object)\n%s / %s\n' % (id_star,header['OBJECT']))

        RADEC = SkyCoord(str(header['RA']),str(header['DEC']), unit=u.deg)

        if 'OBJ_RA' in header:
            RADEC_OBJ = SkyCoord(str(header['OBJ_RA']),str(header['OBJ_DEC']), unit=u.deg)
        elif 'OBJRA' in header:
            RADEC_OBJ = SkyCoord(str(header['OBJRA']),str(header['OBJDEC']), unit=(u.hour,u.deg))

        print('RADEC / RADEC_OBJ\n%s / %s' % (RADEC.ra,RADEC_OBJ.ra))
        print(RADEC.dec,'/',RADEC_OBJ.dec,'\n')

        simbad = query_Simbad(id_star)

        if simbad != None:
            print('From Simbad (querying ID):\n--------------------------')

            RADEC_SB = SkyCoord(simbad['RA'],simbad['DEC'], unit=(u.hour,u.deg))[0]

            try: # Additional coordinates from SkyCoord package
                RADEC_SC = SkyCoord.from_name(id_star)
            except:
                RADEC_SC = SkyCoord('0 0', unit=u.deg)

            print('RADEC (Simbad) / RADEC (SkyCoord)\n%s / %s' % (RADEC_SB.ra,RADEC_SC.ra))
            print(RADEC_SB.dec,'/',RADEC_SC.dec,'\n')

            print('SpC (header) / SpC (Simbad)\n%s (%s) / %s\n' %
                (header['I-SPC'],header['I-SPCREF'],simbad['SP_TYPE'][0]))

        if coords_Simbad == True:

            region = Simbad.query_region(RADEC, radius=radius*u.arcsec)

            if type(region) != type(None):
                print('From Simbad (querying header RA,DEC):\n-------------------------------------')

                print('Sources found:')
                [print(i['MAIN_ID']+' ('+i['SP_TYPE']+')') for i in region]
                print('\n')

        user = '-'; change = False
        while user != '':

            print("To change something from the header type: '<keyword>:<value>'")
            print("Type '+<name>' to obtain a list of alternative names for <name> in Simbad.")
            user = input("Or hit return to continue. Input: ")

            if user.startswith('+') and not ':' in user:

                try:
                    [print(i['ID']) for i in Simbad.query_objectids('HD2905')]
                except:
                    print('No names were found for the input name.')

            if ':' in user:
                try:
                    header[user.split(':')[0]] = user.split(':')[1]
                    change = True
                except:
                    print('Entry not valid. Verify the input.')

        if change == True:
            save = input('FITS file has been modified, do you want to save it? y/n/<filename.fits> ')

            if save.endswith('.fits'):
                filename_new = save
                save = 'y'

            if save == 'y':

                if savepath != '':
                    spectrum = savepath+filename_new

                else:
                    spectrum = spectrum.replace(filename,filename_new)

                hdu.writeto(spectrum, output_verify='ignore', overwrite=True)
