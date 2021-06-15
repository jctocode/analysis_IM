#!/usr/bin/env python3
""" Tools for reading GBT continuum data """
import numpy as np
from numpy.lib.utils import safe_eval
import os
from typing import Tuple
from astropy.io import fits
from astropy.wcs import WCS
from pprint import pprint



# --- For write out to fits
def read_rawdata(data_file: str) -> np.ndarray:
    """Read just the data (assuming metadata is known)

    Args:
        data_file (str): Data file

    Returns:
        np.ndarray: Data array
    """
    # without the meta file, assuming it's known already ...
    data_path = os.path.abspath(data_file)
    # Swapaxes to get right on-sky orientation
    print(f"Reading '{data_path}'")
    data = np.load(data_path)
    freq_n, ra_n, dec_n = data.shape
    data = data.swapaxes(1, 2)
    return data

# --- For write out to fits
def read_data(data_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read full (meta) data from file

    Args:
        data_file (str): Data file

    Returns:
        Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]: Data, freqency, RA, Dec arrays
    """
    data_path = os.path.abspath(data_file)

    print(f"Reading '{data_path}'")
    
    data = np.load(data_path)

    freq_n, ra_n, dec_n = data.shape
    # Swapaxes to get right on-sky orientation
    data = data.swapaxes(1, 2)

    info_path = data_path.replace(".npy", ".npy.meta")
    print(f"Reading '{info_path}'")
    with open(info_path) as info_file:
        infostring = info_file.readline()
        info = safe_eval(infostring)

    freq_axis = (
        info["freq_delta"] * (np.arange(freq_n) - freq_n // 2) + info["freq_centre"]
    )
    ra_axis = info["ra_delta"] * (np.arange(ra_n) - ra_n // 2) + info["ra_centre"]
    dec_axis = info["dec_delta"] * (np.arange(dec_n) - dec_n // 2) + info["dec_centre"]

    return data, freq_axis, ra_axis, dec_axis

# --- For write out in fits
def read_rawdata_wpath(data_path: str, data_name: str) -> np.ndarray:
    """Read just the data (assuming metadata is known)

    Args:
        data_path (str): Data path
        data_name (str): Data file     #without .npy
        
    Returns:
        np.ndarray: Data array
    """
    # without the meta file, assuming it's known already ...
    
    if data_name.split('.')[-1] == 'npy':
        data_name = data_name.replace('.npy', '')
    
    print(f"Reading '{data_path}' '{data_name}'")
    datafile = data_path + data_name + '.npy'
    data     = np.load(datafile)
    freq_n, ra_n, dec_n = data.shape
    # Swapaxes to get right on-sky orientation
    data = data.swapaxes(1, 2)
    return data

# --- For write out in fits
def read_data_wpath(data_path: str, data_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read full (meta) data from file

    Args:
        data_path (str): Data path
        data_file (str): Data file

    Returns:
        Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]: Data, freqency, RA, Dec arrays
    """
    #data_path = os.path.abspath(data_file)

    if data_name.split('.')[-1] == 'npy':
        data_name = data_name.replace('.npy', '')

    print(f"Reading '{data_path}' '{data_name}'")
    
    data_file =data_path + data_name + '.npy'
    data = np.load(data_file)

    freq_n, ra_n, dec_n = data.shape
    # Swapaxes to get right on-sky orientation
    data = data.swapaxes(1, 2)

    info_path = data_file.replace(".npy", ".npy.meta")
    #print(f"Reading '{info_path}'")
    with open(info_path) as info_file:
        infostring = info_file.readline()
        info = safe_eval(infostring)

    freq_axis = (
        info["freq_delta"] * (np.arange(freq_n) - freq_n // 2) + info["freq_centre"]
    )
    ra_axis  = info["ra_delta"] * (np.arange(ra_n) - ra_n // 2) + info["ra_centre"]
    dec_axis = info["dec_delta"] * (np.arange(dec_n) - dec_n // 2) + info["dec_centre"]

    return data, freq_axis, ra_axis, dec_axis


# Save data
def save_data(data_path: str, data_name: str):

    if data_name.split(".")[-1] == "npy":
        data_name = data_name.replace(".npy", "")

    data = np.save(data_path + data_name + ".npy")

    freq_n, ra_n, dec_n = data.shape

    info_file = open(data_path + data_name + ".npy.meta")
    try:
        infostring = info_file.readline()
    finally:
        info_file.close()

    info = safe_eval(infostring)

    freq_axis = (
        info["freq_delta"] * (np.arange(freq_n) - freq_n // 2) + info["freq_centre"]
    )
    ra_axis = info["ra_delta"] * (np.arange(ra_n) - ra_n // 2) + info["ra_centre"]
    dec_axis = info["dec_delta"] * (np.arange(dec_n) - dec_n // 2) + info["dec_centre"]

    # print(info)

    return data, freq_axis, ra_axis, dec_axis

# Save as fits file #
def write_to_fits(
    image: np.ndarray, ra: np.ndarray, dec: np.ndarray, freq: np.ndarray, outfile: str
) -> fits.PrimaryHDU:
    """Write data to FITS

    Args:
        image (np.ndarray): Image data (freq x dec x ra)
        ra (np.ndarray): RA array (in deg)
        dec (np.ndarray): Dec array (in deg)
        freq (np.ndarray): Freq array (in Hz)
        outfile (str): Output file name

    Returns:
        fits.PrimaryHDU: FITS HDU
    """
    w = WCS(naxis=3)
    # Python indices
    # i_r, i_d, i_f = 0, 0, 0
    l_f, l_d, l_r = image.shape
    i_r, i_d, i_f = int(np.median(np.arange(l_r))), int(np.median(np.arange(l_d))), 0
    # Fits indcies
    w.wcs.crpix = [i_r + 1, i_d + 1, i_f + 1]
    w.wcs.cdelt = [np.diff(ra)[0], np.diff(dec)[0], np.diff(freq)[0]]
    w.wcs.crval = [ra[i_r], dec[i_d], freq[i_f]]
    w.wcs.ctype = ["RA---SIN", "DEC--SIN", "FREQ"]
    header = w.to_header()
    # Ignore time reference values
    #del header["DATEREF"]
    #del header["MJDREFI"]
    #del header["MJDREFF"]
    print('Header looks like:')
    pprint(header)
    hdu = fits.PrimaryHDU(image, header=header)
    hdu.writeto(outfile, overwrite=True)
    print(f"Written to '{os.path.abspath(outfile)}'")
    return hdu

# -----#
def write_to_fits(
    image: np.ndarray, ra: np.ndarray, dec: np.ndarray, freq: np.ndarray, outfile: str
) -> fits.PrimaryHDU:
    """Write data to FITS

    Args:
        image (np.ndarray): Image data (freq x dec x ra)
        ra (np.ndarray): RA array (in deg)
        dec (np.ndarray): Dec array (in deg)
        freq (np.ndarray): Freq array (in Hz)
        outfile (str): Output file name

    Returns:
        fits.PrimaryHDU: FITS HDU
    """
    w = WCS(naxis=3)
    # Python indices
    # i_r, i_d, i_f = 0, 0, 0
    l_f, l_d, l_r = image.shape
    i_r, i_d, i_f = int(np.median(np.arange(l_r))), int(np.median(np.arange(l_d))), 0
    # Fits indcies
    w.wcs.crpix = [i_r + 1, i_d + 1, i_f + 1]
    w.wcs.cdelt = [np.diff(ra)[0], np.diff(dec)[0], np.diff(freq)[0]]
    w.wcs.crval = [ra[i_r], dec[i_d], freq[i_f]]
    w.wcs.ctype = ["RA---SIN", "DEC--SIN", "FREQ"]
    header = w.to_header()
    # Ignore time reference values
    #del header["DATEREF"]
    #del header["MJDREFI"]
    #del header["MJDREFF"]
    print('Header looks like:')
    pprint(header)
    hdu = fits.PrimaryHDU(image, header=header)
    hdu.writeto(outfile, overwrite=True)
    print(f"Written to '{os.path.abspath(outfile)}'")
    return hdu




def main():
    """Main script
    """
    import argparse

    # Parse the command line options
    descStr = """
    Tools for reading 
    """
    parser = argparse.ArgumentParser(
        description=descStr, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "data", metavar="data", type=str, help="Data file (stored in .npy format)."
    )

    parser.add_argument(
        '--print',
        action='store_true',
        help='Print data info to screen'
    )

    parser.add_argument(
        '--to_fits',
        type=str,
        default=None,
        help='Write data to this FITS file'
    )    

    args = parser.parse_args()
    data, freq, ra, dec = read_data(args.data)
    if args.print:
        print("Data shape:", data.shape)
        print("Frequency shape:", freq.shape)
        print("RA shape:", ra.shape)
        print("Dec shape:", dec.shape)
        print("Frequencies:", freq)
        print("RAs:", ra)
        print("Decs:", dec)
    if args.to_fits is not None:
        hdu = write_to_fits(data, ra, dec, freq, args.to_fits)


if __name__ == "__main__":
    main()
