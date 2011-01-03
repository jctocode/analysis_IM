"""Module that subtracts map data from time_stream data, leaving only noise.
"""

import cPickle

import scipy as sp
import numpy.ma as ma

import kiyopy.custom_exceptions as ce
import base_single
import map.tools
from core import fits_map

class Subtract(base_single.BaseSingle) :
    """Pipeline module subtracts a map from time stream data.

    This module reads in a map and times stream data.  It then subtracts the
    signal part off of each time bin using the pointing information and the
    map.  This should leave only noise and map residuals (signal that isn't in
    the map).
    """
    
    prefix = 'sm_'
    params_init = {
                   'map_file' : 'testfile_map.fits',
                   'solve_for_gain' : False,
                   # Empty string to not write an output.
                   'gain_output_end' : ''
                   # XXX: What about if I start subtracing off a linear piece.
                   }

    # Add extra stuff to the constructor.
    def __init__(self, parameter_file_or_dict=None, feedback=2):
        
        # Call the base_single init.
        base_single.BaseSingle.__init__(self, parameter_file_or_dict,
                                        feedback)
        # Read in the calibration file.
        map_file_name = self.params['map_file']
        self.Map = fits_map.read(map_file_name, 1, feedback=self.feedback)

    def action(self, Data) :
        if (not self.params['solve_for_gain'] or
            self.params['gain_output_end'] is '') :
            sub_map(Data, self.Map, self.params['solve_for_gain'])
        else :
            block_gain = {}
            Data.calc_freq()
            block_gain['freq'] = sp.copy(Data.freq)
            block_gain['time'] = Data.field['TIMESTAMP']
            block_gain['scan'] = Data.field['SCAN']
            block_gain['gain'] = sub_map(Data, self.Map, True)
            self.gain_list.append(block_gain)

        Data.add_history('Subtracted map from data.', 
                         ('Map file: ' + self.params['map_file'],))
        return Data

    # Overwrite the base single process file method so we can also pickle the
    # gains we've solved for.
    def process_file(self, file_ind) :
        
        # Initialize a list to hold the gains calculated in sub_map.
        self.gain_list = []
        # Do the normal thing from the base class.
        base_single.BaseSingle.process_file(self, file_ind)
        # Pickle the gain list.
        if (self.params['solve_for_gain'] and
            not self.params['gain_output_end'] is '') :
            file_middle = self.params['file_middles'][file_ind]
            gain_fname = (self.params['output_root'] + file_middle
                          + self.params['gain_output_end'])
            f = open(gain_fname, 'w')
            cPickle.dump(self.gain_list, f, 0)




def sub_map(Data, Map, correlate=False) :
    """Subtracts a Map out of Data."""

    # Some dimension checks.  Eventually may want to have a tuple of maps, one
    # for each polaization.  For now, only use I.
    if ((Data.dims[1] > 1) or 
        (hasattr(Map, '__iter__') and len(Map) > 1)):
        raise NotImplementedError('Multiple polarizations not supported.')
    pol_ind = 0
    if hasattr(Map, '__iter__') :
        Map = Map[0]
    if Map.field['POL'].item() != Data.field['CRVAL4'][0] :
        raise ce.DataError("Polarization types don't match.")
        
    Data.calc_pointing()
    Data.calc_freq()
    centre, shape, spacing = map.tools.get_map_params(Map)
    # These indices are the length of the time axis. Integer indicies.
    ra_ind = map.tools.calc_inds(Data.ra, centre[0], shape[0], spacing[0])
    dec_ind = map.tools.calc_inds(Data.dec, centre[1], shape[1], spacing[1])
    # Length of the data frequency axis.
    freq_ind = map.tools.calc_inds(Data.freq, centre[2], shape[2], spacing[2])
    # Exclude indices that are off map or out of band. Boolian indices.
    on_map_inds = sp.logical_and(sp.logical_and(ra_ind>=0, ra_ind<shape[0]),
                                 sp.logical_and(dec_ind>=0, dec_ind<shape[1]))
    in_band_inds = sp.logical_and(freq_ind >= 0, freq_ind < shape[2])
    # Broadcast to the same shape and combine.
    covered_inds = sp.logical_and(on_map_inds[:, sp.newaxis], 
                                  in_band_inds[sp.newaxis, :])
    # Make an array of map data the size of the time stream data.
    submap = Map.data[ra_ind[on_map_inds], dec_ind[on_map_inds], :]
    submap = submap[:, freq_ind[in_band_inds]]
    # submap is the size of the data that is on the map.  Expand to full size 
    # of data.
    subdata = ma.zeros(sp.shape(covered_inds))
    subdata[covered_inds] = submap.flatten()
    del submap
    subdata[sp.logical_not(covered_inds)] = ma.masked
    # If solving for gains, need a place to store them.
    if correlate :
        out_gains = sp.empty(Data.dims[1:4])

    # Now start using the actual data.  Loop over cal and pol indicies.
    for cal_ind in range(Data.dims[2]) :
        data = Data.data[:,pol_ind, cal_ind, :]
        # Correlate to solve for an unknown gain.
        if correlate :
            # Find the common mask.
            un_mask = sp.logical_not(sp.logical_or(data.mask, subdata.mask))
            # Subtract out the mean from both data and map.
            tsubdata = subdata - sp.sum(un_mask*subdata, 0)/sp.sum(un_mask, 0)
            tdata = data - sp.sum(un_mask*data, 0)/sp.sum(un_mask, 0)
            gain = (sp.sum(un_mask*tsubdata*tdata, 0) 
                    / sp.sum(un_mask*tsubdata*tsubdata, 0))
            out_gains[pol_ind,cal_ind,:] = gain
        else :
            gain = 1.0

        # Now do the subtraction and mask the off map data.
        data -= gain*subdata
    if correlate :
        return out_gains
