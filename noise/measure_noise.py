"""Measure the noise parameters of the data."""

import shelve
import multiprocessing as mp
import time as time_module

import scipy as sp
import scipy.linalg as linalg
import numpy.ma as ma

import noise_power as npow
from scipy import optimize
from kiyopy import parse_ini, utils
import kiyopy.pickle_method
import kiyopy.utils
import kiyopy.custom_exceptions as ce
import core.fitsGBT
from map import dirty_map

# XXX
import matplotlib.pyplot as plt


params_init = {
               # IO.
               "input_root" : "./testdata/",
               "file_middles" : ("testfile_guppi_combined",),
               "input_end" : ".fits",
               "output_root" : "./",
               "output_filename" : "noise_parameters.shelve",
               "scans" : (),
               "IFs" : (),
               # What parameters to measure.
               "parameters" : ["channel_var", "mean_over_f"]
               }

prefix = 'mn_'

class Measure(object) :
    """Measures the noise of data files.
    """

    def __init__(self, parameter_file_or_dict=None, feedback=2) :
        # Read the parameter file, store in dictionary named parameters.
        self.params = parse_ini.parse(parameter_file_or_dict, params_init, 
                                      prefix=prefix, feedback=feedback)
        self.feedback = feedback

    def execute(self, nprocesses=1) :
        
        params = self.params
        kiyopy.utils.mkparents(params['output_root'] + 
                               params['output_filename'])
        parse_ini.write_params(params, params['output_root'] + 'params.ini',
                               prefix=prefix)
        output_fname = params['output_root'] + params["output_filename"]
        out_db = shelve.open(output_fname)
        file_middles = params['file_middles']
        n_files = len(file_middles)
        
        n_new = nprocesses-1  # How many new processes to spawn at once.
        if n_new > 0:
            # Loop over files and spawn processes to deal with them, but make 
            # sure that only n_new processes are going at once.
            process_list = range(n_new)
            pipe_list = range(n_new)
            for ii in xrange(n_files + n_new) :
                if ii >= n_new :
                    out_db[file_middles[ii-n_new]] = pipe_list[ii%n_new].recv()
                    process_list[ii%n_new].join()
                    if process_list[ii%n_new].exitcode != 0 : 
                        raise RuntimeError("A thread failed with exit code: "
                                        + str(process_list[ii%n_new].exitcode))
                if ii < n_files :
                    input_fname = (params['input_root'] + file_middles[ii] +
                               params['input_end'])
                    Here, Far = mp.Pipe()
                    pipe_list[ii%n_new] = Here
                    process_list[ii%n_new] = mp.Process(
                        target=self.process_file, args=(input_fname, Far))
                    process_list[ii%n_new].start()
        else :
            for middle in file_middles:
                input_fname = (params['input_root'] + middle +
                               params['input_end'])
                out_db[middle] = self.process_file(input_fname)
        out_db.close()
        if self.feedback > 1 :
            print ("Wrote noise parameters to file: " 
                   + utils.abbreviate_file_path(output_fname))

    def process_file(self, file_name, Pipe=None) :
        
        try :
            band_inds = self.params["IFs"]
            parameter_names = self.params["parameters"]
            Reader = core.fitsGBT.Reader(file_name, feedback=self.feedback)
            n_bands = len(Reader.IF_set)
            if not band_inds:
                band_inds = range(n_bands)
            measured_parameters = {}
            for ii in range(n_bands):
                if ii in band_inds:
                    Blocks = Reader.read(self.params["scans"], ii)
                    Blocks[0].calc_freq()
                    n_chan = Blocks[0].dims[-1]
                    band = (int(round(Blocks[0].freq[n_chan//2]/1e6)))
                    measured_parameters[band] = measure_noise_parameters(
                            Blocks, parameter_names)
            if Pipe:
                Pipe.send(measured_parameters)
            else:
                return measured_parameters
        except :
            if Pipe:
                Pipe.send(-1)
            raise

def measure_noise_parameters(Blocks, parameters):
    """Given a set of data blocks, measure noise parameters.

    Measurement done for all polarizations but only the first cal state.
    """
    
    # Initialize the output.
    out_parameters = {}
    # Calculate the full correlated power spectrum.
    power_mat, window_function, dt, channel_means = npow.full_power_mat(
            Blocks, window="hanning", deconvolve=False, n_time=-1.05)
    # This shouldn't be nessisary, since I've tried to keep things finite in
    # the above function.  However, leave it in for now just in case.
    if not sp.alltrue(sp.isfinite(power_mat)) :
        msg = ("Non finite power spectrum calculated.  Offending data in "
               "file starting with scan %d." % (Blocks[0].field['SCAN']))
        raise ce.DataError(msg)
    # Get frequency axis and do unit conversions.
    n_time = power_mat.shape[0]
    n_chan = power_mat.shape[-1]
    frequency = npow.ps_freq_axis(dt, n_time)
    power_mat = npow.prune_power(power_mat, 0)
    power_mat = npow.make_power_physical_units(power_mat, dt)
    # Discard the mean mode.
    frequency = frequency[1:]
    power_mat = power_mat[1:,...]
    n_f = len(frequency)
    # Loop over polarizations.
    cal_ind = 0
    n_pols = power_mat.shape[1]
    for ii in range(n_pols):
        this_pol_power = power_mat[:,ii,cal_ind,:,:]
        this_pol_window = window_function[:,ii,cal_ind,:,:]
        this_pol = Blocks[0].field['CRVAL4'][ii]
        this_pol_parameters = {}
        # Now figure out what we want to measure and measure it.
        if "channel_var" in parameters:
            power_diag = this_pol_power.view()
            power_diag.shape = (n_f, n_chan**2)
            power_diag = power_diag[:,::n_chan + 1].real
            window_function_diag = this_pol_window.view()
            window_function_diag.shape = (n_time, n_chan**2)
            window_function_diag = window_function_diag[:,::n_chan + 1]
            # Integral of the power spectrum from -BW to BW.
            channel_var = sp.mean(power_diag, 0) / dt
            # If a channel is completly masked Deweight it by giving a high
            # variance
            channel_var[sp.mean(window_function_diag, 0).real < 1.e-3] = \
                        dirty_map.T_infinity
            this_pol_parameters["channel_var"] = channel_var
        if "mean_over_f" in parameters:
            this_pol_parameters["mean_over_f"] = get_mean_over_f(
                    this_pol_power, this_pol_window, frequency)
        for noise_model in parameters:
            if noise_model[:18] == "freq_modes_over_f_":
                n_modes = int(noise_model[18:])
                this_pol_parameters[noise_model] = \
                        get_freq_modes_over_f(this_pol_power, this_pol_window,
                                              frequency, n_modes)
        out_parameters[this_pol] = this_pol_parameters
    return out_parameters

def get_freq_modes_over_f(power_mat, window_function, frequency, n_modes):
    """Fines the most correlated frequency modes and fits thier noise."""
    
    n_f = len(frequency)
    d_f = sp.mean(sp.diff(frequency))
    n_chan = power_mat.shape[-1]
    n_time = window_function.shape[0]
    # Initialize the dictionary that will hold all the parameters.
    output_params = {}
    # First take the low frequency part of the spetrum matrix and average over
    # enough bins to get a well conditioned matrix.
    low_f_mat = sp.mean(power_mat[:4*n_chan,:,:].real, 0)
    # Factor the matrix to get the most correlated modes.
    e, v = linalg.eigh(low_f_mat)
    # Make sure they are sorted.
    if not sp.alltrue(sp.diff(e)[-n_modes:] >= 0):
        raise RuntimeError("Eigenvalues not sorted")
    # Power matrix striped of the biggest modes.
    reduced_power = sp.copy(power_mat)
    # Solve for the spectra of these modes.
    for ii in range(n_modes):
        this_mode_params = {}
        # Get power spectrum and window function for this mode.
        mode = v[:,-1 - ii]
        mode_power = sp.sum(mode * power_mat.real, -1)
        mode_power = sp.sum(mode * mode_power, -1)
        mode_window = sp.sum(mode[:,None]**2 * window_function, 1)
        mode_window = sp.sum(mode_window * mode[None,:]**2, 1)
        # Protect against no data.
        if sp.mean(mode_window).real < 1e-3:
            this_mode_params['amplitude'] = 0.
            this_mode_params['index'] = 0.
            this_mode_params['f_0'] = 1.
            this_mode_params['thermal'] = dirty_map.T_infinity
        else:
            # Fit the spectrum.
            p = fit_overf_const(mode_power, mode_window, frequency)
            # Put all the parameters we measured into the output.
            this_mode_params['amplitude'] = p[0]
            this_mode_params['index'] = p[1]
            this_mode_params['f_0'] = p[2]
            this_mode_params['thermal'] = p[3]
        this_mode_params['mode'] = mode
        output_params['over_f_mode_' + str(ii)] = this_mode_params
        # Remove the mode from the power matrix.
        tmp_amp = sp.sum(power_mat * mode, -1)
        tmp_amp = sp.sum(tmp_amp * mode, -1)
        reduced_power -= tmp_amp[:,None,None] * mode[:,None] * mode
    # Now that we've striped the noisiest modes, measure the auto power
    # spectrum, averaged over channels.
    auto_spec_mean = reduced_power.view()
    auto_spec_mean.shape = (n_f, n_chan**2)
    auto_spec_mean = auto_spec_mean[:,::n_chan + 1].real
    auto_spec_mean = sp.mean(auto_spec_mean, -1)
    diag_window = window_function.view()
    diag_window.shape = (n_time, n_chan**2)
    diag_window = diag_window[:,::n_chan + 1]
    auto_spec_window = sp.mean(diag_window, -1)
    if sp.mean(auto_spec_window).real < 1.e-3:
        auto_cross_over = 0.
        auto_index = 0.
    else:
        auto_spec_params = fit_overf_const(auto_spec_mean, auto_spec_window,
                                           frequency)
        if (auto_spec_params[0] < 0 or auto_spec_params[3] < 0 or
            auto_spec_params[1] > -0.2):
            auto_cross_over = 0.
            auto_index = 0.
        else:
            auto_index = auto_spec_params[1]
            auto_cross_over = auto_spec_params[2] * (auto_spec_params[0]
                                     / auto_spec_params[3])**(-1./auto_index)
            if auto_cross_over < 2*d_f:
                auto_index = 0.
                auto_cross_over = 0.
    output_params['all_channel_index'] = auto_index
    output_params['all_channel_corner_f'] = auto_cross_over
    # Finally measure the thermal part of the noise in each channel.
    cross_over_ind = sp.digitize([auto_cross_over * 4], frequency)[0]
    cross_over_ind = max(cross_over_ind, n_f // 2)
    cross_over_ind = min(cross_over_ind, int(9. * n_f / 10.))
    thermal = reduced_power[cross_over_ind:,:,:].real
    n_high_f = thermal.shape[0]
    thermal.shape = (n_high_f, n_chan**2)
    thermal = sp.mean(thermal[:,::n_chan + 1], 0)
    thermal_norms = sp.mean(diag_window, 0).real
    bad_inds = thermal_norms < 1.e-3
    thermal_norms[bad_inds] = 1.
    thermal /= thermal_norms
    thermal[bad_inds] = dirty_map.T_infinity
    output_params['thermal'] = thermal
    return output_params


def get_mean_over_f(power_mat, window_function, frequency) :
    """Measures noise parameters of a set of scans assuming correlated 1/f.

    Fits model to full f,f' cross power spectrum matrix.  Ignores the fact that
    the entries of this matrix are horribly correlated.
    
    Parameters
    ----------
    power_mat : array
        Pruned of negitive frequencies and 0 frequency.
    """
    
    n_f = len(frequency)
    dt = 1.0 / (frequency[-1] * 2)
    n_chan = power_mat.shape[-1]
    f_0 = 1.0
    n_time = window_function.shape[0]
    # First find the auto correlation part.
    auto_corr = power_mat.view()
    auto_corr.shape = (n_f, n_chan*n_chan)
    auto_corr = auto_corr[:,::n_chan + 1].real    
    # First fit to the thermal free correlated part (cross terms).
    # Take mean excluding auto correlation terms.
    correlated_part = sp.sum(sp.sum(power_mat, -1), -1)
    mean_window = sp.sum(sp.sum(window_function, -1), -1)
    for ii in xrange(n_chan) :
        correlated_part -= power_mat[:, ii, ii]
        mean_window -= window_function[:, ii, ii]
    correlated_part /= (n_chan - 1)*n_chan
    correlated_part = correlated_part.real
    mean_window /= (n_chan - 1)*n_chan
    # Fit power law to this.
    def over_f_spec(params, window) :
        spec = npow.overf_power_spectrum(params[0], params[1],
                                                f_0, dt, n_time)
        spec = npow.convolve_power(spec, window, 0)
        spec = npow.prune_power(spec)
        spec = spec[1:].real
        return spec
    def correlated_part_residuals(params):
        return (correlated_part - over_f_spec(params, mean_window))/weights
    # Initial parameter guesses.
    over_f_params = sp.zeros(2)
    over_f_params[0] = sp.mean(correlated_part * f_0 / frequency)
    over_f_params[1] = -1.0
    thermal_params = sp.mean(auto_corr, 0)
    # Initial weights.
    old_weights = abs(correlated_part)
    old_weights[old_weights < 1e-16] = 1
    old_thermal_weights = abs(auto_corr)
    old_thermal_weights[old_thermal_weights < 1e-16] = 1
    # Perform fit iteratively, updating the weights.  We do a two step fit,
    # doing the thermal part after the correlated part (much faster).
    auto_over_f_specs = sp.empty((n_f, n_chan))
    for ii in range(6) :
        # Update the weights for the correlated part.
        # Memory to eliminate oscillations.  Seems to be nessisary.
        new_weights = (abs(over_f_spec(over_f_params, mean_window)) +
                       abs(sp.mean(thermal_params)))
        new_weights[new_weights < 1e-16] = 1
        weights = 2*sp.sqrt(old_weights * new_weights)
        old_weights = new_weights
        # Fit for the correlated part.
        over_f_params, ier = sp.optimize.leastsq(correlated_part_residuals, 
                                                 over_f_params)
        # Update the weights for the thermal part.
        for ii in xrange(n_chan):
            auto_over_f_specs[:,ii] = over_f_spec(over_f_params,
                                                  window_function[:,ii,ii])
        new_thermal_weights = auto_over_f_specs + thermal_params
        new_thermal_weights[new_thermal_weights < 1e-16] = 1
        thermal_weights = 2*sp.sqrt(old_thermal_weights * new_thermal_weights)
        old_thermal_weights = new_thermal_weights
        # Fit for the thermal part.
        thermal_part = auto_corr - auto_over_f_specs
        thermal_params = (sp.sum(thermal_part/thermal_weights**2, 0)
                          / sp.sum(1.0/thermal_weights**2, 0))
        # XXX
        #plt.figure()
        #plt.loglog(frequency, correlated_part, 'b.')
        #plt.loglog(frequency, -correlated_part, 'r.')
        #plt.loglog(frequency, over_f_spec(over_f_params), 'g')
        #plt.loglog(frequency, sp.mean(thermal_params)*sp.ones(n_f), 'y')
    #print over_f_params, thermal_params
    #plt.show() 
    # XXX
    # Unpack and repack results and return.
    output_params = {}
    output_params['amplitude'] = over_f_params[0]
    output_params['index'] = over_f_params[1]
    output_params['thermal'] = thermal_params
    output_params['f_0'] = f_0

    return output_params

def fit_overf_const(power, window, freq):
    """Fit $A*f**alpha + C$ to a 1D power spectrum.
    
    Power spectrum should be real and contain only positive fequencies.  Window
    function should be complex and have both positive and negitive frequencies.
    """
    

    n_f = len(freq)
    f_0 = 1.0
    dt = 1./2./freq[-1]
    n_time = len(window)
    # Minimum spectral index.
    min_index = 0.1
    if n_f != n_time // 2 - 1:
        raise RuntimeError("Power spectrum and window sizes incompatible.")
    # Make sure the window function is well behaved.
    if ((not window[0].real > 0)
        or (not abs(window[0].imag) < 1.e-6 * window[0].real)):
        raise ValueError("Badly behaved window function.")
    # Spectrum function we will fit too.
    # Instead of fitting for the index directly, we sue another parameter that
    # does not admit index >= -0.2.  This avoids a degeneracy with thermal when
    # index = 0.
    # Also force the amplitudes to be positive by squaring.
    def model(params):
        a = params[0]**2
        i = -(params[1]**2 + min_index)
        t = params[2]**2
        spec = npow.overf_power_spectrum(a, i, f_0, dt, n_time)
        spec += t
        spec = npow.convolve_power(spec, window)
        spec = npow.prune_power(spec)
        spec = spec[1:].real
        return spec
    # Residuals function.
    def residuals(params):
        return (power - model(params))/weights
    # Get good initial guesses for the parameters.  It is extreemly important
    # to do well here.
    params = sp.empty(3, dtype=float)
    # First guess the thermal level by taking the mean at high frequencies.
    norm = n_time / sp.sum(window.real)
    #print norm
    params[2] = sp.sqrt(abs(sp.mean(power[-n_f//10:]) * norm))
    params[1] = sp.sqrt(1 - min_index) # Corresponds to index = -1
    params[0] = sp.sqrt(abs(sp.mean((power * freq)[:n_f//10]) * norm))
    old_weights = abs(power)
    # Iteratively fit then update the weights.
    for ii in range(4):
        new_weights = abs(model(params))
        weights = old_weights + new_weights
        old_weights = new_weights
        # XXX
        #plt.figure()
        #plt.loglog(freq, power)
        #plt.loglog(freq, model(params))
        #plt.figure()
        #plt.semilogx(freq, residuals(params))
        #print params
        ###
        params, cov_x, info, mesg, ier = sp.optimize.leastsq(residuals, 
                                                    params, full_output=True)
    #plt.figure()
    #plt.loglog(freq, power)
    #plt.loglog(freq, model(params))
    #print params
    #plt.show()
    # Check that a solution was found.
    #if ier not in (1, 2, 3, 4):
    #    raise RuntimeError("Could not find a solution. " + repr(params))
    # Unpack results and return.
    amp = params[0]**2
    index = -(params[1]**2 + min_index)
    thermal = params[2]**2
    return amp, index, f_0, thermal



# If this file is run from the command line, execute the main function.
if __name__ == "__main__":
    import sys
    Measure(str(sys.argv[1])).execute()
               
