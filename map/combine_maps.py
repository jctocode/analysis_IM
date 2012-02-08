"""make weighted average of data cubes"""
import numpy as np
import sys
from core import algebra
from correlate import wigglez_xcorr as wxc
import copy
import shelve
# ugly odds and ends that are needed to read the pkl file
import cPickle
from correlate.freq_slices import *
import multiprocessing


cleanmaps_fourway = {
    'root_data': "/mnt/raid-project/gmrt/kiyo/wiggleZ/maps/",
    'root_cov': "/mnt/raid-project/gmrt/kiyo/wiggleZ/maps/",
    'maplist': ["sec_A_15hr_41-73_clean_map_I",
                "sec_B_15hr_41-73_clean_map_I",
                "sec_C_15hr_41-73_clean_map_I",
                "sec_D_15hr_41-73_clean_map_I"],
    "covlist": ["sec_A_15hr_41-73_noise_inv_I",
                "sec_B_15hr_41-73_noise_inv_I",
                "sec_C_15hr_41-73_noise_inv_I",
                "sec_D_15hr_41-73_noise_inv_I"]
}

# TODO: Sec B N^-1 is erroneous so we use Sec A N^-1
# TODO: Sec A N^-1 is inverted
old_twoway = {
    'root_data': "/mnt/raid-project/gmrt/kiyo/wiggleZ/corr/",
    'root_cov': "/mnt/raid-project/gmrt/kiyo/wiggleZ/corr/",
    'maplist': ["sec_A_15hr_41-69_cleaned_clean_map_I",
              "sec_B_15hr_41-69_cleaned_clean_map_I"],
    'covlist': ["sec_A_15hr_41-69_cleaned_noise_inv_I",
              "sec_A_15hr_41-69_cleaned_noise_inv_I"],
    'multiply_cov': [-1., -1.]
}


# /mnt/raid-project/gmrt/calinliv/wiggleZ/corr/test1/
# /mnt/raid-project/gmrt/calinliv/wiggleZ/corr/test/
def make_fourway_list(root_data, root_cov,
                      map_middle = "_22hr_41-84_cleaned_clean_map_I_with_",
                      cov_middle = "_22hr_41-84_cleaned_noise_inv_I_with_"):

    pairs = [('A', 'B'), ('A', 'C'), ('A', 'D'),
             ('B', 'A'), ('B', 'C'), ('B', 'D'),
             ('C', 'A'), ('C', 'B'), ('C', 'D'),
             ('D', 'A'), ('D', 'B'), ('D', 'C')]

    fourway_split = {
        'root_data': root_data,
        'root_cov': root_cov,
        'maplist': ["sec_"+ p1 + map_middle + p2 for (p1, p2) in pairs],
        'covlist': ["sec_"+ p1 + cov_middle + p2 for (p1, p2) in pairs]
    }

    return fourway_split


def combine_maps(param_dict, fullcov=False, verbose=False):
    """combines a list of maps as a weighted mean using a specified list of
    inverse covariance weights
    fullcov indicates that it is not just the diagonal and should be squashed
    """
    print param_dict
    covlist = param_dict["covlist"]
    try:
        mul_cov_list = zip(covlist, param_dict["multiply_cov"])
        print "using user-specified covariance multipliers" + \
               repr(param_dict["multiply_cov"])
    except KeyError:
        mul_cov_list = zip(covlist, [1.] * len(covlist))

    maps = []
    for tagname in param_dict["maplist"]:
        if verbose:
            print tagname
        maps.append(algebra.make_vect(
                    algebra.load(param_dict["root_data"] + tagname + ".npy")))

    weights = []
    for cov_entry in mul_cov_list:
        if verbose:
            print cov_entry
        (tagname, multiplier) = cov_entry

        if fullcov:
            raw_weight = algebra.make_mat(
                            algebra.open_memmap(param_dict["root_cov"] + \
                                                tagname + ".npy", mode='r'))
            raw_weight = raw_weight.mat_diag()
        else:
            raw_weight = algebra.make_vect(algebra.load(
                                param_dict["root_cov"] + tagname + ".npy"))

        # zero out any messy stuff
        raw_weight *= multiplier
        raw_weight[raw_weight < 1.e-20] = 0.
        raw_weight[np.isnan(raw_weight)] = 0.
        raw_weight[np.isinf(raw_weight)] = 0.
        weights.append(raw_weight)

    prodmap = []
    for mapind in range(0, len(maps)):
        prodmap.append(maps[mapind] * weights[mapind])

    for mapind in range(1, len(maps)):
        prodmap[0] += prodmap[mapind]
        weights[0] += weights[mapind]

    algebra.compressed_array_summary(weights[0], "weight map")
    algebra.compressed_array_summary(prodmap[0], "product map")

    newmap = prodmap[0] / weights[0]

    newweights = weights[0]
    newweights[newweights < 1.e-20] = 0.
    # if the new map is nan or inf, set it and the wieghts to zero
    nan_array = np.isnan(newmap)
    newmap[nan_array] = 0.
    newweights[nan_array] = 0.
    inf_array = np.isinf(newmap)
    newmap[inf_array] = 0.
    newweights[inf_array] = 0.
    algebra.compressed_array_summary(newmap, "new map")
    algebra.compressed_array_summary(newweights, "final weight map")

    return (newmap, newweights, prodmap[0])


def make_individual():
    fourway_split = make_fourway_list('/mnt/raid-project/gmrt/calinliv/wiggleZ/corr/84_ABCD_all_15_modes/',
                                      '/mnt/raid-project/gmrt/calinliv/wiggleZ/corr/84_ABCD_all_15_modes/')
    (map_out, weights_out, prodmap_out) = combine_maps(fourway_split)
    algebra.save("combined_41-84_cleaned_clean_test.npy", map_out)
    algebra.save("combined_41-84_cleaned_noise_inv_test.npy", weights_out)
    algebra.save("combined_41-84_cleaned_product_test.npy", prodmap_out)
    #(map_out, weights_out, prodmap_out) = combine_maps(cleanmaps_fourway,
    #                                                   fullcov=True)
    #algebra.save("combined_41-73_clean_test.npy", map_out)
    #algebra.save("combined_41-73_noise_inv_test.npy", weights_out)
    #algebra.save("combined_41-73_product_test.npy", prodmap_out)


def make_modetest_combined_map():
    """combine output maps from a mode subtraction test"""
    modedir = "/mnt/raid-project/gmrt/eswitzer/wiggleZ/modetest/"
    outdir = "/mnt/raid-project/gmrt/eswitzer/wiggleZ/modetest_combined_maps_0_50/"
    dirprefix = "73_ABCD_all_"
    data_dirsuffix = "_modes_real_maponly/"
    cov_dirsuffix = "_modes_real_maponly/"
    for run_index in range(0,55,5):
        fullpath_data = modedir + dirprefix + repr(run_index) + data_dirsuffix
        fullpath_cov = modedir + dirprefix + repr(run_index) + cov_dirsuffix
        print fullpath_data, fullpath_cov
        fourway_split = make_fourway_list(fullpath_data, fullpath_cov)
        (map_out, weights_out, prodmap_out) = combine_maps(fourway_split)

        filename = outdir + "combined_41-73_cleaned_clean_" + \
                   repr(run_index) + ".npy"
        algebra.save(filename, map_out)

        filename = outdir + "combined_41-73_cleaned_noise_inv_" + \
                   repr(run_index) + ".npy"
        algebra.save(filename, weights_out)

        filename = outdir + "combined_41-73_cleaned_product_" + \
                   repr(run_index) + ".npy"
        algebra.save(filename, prodmap_out)


def make_modetest_combined_sim():
    """combine output simulated maps from a mode subtraction test"""
    modedir = "/mnt/raid-project/gmrt/eswitzer/wiggleZ/modetest/"
    outdir = "/mnt/raid-project/gmrt/eswitzer/wiggleZ/modetest_combined_maps_0_50/"
    dirprefix = "73_ABCD_all_"
    data_dirsuffix = "_modes_sim_maponly/"
    cov_dirsuffix = "_modes_real_maponly/"
    for run_index in range(0,55,5):
        fullpath_data = modedir + dirprefix + repr(run_index) + data_dirsuffix
        fullpath_cov = modedir + dirprefix + repr(run_index) + cov_dirsuffix
        print fullpath_data, fullpath_cov
        fourway_split = make_fourway_list(fullpath_data, fullpath_cov)
        (map_out, weights_out, prodmap_out) = combine_maps(fourway_split)

        filename = outdir + "combined_sim_41-73_cleaned_clean_" + \
                   repr(run_index) + ".npy"
        algebra.save(filename, map_out)

        filename = outdir + "combined_sim_41-73_cleaned_noise_inv_" + \
                   repr(run_index) + ".npy"
        algebra.save(filename, weights_out)

        filename = outdir + "combined_sim_41-73_cleaned_product_" + \
                   repr(run_index) + ".npy"
        algebra.save(filename, prodmap_out)


def add_sim_radio():
    """script: go through a list of simulations and add those to a selected map
    """
    root_file = "/mnt/raid-project/gmrt/eswitzer/wiggleZ/"
    radio_file = root_file + "modetest_combined_maps/combined_41-73_cleaned_clean_15.npy"
    root_sim = "/mnt/raid-project/gmrt/calinliv/wiggleZ/simulations/test100/"
    root_out = root_file + "simulations_plus_data/"
    radio_data = algebra.make_vect(algebra.load(radio_file))

    for simindex in range(1,101):
        simname = root_sim + "simulated_signal_map_" + \
                  repr(simindex)+"_with_beam.npy"
        filename = root_out + "simulated_signal_plusdata_map_" + \
                   repr(simindex)+"_with_beam.npy"
        simoutname = root_out + "simulated_signal_map_" + \
                   repr(simindex)+"_with_beam.npy"

        sim_data = algebra.make_vect(algebra.load(simname))
        sim_data /= 1000.
        outmap = copy.deepcopy(radio_data)
        outmap += sim_data

        algebra.save(filename, outmap)
        algebra.save(simoutname, sim_data)

        print filename


def wrap_repackage_pickle(runitem):
    (pklfile, shelvefile) = runitem
    repackage_pickle_as_shelve(pklfile, shelvefile)


def repackage_pickle_as_shelve(pklfile, shelvefile):
    """Take pickled output from Liviu's code and combine the data from various
    sources into a common shelve file. [script, not production]
    """
    print pklfile
    f = open(pklfile, "r")
    F = cPickle.load(f)
    f.close()

    # Setting axis info after pickling.
    map_file = F.params["input_root"] + "sec_A_15hr_41-73_clean_map_I.npy"
    exMap = algebra.make_vect(algebra.load(map_file))
    for Pair in F.Pairs:
        Pair.Map1.info = exMap.info
        Pair.Map2.info = exMap.info
        Pair.Noise_inv1.info = exMap.info
        Pair.Noise_inv2.info = exMap.info

    for corrindex in range(6):
        shelvename = shelvefile + "_" + repr(corrindex) + ".shelve"
        corr_shelve = shelve.open(shelvename)
        print shelvename
        corr_shelve["corr"] = F.Pairs[corrindex].corr
        corr_shelve["counts"] = F.Pairs[corrindex].counts
        corr_shelve["freq_axis"] = F.Pairs[corrindex].Map1.get_axis('freq')
        corr_shelve["params"] = F.params
        corr_shelve.close()


def reprocess_batch_pickle():
    root_dir = "/mnt/raid-project/gmrt/eswitzer/wiggleZ/modetest/"
    out_dir = "/mnt/raid-project/gmrt/eswitzer/wiggleZ/batch_runs/"

    runlist_maps = []
    runlist_sims = []
    for modeindex in range(26):
        sim_dir = "73_ABCD_all_" + repr(modeindex) + "_modes_sim3map/"
        map_dir = "73_ABCD_all_" + repr(modeindex) + "_modes_real3map/"

        runlist_maps.append((root_dir + map_dir + "New_Slices_object.pkl",
                                   out_dir + "corr_slices_mode_loss_maps/" +
                                   "modeloss_"+repr(modeindex)))
        runlist_sims.append((root_dir + sim_dir + "New_Slices_object.pkl",
                                   out_dir + "corr_slices_mode_loss_sims/" +
                                   "modeloss_"+repr(modeindex)))

    count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=count)
    pool.map(wrap_repackage_pickle, runlist_maps)
    pool.map(wrap_repackage_pickle, runlist_sims)


if __name__ == '__main__':
    repackage_pickle_as_shelve("/mnt/raid-project/gmrt/calinliv/wiggleZ/corr/73_ABCD_all_15_modes_realmap_ra_fixed/New_Slices_object.pkl",
                               "/mnt/raid-project/gmrt/eswitzer/wiggleZ/batch_runs/new_autocorr_test/radio_autocorr_15_modes")
    #make_modetest_combined_map()
    #make_modetest_combined_sim()
    #add_sim_radio()
    #reprocess_batch_pickle()
