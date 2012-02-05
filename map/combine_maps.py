from utils import data_paths
from core import algebra
import numpy as np

def combine_maps(source_key, outputdir_key, alt_weight=None,
                 signal='map', weight='noise_inv', divider_token=";",
                 fullcov=False):
    r"""
    `source_key` is the file db key for the maps to combine
    `outputdir_key` is the file db path for the combined maps
    `alt_weight` is an optional alternate weighting for the maps
    `signal` is the tag in the file db entry for the signal maps
    `weight` is the tag in the file db entry for the N^-1 weights
    `fullcov` uses a memory map for large N^-1 and pulls the diagonal
    `divider_token` is the token that divides the map section name
            from the data type e.g. "A_with_B;noise_inv"
    """
    datapath_db = data_paths.DataPath()
    source_fdb = datapath_db.fetch(source_key, intend_read=True,
                                   silent=True)
    outputdir = datapath_db.fetch(outputdir_key, intend_write=True)
    source_fdict = source_fdb[1]

    # accumulate all the files to combine
    weightkeys = {}
    signalkeys = {}
    for filekey in source_fdb[0]:
        if divider_token in filekey:
            data_type = filekey.split(divider_token)[1]
            map_section = filekey.split(divider_token)[0]

            if data_type == signal:
                signalkeys[map_section] = source_fdict[filekey]

            if data_type == weight:
                weightkeys[map_section] = source_fdict[filekey]

    signal_list = []
    weight_list = []
    for mapkey in signalkeys:
        signalfile = signalkeys[mapkey]
        weightfile = weightkeys[mapkey]
        print "loading pair: %s %s" % (signalfile, weightfile)
        signal_list.append(algebra.make_vect(algebra.load(signalfile)))

        if fullcov:
            raw_weight = algebra.make_mat(
                            algebra.open_memmap(weightfile))
            raw_weight = raw_weight.mat_diag()
        else:
            raw_weight = algebra.make_vect(algebra.load(weightfile))

        # zero out any messy stuff
        raw_weight[raw_weight < 1.e-20] = 0.
        raw_weight[np.isnan(raw_weight)] = 0.
        raw_weight[np.isinf(raw_weight)] = 0.
        weight_list.append(raw_weight)

    prodmap = []
    for mapind in range(0, len(signal_list)):
        prodmap.append(signal_list[mapind] * weight_list[mapind])

    print "CHECK THESE: %d %d %d" % (len(signal_list), len(weight_list),
                                     len(prodmap))

    cumulative_product = algebra.zeros_like(prodmap[0])
    cumulative_weight = algebra.zeros_like(prodmap[0])
    for mapind in range(0, len(signal_list)):
        cumulative_product += prodmap[mapind]
        cumulative_weight += weight_list[mapind]

    algebra.compressed_array_summary(cumulative_weight, "weight map")
    algebra.compressed_array_summary(cumulative_product, "product map")

    newmap = cumulative_product / cumulative_weight

    cumulative_weight[cumulative_weight < 1.e-20] = 0.
    cumulative_product[cumulative_weight < 1.e-20] = 0.

    # if the new map is nan or inf, set it and the wieghts to zero
    nan_array = np.isnan(newmap)
    newmap[nan_array] = 0.
    cumulative_product[nan_array] = 0.
    cumulative_weight[nan_array] = 0.
    inf_array = np.isinf(newmap)
    newmap[inf_array] = 0.
    cumulative_product[inf_array] = 0.
    cumulative_weight[inf_array] = 0.
    algebra.compressed_array_summary(newmap, "new map")
    algebra.compressed_array_summary(cumulative_product, "final map * weight")
    algebra.compressed_array_summary(cumulative_weight, "final weight map")

    signal_out = "%s/%s_signal.npy" % (outputdir, source_key)
    prodmap_out = "%s/%s_product.npy" % (outputdir, source_key)
    weight_out = "%s/%s_weight.npy" % (outputdir, source_key)
    ones_out = "%s/%s_ones.npy" % (outputdir, source_key)
    algebra.save(signal_out, newmap)
    algebra.save(prodmap_out, cumulative_product)
    algebra.save(weight_out, cumulative_weight)
    algebra.save(ones_out, algebra.ones_like(newmap))


if __name__ == '__main__':
    #fieldlist = ['15hr', '22hr']
    #fieldlist = ['22hr']
    fieldlist = ['15hr']

    # fluxcal cleaned
    #for field in fieldlist:
    #    for mode_num in range (0, 55, 5):
    #        source_key = 'GBT_%s_map_cleaned_fluxcal_%dmode' % (field, mode_num)
    #        parent = 'GBT_combined_%s_maps_Eric' % field
    #        combine_maps(source_key, parent)

    #for field in fieldlist:
    #    for mode_num in range (0, 55, 5):
    #        source_key = 'GBT_%s_map_cleaned_noconv_fluxcal_%smode' % (field, mode_num)
    #        parent = 'GBT_combined_%s_maps_Eric' % field
    #        combine_maps(source_key, parent)

    # sims with fluxcal cleaned
    #for field in fieldlist:
    #    for mode_num in range (0, 55, 5):
    #        source_key = 'sim_%s_cleaned_fluxcal_%dmode' % (field, mode_num)
    #        parent = 'GBT_combined_%s_maps_Eric' % field
    #        combine_maps(source_key, parent)

    #for field in fieldlist:
    #    for mode_num in range (0, 55, 5):
    #        source_key = 'sim_%s_cleaned_noconv_fluxcal_%smode' % (field, mode_num)
    #        parent = 'GBT_combined_%s_maps_Eric' % field
    #        combine_maps(source_key, parent)

    #for field in fieldlist:
    #    for mode_num in range (0, 55, 5):
    #        source_key = 'GBT_%s_map_cleaned_%dmode' % (field, mode_num)
    #        parent = 'GBT_combined_%s_maps_Eric' % field
    #        combine_maps(source_key, parent)

    #for field in fieldlist:
    #    for mode_num in range (0, 55, 5):
    #        source_key = 'GBT_%s_map_cleaned_noconv_%smode' % (field, mode_num)
    #        parent = 'GBT_combined_%s_maps_Eric' % field
    #        combine_maps(source_key, parent)

    #for field in fieldlist:
    #    for mode_num in range (0, 55, 5):
    #        source_key = 'sim_%s_cleaned_%dmode' % (field, mode_num)
    #        parent = 'GBT_combined_%s_maps_Eric' % field
    #        combine_maps(source_key, parent)

    #for field in fieldlist:
    #    for mode_num in range (0, 55, 5):
    #        source_key = 'sim_%s_cleaned_noconv_%smode' % (field, mode_num)
    #        parent = 'GBT_combined_%s_maps_Eric' % field
    #        combine_maps(source_key, parent)

    # old stuff
    #fieldlist = ['15hr']
    #for field in fieldlist:
    #    for mode_num in range (0, 55, 5):
    #        source_key = 'GBT_%s_map_cleaned_nomeanconv_%smode' % (field, mode_num)
    #        parent = 'GBT_combined_%s_maps_Eric' % field
    #        combine_maps(source_key, parent)

    fieldlist = ['15hr']
    for field in fieldlist:
        for mode_num in range (0, 55, 5):
            source_key = 'GBT_%s_map_cleaned_optimalmap_%smode' % (field, mode_num)
            parent = 'GBT_combined_%s_maps_Eric' % field
            combine_maps(source_key, parent)
