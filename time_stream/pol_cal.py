"""Module that performs polarization calibration."""

import scipy as sp
import numpy.ma as ma
import pylab as pl
import numpy as np

import kiyopy.custom_exceptions as ce
import base_single
import map.tools
from core import fits_map

# base_single.BaseSingle is a base class that knows how to read an input file
# loop over files, scans and IFs.  It does all the input and output.  The only
# thing it doesn't know how to do is the science, which is what will be added
# here.
class Calibrate(base_single.BaseSingle) :
    """Pipeline module that corrects for polarization leakage."""
    # Here we define a bunch of stuff that BaseSingle needs to know to do its
    # thing.
    # prefix is a few letters that are added to all parameter names that are
    # read from a configureation file.
    prefix = 'pc_'
    # These are the parameters that should be read from file.  These are in
    # addition to the ones defined at the top of base_single.py.
    params_init = {'mueler_file' : 'default_fname' }

    # The base single initialization method does a bunch of stuff, but we want
    # to add one thing.  We want to read a mueler matrix from file.
    def __init__(self, parameter_file_or_dict=None, feedback=2):
        
        # Call the base_single init.
       	base_single.BaseSingle.__init__(self, parameter_file_or_dict,
                                        feedback)
        # Read in the mueler matrix file.
       	mueler_file_name = self.params['mueler_file']
       	self.mueler = mueller()
    
    # This function tells BaseSingle what science to do.  Data is a
    # core.data_block.DataBlock object.  It holds all the data for a single
    # scan and a single IF.  BaseSingle knows how to loop over all of these.
    # More on DataBlock objects in the calibrate function below.
    def action(self, Data) :
        #setting parameters
#        Data.calc_freq()
#        frequency = Data.freq/1000000
        #Center_freq = int(Data.field['CRVAL1']/1000000)

        # Generating pre-mod plots
#        pl.plot(frequency,Data.data[0,0,0,:])
#        pl.plot(frequency,Data.data[0,1,0,:])
#        pl.plot(frequency,Data.data[0,2,0,:])
#        pl.plot(frequency,Data.data[0,3,0,:])
        #Swapped self.params['file_middles'] for  str(Data.field['SCAN'])

        # Main Action
       	calibrate_pol(Data, self.mueler)
       	Data.add_history('Corrected for polarization leakage.', 
               	         ('Mueler matrix file: ' + self.params['mueler_file'],))

        # Generaing post-mod plots
#        pl.plot(frequency,Data.data[0,0,0,:]) 
#        pl.plot(frequency,Data.data[0,1,0,:]) 
#        pl.plot(frequency,Data.data[0,2,0,:]) 
#        pl.plot(frequency,Data.data[0,3,0,:]) 
#        pl.legend(("I-init","Q-init","U-init","V-init",'I-mod','Q-mod','U-mod','V-mod')) 
#        pl.xlabel("Freqency (MHz)") 
#        pl.ylabel("Polarization")
#        title0 = str(Data.field['SCAN'])+'_'+"_caloff_pol"
        #pl.suptitle(title0) 
#        pl.savefig(title0+"_comparison_test_for_3C286.png")
#        pl.clf() 

       	return Data

# m_total is the final Mueller matrix that needs to multiply by the stokes parameters.

# Note: I have it set up so that the third index represents the frequency 
# bin where 0 is the lowest frequency bin and 7 is the highest, and the 
# first and second indices represent the mueller matrix for each frquency bin. 

def mueller() :
# Mueller Matrix parameters from polcal calculations
#    deltaG = [-0.185,-0.113,-0.213,-0.295,-0.394,-0.012,0.546,0.680]
#    psi = [-21.9,179.9,-175.9,178.9,171.2,152.5,159.9,122.5]
#    alpha = [-1.2,3.5,2.9,-1.3,10.0,-6.2,-2.2,-7.5]
#    epsilon = [0.008,0.013,0.015,0.019,0.024,0.016,0.016,0.017]
#    phi = [-164.9,-4.0,-7.9,0.5,6.1,23.2,17.6,55.4]
#    CFR = [694,724,754,784,814,844,874,904]
#    delf = sp.array([(CFR[i]-1485.8) for i in range(0,8)])
#    psi = sp.array([(psi[i]-0.0086*delf[i]) for i in range(0,8)])
#    mp = sp.array([deltaG,psi,alpha,epsilon,phi,CFR])


     mp = np.loadtxt('mueller_params_calc.txt')
#     print mp
#This is a file with first index being freq, second index being parameter where the values are:
# 0 = Freq, 1 = deltaG, 2 = alpha, 3 = psi, 4 = phi, 5 = epsilon, 6 = Q_src, 7 = U_src, 8 = chi

# Matrix for rotating Mueller Matrix into Astronomical frame - assumes parallactic angle = 0
#    m_astron = sp.zeros((4,4,8), float)
#    for i in range(0,8):
#        m_astron[0,0,i] = 1
#        m_astron[3,3,i] = 1
#        m_astron[1,1,i] = -1
#        m_astron[2,2,i] = -1

# Generates Inverse Mueller Matrix for use
     freq_limit = len(mp[:,0])
#     print freq_limit
     m_total = sp.zeros((4,4,freq_limit),float)
     for i in range(0,freq_limit):
#Note that this version is based on the combined matrix I derived corrected for linear feed:
        dG = mp[i,1]
        al = mp[i,2]*sp.pi/180
        ps = mp[i,3]*sp.pi/180
        ph = mp[i,4]*sp.pi/180
        ep = mp[i,5]
        ch = mp[i,8]*sp.pi/180
        
        m_total[0,0,i] = 1
        m_total[0,1,i] = 0.5*dG*sp.cos(2*al)-2*ep*sp.sin(2*al)*sp.cos(ph-ch)
        m_total[0,2,i] = 0.5*dG*sp.sin(2*al)*sp.cos(ch)+2*ep*(sp.cos(al)*sp.cos(al)*sp.sin(ph)-sp.sin(al)*sp.sin(al)*sp.cos(ph-2*ch))
        m_total[0,3,i] = 0.5*dG*sp.sin(2*al)*sp.sin(ch)+2*ep*(sp.cos(al)*sp.cos(al)*sp.sin(ph)+sp.sin(al)*sp.sin(al)*sp.sin(ph-2*ch))
        m_total[1,0,i] = 0.5*dG
        m_total[1,1,i] = sp.cos(2*al)
        m_total[1,2,i] = sp.sin(2*al)*sp.cos(ch)
        m_total[1,3,i] = sp.sin(2*al)*sp.sin(ch)
        m_total[2,0,i] = 2*ep*sp.cos(ph+ps)
        m_total[2,1,i] = -sp.sin(2*al)*sp.cos(ps+ch)
        m_total[2,2,i] = sp.cos(al)*sp.cos(al)*sp.cos(ps)-sp.sin(al)*sp.sin(al)*sp.cos(ps+2*ch)
        m_total[2,3,i] = -sp.cos(al)*sp.cos(al)*sp.sin(ps)-sp.sin(al)*sp.sin(al)*sp.sin(ps+2*ch)
        m_total[3,0,i] = 2*ep*sp.sin(ps+ph)
        m_total[3,1,i] = -sp.sin(2*al)*sp.sin(ps+ch)
        m_total[3,2,i] = sp.cos(al)*sp.cos(al)*sp.sin(ps)-sp.sin(al)*sp.sin(al)*sp.sin(ps+2*ch)
        m_total[3,3,i] = sp.cos(al)*sp.cos(al)*sp.cos(ps)+sp.sin(al)*sp.sin(al)*sp.cos(ps+2*ch)
        M_total = sp.mat(m_total[:,:,i])
#        print M_total
        M_total = M_total.I
#        M_astron = sp.mat(m_astron[:,:,i])
#        M_total = M_astron*M_total
        for j in range(0,4):
            for k in range(0,4):
                m_total[j,k,i] = M_total[j,k]
     np.savetxt('mueller_matrix_first_row.txt', m_total[:,0,:], delimiter=' ')
     np.savetxt('mueller_matrix_second_row.txt', m_total[:,1,:], delimiter=' ')
     np.savetxt('mueller_matrix_third_row.txt', m_total[:,2,:], delimiter=' ')
     np.savetxt('mueller_matrix_fourth_row.txt', m_total[:,3,:], delimiter=' ') 
     return m_total

def calibrate_pol(Data, m_total) :
    """Subtracts a Map out of Data."""
        
    # Data is a DataBlock object.  It holds everything you need to know about
    # the data in a single scan and IF.  You should get to know them very well.
    # Data.data is a numpy masked array (see numpy documentation) and holds the
    # acctual data.  It is a 4 dimensional array.  The demensions are (in
    # order): (time, pol, cal, freq).  Each dimension can be any length which
    # you can figure out by looking at Data.dims = sp.shape(Data.data).
    # Data.field is a python dictionary that holds all the other data that you
    # might care about from the origional fits file.  For instance,
    # Data.field['CAL'] is an array with length dims[2].  It normally has
    # values ['T', 'F']. Data.field['CRVAL4'] tells you about the polarization
    # axis of Data.data.  By SDfits convension each polarization is represented
    # by an integer: 1=I, 2=Q, 3=U, 4=V, -5=XX, -6=YY, -7=XY, -8=YX.

    # Also this depends on having the polarizations rotated correctly to IQUV.
    # Some code to do this has been hacked together in the rotate_pol module,
    # but I don't trust it yet.

    # Some dimension checks.
    # We expect 4 polarizations.
    if not Data.dims[1] == 4 :
       	raise ce.DataError('Require 4 polarizations.')
    # We expect polarizations to be in order IQUV.
    if (Data.field['CRVAL4'][0] != 1 or Data.field['CRVAL4'][1] != 2 or
        Data.field['CRVAL4'][2] != 3 or Data.field['CRVAL4'][3] != 4) :
       	raise ce.DataError('Expected the polarization basis to be IQUV.')

    # A useful function that might need:
    Data.calc_freq()
    # Now data has an atribute Data.freq which is an array that gives the
    # frequency along the last axis.
          
    # Data.field['CRVAL1'] is center frequency in Hz. 
    # Data.data 4 dim array 2nd index polarization, 4th index frequency. 

    # Need to get parallactic angle:
    Data.calc_PA()
    # This gives an array (Data.PA) of PA values of length = time dim.

    for time_index in range(0,Data.dims[0]):

    #Generate a sky matrix for this time index:
        m_sky = sp.zeros((4,4))
        m_sky[0,0] = 1
        m_sky[1,1] = ma.cos(2*Data.PA[time_index]*sp.pi/180)
        m_sky[1,2] = ma.sin(2*Data.PA[time_index]*sp.pi/180)
        m_sky[2,1] = -ma.sin(2*Data.PA[time_index]*sp.pi/180)
        m_sky[2,2] = ma.cos(2*Data.PA[time_index]*sp.pi/180)
        m_sky[3,3] = 1

        M_sky = sp.mat(m_sky)
        M_sky = M_sky.I
#        print M_sky

        for cal_index in range(0,Data.dims[2]):
        # Determines the Inverse Mueller Matrix to use   
#            CenterFrequency = int(Data.field['CRVAL1']/1000000)
            for freq in range(0,Data.dims[3]):
#               if Data.freq[freq] in range(669,699):
#                  bin = 0
#               elif Data.freq[freq] in range(700,729):
#                  bin = 1
#               elif Data.freq[freq] in range(730,759):
#                  bin = 2
#               elif Data.freq[freq] in range(760,789):
#                  bin = 3
#               elif Data.freq[freq] in range(790,819):
#                  bin = 4
#               elif Data.freq[freq] in range(820,849):
#                  bin = 5
#               elif Data.freq[freq] in range(850,879):
#                  bin = 6
#               elif Data.freq[freq] in range(880,929):
#                  bin = 7
#               else :
#                  raise ce.DataError('The frequency outside viable window') 

     # Tells which mueller matrix to use. Assumes at least 1 MHz bins.
               frequency = int(Data.freq[freq]/1000) 
               freq_limit=len(m_total[0,0,:])
               bin = int((900000-frequency)*freq_limit/200000)
#               if freq_limit == 200:
#                   bin = 900-frequency
#               elif freq_limit == 260:
#                   bin = 929-frequency
#               print bin
    # Converts files into matrix format 
               STOKES = Data.data[time_index,:,cal_index,freq]       
#               print STOKES
               MUELLER = sp.mat(m_total[:,:,bin])
#               print MUELLER

    # Next there is a matrix multiplication that will generate 
    # a new set of stokes values.
               stokesmod = np.dot(MUELLER,STOKES)
               stokesmod = np.dot(M_sky,stokesmod)
#               print stokesmod
               for i in range(0,Data.dims[1]):
                    Data.data[time_index,i,cal_index,freq] = stokesmod[i]	

    # At this point the polarization values should be adjusted. 


# If this file is run from the command line, execute the main function.
if __name__=="__main__":
    import sys
    Calibrate(str(sys.argv[1])).execute()
