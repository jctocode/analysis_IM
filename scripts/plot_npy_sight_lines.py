#This script generates a set of maps in RA and DEC for each frequency in the map data. Can be used to make plots of maps.

import pylab
import pyfits
import sys
from numpy import *
#from core import algebra as al
#from utils import * 
import core.algebra as al
import scipy
import scipy.optimize as opt
import scipy.stats as stat
#from pylab import *

def function(freq,params):
   base = 750./freq
   function = params[0]*power(base,params[1])
   return function

def residuals(params,freq,signal):
   err = signal-function(freq,params)
   return err

filename1 = sys.argv[1]

filename2 = filename1.split('.')[0]

array = al.load(filename1)
array = al.make_vect(array)

#print array
#   creates a 3D array with indices (freq, ra, dec)

ras = array.get_axis('ra')
decs = array.get_axis('dec')
freqs = array.get_axis('freq')
freqs = freqs/1e6


#total_outliers = array
#total_freq = []
#total_ra = []
#total_dec = []
sizing = scipy.zeros((len(ras),len(decs)))

for slice, dec in enumerate(decs):
    for line, ra in enumerate(ras):
        data = array[:,line,slice]
        bad_pts = logical_or(isnan(data),isinf(data))
        good_pts = where(logical_not(bad_pts))
        data = data[good_pts]
        sizing[line,slice] = len(data)

#print sizing
total_outliers = scipy.zeros((4094,len(ras),len(decs)))        
total_freq = scipy.zeros((4094,len(ras),len(decs)))
total_stats = scipy.zeros((2,len(ras),len(decs)))

for slice, dec in enumerate(decs):
    for line, ra in enumerate(ras):
        # Now have "slice" being the dec index, "line" being the ra index
# Making initial sight line plots
#        pylab.plot(freqs,array[:,line,slice])
#        pylab.ylim(-1.0,1.0)
#        pylab.savefig(filename2+'_'+str(ra)+'_'+str(dec)+'_.png')
#        pylab.clf()

# Subtracting off the "first mode" power law slope to the data
        data = array[:,line,slice]
#       total_outliers[:,line,slice] = array[:,line,slice]
        bad_pts = logical_or(isnan(data),isinf(data))
#        print bad_pts        
        good_pts = where(logical_not(bad_pts))
        data = data[good_pts]
#        total_outliers = total_outliers[good_pts]
#        total_outliers = total_outliers[good_pts]
#        mean_sig = mean(data)
#        print mean_sig
#        data = data/mean_sig
#        print data
        freqs_good = freqs[good_pts]
#        freqs_good = freqs
        params0=[19.6,0.495]
        plsq = opt.leastsq(residuals,params0,args=(freqs_good,data),full_output=1,maxfev=5000)
#        print ra, dec, plsq[0]
        remainder = data-function(freqs_good,plsq[0])
 #       pylab.plot(freqs_good,remainder)
#        pylab.ylim(-0.2,0.2)
#        pylab.savefig(filename2+'_dev_'+str(ra)+'_'+str(dec)+'.png')
#        pylab.clf()

#Applying a filter to the remainder to extract the significant points
#        filter = data
        filter = remainder
        for f in range(5, len(data)-5):
#            filter[f] = filter[f]-filter[f-1]
#            filter[f] = filter[f]-0.5*(filter[f+1]+filter[f-1])
            filter[f] = filter[f]-0.1*(filter[f+1]+filter[f+2]+filter[f+3]+filter[f+4]+filter[f+5]+filter[f-1]+filter[f-2]+filter[f-3]+filter[f-4]+filter[f-5])
        filter[0] = 0
        filter[1] = 0
        filter[2] = 0
        filter[3] = 0
        filter[4] = 0
        filter[-1] = 0
        filter[-2] = 0
        filter[-3] = 0
        filter[-4] = 0
        filter[-5] = 0
#        print len(filter)
#       pylab.plot(freqs_good,filter)
#       pylab.ylim(-0.2,0.2)
#       pylab.savefig(filename2+'_filtered_'+str(ra)+'_'+str(dec)+'.png')
#       pylab.clf()
#       pylab.hist(filter,200,normed=1)
#       pylab.savefig(filename2+'_hist_'+str(ra)+'_'+str(dec)+'.png')
#       pylab.clf()
#        pylab.hist(filter,bins=200, range=(-1.0,-0.05))
#        pylab.savefig(filename2+'_hist_trunc_'+str(ra)+'_'+str(dec)+'.png')
#        pylab.clf()

# Manipulating the histogram data for fitting gaussians
#        mean = stat.gmean(filter)
#        var = stat.variation(filter)
        f_hist = stat.histogram(filter,500)
#        print f_hist
        scale = arange(f_hist[1],f_hist[1]+500*f_hist[2],f_hist[2])
#        print len(scale), len(f_hist[0])
        hist_out = []
        for l in range(0,len(f_hist[0])):
            hist_out.append(f_hist[0][l])
#        print hist_out
#        print f_hist
        if len(f_hist) !=len(scale):
            if len(f_hist[0])>len(scale):
                scale.append(f_hist[1]+500*f_hist[2])
                print scale
            elif len(f_hist[0])<len(scale):
                hist_out.append(0.0)                
#                print len(scale),len(f_hist[0])
        fitfunc = lambda p,x: p[0]*exp(-(p[1]-x)**2/(2*p[2]**2))
        errfunc = lambda p,x,y: fitfunc(p,x)-y
#       pylab.plot(scale,hist_out)
#        a =1.
#        mean = float(mean)
#        var = float(var)
#        print a
#        p0 = [a,mean,var]
        p0 = [100.0,0.001,0.001]
#        print any(isnan(f_hist[0])),any(isinf(f_hist[0])), any(isnan(scale)), any(isinf(scale))
        fit = opt.leastsq(errfunc,p0[:],args=(scale,hist_out),maxfev=50000)
#        print mean,var
#        print fit[0]
#        print '________________'
#       pylab.plot(scale,fitfunc(fit[0],scale))
#       pylab.xlim(-0.2,0.2)
#       pylab.savefig(filename2+'_gauss_fit_'+str(ra)+'_'+str(dec)+'.png')
#       pylab.clf()
#        pylab.plot(scale,abs(fitfunc(fit[0],scale)-f_hist[0]))
#        pylab.savefig(filename2+'_gauss_resid_'+str(ra)+'_'+str(dec)+'.png')
#        pylab.clf()

# Isolating frequencies that may be absorption features
        #pick limit based on gaussian. start at 2 sigma
        edge = fit[0][1]-3*fit[0][2]
#       edge = -1

        outliers = []
        out_freq = []
        out_ra = []
        out_dec = []
        for f in range(0,len(data)):
            if filter[f]<edge:
#                if freqs_good[f]<850:
                if freqs_good[f]<795:
                    outliers.append(filter[f])
                    out_freq.append(freqs_good[f])
                    out_ra.append(ra)
                    out_dec.append(dec)
                elif 815>freqs_good[f]>798:
                    outliers.append(filter[f])
                    out_freq.append(freqs_good[f])
                    out_ra.append(ra)
                    out_dec.append(dec)
                elif freqs_good[f]>819:
                    outliers.append(filter[f])
                    out_freq.append(freqs_good[f])
                    out_ra.append(ra)
                    out_dec.append(dec)
        for f in range(0,len(filter)):
            total_outliers[f,line,slice] = filter[f]
            total_freq[f,line,slice] = freqs_good[f]
        total_stats[:,line,slice] = [fit[0][1],fit[0][2]]
#        for f in range(0,len(freqs_good)):

#            total_ra.append(ra)
#            total_dec.append(dec) 
        cutoff_out = []
        cutoff_freq = []
        for f in range(0,len(outliers)):
            if 900>out_freq[f]>800:
                cutoff_out.append(outliers[f])
                cutoff_freq.append(out_freq[f])
#       print ra,dec
#       print len(outliers)
#       if len(outliers)<200:
#            print ra, dec
#            print outliers
#            print out_freq
#          print cutoff_out
#          print cutoff_freq
#       print '__________'


##Now total_outliers is the filtered data, total_freq is the corresponding frequencies, total_stat is mean and std for each pixel
#print total_outliers[100,1,1]
#print total_freq
#print total_ra
#print total_dec
#print total_stats

#Want to create an array that will simply flag which frequencies are "bad"
total_3sig_outliers = scipy.zeros((len(total_outliers[:,0,0]),len(total_outliers[0,:,0]),len(total_outliers[0,0,:])))
#outlier_freqs = scipy.zeros(len(total_outliers[:,0,0]))
#print len(total_3sig_outliers[:,0,0]),len(total_3sig_outliers[0,:,0]),len(total_3sig_outliers[0,0,:])
num = scipy.zeros(len(total_freq[:,0,0]))
for f in range(0,len(total_freq[:,0,0])):
    for i in range(0,len(total_stats[0,:,0])):
        for j in range(0,len(total_stats[0,0,:])):
            if total_outliers[f,i,j]<(total_stats[0,i,j]-3*total_stats[1,i,j]):
                total_3sig_outliers[f,i,j] = 1.0
    num[f] = sum(total_3sig_outliers[f])
#    print num
#    if num>3.0:
#        outlier_freqs[f] = 1
outlier_freqs = where(num<4,1,0)
#print outlier_freqs
good_freq = len(compress(outlier_freqs,total_outliers[:,0,0]))
limited_tot_out = scipy.zeros((good_freq,len(total_outliers[0,:,0]),len(total_outliers[0,0,:])))
limited_tot_freq = scipy.zeros((good_freq,len(total_outliers[0,:,0]),len(total_outliers[0,0,:])))
for i in range(0,len(total_outliers[0,:,0])):
    for j in range(0,len(total_outliers[0,0,:])):
#        test = compress(outlier_freqs,total_outliers[:,i,j])
#        print len(test)
        limited_tot_out[:,i,j] = compress(outlier_freqs,total_outliers[:,i,j])
        limited_tot_freq[:,i,j] = compress(outlier_freqs,total_freq[:,i,j])   

#now limited_tot_out is the total_outliers data with the outlier bad frequencies removed. Need to redo histograms
outliers = []
out_freq = []
out_ra = []
out_dec = []

for slice, dec in enumerate(decs):
    for line, ra in enumerate(ras):
        f_hist = stat.histogram(limited_tot_out[:,line,slice],500)
#        print f_hist
        scale = arange(f_hist[1],f_hist[1]+500*f_hist[2],f_hist[2])
#        print len(scale), len(f_hist[0])
        hist_out = []
        for l in range(0,len(f_hist[0])):
            hist_out.append(f_hist[0][l])
#        print hist_out
#        print f_hist
        if len(f_hist) !=len(scale):
            if len(f_hist[0])>len(scale):
                scale.append(f_hist[1]+500*f_hist[2])
#                print scale
            elif len(f_hist[0])<len(scale):
                hist_out.append(0.0)
#                print len(scale),len(f_hist[0])
        fitfunc = lambda p,x: p[0]*exp(-(p[1]-x)**2/(2*p[2]**2))
        errfunc = lambda p,x,y: fitfunc(p,x)-y
        pylab.scatter(scale,hist_out)
#        a =1.
#        mean = float(mean)
#        var = float(var)
#        print a
#        p0 = [a,mean,var]
        p0 = [100.0,0.001,0.001]
#        print any(isnan(f_hist[0])),any(isinf(f_hist[0])), any(isnan(scale)), any(isinf(scale))
        fit = opt.leastsq(errfunc,p0[:],args=(scale,hist_out),maxfev=50000)
#        print mean,var
#        print fit[0]
#        print '________________'
        pylab.plot(scale,fitfunc(fit[0],scale))
#        pylab.xlim(-0.2,0.2)
        pylab.savefig(filename2+'_limited_gauss_fit_'+str(ra)+'_'+str(dec)+'.png')
        pylab.clf()
#        pylab.plot(scale,abs(fitfunc(fit[0],scale)-f_hist[0]))
#        pylab.savefig(filename2+'_gauss_resid_'+str(ra)+'_'+str(dec)+'.png')
#        pylab.clf()
# Isolating frequencies that may be absorption features
        #pick limit based on gaussian. start at 3 sigma
        edge = fit[0][1]-3*fit[0][2]
#        outliers = []
#        out_freq = []
#        out_ra = []
#        out_dec = []
        for f in range(0,len(limited_tot_out[:,0,0])):
            if limited_tot_out[f,line,slice]<edge:
                outliers.append(limited_tot_out[f,line,slice])
                out_freq.append(limited_tot_freq[f,line,slice])
                out_ra.append(ra)
                out_dec.append(dec)

output_array = scipy.zeros((4,len(outliers)))
output_array[0] = outliers
output_array[1] = out_freq
output_array[2] = out_ra
output_array[3] = out_dec
savetxt('test.txt',output_array,delimiter=' ')
print len(outliers)
#print out_freq
#print out_ra
#print out_dec       



# For making 2D plots, setting scale based upon mean 
#smallest_min = 0.0 
#test_array = []
#test_array_2 = []
#for i in range(0,len(total_stats[0,:,0])):
#    for j in range(0,len(total_stats[0,0,:])):
#        test_array.append(total_stats[0,i,j]-3*total_stats[1,i,j])
#        test_array_2.append(total_stats[0,i,j]+3*total_stats[1,i,j])
#        if total_stats[0,i,j]-3*total_stats[1,i,j]<smallest_min:
#            smallest_min = total_stats[0,i,j]-3*total_stats[1,i,j]

#print mean(test_array)
#print test_array 
#print smallest_min
#min_value = mean(test_array)
#max_value = mean(test_array_2)
#print min_value, max_value
#for freq in range(0,len(total_freq[:,0,0])):
#    new_array = total_outliers[freq]
#   Need to rotate array[slice] because axes were flipped
#    new_array = scipy.transpose(new_array)

#    pylab.imshow(new_array, cmap='hot', interpolation='nearest',vmin=min_value, vmax=max_value, extent=(ras.max(),ras.min(),decs.min(),decs.max()), origin='lower')
#    pylab.colorbar() #For some reason this isn't working, fixed...
#    pylab.savefig(filename2+str(total_freq[freq,0,0])[:6]+'.png')
#    pylab.savefig('v_'+filename2+str(freq)[:3]+'.png')
#    pylab.clf()    


# List of outliers to investigate based upon analysis of lower frequency data.
#total_freq = [794.48,791.45,784.72,776.07,771.68,771.19,771.04,771.0,768.21,767.92,758.64,752.05,750.88,747.75,747.17,746.68,745.17,744.73,744.04,743.9,743.85,743.6,743.6,743.55,743.46,742.63,742.63,742.63,742.48,742.33,742.09,741.89,741.85,741.85,741.75,741.6,741.55,741.55,741.5,739.11,737.35,735.99,735.79,735.74,733.4,733.25,730.37,729.74,729.49,729.39,729.39,728.96,728.91,728.76,727.93,727.05,725.1,724.61,723.97,721.58,721.58,720.41,720.31,720.02,719.68,718.75,718.65,718.6,718.55,718.07,717.87,717.82,717.43,716.85,716.46,716.41,716.31,715.09,714.84,714.45,714.36,714.11,713.13,711.91,711.72,711.13,711.08,710.84,710.3,710.21,709.67,709.57,709.47,708.54,707.67,707.28,706.69,706.69,706.69,706.59,706.49,706.49,706.45,705.47,705.47,705.42,705.37,705.27,705.18,705.13,705.08,705.08]
#total_freq_2 = [899.56, 899.46, 899.46, 899.22, 899.22, 899.17, 899.17, 899.17, 898.83, 898.83, 898.58, 897.75, 897.61, 895.41, 895.36, 894.48, 893.65, 893.51, 893.41, 893.26, 890.63, 890.58, 890.19, 890.14, 889.26, 889.11, 889.11, 888.96, 888.82, 888.28, 887.94, 887.94, 887.94, 887.84, 887.79, 887.55, 887.55, 887.40, 887.40, 887.26, 887.21, 884.81, 884.33, 883.89, 883.79, 883.69, 883.69, 883.45, 883.40, 883.40, 883.30, 883.30, 882.91, 882.76, 882.76, 882.18, 882.18, 881.88, 881.20, 881.20, 881.20, 881.15, 881.15, 881.10, 881.05, 881.05, 880.81, 879.79, 879.79, 879.74, 876.27, 873.49, 873.44, 873.29, 873.24, 873.19, 873.14, 873.14, 872.66, 871.92, 871.92, 871.88, 871.88, 871.83, 871.83, 871.63, 871.63, 871.53, 871.48, 871.48, 871.48, 870.80, 870.80, 870.70, 870.56, 867.87, 865.67, 865.33, 865.33, 864.65, 864.40, 864.11, 864.01, 863.53, 860.99, 860.30, 858.01, 857.96, 857.91, 857.71, 857.71, 857.71, 856.64, 842.68, 835.30, 835.25, 830.13, 819.63, 819.58, 801.37, 801.27, 801.27]
#total_ra = [219.37,217.87,219.37,219.37,216.37,216.37,217.87,220.37,219.87,219.87,218.37,216.37,218.37,220.37,217.37,219.37,217.37,220.37,219.37,219.37,219.37,218.37,219.37,219.37,218.37,218.87,219.37,219.87,217.87,219.37,218.87,217.87,215.87,216.37,215.87,219.37,216.37,216.87,219.37,217.37,220.37,220.37,220.37,218.87,216.87,219.87,219.87,216.37,219.37,216.87,216.87,219.37,217.37,218.37,220.37,219.87,219.87,218.87,218.37,217.37,217.87,216.37,215.87,220.37,218.87,219.87,217.37,219.37,219.87,219.37,219.37,220.37,216.87,219.37,219.37,219.37,217.37,219.87,216.37,216.87,220.37,219.87,219.37,219.87,219.87,220.37,215.87,219.87,215.87,220.37,219.37,219.37,218.37,219.37,219.87,216.87,218.37,218.87,218.87,219.37,215.87,220.37,215.87,219.37,219.37,219.37,219.37,220.37,219.87,220.37,219.37,219.37]
#total_ra = [1,5,2,2,8,8,5,0,1,1,4,8,4,0,6,2,6,0,2,2,2,4,2,2,4,3,2,1,5,2,3,5,9,8,9,2,8,7,2,6,0,0,0,3,7,1,1,8,2,7,7,2,6,4,0,1,1,3,4,6,5,8,9,0,3,1,6,2,1,2,2,0,7,2,2,2,6,1,8,7,0,1,2,1,1,0,9,1,9,0,2,2,4,2,1,7,4,3,3,2,9,0,9,2,2,2,2,0,1,0,2,2]
#total_ra_2 = [1, 2, 0, 5, 8, 9, 7, 1, 8, 4, 7, 7, 9, 5, 9, 9, 1, 9, 9, 2, 4, 0, 3, 9, 7, 7, 8, 1, 8, 2, 8, 1, 9, 1, 2, 0, 2, 3, 8, 8, 2, 4, 2, 1, 4, 4, 9, 1, 4, 6, 6, 7, 0, 1, 6, 6, 7, 6, 6, 9, 9, 1, 3, 8, 1, 8, 9, 3, 0, 4, 2, 8, 2, 1, 2, 0, 0, 7, 4, 4, 0, 1, 2, 7, 9, 4, 9, 8, 7, 2, 2, 1, 8, 5, 2, 4, 4, 0, 1, 2, 1, 7, 2, 2, 2, 1, 5, 6, 3, 1, 2, 9, 0, 7, 1, 1, 5, 2, 2, 3, 6, 0] 
#total_dec = [3.0,0.5,0.5,0.5,0.5,2.0,1.0,1.0,1.5,3.0,0.5,3.0,1.5,1.5,3.0,0.5,0.5,2.5,2.0,2.0,2.0,3.0,2.0,2.0,1.0,3.0,3.0,0.5,1.0,2.0,1.5,0.5,1.0,1.0,1.5,0.5,0.5,0.5,2.0,1.0,1.5,0.5,0.5,1.5,3.0,1.5,0.5,0.5,0.5,1.0,0.5,1.0,3.0,0.5,1.5,1.5,1.0,2.0,1.0,0.5,0.5,1.0,1.5,1.5,3.0,2.0,0.5,2.5,0.5,3.0,2.0,0.5,0.5,1.5,3.0,1.0,3.0,0.5,3.0,1.0,2.5,2.5,2.5,1.0,1.5,0.5,0.5,3.0,0.5,3.0,3.0,3.0,2.0,1.5,0.5,0.5,2.5,2.0,1.5,0.5,0.5,0.5,0.5,3.0,2.0,3.0,3.0,1.5,0.5,0.5,3.0,2.0]
#total_dec = [5,0,0,0,0,3,1,1,2,5,0,5,2,2,5,0,0,4,3,3,3,5,3,3,1,5,5,0,1,3,2,0,1,1,2,0,0,0,3,1,2,0,0,3,5,2,0,0,0,1,0,1,5,0,2,2,1,3,1,0,0,1,2,2,5,3,0,4,0,5,3,0,0,2,5,1,5,0,5,1,4,4,4,1,2,0,0,5,0,5,5,5,3,2,0,0,4,3,2,0,0,0,0,5,3,5,5,2,0,0,5,3]
#total_dec_2 = [0, 3, 5, 0, 1, 1, 2, 5, 3, 4, 0, 0, 0, 0, 2, 1, 1, 0, 0, 3, 0, 0, 0, 4, 1, 2, 2, 1, 0, 3, 0, 3, 3, 1, 3, 2, 3, 1, 5, 4, 1, 0, 5, 0, 2, 1, 4, 0, 1, 5, 0, 0, 4, 0, 5, 0, 0, 5, 3, 3, 4, 2, 2, 0, 0, 5, 0, 0, 5, 0, 5, 3, 2, 2, 2, 0, 0, 0, 4, 0, 4, 0, 0, 0, 4, 2, 3, 2, 2, 3, 5, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 0, 3, 2, 3, 2, 0, 0, 0, 2, 2, 4, 0, 5, 5, 1, 0, 5, 5, 0, 0, 5] 

#print len(total_freq_2),len(total_ra_2),len(total_dec_2)
#print len(array[0,:,0])
#print len(array[0,0,:])
#print ras[0],decs[0]
#total_freqs=zeros(len(total_freq))
#total_freqs_2 = zeros(len(total_freq_2))
#print total_freq
#print freqs
#for i in range(0,len(total_freq)):
#    for j in range(0,len(freqs)):
#	if freqs[j]>total_freq[i]:
#            print j
#            total_freqs[i]=int(j)

#for i in range(0,len(total_freqs_2)):
#    for j in range(0,len(freqs)):
#        if freqs[j]>total_freq_2[i]:
#            total_freqs_2[i]=int(j)

#print total_freqs
#for i in range(0,len(total_freq)):
#    sub_data = array[total_freqs[i]-40:total_freqs[i]+40,total_ra[i],total_dec[i]]
#    sub_freq = freqs[total_freqs[i]-40:total_freqs[i]+40]

#for i in range(0,len(total_freq_2)):
#    if total_freqs_2[i]>40:
#        sub_data_2 = array[total_freqs_2[i]-40:total_freqs_2[i]+40,total_ra_2[i],total_dec_2[i]]
#        sub_freq_2 = freqs[total_freqs_2[i]-40:total_freqs_2[i]+40]
#    else:
#        limit = total_freqs_2[i]
#        sub_data_2 = array[total_freqs_2[i]-limit:total_freqs_2[i]+limit,total_ra_2[i],total_dec_2[i]]
#        sub_freq_2 = freqs[total_freqs_2[i]-limit:total_freqs_2[i]+limit]
#        print limit
#    print sub_data_2
#    print sub_freq_2
#    pylab.scatter(sub_freq_2,sub_data_2)
#    pylab.plot(sub_freq_2,sub_data_2)
#    pylab.savefig(filename2+'test_'+str(i)+'.png')
#    pylab.clf()


# Now have 4 arrays with RA, DEC, Outlier values, outlier frequencies these are nested arrays, so call array[i][j]
#print total_ra 
#for i in range(0,len(total_outliers)):
#    for j in range(0,len(total_outliers[0])):
     


# Old code for npy plotting maps
#for slice, freq in enumerate(freqs):
#   nancut = (array[slice] < 10e10) & ( array[slice] != NaN )
#   cut = ( array[slice] > 3.0*array[slice][nancut].std() ) 
#   array[slice][cut] = 3.0*array[slice][nancut].std()
#   cut = ( array[slice] < -3.0*array[slice][nancut].std() ) 
#   array[slice][cut] = -3.0*array[slice][nancut].std()

#   Need to rotate array[slice] because axes were flipped
#   new_array = scipy.transpose(array[slice])
#   medianv = median(array[slice][nancut])

#   Alternate plotting command to set temperature limits
#   pylab.imshow(new_array, cmap='hot', vmin=-0.1, vmax=0.1, extent=(ras.max(),ras.min(),decs.min(),decs.max()), origin='lower')
#   pylab.imshow(new_array, interpolation='gaussian', cmap='hot', extent=(ras.max(),ras.min(),decs.min(),decs.max()), origin='lower')
#   pylab.colorbar() #For some reason this isn't working, fixed...
#   pylab.savefig(filename2+str(freq)[:3]+'.png')
#   pylab.savefig('v_'+filename2+str(freq)[:3]+'.png')
#   pylab.clf()

#Alternate code if want to plot in term of dec instead of freq.
#for slice, dec in enumerate(decs):
#   print slice
#   print dec
#   nancut = (array[:,:,slice] < 10e10) & ( array[:,:,slice] != NaN )
#   cut = ( array[:,:,slice] > 3.0*array[:,:,slice][nancut].std() )
#   array[:,:,slice][cut] = 3.0*array[:,:,slice][nancut].std()
#   cut = ( array[:,:,slice] < -3.0*array[:,:,slice][nancut].std() )
#   array[:,:,slice][cut] = -3.0*array[:,:,slice][nancut].std()

#   print array[:,:,slice]
#   Need to rotate array[slice] because axes were flipped
#   new_array = scipy.transpose(array[:,:,slice])
#   medianv = median(array[slice][nancut])
 
#   Alternate plotting command to set temperature limits
#   pylab.imshow(array[:,:,slice], aspect = 0.02, interpolation='gaussian', vmin=-0.2, vmax=0.2, extent=(ras.max(),ras.min(),freqs.min(),freqs.max()), origin='lower')
#   pylab.imshow(array[:,:,slice], interpolation='gaussian', extent=(freqs.max(),freqs.min(),ras.min(),ras.max()), origin='lower')
#   pylab.colorbar() #For some reason this isn't working, fixed...
#   pylab.savefig(filename2+str(dec)[:3]+'.png')
#   pylab.savefig('v_'+filename2+str(freq)[:3]+'.png')
#   pylab.clf()


