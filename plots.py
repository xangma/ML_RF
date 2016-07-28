# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 14:59:20 2016

@author: moricex
"""
import settings
import matplotlib.pyplot as plt
import numpy
from matplotlib.colors import LogNorm
import os
import logging
from mpl_toolkits.mplot3d import Axes3D
import numpy.ma as ma

plots_log=logging.getLogger('plots')


path=settings.programpath+'plots/'
plt.ioff()

# Plot histogram of subclasses
def plot_subclasshist(XX,XXpredict,classnames_tr,classnames_pr):
    if settings.plotsubclasshist == 1:
        uniquesubclass=numpy.unique(classnames_tr,return_index=True)
        fig = plt.figure()
        plt.hist(XX[:,-1],bins=len(uniquesubclass[1]))
        plt.close(fig)

def plot_bandvprob(XXpredict,probs,filtstats,probshape): # Plots each filter band vs probability of each class.
    outnamelist=[]
    if settings.plotbandvprob == 1: # If plotting selected
        bottom=0
        plots_log.info('')
        plot_bandvprob_log=logging.getLogger('plot_bandvprob')
        plot_bandvprob_log.info('Plotting plot_bandvprob')
        dirs=os.listdir(path)
        savedir='plot_bandvprob' # Check if directory exists, if not, create
        fullsavedir=path+savedir+'/'
        if savedir not in dirs:
            os.mkdir(fullsavedir)
        names=['Galaxy','QSO','Star']
        for k in range(0,probshape):
            for j in range(len(filtstats)):
                fig=plt.figure()
                for i in range(filtstats[j][0]): # Plot for all filtersrpr
                    plt.subplot(2,3,i+1)
                    #print('k: %s' %k)
                    #print('j: %s' %j)
                    #print('i: %s' %i)
                    #print('bottom: %s' %bottom)
                    #print(names[k])
                    mask = XXpredict[:,-1] == k
                    H, yedges, xedges, img = plt.hist2d(XXpredict[:,bottom+i][mask],probs[:,k][mask],bins=100,norm=LogNorm())
                    cb=plt.colorbar(norm=LogNorm())
                    cb.ax.tick_params(labelsize=8)
                    plt.ylabel('Prob. of %s' %names[k],size=8)
                    plt.xlabel('%s apparent mag' %settings.filters[j][i],size=8)
                    plt.tick_params(labelsize=8)
                    plt.xlim(5,32)
#                plt.show()
                outname='plots/'+savedir+'/bandvprob%s_allfilt_%s.png' %(names[k],j)
                outnamelist.append(outname)
                plt.tight_layout()
                plt.savefig(outname)
                plt.close(fig)
                
                for i in range(filtstats[j][0]): # Plot for all filters
                    fig=plt.figure()
                    H, yedges, xedges, img = plt.hist2d(XXpredict[:,bottom+i][mask],probs[:,k][mask],bins=100,norm=LogNorm())
                    plt.colorbar(norm=LogNorm())
                    plt.ylabel('Prob. of %s' %names[k])
                    plt.xlabel('%s apparent mag' %settings.filters[j][i])
                    outname='plots/'+savedir+'/bandvprob%s_filt_%s.png' %(names[k],settings.filters[j][i])
                    outnamelist.append(outname)
                    plt.tight_layout()
                    plt.savefig(outname)
                    plt.close(fig)
                bottom=bottom + filtstats[j][0] + filtstats[j][1]
            bottom=0
    return outnamelist

def plot_colourvprob(XXpredict,probs,filtstats,probshape,combs): # Plots each colour vs probability of each class
    outnamelist=[]
    if settings.plotcolourvprob == 1: # If plotting selected
        bottom=0
        plots_log.info('')
        plot_colourvprob_log=logging.getLogger('plot_colourvprob')
        plot_colourvprob_log.info('Plotting plot_colourvprob')
        dirs=os.listdir(path)
        savedir='plot_colourvprob' # Check if directory exists, if not, create
        fullsavedir=path+savedir+'/'
        if savedir not in dirs:
            os.mkdir(fullsavedir)
        names=['Galaxy','QSO','Star']
        numplots=0
        for k in range(len(filtstats)):
            numplots = numplots+len(combs[k])
        numplots=numplots*probshape
        plot_colourvprob_log.info('%s Total number of plots: %s' %(savedir,numplots))
        for k in range(0,probshape): # For all the names
            plot_colourvprob_log.info('%s Plotting: %s' %(savedir,names[k]))
            for j in range(len(filtstats)): # For all the filter sets
                fig=plt.figure()
                for i in range(0,filtstats[j][1]): # Plot for all filters
                    plt.subplot(4,3,i+1)
#                    plot_colourvprob_log.info('k: %s' %k)
#                    plot_colourvprob_log.info('j: %s' %j)
#                    plot_colourvprob_log.info('i: %s' %i)
#                    plot_colourvprob_log.info('bottom: %s' %bottom)
#                    plot_colourvprob_log.info('bottom+filtstats[j][0]: %s' %(bottom+filtstats[j][0]))
#                    plot_colourvprob_log.info('Colour: %s - %s' %(settings.filters[j][combs[j][i][0]],settings.filters[j][combs[j][i][1]]))
#                    plot_colourvprob_log.info(names[k])
                    mask = XXpredict[:,-1] == k
                    H, yedges, xedges, img = plt.hist2d(XXpredict[:,bottom+filtstats[j][0]+i][mask],probs[:,k][mask],bins=100,norm=LogNorm())
                    cb=plt.colorbar(norm=LogNorm())
                    cb.ax.tick_params(labelsize=8)
                    plt.ylabel('Prob. of %s' %names[k],size=8)
                    plt.xlabel('%s - %s' %(settings.filters[j][combs[j][i][0]],settings.filters[j][combs[j][i][1]]),size=8)
                    plt.tick_params(labelsize=8)
                    plt.xlim(-10,10)
#                plt.show()
                plt.tight_layout()
                outname='plots/'+savedir+'/colourvprob%s_allfilt_%s.png' %(names[k],j)
                outnamelist.append(outname)
                plt.savefig(outname)
                plt.close(fig)
                
                for i in range(filtstats[j][1]): # Plot for all filters
                    fig=plt.figure()
                    H, yedges, xedges, img = plt.hist2d(XXpredict[:,bottom+filtstats[j][0]+i][mask],probs[:,k][mask],bins=100,norm=LogNorm())
                    plt.colorbar(norm=LogNorm())
                    plt.ylabel('Prob. of %s' %names[k])
                    plt.xlabel('%s - %s' %(settings.filters[j][combs[j][i][0]],settings.filters[j][combs[j][i][1]]))
                    plt.xlim(-10,10)
                    outname='plots/'+savedir+'/colourvprob%s_filt_%s_%s_%s-%s.png' %(names[k],j,i,settings.filters[j][combs[j][i][0]],settings.filters[j][combs[j][i][1]])
                    outnamelist.append(outname)
                    plt.tight_layout()
                    plt.savefig(outname)
                    plt.close(fig)
                bottom=bottom + filtstats[j][1] +filtstats[j][0]
            bottom=0
    return outnamelist

def plot_feat(feat_importance,feat_names,n_run):
    outname=[]
    if settings.plotfeatimp == 1:
        plt.figure()
#        plt.scatter(numpy.array(range(len(feat_importance)))+1,feat_importance,s=2)
        plt.step(numpy.array(range(len(feat_importance)+2)),numpy.concatenate(([0],feat_importance,[0])))
#        plt.plot(numpy.array(range(len(feat_importance)))+1,feat_importance)
        plt.axvspan(5.0, 15.0, color='red', alpha=0.3)
        plt.axvspan(20.0, 30.0, color='red', alpha=0.3)
        plt.axvspan(35.0, 45.0, color='red', alpha=0.3)
        plt.title('Feature importance')
        plt.xlabel('Features')
        plt.ylabel('Feat_Importance')
        plt.xlim(0.0,len(feat_importance))
        plt.ylim(0,0.25)
        plt.xticks(numpy.array(range(len(feat_importance)))+0.5,feat_names, size=8,rotation='vertical')
    #    plt.xticks(numpy.array(range(len(feat_arr[0]))))
#        plt.minorticks_on()
        plt.grid(alpha=0.4,which='both')
        plt.tight_layout()
        outname = 'plots/Feature_imp_%s.png' %n_run
        plt.savefig(outname)
        plt.close()
    return outname

def plot_feat_per_class(one_vs_all_results,feat_names,n):
    outname=[]
    plt.figure()
    for i in range(len(one_vs_all_results)):
        plt.step(numpy.array(range(len(one_vs_all_results[i]['feat_importance'])+2)),numpy.concatenate(([0],one_vs_all_results[i]['feat_importance'],[0])),label='class == %s' %one_vs_all_results[i]['uniquetarget_tr_loop'][0][0])
#        plt.scatter(numpy.array(range(len(one_vs_all_results[i]['feat_importance'])))+1,one_vs_all_results[i]['feat_importance'],s=2)
#        plt.plot(numpy.array(range(len(one_vs_all_results[i]['feat_importance'])))+1,one_vs_all_results[i]['feat_importance'],label='class == %s' %one_vs_all_results[i]['class_ID'])
    plt.axvspan(5.0, 15.0, color='red', alpha=0.3)
    plt.axvspan(20.0, 30.0, color='red', alpha=0.3)
    plt.axvspan(35.0, 45.0, color='red', alpha=0.3)
    plt.legend()
    plt.title('Feature importance per class')
    plt.xlabel('Features')
    plt.ylabel('Feat_Importance')
    plt.xlim(0.0,len(feat_names))
    plt.ylim(0,0.25)
    plt.xticks(numpy.array(range(len(one_vs_all_results[0]['feat_importance'])))+0.5,feat_names, size=8,rotation='vertical')
#    plt.xticks(numpy.array(range(len(feat_arr[0]))))
#    plt.minorticks_on()
    plt.grid(alpha=0.4,which='both')
    plt.tight_layout()
    outname = 'plots/Feature_imp_per_class_%s.png' %n
    plt.savefig(outname)
    plt.close()
    return outname

def plot_feat_per_class_oth(one_vs_all_results,n_filt,n_colours):
    plt.figure()
    for i in range(len(one_vs_all_results)):
        plt.scatter(numpy.array(range(n_filt+n_colours+1,len(one_vs_all_results[i]['feat_importance']))),one_vs_all_results[i]['feat_importance'][n_filt+n_colours+1:len(one_vs_all_results[i]['feat_importance'])],s=2)
        plt.plot(numpy.array(range(n_filt+n_colours+1,len(one_vs_all_results[i]['feat_importance']))),one_vs_all_results[i]['feat_importance'][n_filt+n_colours+1:len(one_vs_all_results[i]['feat_importance'])],label='class == %s' %one_vs_all_results[i]['class_ID'])
#    plt.axvspan(5.5, 15.5, color='red', alpha=0.3)
#    plt.axvspan(20.5, 30.5, color='red', alpha=0.3)
#    plt.axvspan(35.5, 45.5, color='red', alpha=0.3)
    plt.legend()
    plt.title('Feature importance per class')
    plt.xlabel('Features')
    plt.ylabel('Feat_Importance')
#    plt.xlim(0.5,45.5)
    plt.ylim(0,0.2)
#    plt.xticks(numpy.array(range(len(feat_arr[0]))))
    plt.minorticks_on()
    plt.grid(alpha=0.4,which='both')
    plt.savefig('plots/Feature_imp_per_class_oth.png')
    plt.close()



def plot_col_rad(XXpredict,result,yypredict,feat_names,filtstats,uniquetarget_tr):
    if settings.plot_col_rad==1:
        dirs=os.listdir(path)
        savedir='plot_col_rad' # Check if directory exists, if not, create
        fullsavedir=path+savedir+'/'
        if savedir not in dirs:
            os.mkdir(fullsavedir)
        
        for l in range(len(uniquetarget_tr[0])):
            for k in range(len(settings.othertrain)):
                ot_index=feat_names.index(settings.othertrain[k])
                bottom=0
                for j in range(len(filtstats)): # For all the filter sets
                    for i in range(filtstats[j][1]): # Plot for all filters
                        plt.figure()
                        mask_pred = result == l
                        mask_true = yypredict == l
                        outliermask1=is_outlier(XXpredict[:,bottom+filtstats[j][0]+i])
                        outliermask2 = is_outlier(XXpredict[:,ot_index])
                        totalmask_pred = (~outliermask1) & (~outliermask2) & (mask_pred)
                        totalmask_true = (~outliermask1) & (~outliermask2) & (mask_true)
                        # plot
                        hist_pred, xedges, yedges = numpy.histogram2d(XXpredict[:,bottom+filtstats[j][0]+i][totalmask_pred],XXpredict[:,ot_index][totalmask_pred], bins=80)#,range=[[min(XXpredict[:,42]),max(XXpredict[:,42])],[min(XXpredict[:,49]),max(XXpredict[:,49])]])
                        hist_true, xedges, yedges = numpy.histogram2d(XXpredict[:,bottom+filtstats[j][0]+i][totalmask_true],XXpredict[:,ot_index][totalmask_true], bins=80)#,range=[[min(XXpredict[:,42]),max(XXpredict[:,42])],[min(XXpredict[:,49]),max(XXpredict[:,49])]])
                        hist = (hist_pred/hist_true)
                        Zm = ma.masked_where(~numpy.isfinite(hist),hist)
                        xpos, ypos = numpy.meshgrid(xedges,yedges)
                        plt.pcolormesh(xpos, ypos, Zm.T)
                        plt.xlabel('%s' %feat_names[bottom+filtstats[j][0]+i]), plt.ylabel('%s' %feat_names[ot_index])
                        plt.title('%s Precision' %uniquetarget_tr[0][l])
                        cb=plt.colorbar()
                        cb.ax.tick_params(labelsize=8)
                        plt.tight_layout()
                        outname='plots/'+savedir+'/col_rad _%s_filt_%s_%s_%s_vs_%s.png' %(uniquetarget_tr[0][l],j,i,feat_names[bottom+filtstats[j][0]+i], feat_names[ot_index])
                        plt.savefig(outname)
                        plt.close()
                    bottom=bottom + filtstats[j][1] +filtstats[j][0]


def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = numpy.median(points, axis=0)
    diff = numpy.sum((points - median)**2, axis=-1)
    diff = numpy.sqrt(diff)
    med_abs_deviation = numpy.median(diff)
    
    modified_z_score = 0.6745 * diff / med_abs_deviation
    
    return modified_z_score > thresh
