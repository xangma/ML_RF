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

def plot_bandvprob(resultsstack,filtstats,probshape): # Plots each filter band vs probability of each class.
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
        names=['Star','QSO','Galaxy']
        for k in range(0,probshape):
            for j in range(len(filtstats)):
                fig=plt.figure()
                for i in range(filtstats[j][0]): # Plot for all filters
                    plt.subplot(2,3,i+1)
                    #print('k: %s' %k)
                    #print('j: %s' %j)
                    #print('i: %s' %i)
                    #print('bottom: %s' %bottom)
                    #print(names[k])
                    H, yedges, xedges, img = plt.hist2d(resultsstack[:,bottom+i],resultsstack[:,-(k+1)],bins=100,norm=LogNorm())
                    plt.colorbar(norm=LogNorm())
                    plt.ylabel('Prob. of %s' %names[k])
                    plt.xlabel('%s apparent mag' %settings.filters[j][i])
                    plt.xlim(5,32)
#                plt.show()
                plt.savefig('%sbandvprob%s_allfilt_%s.png' %(fullsavedir,names[k],j))
                plt.close(fig)
                
                for i in range(filtstats[j][0]): # Plot for all filters
                    fig=plt.figure()
                    H, yedges, xedges, img = plt.hist2d(resultsstack[:,bottom+i],resultsstack[:,-(k+1)],bins=100,norm=LogNorm())
                    plt.colorbar(norm=LogNorm())
                    plt.ylabel('Prob. of %s' %names[k])
                    plt.xlabel('%s apparent mag' %settings.filters[j][i])
                    plt.savefig('%sbandvprob%s_filt_%s_%s.png' %(fullsavedir,names[k],j,i))
                    plt.close(fig)
                bottom=bottom + filtstats[j][0] + filtstats[j][1]
            bottom=0

def plot_colourvprob(resultsstack,filtstats,probshape,combs): # Plots each colour vs probability of each class
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
        names=['Star','QSO','Galaxy']
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
                    H, yedges, xedges, img = plt.hist2d(resultsstack[:,bottom+filtstats[j][0]+i],resultsstack[:,-(k+1)],bins=100,norm=LogNorm())
                    plt.colorbar(norm=LogNorm())
                    plt.ylabel('Prob. of %s' %names[k])
                    plt.xlabel('%s - %s' %(settings.filters[j][combs[j][i][0]],settings.filters[j][combs[j][i][1]]))
                    plt.xlim(-10,10)
#                plt.show()
                plt.savefig('%scolourvprob%s_allfilt_%s.png' %(fullsavedir,names[k],j))
                plt.close(fig)
                
                for i in range(filtstats[j][1]): # Plot for all filters
                    fig=plt.figure()
                    H, yedges, xedges, img = plt.hist2d(resultsstack[:,bottom+filtstats[j][0]+i],resultsstack[:,-(k+1)],bins=100,norm=LogNorm())
                    plt.colorbar(norm=LogNorm())
                    plt.ylabel('Prob. of %s' %names[k])
                    plt.xlabel('%s - %s' %(settings.filters[j][combs[j][i][0]],settings.filters[j][combs[j][i][1]]))
                    plt.xlim(-10,10)
                    plt.savefig('%scolourvprob%s_filt_%s_%s_%s-%s.png' %(fullsavedir,names[k],j,i,settings.filters[j][combs[j][i][0]],settings.filters[j][combs[j][i][1]]))
                    plt.close(fig)
                bottom=bottom + filtstats[j][1] +filtstats[j][0]
            bottom=0

def plot_feat_per_class(one_vs_all_results):
    plt.figure()
    for i in range(len(one_vs_all_results)):
        plt.scatter(numpy.array(range(len(one_vs_all_results[i]['feat_importance'])))+1,one_vs_all_results[i]['feat_importance'],s=2)
        plt.plot(numpy.array(range(len(one_vs_all_results[i]['feat_importance'])))+1,one_vs_all_results[i]['feat_importance'],label='class == %s' %one_vs_all_results[i]['class_ID'])
    plt.axvspan(5.5, 15.5, color='red', alpha=0.3)
    plt.axvspan(20.5, 30.5, color='red', alpha=0.3)
    plt.axvspan(35.5, 45.5, color='red', alpha=0.3)
    plt.legend()
    plt.title('Feature importance per class')
    plt.xlabel('Features')
    plt.ylabel('Feat_Importance')
    plt.xlim(0.5,45.5)
    plt.ylim(0,0.2)
#    plt.xticks(numpy.array(range(len(feat_arr[0]))))
    plt.minorticks_on()
    plt.grid(alpha=0.4,which='both')
    plt.savefig('plots/Feature_imp_per_class.png')
    plt.close()
