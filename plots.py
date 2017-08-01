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
import numpy.ma as ma
import itertools as it
from sklearn import tree
from sklearn import metrics
from sklearn import covariance
from collections import OrderedDict

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
    if settings.one_vs_all == 1:
        plt.figure()
        for i in range(len(one_vs_all_results)):
            plt.step(numpy.array(range(len(one_vs_all_results[i]['feat_importance'])+2)),numpy.concatenate(([0],one_vs_all_results[i]['feat_importance'],[0])),label='class == %s' %one_vs_all_results[i]['uniquetarget_tr'][0][0])
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
    outnames_list=[]
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
                        f, (axarr) = plt.subplots(2, 2)#,figsize=(14, 5))
                        plt.ylabel('%s' %feat_names[ot_index])
                        true_class = yypredict==l
                        T_p = (yypredict == l) & (result == l)
                        T_p_F_p = ((yypredict == l)& (result == l)) | ((result != l) & (yypredict==l))
                        T_p_F_n = ((result == l) & (yypredict != l)) | ((yypredict==l)& (result == l))
                        #
                        outliermask1=is_outlier(XXpredict[:,bottom+filtstats[j][0]+i])
                        outliermask2 = is_outlier(XXpredict[:,ot_index])
                        totalmask_tpfp = (T_p_F_p) & (~outliermask2) & (~outliermask1)
                        totalmask_tpfn = (T_p_F_n) & (~outliermask2) & (~outliermask1)
                        totalmask_tp = (T_p) & (~outliermask2) & (~outliermask1)
                        totalmask_true = (true_class) & (~outliermask2) & (~outliermask1)
                        # plot
                        hist_true, xedges, yedges,img = plt.hist2d(XXpredict[:,bottom+filtstats[j][0]+i][totalmask_true],XXpredict[:,ot_index][totalmask_true], bins=[80,120])
                        hist_tp, xedges_tp, yedges_tp,img_tp = plt.hist2d(XXpredict[:,bottom+filtstats[j][0]+i][totalmask_tp],XXpredict[:,ot_index][totalmask_tp], bins=[80,120],range=[[xedges.min(),xedges.max()],[yedges.min(),yedges.max()]])
                        hist_tpfp, xedges_tpfp, yedges_tpfp,img_tpfp = plt.hist2d(XXpredict[:,bottom+filtstats[j][0]+i][totalmask_tpfp],XXpredict[:,ot_index][totalmask_tpfp], bins=[80,120],range=[[xedges.min(),xedges.max()],[yedges.min(),yedges.max()]])
                        hist_tpfn, xedges_tpfn, yedges_tpfn,img_tpfn = plt.hist2d(XXpredict[:,bottom+filtstats[j][0]+i][totalmask_tpfn],XXpredict[:,ot_index][totalmask_tpfn], bins=[80,120],range=[[xedges.min(),xedges.max()],[yedges.min(),yedges.max()]])
                        hist_prec = (hist_tp/hist_tpfp)
                        hist_reca = (hist_tp/hist_tpfn)
                        Zm_prec = ma.masked_where(~numpy.isfinite(hist_prec),hist_prec)
                        Zm_reca = ma.masked_where(~numpy.isfinite(hist_reca),hist_reca)
                        Zm_true = ma.masked_where(hist_true == 0 , hist_true)
                        Zm_tp = ma.masked_where(hist_tp == 0 , hist_tp)
                        xpos, ypos = numpy.meshgrid(xedges,yedges)
                        cmap = plt.cm.jet
                        cmap.set_bad('w',1)
                        
                        im0 = axarr[0,0].pcolormesh(xpos, ypos, Zm_true.T,cmap=cmap)
                        axarr[0,0].set_title('%s True' %uniquetarget_tr[0][l])
                        axarr[0,0].set_ylim(min(yedges),max(yedges))
                        axarr[0,0].set_xlim(min(xedges),max(xedges))
                        axarr[0,0].set_xlabel('%s' %feat_names[bottom+filtstats[j][0]+i])
                        cb=plt.colorbar(im0,ax=axarr[0,0])
                        cb.ax.tick_params(labelsize=8)
                        im1 = axarr[0,1].pcolormesh(xpos, ypos, Zm_tp.T,cmap=cmap)
                        axarr[0,1].set_title('%s T_p' %uniquetarget_tr[0][l])
                        axarr[0,1].set_ylim(min(yedges),max(yedges))
                        axarr[0,1].set_xlim(min(xedges),max(xedges))
                        axarr[0,1].set_xlabel('%s' %feat_names[bottom+filtstats[j][0]+i])
                        cb=plt.colorbar(im1,ax=axarr[0,1])
                        cb.ax.tick_params(labelsize=8)
                        im2 = axarr[1,0].pcolormesh(xpos, ypos, Zm_prec.T,cmap=cmap)
                        axarr[1,0].set_title('%s Precision' %uniquetarget_tr[0][l])
                        axarr[1,0].set_ylim(min(yedges),max(yedges))
                        axarr[1,0].set_xlim(min(xedges),max(xedges))
                        axarr[1,0].set_xlabel('%s' %feat_names[bottom+filtstats[j][0]+i])
                        cb=plt.colorbar(im2,ax=axarr[1,0])
                        cb.ax.tick_params(labelsize=8)
                        im3 = axarr[1,1].pcolormesh(xpos, ypos, Zm_reca.T,cmap=cmap)
                        axarr[1,1].set_title('%s Recall' %uniquetarget_tr[0][l])
                        axarr[1,1].set_ylim(min(yedges),max(yedges))
                        axarr[1,1].set_xlim(min(xedges),max(xedges))
                        axarr[1,1].set_xlabel('%s' %feat_names[bottom+filtstats[j][0]+i])
                        cb=plt.colorbar(im3,ax=axarr[1,1])
                        cb.ax.tick_params(labelsize=8)                        
                        
                        plt.tight_layout()
                        outname='plots/'+savedir+'/col_rad _%s_filt_%s_%s_%s_vs_%s.png' %(uniquetarget_tr[0][l],j,i,feat_names[bottom+filtstats[j][0]+i], feat_names[ot_index])
                        outnames_list.append(outname)
                        plt.savefig(outname)
                        plt.close(f)
                        plt.close()

                    bottom=bottom + filtstats[j][1] +filtstats[j][0]
    return outnames_list

def plot_col_cont(XXpredict,result,yypredict,feat_names,filtstats,uniquetarget_tr):
    outnamelist=[]
    if settings.plot_col_cont==1:
        contributions = numpy.load('contributions.npy')
        cont=numpy.transpose(contributions)
        dirs=os.listdir(path)
        savedir='plot_col_cont' # Check if directory exists, if not, create
        fullsavedir=path+savedir+'/'
        if savedir not in dirs:
            os.mkdir(fullsavedir)
        
        for l in range(len(uniquetarget_tr[0])):
            for i in range(len(feat_names)): # Plot for all filters
#                fig=plt.figure()
                plt.figure()
                plt.ylabel('Contributions to P(%s) from %s' %(uniquetarget_tr[0][l],feat_names[i]))
                f, (axarr) = plt.subplots(2, 2)#,figsize=(14, 5))
                
                true_class = yypredict==l
                T_p = (yypredict == l) & (result == l)
                T_p_F_p = ((yypredict == l)& (result == l)) | ((result != l) & (yypredict==l))
                T_p_F_n = ((result == l) & (yypredict != l)) | ((yypredict==l)& (result == l))
                #
                outliermask1=is_outlier(XXpredict[:,i])
                outliermask2 = is_outlier(cont[l][i])
                totalmask_tpfp = (T_p_F_p)& (~outliermask2) & (~outliermask1)
                totalmask_tpfn = (T_p_F_n) & (~outliermask2) & (~outliermask1)
                totalmask_tp = (T_p) & (~outliermask2) & (~outliermask1)
                totalmask_true = (true_class) & (~outliermask2) & (~outliermask1)
                # plot
                hist_true, xedges, yedges,img = plt.hist2d(XXpredict[:,i][totalmask_true],cont[l][i][totalmask_true], bins=[80,120])
                hist_tp, xedges_tp, yedges_tp,img_tp = plt.hist2d(XXpredict[:,i][totalmask_tp],cont[l][i][totalmask_tp], bins=[80,120],range=[[xedges.min(),xedges.max()],[yedges.min(),yedges.max()]])
                hist_tpfp, xedges_tpfp, yedges_tpfp,img_tpfp = plt.hist2d(XXpredict[:,i][totalmask_tpfp],cont[l][i][totalmask_tpfp], bins=[80,120],range=[[xedges.min(),xedges.max()],[yedges.min(),yedges.max()]])
                hist_tpfn, xedges_tpfn, yedges_tpfn,img_tpfn = plt.hist2d(XXpredict[:,i][totalmask_tpfn],cont[l][i][totalmask_tpfn], bins=[80,120],range=[[xedges.min(),xedges.max()],[yedges.min(),yedges.max()]])
                hist_prec = (hist_tp/hist_tpfp)
                hist_reca = (hist_tp/hist_tpfn)
                Zm_prec = ma.masked_where(~numpy.isfinite(hist_prec),hist_prec)
                Zm_reca = ma.masked_where(~numpy.isfinite(hist_reca),hist_reca)
                Zm_true = ma.masked_where(hist_true == 0 , hist_true)
                Zm_tp = ma.masked_where(hist_tp == 0 , hist_tp)
                xpos, ypos = numpy.meshgrid(xedges,yedges)
                cmap = plt.cm.jet
                cmap.set_bad('w',1)
                
                im0 = axarr[0,0].pcolormesh(xpos, ypos, Zm_true.T,cmap=cmap)
                axarr[0,0].axhline(0, color='black',linewidth=1)
                axarr[0,0].set_title('%s True Contprob' %uniquetarget_tr[0][l])
                axarr[0,0].set_ylim(min(yedges),max(yedges))
                axarr[0,0].set_xlim(min(xedges),max(xedges))
                axarr[0,0].set_xlabel('%s' %feat_names[i])
                cb=plt.colorbar(im0,ax=axarr[0,0])
                cb.ax.tick_params(labelsize=8)
                im1 = axarr[0,1].pcolormesh(xpos, ypos, Zm_tp.T,cmap=cmap)
                axarr[0,1].axhline(0, color='black',linewidth=1)
                axarr[0,1].set_title('%s T_p Contprob' %uniquetarget_tr[0][l])
                axarr[0,1].set_ylim(min(yedges),max(yedges))
                axarr[0,1].set_xlim(min(xedges),max(xedges))
                axarr[0,1].set_xlabel('%s' %feat_names[i])
                cb=plt.colorbar(im1,ax=axarr[0,1])
                cb.ax.tick_params(labelsize=8)
                im2 = axarr[1,0].pcolormesh(xpos, ypos, Zm_prec.T,cmap=cmap)
                axarr[1,0].axhline(0, color='black',linewidth=1)
                axarr[1,0].set_title('%s Precision Contprob' %uniquetarget_tr[0][l])
                axarr[1,0].set_ylim(min(yedges),max(yedges))
                axarr[1,0].set_xlim(min(xedges),max(xedges))
                axarr[1,0].set_xlabel('%s' %feat_names[i])
                cb=plt.colorbar(im2,ax=axarr[1,0])
                cb.ax.tick_params(labelsize=8)
                im3 = axarr[1,1].pcolormesh(xpos, ypos, Zm_reca.T,cmap=cmap)
                axarr[1,1].axhline(0, color='black',linewidth=1)
                axarr[1,1].set_title('%s Recall Contprob' %uniquetarget_tr[0][l])
                axarr[1,1].set_ylim(min(yedges),max(yedges))
                axarr[1,1].set_xlim(min(xedges),max(xedges))
                axarr[1,1].set_xlabel('%s' %feat_names[i])
                cb=plt.colorbar(im3,ax=axarr[1,1])
                cb.ax.tick_params(labelsize=8)
                                
                outname='plots/'+savedir+'/col_cont_%s_%s_%s.png' %(uniquetarget_tr[0][l],i,feat_names[i])
                outnamelist.append(outname)
                plt.tight_layout()
                plt.savefig(outname)
                plt.close(f)
                plt.close()
    return outnamelist


def plot_col_cont_true(XXpredict,result,yypredict,feat_names,filtstats,uniquetarget_tr):
    outnamelist=[]
    if settings.plot_col_cont_true==1:
        contributions=numpy.load('perfect_contributions.npy')
        cont=numpy.transpose(contributions)
        dirs=os.listdir(path)
        savedir='plot_col_cont_true' # Check if directory exists, if not, create
        fullsavedir=path+savedir+'/'
        if savedir not in dirs:
            os.mkdir(fullsavedir)
        
        for l in range(len(uniquetarget_tr[0])):
            for i in range(len(feat_names)): # Plot for all filters
                fig=plt.figure()
                #                        mask_true = yypredict == l
                #                        mask_pred = result == l
                #
                findalltrue = yypredict == l
#                findpredtrue = (yypredict == l) & (result == yypredict)
                #
                outliermask1=is_outlier(XXpredict[:,i])
                outliermask2 = is_outlier(cont[l][i])
#                totalmask_pred = (findpredtrue) & (~outliermask2) & (~outliermask1)
                totalmask_true = (findalltrue) & (~outliermask2) & (~outliermask1)
                # plot
                hist_true, xedges, yedges,img = plt.hist2d(XXpredict[:,i][totalmask_true],cont[l][i][totalmask_true], bins=[80,120])
 #               hist_pred, xedges, yedges,img = plt.hist2d(XXpredict[:,i][totalmask_pred],cont[l][i][totalmask_pred], bins=[80,120],range=[[xedges.min(),xedges.max()],[yedges.min(),yedges.max()]])
#                hist = (hist_pred/hist_true)
                Zm = ma.masked_where(hist_true==0,hist_true)
                xpos, ypos = numpy.meshgrid(xedges,yedges)
                cmap = plt.cm.jet
                cmap.set_bad('w',1)
                plt.pcolormesh(xpos, ypos, Zm.T,cmap=cmap)
                plt.xlabel('%s' %feat_names[i]), plt.ylabel('Contributions to P(%s) of %s' %(feat_names[i],uniquetarget_tr[0][l]))
                plt.title('%s Contprob True' %uniquetarget_tr[0][l])
                cb=plt.colorbar()
                cb.ax.tick_params(labelsize=8)
                plt.axhline(0, color='black',linewidth=1)
                plt.tight_layout()
                outname='plots/'+savedir+'/col_cont_true_%s_%s_%s.png' %(uniquetarget_tr[0][l],i,feat_names[i])
                outnamelist.append(outname)
                plt.savefig(outname)
                plt.close(fig)
    return outnamelist

def plot_mic(feat_names):
    outnamelist=[]
    if settings.plot_mic == 1:
        mic_runs1=[s for s in os.listdir() if 'mic_OvsA' in s]
        mic_runs2=[s for s in os.listdir() if 'A_mic' in s]
        mic_runs=mic_runs1+mic_runs2
        for k in range(len(mic_runs)):
            data=numpy.load(mic_runs[k])
            #mic_combs=data[0]
            mic_all=data[1]
            triangle=[]
            bottom,l=0,0
            for i in range(len(feat_names),-1,-1):
                triangle.append(mic_all[bottom:bottom+i])
                bottom=bottom+i
                l=l+1
            
            for i in range(len(triangle)):
                diff = len(feat_names)+1 - len(triangle[i])
                for j in range(diff):
                    triangle[i]=list(numpy.hstack((numpy.NaN,triangle[i])))
            xpos,ypos=numpy.meshgrid(numpy.array(range(len(feat_names)+1)),numpy.array(range(len(feat_names)+1)))
            square=numpy.array(triangle)
            Zm_sq=ma.masked_where(numpy.isnan(square),square)
            fig,ax=plt.subplots()
            cmap=plt.cm.jet
            cmap.set_bad('w',1)
            #cb=fig.colorbar()
            #cb.ax.tick_params(labelsize=8)
            ax.pcolormesh(xpos,ypos,Zm_sq,cmap=cmap)
            ax.set_xticks(numpy.array(range(len(feat_names)+1))),ax.set_yticks(numpy.array(range(len(feat_names)+1)))
            ax.set_xticklabels(feat_names,rotation='vertical',size=8,ha='left'),ax.set_yticklabels(feat_names,size=8,va='bottom')
            ax.grid(alpha=0.7,linestyle='-')
            fig.tight_layout()
            plt.title('%s' %mic_runs[k])
            outname='plots/'+mic_runs[k]+'.png'
            plt.savefig(outname)
            plt.close(fig)
            outnamelist.append(outname)
    return outnamelist

def plot_mic_cont(feat_names):
    outnamelist=[]
    if settings.compute_contribution_mic==1:
        mic_runs=[s for s in os.listdir() if 'mic_cont' in s]
        for k in range(len(mic_runs)):
            data=numpy.load(mic_runs[k])
            #mic_combs=data[0]
            mic_all=data[1]
            triangle=[]
            bottom,l=0,0
            for i in range(len(feat_names)-1,-1,-1):
                triangle.append(mic_all[bottom:bottom+i])
                bottom=bottom+i
                l=l+1
            
            for i in range(len(triangle)):
                diff = len(feat_names) - len(triangle[i])
                for j in range(diff):
                    triangle[i]=list(numpy.hstack((numpy.NaN,triangle[i])))
            xpos,ypos=numpy.meshgrid(numpy.array(range(len(feat_names)+1)),numpy.array(range(len(feat_names)+1)))
            square=numpy.array(triangle)
            Zm_sq=ma.masked_where(numpy.isnan(square),square)
            fig,ax=plt.subplots()
            cmap=plt.cm.jet
            cmap.set_bad('w',1)
            #cb=fig.colorbar()
            #cb.ax.tick_params(labelsize=8)
            ax.pcolormesh(xpos,ypos,Zm_sq,cmap=cmap)
            ax.set_xticks(numpy.array(range(len(feat_names)+1))),ax.set_yticks(numpy.array(range(len(feat_names)+1)))
            ax.set_xticklabels(feat_names,rotation='vertical',size=8,ha='left'),ax.set_yticklabels(feat_names,size=8,va='bottom')
            ax.grid(alpha=0.7,linestyle='-')
            fig.tight_layout()
            plt.title('%s' %mic_runs[k])
            outname='plots/'+mic_runs[k]+'.png'
            plt.savefig(outname)
            plt.close(fig)
            outnamelist.append(outname)
    return outnamelist

def plot_pearson(feat_names):
    outnamelist=[]
    if settings.plot_pearson == 1:
        pearson_runs1=[s for s in os.listdir() if 'mpearson_OvsA' in s]
        pearson_runs2=[s for s in os.listdir() if 'A_mpearson' in s]
        pearson_runs=pearson_runs1+pearson_runs2
        for k in range(len(pearson_runs)):
            data=numpy.load(pearson_runs[k])
            #pearson_combs=data[0]
            pearson_all=data[1]
            triangle=[]
            bottom,l=0,0
            for i in range(len(feat_names),-1,-1):
                triangle.append(pearson_all[bottom:bottom+i])
                bottom=bottom+i
                l=l+1
            
            for i in range(len(triangle)):
                diff = len(feat_names)+1 - len(triangle[i])
                for j in range(diff):
                    triangle[i]=list(numpy.hstack((numpy.NaN,triangle[i])))
            xpos,ypos=numpy.meshgrid(numpy.array(range(len(feat_names)+1)),numpy.array(range(len(feat_names)+1)))
            square=numpy.array(triangle)
            Zm_sq=ma.masked_where(numpy.isnan(square),square)
            fig,ax=plt.subplots()
            cmap=plt.cm.jet
            cmap.set_bad('w',1)
            #cb=fig.colorbar()
            #cb.ax.tick_params(labelsize=8)
            ax.pcolormesh(xpos,ypos,Zm_sq,cmap=cmap)
            ax.set_xticks(numpy.array(range(len(feat_names)+1))),ax.set_yticks(numpy.array(range(len(feat_names)+1)))
            ax.set_xticklabels(feat_names,rotation='vertical',size=8,ha='left'),ax.set_yticklabels(feat_names,size=8,va='bottom')
            ax.grid(alpha=0.7,linestyle='-')
            fig.tight_layout()
            plt.title('%s' %pearson_runs[k])
            outname='plots/'+pearson_runs[k]+'.png'
            plt.savefig(outname)
            plt.close(fig)
            outnamelist.append(outname)
    return outnamelist

def decision_boundaries_MINT(XX,XXpredict,yy,MINT_feats,MINT_feat_names,uniquetarget_tr):
    outnamelist=[]
    dirs=os.listdir(path)
    savedir='db_MINT' # Check if directory exists, if not, create
    fullsavedir=path+savedir+'/'
    if savedir not in dirs:
        os.mkdir(fullsavedir)
    if settings.plot_decision_boundaries_MINT == 1:
        plot_step = 0.1  # fine step width for decision surface contours
        plot_step_coarser = 0.5  # step widths for coarse classifier guesses
        if settings.make_binary == 0:
            plot_colors = "ryb"
            cmap = plt.cm.RdYlBu
        else:
            plot_colors="rb"
            cmap = plt.cm.RdBu
        n_classes = len(uniquetarget_tr[0])
        n_estimators = 256
        
        MLA = get_function(settings.MLA)        
        clf = MLA().set_params(**settings.MLAset)
        
        combs_MINT_index = list(it.combinations(MINT_feats['best_feats'],2))
        combs = list(it.combinations(range(len(MINT_feats['best_feats'])),2))
        
        for i in range(len(combs_MINT_index)):
            fig=plt.figure()
            x_min, x_max = XX[:, combs_MINT_index[i][0]].min() - 1, XX[:, combs_MINT_index[i][0]].max() + 1
            y_min, y_max = XX[:, combs_MINT_index[i][1]].min() - 1, XX[:, combs_MINT_index[i][1]].max() + 1
            xxx, yyy = numpy.meshgrid(numpy.arange(x_min, x_max, plot_step),numpy.arange(y_min, y_max, plot_step))
            
            clf = clf.fit(numpy.transpose(numpy.vstack((XX[:,combs_MINT_index[i][0]],XX[:,combs_MINT_index[i][1]]))),yy)        
            
            estimator_alpha = 1.0 / len(clf.estimators_)
            for tree in clf.estimators_:
                Z = tree.predict(numpy.c_[xxx.ravel(), yyy.ravel()])
                Z = Z.reshape(xxx.shape)
                cs = plt.contourf(xxx, yyy, Z, alpha=estimator_alpha, cmap=cmap)  
            xx_coarser, yy_coarser = numpy.meshgrid(numpy.arange(x_min, x_max, plot_step_coarser),numpy.arange(y_min, y_max, plot_step_coarser))
            Z_points_coarser = clf.predict(numpy.c_[xx_coarser.ravel(), yy_coarser.ravel()]).reshape(xx_coarser.shape)
            cs_points = plt.scatter(xx_coarser, yy_coarser, s=15, c=Z_points_coarser, cmap=cmap, edgecolors="none")
            for j, c in zip(range(n_classes), plot_colors):
                idx = numpy.where(yy == j)
                plt.scatter(XX[idx, combs_MINT_index[i][0]], XX[idx, combs_MINT_index[i][1]], c=c, label=uniquetarget_tr[0][j], cmap=cmap)
            plt.legend()
            plt.xlabel(MINT_feat_names[combs[i][0]])
            plt.ylabel(MINT_feat_names[combs[i][1]])
            outname=fullsavedir+'db'+str(i)+'.png'
            plt.savefig(outname)
            plt.close(fig)
            outnamelist.append(outname)
    return outnamelist

def decision_boundaries(XX,XXpredict,yy,yypredict,feat_names,uniquetarget_tr):
    outnamelist=[]
    dirs=os.listdir(path)
    savedir='db' # Check if directory exists, if not, create
    fullsavedir=path+savedir+'/'
    if savedir not in dirs:
        os.mkdir(fullsavedir)
    if settings.plot_decision_boundaries == 1:
        plot_step = 0.1  # fine step width for decision surface contours
        plot_step_coarser = 0.5  # step widths for coarse classifier guesses
        if settings.make_binary == 0:
            plot_colors = "ryb"
            cmap = plt.cm.RdYlBu
        else:
            plot_colors="rb"
            cmap = plt.cm.RdBu
        n_classes = len(uniquetarget_tr[0])
        n_estimators = 256
        
        MLA = get_function(settings.MLA)        
        clf = MLA().set_params(**settings.MLAset)
        
    #        combs_MINT_index = list(it.combinations(MINT_feats['best_feats'],2))
        combs = list(it.combinations(range(len(feat_names)),2))
        
        for i in range(len(combs)):
            fig=plt.figure()
            x_min, x_max = XX[:, combs[i][0]].min() - 1, XX[:, combs[i][0]].max() + 1
            y_min, y_max = XX[:, combs[i][1]].min() - 1, XX[:, combs[i][1]].max() + 1
            xxx, yyy = numpy.meshgrid(numpy.arange(x_min, x_max, plot_step),numpy.arange(y_min, y_max, plot_step))
            
            clf = clf.fit(numpy.transpose(numpy.vstack((XX[:,combs[i][0]],XX[:,combs[i][1]]))),yy)        
            result= clf.predict(numpy.transpose(numpy.vstack((XXpredict[:,combs[i][0]],XXpredict[:,combs[i][1]]))))
            accuracy = metrics.accuracy_score(result,yypredict)
            estimator_alpha = 1.0 / len(clf.estimators_)
            for tree in clf.estimators_:
                Z = tree.predict(numpy.c_[xxx.ravel(), yyy.ravel()])
                Z = Z.reshape(xxx.shape)
                cs = plt.contourf(xxx, yyy, Z, alpha=estimator_alpha, cmap=cmap)  
            xx_coarser, yy_coarser = numpy.meshgrid(numpy.arange(x_min, x_max, plot_step_coarser),numpy.arange(y_min, y_max, plot_step_coarser))
            Z_points_coarser = clf.predict(numpy.c_[xx_coarser.ravel(), yy_coarser.ravel()]).reshape(xx_coarser.shape)
            cs_points = plt.scatter(xx_coarser, yy_coarser, s=15, c=Z_points_coarser, cmap=cmap, edgecolors="none")
            for j, c in zip(range(n_classes), plot_colors):
                idx = numpy.where(yy == j)
                plt.scatter(XX[idx, combs[i][0]], XX[idx, combs[i][1]], c=c, label=uniquetarget_tr[0][j], cmap=cmap)
            plt.legend()
            axes = plt.gca()
            yaxes = axes.get_ylim()
            absy=yaxes[1]-yaxes[0]
            deltay=0.145-yaxes[0]
            xaxes = axes.get_xlim()
            absx=xaxes[1]-xaxes[0]
            deltax=0.145-xaxes[0]
            plt.axhline(y=.145,xmax=deltax/absx, linewidth=2,color='black')
            plt.axvline(x=.145,ymax=deltay/absy, linewidth=2,color='black')
            plt.xlabel(feat_names[combs[i][0]])
            plt.ylabel(feat_names[combs[i][1]])
            plt.title('Accuracy: %s' %accuracy)
            outname=fullsavedir+'db'+str(i)+'.png'
            plt.savefig(outname)
            plt.close(fig)
            outnamelist.append(outname)
    return outnamelist

def decision_boundaries_DT(XX,XXpredict,yy,yypredict,feat_names,uniquetarget_tr):
    outnamelist=[]
    dirs=os.listdir(path)
    savedir='db' # Check if directory exists, if not, create
    fullsavedir=path+savedir+'/'
    if savedir not in dirs:
        os.mkdir(fullsavedir)
    if settings.plot_decision_boundaries_DT == 1:
        plot_step = 0.1  # fine step width for decision surface contours
        plot_step_coarser = 0.5  # step widths for coarse classifier guesses
        if settings.make_binary == 0:
            plot_colors = "ryb"
            cmap = plt.cm.RdYlBu
        else:
            plot_colors="rb"
            cmap = plt.cm.RdBu
        n_classes = len(uniquetarget_tr[0])
        n_estimators = 256
        MLA = 'sklearn.tree.DecisionTreeClassifier'                             # Which MLA to load
        MLAset = {'max_depth':10} 
        MLA = get_function(MLA)        
        clf = MLA().set_params(**MLAset)
        
    #        combs_MINT_index = list(it.combinations(MINT_feats['best_feats'],2))
        combs = list(it.combinations(range(len(feat_names)),2))
        
        for i in range(len(combs)):
            fig=plt.figure()
            x_min, x_max = XX[:, combs[i][0]].min() - 1, XX[:, combs[i][0]].max() + 1
            y_min, y_max = XX[:, combs[i][1]].min() - 1, XX[:, combs[i][1]].max() + 1
            xxx, yyy = numpy.meshgrid(numpy.arange(x_min, x_max, plot_step),numpy.arange(y_min, y_max, plot_step))
            
            clf = clf.fit(XX[:,combs[i][0],None],yy)        
            result= clf.predict(XXpredict[:,combs[i][0],None])
            accuracy = metrics.accuracy_score(result,yypredict)
#            estimator_alpha = 1.0 / len(clf.estimators_)
#            for tree in clf.estimators_:
            Z = clf.predict(numpy.c_[xxx.ravel()])
            Z = Z.reshape(xxx.shape)
            cs = plt.contourf(xxx, yyy, Z, cmap=cmap)  
            xx_coarser, yy_coarser = numpy.meshgrid(numpy.arange(x_min, x_max, plot_step_coarser),numpy.arange(y_min, y_max, plot_step_coarser))
            Z_points_coarser = clf.predict(numpy.c_[xx_coarser.ravel()]).reshape(xx_coarser.shape)
            cs_points = plt.scatter(xx_coarser, yy_coarser, s=15, c=Z_points_coarser, cmap=cmap, edgecolors="none")
            for j, c in zip(range(n_classes), plot_colors):
                idx = numpy.where(yy == j)
                plt.scatter(XX[idx, combs[i][0]], XX[idx, combs[i][1]], c=c, label=uniquetarget_tr[0][j], cmap=cmap)
            plt.legend()
            axes = plt.gca()
            yaxes = axes.get_ylim()
            absy=yaxes[1]-yaxes[0]
            deltay=0.145-yaxes[0]
            xaxes = axes.get_xlim()
            absx=xaxes[1]-xaxes[0]
            deltax=0.145-xaxes[0]
            plt.axhline(y=.145,xmax=deltax/absx, linewidth=2,color='black')
            plt.axvline(x=.145,ymax=deltay/absy, linewidth=2,color='black')
            plt.xlabel(feat_names[combs[i][0]])
            plt.ylabel(feat_names[combs[i][1]])
            plt.title('Decision Tree Accuracy: %s' %accuracy)
            outname=fullsavedir+'db'+str(i)+'.png'
            plt.savefig(outname)
            plt.close(fig)
            outnamelist.append(outname)
    return outnamelist

def plot_depth_acc(XXpredict,result,yypredict,feat_names,filtstats,uniquetarget_tr,dered_tr_r,dered_pr_r):
    outnamelist=[]
    if settings.plot_depth_acc==1:
        dirs=os.listdir(path)
        savedir='plot_depth_acc' # Check if directory exists, if not, create
        fullsavedir=path+savedir+'/'
        if savedir not in dirs:
            os.mkdir(fullsavedir)       
#        for i in range(len(feat_names)): # Plot for all filters
#                fig=plt.figure()
#            f, (axarr) = plt.subplots(2, 2)#,figsize=(14, 5))
            
        true_class = yypredict==result
        outliermask1=is_outlier(dered_pr_r)
        totalmask_true = (true_class) & (~outliermask1)
        # plot
        hist_true,bin_edges,patches = plt.hist(dered_pr_r[totalmask_true], bins=80)
        hist_total,bin_edges_tot,patches_tot= plt.hist(dered_pr_r[~outliermask1], bins=80,range=(bin_edges.min(),bin_edges.max()))
        hist_acc = (hist_true/hist_total)
        hist_acc = numpy.nan_to_num(hist_acc)
        hist_acc = hist_acc*100
        plt.figure()
        plt.step(bin_edges[:-1],hist_acc)
        plt.ylim(0,110)
        plt.title('Depth Accuracy in DERED_R')
        plt.xlabel('DERED_R')
        plt.ylabel('Accuracy (%)')
        outname='plots/'+savedir+'/depth_acc_DR_R.png'
        outnamelist.append(outname)
        plt.tight_layout()
        plt.savefig(outname)
        plt.close()
    return outnamelist

def plot_oob_err_rate(XX,yy):
    outnamelist=[]
    if settings.plot_oob_err_rate==1:
        dirs=os.listdir(path)
        savedir='plot_oob_err_rate' # Check if directory exists, if not, create
        fullsavedir=path+savedir+'/'
        if savedir not in dirs:
            os.mkdir(fullsavedir)    
        MLA = get_function(settings.MLA)        
        clf = MLA().set_params(**settings.MLAset)
        ensemble_clfs = [
            ("RandomForestClassifier, max_features='sqrt'",
                MLA(warm_start=True, oob_score=True,
                                       max_features="sqrt")),
            ("RandomForestClassifier, max_features='log2'",
                MLA(warm_start=True, max_features='log2',
                                       oob_score=True)),
            ("RandomForestClassifier, max_features=None",
                MLA(warm_start=True, max_features=None,
                                       oob_score=True))]
        # Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
        error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)
        
        # Range of `n_estimators` values to explore.
        min_estimators = 15
        max_estimators = 175
        
        for label, clf in ensemble_clfs:
            for i in range(min_estimators, max_estimators + 1):
                clf.set_params(n_estimators=i)
                clf.fit(XX, yy)
                # Record the OOB error for each `n_estimators=i` setting.
                oob_error = 1 - clf.oob_score_
                error_rate[label].append((i, oob_error))
        # Generate the "OOB error rate" vs. "n_estimators" plot.
        for label, clf_err in error_rate.items():
            xs, ys = zip(*clf_err)
            plt.plot(xs, ys, label=label)
        plt.xlim(min_estimators, max_estimators)
        plt.xlabel("n_estimators")
        plt.ylabel("OOB error rate")
        plt.legend(loc="upper right")
#        plt.show()
        outname='plots/'+savedir+'/oob_err_rate.png'
        outnamelist.append(outname)
        plt.tight_layout()
        plt.savefig(outname)
        plt.close()
    return outnamelist

def get_function(function_string):
    import importlib
    module, function = function_string.rsplit('.', 1)
    module = importlib.import_module(module)
    function = getattr(module, function)
    return function

def is_outlier(points, thresh=6):
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
