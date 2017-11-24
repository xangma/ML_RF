# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 19:23:59 2016

@author: moricex
"""
import os
import numpy
import sys
import shutil
import importlib
import settings
import matplotlib.pyplot as plt
import time

plt.ioff()

root_path='/users/moricex/ML_RF'
#run_name='RUN_20161003-134245'
#run_name='RUN_20161003-143643'
run_name='RUN_20161003-143643'
os.chdir(root_path) # Change directory
os.chdir('runresults')
run_path=os.getcwd()
runresults=os.listdir(run_path)
os.chdir(run_name+'_MINT')
runresults_path=os.getcwd()
runresults_dirs=os.listdir(runresults_path)

stats_name='stats_'+run_name
paths_name='paths_'+run_name
dirs_run = [s for s in runresults_dirs if run_name in s]
stats_file= [s for s in runresults if stats_name in s]
path_file= [s for s in runresults if paths_name in s]
#dirs_run.remove(stats_file[0])
#dirs_run.remove(path_file[0])
os.chdir(root_path+'/runresults')
run_stats=numpy.load(stats_file[0])
unique_n_depth = run_stats['n_depth']
unique_n_est=run_stats['n_estimators']
unique_n_train=run_stats['n_train']
#filters = run_stats['filters']
#usecolours= run_stats['use_colours']

if 'run_plots' not in runresults:
    os.mkdir('run_plots')

run_plots_dir = os.listdir('run_plots')
if run_name not in run_plots_dir:
    os.mkdir('run_plots/%s'%run_name)
    
# READ DATA
scores_arr,test_arr,nfeat_arr,n_estimators_arr,n_train_arr,n_depth_arr,result_arr,feat_arr,featnames_arr=\
numpy.array([]),numpy.array([]),numpy.array([]),numpy.array([]),numpy.array([]),numpy.array([]),numpy.array([]),{},{}
for i in range(len(dirs_run)):
    os.chdir(runresults_path+'/'+dirs_run[i])
    run_path=os.getcwd()
    sub_dir=os.listdir(run_path)
#    stat_file=[s for s in sub_dir if 'ML_RF_stats' in s]
    feat_file=[s for s in sub_dir if 'ML_RF_feat_' in s]
    featnames_file=[s for s in sub_dir if 'ML_RF_featnames_' in s]
    scores_file=[s for s in sub_dir if 'ML_RF_scores_' in s]
    exec(open(run_path+'/settings.py').read())
#    test_arr=numpy.append(test_arr,MLAset)
#    stat_arr=numpy.genfromtxt(stat_file[0])
    featrun_arr=numpy.genfromtxt(feat_file[0])
    featnamesrun_arr=numpy.load(featnames_file[0])
    scoresrun_arr=numpy.genfromtxt(scores_file[0])
#    scores_arr= numpy.append(scores_arr,scoresrun_arr)
    nfeat_arr = numpy.append(nfeat_arr,len(featnamesrun_arr))
    n_estimators_arr=numpy.append(n_estimators_arr,MLAset['n_estimators'])
    n_train_arr=numpy.append(n_train_arr,traindatanum)
    result_arr=numpy.append(result_arr,scoresrun_arr[0,2])
    n_depth_arr = numpy.append(n_depth_arr,MLAset['max_depth'])
    featnames_arr[i] = featnamesrun_arr
    feat_arr[i]=featrun_arr
n_depth_arr =n_depth_arr.tolist()

for i in range(len(n_depth_arr)):
    if n_depth_arr[i] == None: 
        n_depth_arr[i] = 'None'

n_depth_arr=numpy.transpose(n_depth_arr)

os.chdir(runresults_path)

# CUT IN:
for cut_i in range(len(unique_n_depth)):
    CUT = n_depth_arr == unique_n_depth[cut_i]
    
    # PLOTS START
#    for i in range(len(unique_n_est)):
#    #    print(unique_n_est[i])
#        mask=(n_estimators_arr==unique_n_est[i]) & CUT# KEEP EST CONSTANT
#        sort= n_train_arr[mask].argsort()
#        feat_sub_arr={}
#        n_est_index=numpy.where(mask==True)
#        n_est_ind_sorted=n_est_index[0][sort]
#        for j in range(len(mask)):
#            if mask[j] == True:
#                feat_sub_arr[j]=feat_arr[j]
#        plt.figure()
#        for j in range(len(sort))[2:-2]:
#            plt.scatter(numpy.array(range(len(feat_arr[0])))+1,feat_sub_arr[n_est_ind_sorted[j]],s=2)
#            plt.plot(numpy.array(range(len(feat_arr[0])))+1,feat_sub_arr[n_est_ind_sorted[j]],label='n_train == %s' %n_train_arr[mask][sort][j])
#    #    plt.axvspan(5.5, 15.5, color='red', alpha=0.3)
#    #    plt.axvspan(20.5, 30.5, color='red', alpha=0.3)
#    #    plt.axvspan(35.5, 45.5, color='red', alpha=0.3)
#        plt.legend()
#        plt.title('n_estimators == %s'%unique_n_est[i])
#        plt.xlabel('Features')
#        plt.ylabel('Feat_Importance')
#        plt.xlim(0.5,(len(featrun_arr)+0.5))
#    #    plt.ylim(0,0.2)
#    #    plt.xticks(numpy.array(range(len(feat_arr[0]))))
#        plt.minorticks_on()
#        plt.grid(alpha=0.4,which='both')
#        plt.savefig('../run_plots/'+run_name+'/feat_imp_vs_value_n_est_%s_n_depth_%s.png'%(unique_n_est[i],unique_n_depth[cut_i]))
#    #    plt.show()
#        plt.close()
    
#    for i in range(len(unique_n_train)):
#    #    print(unique_n_train[i])
#        mask=(n_train_arr==unique_n_train[i])& CUT # KEEP EST CONSTANT
#        sort= n_estimators_arr[mask].argsort()
#        feat_sub_arr={}
#        n_train_index=numpy.where(mask==True)
#        n_train_ind_sorted=n_train_index[0][sort]
#        for j in range(len(mask)):
#            if mask[j] == True:
#                feat_sub_arr[j]=feat_arr[j]
#        plt.figure()
#        for j in range(len(sort))[5:-2]:
#            plt.scatter(numpy.array(range(len(feat_arr[0])))+1,feat_sub_arr[n_train_ind_sorted[j]],s=2)
#            plt.plot(numpy.array(range(len(feat_arr[0])))+1,feat_sub_arr[n_train_ind_sorted[j]],label='n_est == %s' %n_estimators_arr[mask][sort][j])
#    #    plt.axvspan(5.5, 15.5, color='red', alpha=0.3)
#    #    plt.axvspan(20.5, 30.5, color='red', alpha=0.3)
#    #    plt.axvspan(35.5, 45.5, color='red', alpha=0.3)
#        plt.legend()
#        plt.title('n_train == %s'%unique_n_train[i])
#        plt.xlabel('Features')
#        plt.ylabel('Feat_Importance')
#        plt.ylim(0,0.2)
#        plt.xlim(0.5,(len(featnamesrun_arr)+0.5))
#    #    plt.xticks(numpy.array(range(len(feat_arr[0]))))
#        plt.minorticks_on()
#        plt.grid(alpha=0.4,which='both')
#        plt.savefig('../run_plots/'+run_name+'/feat_imp_vs_value_n_train_%s_n_depth_%s.png'%(unique_n_train[i],unique_n_depth[cut_i]))
#    #    plt.show()
#        plt.close()
    
#    for i in range(len(unique_n_depth)):
#    #    print(unique_n_est[i])
#        print(i)
#        mask=(n_depth_arr==unique_n_depth[i]) & CUT # KEEP EST CONSTANT
#        sort= n_train_arr[mask].argsort()
#        feat_sub_arr={}
#        n_depth_index=numpy.where(mask==True)
#        n_depth_ind_sorted=n_depth_index[0][sort]
#        for j in range(len(mask)):
#            if mask[j] == True:
#                feat_sub_arr[j]=feat_arr[j]
#        plt.figure()
#        for j in range(len(sort))[2:-2]:
#            plt.scatter(numpy.array(range(len(feat_arr[0])))+1,feat_sub_arr[n_depth_ind_sorted[j]],s=2)
#            plt.plot(numpy.array(range(len(feat_arr[0])))+1,feat_sub_arr[n_depth_ind_sorted[j]],label='n_train == %s' %n_train_arr[mask][sort][j])
#    #    plt.axvspan(5.5, 15.5, color='red', alpha=0.3)
#    #    plt.axvspan(20.5, 30.5, color='red', alpha=0.3)
#    #    plt.axvspan(35.5, 45.5, color='red', alpha=0.3)
#        plt.legend()
#        plt.title('n_depth == %s'%unique_n_depth[i])
#        plt.xlabel('Features')
#        plt.ylabel('Feat_Importance')
#        plt.xlim(0.5,(len(featnamesrun_arr)+0.5))
#    #    plt.ylim(0,0.2)
#    #    plt.xticks(numpy.array(range(len(feat_arr[0]))))
#    #    plt.minorticks_on()
#        plt.grid(alpha=0.4,which='both')
#        plt.savefig('../run_plots/'+run_name+'/feat_imp_vs_value_n_depth_%s.png'%unique_n_depth[i])
#    #    plt.show()
#        plt.close()
    
#    for i in range(len(unique_n_train)):
#    #    print(unique_n_train[i])
#        mask=(n_train_arr==unique_n_train[i]) & CUT # KEEP EST CONSTANT
#        sort= n_depth_arr[mask].argsort()
#        feat_sub_arr={}
#        n_train_index=numpy.where(mask==True)
#        n_train_ind_sorted=n_train_index[0][sort]
#        for j in range(len(mask)):
#            if mask[j] == True:
#                feat_sub_arr[j]=feat_arr[j]
#        plt.figure()
#        for j in range(len(sort))[5:-2]:
#            plt.scatter(numpy.array(range(len(feat_arr[0])))+1,feat_sub_arr[n_train_ind_sorted[j]],s=2)
#            plt.plot(numpy.array(range(len(feat_arr[0])))+1,feat_sub_arr[n_train_ind_sorted[j]],label='n_depth == %s' %n_depth_arr[mask][sort][j])
#    #    plt.axvspan(5.5, 15.5, color='red', alpha=0.3)
#    #    plt.axvspan(20.5, 30.5, color='red', alpha=0.3)
#    #    plt.axvspan(35.5, 45.5, color='red', alpha=0.3)
#        plt.legend()
#        plt.title('n_train == %s'%unique_n_train[i])
#        plt.xlabel('Features')
#        plt.ylabel('Feat_Importance')
#    #    plt.ylim(0,0.2)
#        plt.xlim(0.5,(len(featnamesrun_arr)+0.5))
#    #    plt.xticks(numpy.array(range(len(feat_arr[0]))))
#    #    plt.minorticks_on()
#        plt.grid(alpha=0.4,which='both')
#        plt.savefig('../run_plots/'+run_name+'/feat_imp_vs_value_n_train_%s_n_depth_%s.png'%(unique_n_train[i],unique_n_depth[cut_i]))
#    #    plt.show()
#        plt.close()
    
    ## Individual n_train vs success
    #for i in range(len(unique_n_est)):
    #    mask=n_estimators_arr==unique_n_est[i]
    #    sort= n_train_arr[mask].argsort()
    #    plt.figure()
    #    plt.scatter(n_train_arr[mask][sort],result_arr[mask][sort])
    #    plt.plot(n_train_arr[mask][sort],result_arr[mask][sort])
    #    plt.title('n_estimators == %s'%unique_n_est[i])
    #    plt.xlabel('n_train')
    #    plt.ylabel('success (%)')
    #    plt.ylim(75,100)
    #    plt.savefig('run_plots/'+run_name+'/n_train_vs_success_n_est_%s.png'%unique_n_est[i])
    #    plt.close()
    
    # Total n_train vs success
    plt.figure()
    for i in range(len(unique_n_est))[3:-2]:
        mask=(n_estimators_arr==unique_n_est[i]) & CUT
        sort= n_train_arr[mask].argsort()
        plt.scatter(n_train_arr[mask][sort],result_arr[mask][sort]*100,s=2)
        plt.plot(n_train_arr[mask][sort],result_arr[mask][sort]*100,label='n_est == %s' %unique_n_est[i])
    plt.legend(loc=4)
    plt.title('n_estimators')
    plt.xlabel('n_train')
    plt.ylabel('success (%)')
    plt.xscale('log')
    plt.ylim(95,100)
    #plt.show()
    plt.grid(False)
    plt.savefig('../run_plots/'+run_name+'/n_train_vs_success_n_est_total_n_depth_%s.png'%unique_n_depth[cut_i])
    plt.close()
    
    ## Individual n_estimators vs n_train
    #for i in range(len(unique_n_train)):
    #    mask=n_train_arr==unique_n_train[i]
    #    sort= n_estimators_arr[mask].argsort()
    #    plt.figure()
    #    plt.scatter(n_estimators_arr[mask][sort],result_arr[mask][sort])
    #    plt.plot(n_estimators_arr[mask][sort],result_arr[mask][sort])
    #    plt.title('n_train == %s'%unique_n_train[i])
    #    plt.xlabel('n_estimators')
    #    plt.ylabel('success (%)')
    #    plt.ylim(75,100)
    #    plt.savefig('run_plots/'+run_name+'/n_estimators_vs_success_n_train_%s.png'%unique_n_train[i])
    #    plt.close()
    
    # Total n_estimatos vs n_train
    plt.figure()
    for i in range(len(unique_n_train))[2:]:
        mask=(n_train_arr==unique_n_train[i]) & CUT
        sort= n_estimators_arr[mask].argsort()
        plt.scatter(n_estimators_arr[mask][sort],result_arr[mask][sort]*100,s=2)
        plt.plot(n_estimators_arr[mask][sort],result_arr[mask][sort]*100,label='n_train == %s'%unique_n_train[i])
    plt.legend(loc=4)
    plt.title('n_train')
    plt.xlabel('n_estimators')
    plt.ylabel('success (%)')
    plt.xscale('log')
    plt.grid()
    plt.ylim(95,100)
    #plt.show()
    plt.grid(False)
    plt.savefig('../run_plots/'+run_name+'/n_estimators_vs_success_n_train_total_n_depth_%s.png'%unique_n_depth[cut_i])
    plt.close()
    
    # Total n_train vs success
    plt.figure()
    for i in range(len(unique_n_depth)):
        mask=(n_depth_arr==unique_n_depth[i])& CUT
        sort= n_train_arr[mask].argsort()
        plt.scatter(n_train_arr[mask][sort],result_arr[mask][sort]*100,s=2)
        plt.plot(n_train_arr[mask][sort],result_arr[mask][sort]*100,label='n_depth == %s' %unique_n_depth[i])
    plt.legend(loc=3)
    plt.title('n_depth')
    plt.xlabel('n_train')
    plt.ylabel('success (%)')
    plt.xscale('log')
    plt.ylim(95,100)
    #plt.show()
    plt.grid(False)
    plt.savefig('../run_plots/'+run_name+'/n_train_vs_success_n_depth_total_%s.png'%unique_n_depth[cut_i])
    plt.close()
    
#    # Total n_estimatos vs n_train vs nfeat
#    plt.figure()
#    for j in range(len(unique_n_est)):
#        for i in range(len(unique_n_train))[2:]:
#            mask=(n_train_arr==unique_n_train[i]) & CUT & (n_estimators_arr==unique_n_est[j])
#            sort= nfeat_arr[mask].argsort()
#            plt.scatter(nfeat_arr[mask][sort],result_arr[mask][sort]*100,s=2)
#            plt.plot(nfeat_arr[mask][sort],result_arr[mask][sort]*100,label='n_train == %s'%unique_n_train[i])
#        plt.legend(loc=4)
#        plt.title('n_train, n_est=%s'%unique_n_est[j])
#        plt.xlabel('nfeat')
#        plt.ylabel('success (%)')
#    #    plt.xscale('log')
#        plt.grid()
#        plt.ylim(95,100)
#        #plt.show()
#        plt.grid(False)
#        plt.savefig('../run_plots/'+run_name+'/nfeat_vs_success_n_train_total_n_depth_%s_n_est_%s.png'%(unique_n_depth[cut_i],unique_n_est[j]))
#        plt.close()
#        
#    # Total n_estimatos vs n_train vs nfeat
#    plt.figure()
#    for j in range(len(unique_n_train)):
#        for i in range(len(unique_n_est))[2:]:
#            mask=(n_estimators_arr==unique_n_est[i]) & CUT & (n_train_arr==unique_n_train[j])
#            sort= nfeat_arr[mask].argsort()
#            plt.scatter(nfeat_arr[mask][sort],result_arr[mask][sort]*100,s=2)
#            plt.plot(nfeat_arr[mask][sort],result_arr[mask][sort]*100,label='n_est == %s'%unique_n_est[i])
#        plt.legend(loc=4)
#        plt.title('n_est, n_train=%s'%unique_n_train[j])
#        plt.xlabel('nfeat')
#        plt.ylabel('success (%)')
#    #    plt.xscale('log')
#        plt.grid()
#        plt.ylim(95,100)
#        #plt.show()
#        plt.grid(False)
#        plt.savefig('../run_plots/'+run_name+'/nfeat_vs_success_n_est_total_n_depth_%s_n_train_%s.png'%(unique_n_depth[cut_i],unique_n_train[j]))
#        plt.close()
    
bestres=result_arr.argsort()
print('Best result n_estimators: %s' %n_estimators_arr[bestres[-1]])
print('Best result n_train: %s' %n_train_arr[bestres[-1]])
print('Best result n_depth: %s' %n_depth_arr[bestres[-1]])
print('Best result features: %s' %featnames_arr[bestres[-1]])
print('Best result accuracy: %s' %result_arr[bestres[-1]])
findex= n_train_arr[bestres] <10000
lown_arr=numpy.where(findex==True)
idofobj=bestres[lown_arr]
print('Found settings with highest accuracy but n_train below 10000')
print('Best result n_estimators: %s' %n_estimators_arr[idofobj[-1]])
print('Best result n_train: %s' %n_train_arr[idofobj[-1]])
print('Best result n_depth: %s' %n_depth_arr[idofobj[-1]])
print('Best result features: %s' %featnames_arr[idofobj[-1]])
print('Best result accuracy: %s' %result_arr[idofobj[-1]])