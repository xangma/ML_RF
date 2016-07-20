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
os.chdir(root_path) # Change directory
os.chdir('runresults')
runresults_path=os.getcwd()
runresults_dirs=os.listdir(runresults_path)

run_name='RUN_20160719-2347'
stats_name='stats_'+run_name
dirs_run = [s for s in runresults_dirs if run_name in s]
stats_file= [s for s in runresults_dirs if stats_name in s]
dirs_run.remove(stats_file[0])
run_stats=numpy.load(stats_file[0])
unique_n_depth = run_stats['n_depth']
unique_n_est=run_stats['n_estimators']
unique_n_train=run_stats['n_train']
filters = run_stats['filters']
usecolours= run_stats['use_colours']

if 'run_plots' not in runresults_dirs:
    os.mkdir('run_plots')

run_plots_dir = os.listdir('run_plots')
if run_name not in run_plots_dir:
    os.mkdir('run_plots/%s'%run_name)
    
# READ DATA
n_estimators_arr,n_train_arr,n_depth_arr,result_arr,feat_arr=numpy.array([]),numpy.array([]),numpy.array([]),numpy.array([]),{}
for i in range(len(dirs_run)):
    os.chdir(runresults_path+'/'+dirs_run[i])
    run_path=os.getcwd()
    sub_dir=os.listdir(run_path)
    stat_file=[s for s in sub_dir if 'ML_RF_stats' in s]
    feat_file=[s for s in sub_dir if 'ML_RF_feat_' in s]
    stat_arr=numpy.genfromtxt(stat_file[0])
    featrun_arr=numpy.genfromtxt(feat_file[0])
    n_estimators_arr=numpy.append(n_estimators_arr,stat_arr[0])
    n_train_arr=numpy.append(n_train_arr,stat_arr[1])
    result_arr=numpy.append(result_arr,stat_arr[3])
    n_depth_arr = numpy.append(n_depth_arr,stat_arr[4])
    feat_arr[i]=featrun_arr
n_depth_arr =n_depth_arr.tolist()

for i in range(len(n_depth_arr)):
    if numpy.isnan(n_depth_arr[i]) == True: 
        n_depth_arr[i] = 'None'

n_depth_arr=numpy.transpose(n_depth_arr)

os.chdir(runresults_path)

# PLOTS START
for i in range(len(unique_n_est)):
#    print(unique_n_est[i])
    mask=n_estimators_arr==unique_n_est[i] # KEEP EST CONSTANT
    sort= n_train_arr[mask].argsort()
    feat_sub_arr={}
    n_est_index=numpy.where(mask==True)
    n_est_ind_sorted=n_est_index[0][sort]
    for j in range(len(mask)):
        if mask[j] == True:
            feat_sub_arr[j]=feat_arr[j]
    plt.figure()
    for j in range(len(sort))[2:-2]:
        plt.scatter(numpy.array(range(len(feat_arr[0])))+1,feat_sub_arr[n_est_ind_sorted[j]],s=2)
        plt.plot(numpy.array(range(len(feat_arr[0])))+1,feat_sub_arr[n_est_ind_sorted[j]],label='n_train == %s' %n_train_arr[mask][sort][j])
    plt.axvspan(5.5, 15.5, color='red', alpha=0.3)
    plt.axvspan(20.5, 30.5, color='red', alpha=0.3)
    plt.axvspan(35.5, 45.5, color='red', alpha=0.3)
    plt.legend()
    plt.title('n_estimators == %s'%unique_n_est[i])
    plt.xlabel('Features')
    plt.ylabel('Feat_Importance')
    plt.xlim(0.5,45.5)
    plt.ylim(0,0.2)
#    plt.xticks(numpy.array(range(len(feat_arr[0]))))
    plt.minorticks_on()
    plt.grid(alpha=0.4,which='both')
    plt.savefig('run_plots/'+run_name+'/feat_imp_vs_value_n_est_%s.png'%unique_n_est[i])
#    plt.show()
    plt.close()

for i in range(len(unique_n_train)):
#    print(unique_n_train[i])
    mask=n_train_arr==unique_n_train[i] # KEEP EST CONSTANT
    sort= n_estimators_arr[mask].argsort()
    feat_sub_arr={}
    n_train_index=numpy.where(mask==True)
    n_train_ind_sorted=n_train_index[0][sort]
    for j in range(len(mask)):
        if mask[j] == True:
            feat_sub_arr[j]=feat_arr[j]
    plt.figure()
    for j in range(len(sort))[5:-2]:
        plt.scatter(numpy.array(range(len(feat_arr[0])))+1,feat_sub_arr[n_train_ind_sorted[j]],s=2)
        plt.plot(numpy.array(range(len(feat_arr[0])))+1,feat_sub_arr[n_train_ind_sorted[j]],label='n_est == %s' %n_estimators_arr[mask][sort][j])
    plt.axvspan(5.5, 15.5, color='red', alpha=0.3)
    plt.axvspan(20.5, 30.5, color='red', alpha=0.3)
    plt.axvspan(35.5, 45.5, color='red', alpha=0.3)
    plt.legend()
    plt.title('n_train == %s'%unique_n_train[i])
    plt.xlabel('Features')
    plt.ylabel('Feat_Importance')
    plt.ylim(0,0.2)
    plt.xlim(0.5,45.5)
#    plt.xticks(numpy.array(range(len(feat_arr[0]))))
    plt.minorticks_on()
    plt.grid(alpha=0.4,which='both')
    plt.savefig('run_plots/'+run_name+'/feat_imp_vs_value_n_train_%s.png'%unique_n_train[i])
#    plt.show()
    plt.close()

for i in range(len(unique_n_depth)):
#    print(unique_n_est[i])
    print(i)
    mask=n_depth_arr==unique_n_depth[i] # KEEP EST CONSTANT
    sort= n_train_arr[mask].argsort()
    feat_sub_arr={}
    n_depth_index=numpy.where(mask==True)
    n_depth_ind_sorted=n_depth_index[0][sort]
    for j in range(len(mask)):
        if mask[j] == True:
            feat_sub_arr[j]=feat_arr[j]
    plt.figure()
    for j in range(len(sort))[2:-2]:
        plt.scatter(numpy.array(range(len(feat_arr[0])))+1,feat_sub_arr[n_depth_ind_sorted[j]],s=2)
        plt.plot(numpy.array(range(len(feat_arr[0])))+1,feat_sub_arr[n_depth_ind_sorted[j]],label='n_train == %s' %n_train_arr[mask][sort][j])
    plt.axvspan(5.5, 15.5, color='red', alpha=0.3)
    plt.axvspan(20.5, 30.5, color='red', alpha=0.3)
    plt.axvspan(35.5, 45.5, color='red', alpha=0.3)
    plt.legend()
    plt.title('n_depth == %s'%unique_n_depth[i])
    plt.xlabel('Features')
    plt.ylabel('Feat_Importance')
    plt.xlim(0.5,45.5)
    plt.ylim(0,0.2)
#    plt.xticks(numpy.array(range(len(feat_arr[0]))))
#    plt.minorticks_on()
    plt.grid(alpha=0.4,which='both')
    plt.savefig('run_plots/'+run_name+'/feat_imp_vs_value_n_depth_%s.png'%unique_n_depth[i])
#    plt.show()
    plt.close()

for i in range(len(unique_n_train)):
#    print(unique_n_train[i])
    mask=n_train_arr==unique_n_train[i] # KEEP EST CONSTANT
    sort= n_depth_arr[mask].argsort()
    feat_sub_arr={}
    n_train_index=numpy.where(mask==True)
    n_train_ind_sorted=n_train_index[0][sort]
    for j in range(len(mask)):
        if mask[j] == True:
            feat_sub_arr[j]=feat_arr[j]
    plt.figure()
    for j in range(len(sort))[5:-2]:
        plt.scatter(numpy.array(range(len(feat_arr[0])))+1,feat_sub_arr[n_train_ind_sorted[j]],s=2)
        plt.plot(numpy.array(range(len(feat_arr[0])))+1,feat_sub_arr[n_train_ind_sorted[j]],label='n_depth == %s' %n_depth_arr[mask][sort][j])
    plt.axvspan(5.5, 15.5, color='red', alpha=0.3)
    plt.axvspan(20.5, 30.5, color='red', alpha=0.3)
    plt.axvspan(35.5, 45.5, color='red', alpha=0.3)
    plt.legend()
    plt.title('n_train == %s'%unique_n_train[i])
    plt.xlabel('Features')
    plt.ylabel('Feat_Importance')
    plt.ylim(0,0.2)
    plt.xlim(0.5,45.5)
#    plt.xticks(numpy.array(range(len(feat_arr[0]))))
#    plt.minorticks_on()
    plt.grid(alpha=0.4,which='both')
    plt.savefig('run_plots/'+run_name+'/feat_imp_vs_value_n_train_%s.png'%unique_n_train[i])
#    plt.show()
    plt.close()

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
for i in range(len(unique_n_est))[:-3]:
    mask=n_estimators_arr==unique_n_est[i]
    sort= n_train_arr[mask].argsort()
    plt.scatter(n_train_arr[mask][sort],result_arr[mask][sort],s=2)
    plt.plot(n_train_arr[mask][sort],result_arr[mask][sort],label='n_est == %s' %unique_n_est[i])
plt.legend(loc=4)
plt.title('n_estimators')
plt.xlabel('n_train')
plt.ylabel('success (%)')
plt.xscale('log')
plt.ylim(75,100)
#plt.show()
plt.grid(False)
plt.savefig('run_plots/'+run_name+'/n_train_vs_success_n_est_total.png')
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
    mask=n_train_arr==unique_n_train[i]
    sort= n_estimators_arr[mask].argsort()
    plt.scatter(n_estimators_arr[mask][sort],result_arr[mask][sort],s=2)
    plt.plot(n_estimators_arr[mask][sort],result_arr[mask][sort],label='n_train == %s'%unique_n_train[i])
plt.legend(loc=4)
plt.title('n_train')
plt.xlabel('n_estimators')
plt.ylabel('success (%)')
plt.xscale('log')
plt.grid()
plt.ylim(75,100)
#plt.show()
plt.grid(False)
plt.savefig('run_plots/'+run_name+'/n_estimators_vs_success_n_train_total.png')
plt.close()

# Total n_train vs success
plt.figure()
for i in range(len(unique_n_depth)):
    mask=n_depth_arr==unique_n_depth[i]
    sort= n_train_arr[mask].argsort()
    plt.scatter(n_train_arr[mask][sort],result_arr[mask][sort],s=2)
    plt.plot(n_train_arr[mask][sort],result_arr[mask][sort],label='n_depth == %s' %unique_n_depth[i])
plt.legend(loc=3)
plt.title('n_depth')
plt.xlabel('n_train')
plt.ylabel('success (%)')
plt.xscale('log')
plt.ylim(75,100)
#plt.show()
plt.grid(False)
plt.savefig('run_plots/'+run_name+'/n_train_vs_success_n_depth_total.png')
plt.close()