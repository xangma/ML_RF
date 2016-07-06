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

root_path=os.getcwd()
#os.chdir(programpath) # Change directory
os.chdir('runresults')
runresults_path=os.getcwd()
runresults_dirs=os.listdir(runresults_path)

run_name='RUN_20160706-16'
stats_name='stats_'+run_name
dirs_run = [s for s in runresults_dirs if run_name in s]
stats_file= [s for s in runresults_dirs if stats_name in s]
dirs_run.remove(stats_file[0])
run_stats=numpy.load(stats_file[0])
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
n_estimators_arr,n_train_arr,result_arr,feat_arr=numpy.array([]),numpy.array([]),numpy.array([]),{}
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
    feat_arr[i]=featrun_arr

os.chdir(runresults_path)

# PLOTS START


for i in range(len(unique_n_est)):
    mask=n_estimators_arr==unique_n_est[i]
    sort= n_train_arr[mask].argsort()
    plt.figure()
    plt.scatter(n_train_arr[mask][sort],result_arr[mask][sort])
    plt.plot(n_train_arr[mask][sort],result_arr[mask][sort])
    plt.title('n_estimators == %s'%unique_n_est[i])
    plt.xlabel('n_train')
    plt.ylabel('success (%)')
    plt.ylim(75,100)
    plt.savefig('run_plots/'+run_name+'/n_train_vs_success_n_est_%s.png'%unique_n_est[i])
    plt.close()

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
plt.ylim(75,100)
plt.show()
plt.savefig('run_plots/'+run_name+'/n_train_vs_success_n_est_total.png')
plt.close()

for i in range(len(unique_n_train)):
    mask=n_train_arr==unique_n_train[i]
    sort= n_estimators_arr[mask].argsort()
    plt.figure()
    plt.scatter(n_estimators_arr[mask][sort],result_arr[mask][sort])
    plt.plot(n_estimators_arr[mask][sort],result_arr[mask][sort])
    plt.title('n_train == %s'%unique_n_train[i])
    plt.xlabel('n_estimators')
    plt.ylabel('success (%)')
    plt.ylim(75,100)
    plt.savefig('run_plots/'+run_name+'/n_estimators_vs_success_n_train_%s.png'%unique_n_train[i])
    plt.close()

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
plt.ylim(75,100)
plt.show()
plt.savefig('run_plots/'+run_name+'/n_estimators_vs_success_n_train_total.png')
plt.close()