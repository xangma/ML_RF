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
root_path=os.getcwd()
#os.chdir(programpath) # Change directory
os.chdir('runresults')
runresults_path=os.getcwd()
runresults_dirs=os.listdir(runresults_path)

run_name='RUN_20160705-23'
stats_name='stats_'+run_name
dirs_run = [s for s in runresults_dirs if run_name in s]
stats_file= [s for s in runresults_dirs if stats_name in s]
dirs_run.remove(stats_file[0])
run_stats=numpy.load(stats_file[0])
# RUN_PLOTS
n_estimators_arr,traindatanum_arr,result_arr=numpy.array([]),numpy.array([]),numpy.array([])
for i in range(len(dirs_run)):
    start,end=[],[]
    start=time.time()
    os.chdir(runresults_path+'/'+dirs_run[i])
    importlib.reload(settings)
    run_path=os.getcwd()
    sub_dir=os.listdir(run_path)
    stat_file=[s for s in sub_dir if 'ML_RF_stats' in s]
    arr=numpy.genfromtxt(stat_file[0])
    n_estimators_arr=numpy.append(n_estimators_arr,arr[0])
    traindatanum_arr=numpy.append(traindatanum_arr,arr[1])
    result_arr=numpy.append(result_arr,arr[3])
