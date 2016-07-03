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

import time
root_path=os.getcwd()
#os.chdir(programpath) # Change directory
os.chdir('runresults')
runresults_path=os.getcwd()
runresults_dirs=os.listdir(runresults_path)

run_name='RUN_20160703-19'

dirs_run = [s for s in runresults_dirs if run_name in s]
# RUN_PLOTS
run_results=[]
for i in range(len(dirs_run)):
    vals={}
    os.chdir(runresults_path+'/'+dirs_run[i])
    importlib.reload(settings)
    run_path=os.getcwd()
    sub_dir=os.listdir(run_path)
    from settings import MLAset,traindatanum,predictdatanum,result_outfile
    result_file=numpy.genfromtxt(result_outfile)
    result = ((sum(result_file[:,0]==result_file[:,1]))/predictdatanum)*100
    vals['n_estimators'] = MLAset['n_estimators']
    vals['traindatanum'] = traindatanum
    vals['result'] = result
    run_results.append(vals)