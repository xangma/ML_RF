# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 12:58:37 2016

@author: moricex
"""

import os
import numpy
import sys
from settings import *
import shutil
from subprocess import Popen, PIPE

import time

#os.chdir(programpath) # Change directory
cwd=os.getcwd()
dirs=os.listdir(cwd)

# CREATE DIRECTORIES AND COPY CODE UP

n_estimators=[2,4,8,16,32,64,128,256,512,1024,2048,4096,8192]
n_runs=len(n_estimators)

for i in range(0,n_runs):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    run_namewo='RUN_'+timestr+'n_est_'+str(n_estimators[i]) 
    run_name = '/'+run_namewo+'/'
    
    fullpath=programpath+'runresults'+run_name
    if 'runresults' not in dirs:
        os.mkdir('runresults')
    
    dirs_runresults = os.listdir(programpath+'runresults')
    if run_namewo not in dirs_runresults:
        os.mkdir(fullpath)
    
    shutil.copy('MLdr12_RF.py',fullpath+'MLdr12_RF.py')
    shutil.copy('plots.py',fullpath+'plots.py')
    shutil.copy('settings.py',fullpath+'settings.py')
    shutil.copy('run_opts.py',fullpath+'run_opts.py')
    os.chdir(fullpath)
    cwd=os.getcwd()
    print(cwd)
    with open(fullpath+'settings.py',"a") as set_file: 
        set_file.write("\n# MODIFIED SETTINGS ADDED BY run_sciama.py for run_name:%s"%run_namewo)             # Add settings to bottom of settings.py file here
        set_file.write("\nprogrampath='"+fullpath+"'")
        set_file.write("\n")
        set_file.write("\nMLAset = {'n_estimators': %s, 'n_jobs': 4,'bootstrap':True,'verbose':True}"%n_estimators[i])
    
    dirs_run=os.listdir(fullpath)
    runfull=fullpath+'MLdr12_RF.py'
    
    # Open a pipe to the qsub command.
    p = Popen(["qsub"],stdin=PIPE, stdout=PIPE, close_fds=True,universal_newlines=True)
    output, input = p.stdout, p.stdin
     
    # Customize your options here
    job_name = "ML_RF_sciamajob_%s" %run_namewo
    walltime = "1:00:00"
    processors = "nodes=1:ppn=4"
    command = "python MLdr12_RF.py"
     
    job_string = """#!/bin/bash
    #PBS -N %s
    #PBS -l walltime=%s
    #PBS -l %s
    #PBS -j oe
    #PBS -V
    cd %s
    echo "Current working directory is `pwd`"
    %s""" % (job_name, walltime, processors,fullpath, command)
     
    # Send job_string to qsub
    out,errs=p.communicate(input=job_string)
#    input.close()
     
    # Print your job and the system response to the screen as it's submitted
    print(job_string)
    print(out)
     
    #    time.sleep(0.1)
    
    #exec(open(runfull).read())
