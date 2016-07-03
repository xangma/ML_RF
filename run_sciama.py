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

run_namewo='test'
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
with open(fullpath+'settings.py',"a") as att_file:
    att_file.write("\nprogrampath='"+fullpath+"'\n")

dirs_run=os.listdir(fullpath)
runfull=fullpath+'MLdr12_RF.py'

for i in range(1, 2):
    # Open a pipe to the qsub command.
    p = Popen(["qsub"],stdin=PIPE, stdout=PIPE, close_fds=True,universal_newlines=True)
    output, input = p.stdout, p.stdin
     
    # Customize your options here
    job_name = "my_job_%d" % i
    walltime = "1:00:00"
    processors = "nodes=1:ppn=16"
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
    input.close()
     
    # Print your job and the system response to the screen as it's submitted
    print(job_string)
    print(out)
     
#    time.sleep(0.1)

#exec(open(runfull).read())
    