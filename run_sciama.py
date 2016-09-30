# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 12:58:37 2016

@author: moricex
"""

import os
import numpy
from settings import *
import shutil
from subprocess import Popen, PIPE
import time

#os.chdir(programpath) # Change directory
cwd=os.getcwd()
dirs=os.listdir(cwd)

# Variables to iterate through

n_estimators=numpy.array([2,4,8,16,32,64,128,256,512,1024,2048,4096,8192])
#n_estimators=numpy.array([16])
n_train=numpy.array([50,100,500,1000,2500,5000,10000,25000])
n_depth=numpy.array([2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,'None'])

# CREATE DIRECTORIES AND COPY CODE UP

timestr = time.strftime("%Y%m%d-%H%M%S")
os.chdir(programpath+'runresults')
numpy.savez("stats_%s" %('RUN_'+timestr),n_estimators=n_estimators,n_depth=n_depth,n_train=n_train,filters=filters,use_colours=use_colours)

runpaths=[]
runnames=[]
for k in range(0,len(n_estimators)):
    for j in range(0,len(n_train)):
        for i in range(0,len(n_depth)):
            run_namewo='RUN_'+timestr+'MINT'+'_n_depth_'+str(n_depth[i])+'_n_tr_'+str(n_train[j])+'_n_est_'+str(n_estimators[k])#+'max_feat_None' 
            run_name = '/'+run_namewo+'/'
            
            fullpath=programpath+'runresults'+run_name
            if 'runresults' not in dirs:
                os.mkdir('runresults')
            
            dirs_runresults = os.listdir(programpath+'runresults')
            if run_namewo not in dirs_runresults:
                os.mkdir(fullpath)
            runpaths.append(fullpath)
            runnames.append(run_namewo)
            shutil.copy(programpath+'MLdr12_RF.py',fullpath+'MLdr12_RF.py')
            shutil.copy(programpath+'plots.py',fullpath+'plots.py')
            shutil.copy(programpath+'settings.py',fullpath+'settings.py')
            shutil.copy(programpath+'run_opts.py',fullpath+'run_opts.py')
            shutil.copy(programpath+'htmloutput.py',fullpath+'htmloutput.py')
            os.chdir(fullpath)
            cwd=os.getcwd()
            print(cwd)
            with open(fullpath+'settings.py',"a") as set_file: 
                set_file.write("\n# MODIFIED SETTINGS ADDED BY run_sciama.py for run_name:%s"%run_namewo)             # Add settings to bottom of settings.py file here
                set_file.write("\nprogrampath='"+fullpath+"'")
                set_file.write("\n")
                set_file.write("\nMLAset = {'n_estimators': %s, 'n_jobs': 4,'bootstrap':True,'verbose':True,'max_depth':%s}"%(n_estimators[k],n_depth[i]))
                set_file.write("\ntraindatanum=%s"%n_train[j])
                set_file.write("\nfeat_outfile = 'ML_RF_feat_%s'" %run_namewo)
                set_file.write("\nresult_outfile = 'ML_RF_results_%s'" %run_namewo)
                set_file.write("\nprob_outfile = 'ML_RF_probs_%s'" %run_namewo)
                set_file.write("\nlog_outfile='ML_RF_log_%s'" %run_namewo)
                set_file.write("\nstats_outfile='ML_RF_stats_%s'" %run_namewo)
                set_file.write("\nfeatnames_outfile='ML_RF_featnames_%s'" %run_namewo)
                set_file.write("\nscores_outfile='ML_RF_scores_%s'" %run_namewo)
            dirs_run=os.listdir(fullpath)
            runfull=fullpath+'MLdr12_RF.py'
            
            # Open a pipe to the qsub command.
            p = Popen(["qsub"],stdin=PIPE, stdout=PIPE, close_fds=True,universal_newlines=True)
    #        output, input = p.stdout, p.stdin
             
            # Customize your options here
            job_name = "ML_RF_sciamajob_%s" %run_namewo
            walltime = "1:00:00"
            if n_depth[i]=='None':
                processors = "nodes=1:ppn=6"
            else:
                processors = "nodes=1:ppn=4"
            command = "python MLdr12_RF.py"
             
            job_string = """#!/bin/bash
            #PBS -N %s_numcat*10
            #PBS -l walltime=%s
            #PBS -l %s
            #PBS -j oe
            #PBS -V
            #PBS -q sciama1.q
            cd %s
            echo "Current working directory is `pwd`"
            %s""" % (job_name, walltime, processors,fullpath, command)
             
            # Send job_string to qsub
            out,errs=p.communicate(input=job_string)
        #    input.close()
             
            # Print your job and the system response to the screen as it's submitted
            print(job_string)
            print(out)
             
            time.sleep(1)

os.chdir(programpath+'runresults')
numpy.save("paths_%s" %('RUN_'+timestr),numpy.column_stack((runpaths,runnames)))