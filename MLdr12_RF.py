# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 18:00:54 2016

@author: moricex
"""
# Dependencies
import os
import run_opts
import settings
import numpy
import astropy.io.fits as fits
import time
import plots

import logging

# Change directory
os.chdir(settings.programpath)
cwd=os.getcwd()
dirs=os.listdir(cwd)

logging.basicConfig(level=logging.DEBUG,\
                    format='%(asctime)s %(name)-20s %(levelname)-6s %(message)s',\
                    datefmt='%d-%m-%y %H:%M',\
                    filename='ML_RF.log',\
                    filemode='w')
# define a Handler which writes INFO messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logger=logging.getLogger('')
logger.addHandler(console)

if 'plots' not in dirs:
    os.mkdir('plots')
# Number to train and predict
traindatanum=settings.traindatanum
predictdatanum=settings.predictdatanum

# Clear the last fit if there was one
if 'clf' in locals():
    del clf

logger.info('Program start')
logger.info('------------')
logger.info('Loading data')
logger.info('------------')

# Check if data is loaded, else load it (Might remove this)
if 'traindata' in locals():
    logger.info('Data already loaded, skipping')
    logger.info('------------')
else:
    traindata=fits.open(settings.trainpath)
    traindata=traindata[1].data
    preddata=fits.open(settings.predpath)
    preddata=preddata[1].data

# Extra options before running

# Find and exclude unclassified objects (subclass)
traindata, preddata = run_opts.find_only_classified(traindata,preddata)

filt_train_all= {}
filt_predict_all = {}
combs={}
for j in range(len(settings.filters)):
    n_filt=len(settings.filters[j])
    filt_train=[]
    for i in range(numpy.size(settings.filters[j])): # Create filter array (training set)
        filt_train.append(traindata[settings.filters[j][i]])
    filt_train=numpy.transpose(filt_train)
    filt_predict=[]
    for i in range(numpy.size(settings.filters[j])): # Create filter array (prediction set)
        filt_predict.append(preddata[settings.filters[j][i]])
    filt_predict=numpy.transpose(filt_predict)
    
    # Section that calculates all possible colours
    filt_train,filt_predict,combs[j] = run_opts.calculate_colours(filt_train,filt_predict,n_filt) 
    # Section that checks use_colours and cuts colours accordingly
    filt_train,filt_predict,n_colour = run_opts.use_filt_colours(filt_train,filt_predict,j,n_filt)
    filt_train_all[j]=filt_train,n_filt,n_colour
    filt_predict_all[j]=filt_predict,n_filt,n_colour

n_filt=0
n_colours=0
filtstats={}
for i in range(len(filt_train_all)):
    filtstats[i]=filt_train_all[i][1],filt_train_all[i][2] # Make filtstats var with n_filt and n_colours to be passed to runopts.checkmagspos
    n_filt=n_filt+filt_train_all[i][1]# Number of filters
    n_colours=n_colours+filt_train_all[i][2] # Number of colours
n_oth=len(settings.othertrain) # Number of other features
n_feat=n_filt+n_colours+n_oth # Number of total features
logger.info('Number of filters: %s, Number of colours: %s, Number of other features: %s' %(n_filt,n_colours,n_oth))
logger.info('Number of total features = %s + 1 target' %(n_feat))
    
# Stack arrays.
XX=numpy.array(filt_train_all[0][0])
if len(filt_train_all) > 1:
    for i in range(1,len(filt_train_all)):
        XX=numpy.column_stack((XX,numpy.array(filt_train_all[i][0])))
for i in range(len(settings.othertrain)): # Tack on other training features (not mags, like redshift)
    XX = numpy.column_stack((XX,traindata[settings.othertrain[i]]))
classnames_tr=traindata[settings.predict[:-3]]
XX=numpy.column_stack((XX,traindata[settings.predict]))

XXpredict=numpy.array(filt_predict_all[0][0])
if len(filt_predict_all) > 1:
    for i in range(1,len(filt_predict_all)):
        XXpredict=numpy.column_stack((XXpredict,filt_predict_all[i][0]))
for i in range(len(settings.othertrain)): # Tack on other prediction features
    XXpredict = numpy.column_stack((XXpredict,preddata[settings.othertrain[i]]))
classnames_pr=preddata[settings.predict[:-3]]
XXpredict=numpy.column_stack((XXpredict,preddata[settings.predict]))

# Filter out negative magnitudes
# THIS MUST BE DONE LAST IN THIS PROCESSING PART.
XX,XXpredict,classnames_tr,classnames_pr = run_opts.checkmagspos(XX,XXpredict,classnames_tr,classnames_pr,filtstats)

XX,classnames_tr = run_opts.weightinput(XX,classnames_tr)

XX = XX[0:traindatanum]
XXpredict=XXpredict[0:predictdatanum]
classnames_tr=classnames_tr[0:traindatanum]
classnames_pr=classnames_pr[0:predictdatanum]

del traindata,preddata,filt_train,filt_predict,filt_train_all,filt_predict_all # Clean up

unique_IDS_tr, unique_IDS_pr,uniquetarget_tr,uniquetarget_pr = \
run_opts.diagnostics([XX[:,-1],XXpredict[:,-1],classnames_tr,classnames_pr],'inputdata') # Total breakdown of types going in

yy = XX[:,-1] # Training answers
yypredict = XXpredict[:,-1] # Prediction answers

if settings.actually_run == 1: # Run MLA switch

    clf = settings.MLA # Pulls in machine learning algorithm from settings
    logger.info('MLA settings') 
    logger.info(clf)
    logger.info('------------')    
    start, end=[],[] # Timer
    logger.info('Fit start')
    logger.info('------------')
    start = time.time()
    clf = clf.fit(XX[:,0:n_feat],yy) # XX is train array
    end = time.time()
    logger.info('Fit ended in %s seconds' %(end-start))
    logger.info('------------')
    
    start, end=[],[]
    logger.info('Predict start')
    logger.info('------------')
    start = time.time()
    result = clf.predict(XXpredict[:,0:n_feat]) # XX is predict array.
    probs = clf.predict_proba(XXpredict[:,0:n_feat]) # Only take from 0:n_feat because answers are tacked on end
    feat_importance = clf.feature_importances_
    end = time.time()
    logger.info('Predict ended in %s seconds' %(end-start))
    logger.info('------------')
    
    logger.info('Totalling results')
    n = sum(result == yypredict)
    logger.info('%s / %s were correct' %(n,predictdatanum))
    logger.info('That''s %s percent' %((n/predictdatanum)*100))
    logger.info('------------')
    
    resultsstack = numpy.column_stack((XXpredict,result,probs)) # Compile results into table
    
    run_opts.diagnostics([result,yypredict,unique_IDS_tr, unique_IDS_pr,uniquetarget_tr,uniquetarget_pr],'result')
    # SAVE
    if settings.saveresults == 1:
        logger.info('Saving results')
        logger.info('------------')
        numpy.savetxt(settings.outfile,resultsstack)
        numpy.savetxt(settings.feat_outfile,feat_importance)
    
    # PLOTS
    logger.info('Plotting ...')
    plots.plot_subclasshist(XX,XXpredict,classnames_tr,classnames_pr) # Plot a histogram of the subclasses in the data
    plots.plot_bandvprob(resultsstack,filtstats,probs.shape[1]) # Plot band vs probability.
    plots.plot_colourvprob(resultsstack,filtstats,probs.shape[1],combs) # Plot colour vs probability

logger.removeHandler(console)