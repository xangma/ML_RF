# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 15:53:16 2016

@author: moricex
"""
import settings
import numpy
import itertools as it
import logging
run_opts_log=logging.getLogger('run_opts') # Set up overall logger for file

# This checks all the mags in the whole catalogue are positive.
# It cuts ones that aren't
def checkmagspos(XX,XXpredict,classnames_tr,classnames_pr,subclass_tr,subclass_names_tr,subclass_pr,subclass_names_pr,OBJID_tr,OBJID_pr,RA_tr,DEC_tr,RA_pr,DEC_pr,filtstats):
    if settings.checkmagspos == 1: # If set to check for neg mags
        run_opts_log.info('')
        checkmagspos_log=logging.getLogger('checkmagspos')
        checkmagspos_log.info('Checking mags aren''t below 0 ...')
        checkmagspos_log.info('------------')
#        checkmagspos_log.info(len(XX))
        bottom=0
        for i in range(len(filtstats)): # For every filter
            n=bottom+filtstats[i][0]
            checkmagspos_log.info('Checking mags in XX: %s:%s' %(bottom, n))
            negmagsXX = XX[:,bottom:n] < 0
            negmagsXXpred = XXpredict[:,bottom:n] < 0
            bottom=n+filtstats[i][1]
            negmagXXsum = numpy.sum(negmagsXX,axis=1)
            negmagXXpredsum = numpy.sum(negmagsXXpred,axis=1)
            
            XX_neg_index = negmagXXsum == 0
            XXpred_neg_index = negmagXXpredsum == 0
            
            XX = XX[XX_neg_index]
            XXpredict = XXpredict[XXpred_neg_index]
            
            classnames_tr=classnames_tr[XX_neg_index]
            classnames_pr=classnames_pr[XXpred_neg_index]
            subclass_tr = subclass_tr[XX_neg_index]
            subclass_names_tr=subclass_names_tr[XX_neg_index]
            subclass_pr=subclass_pr[XXpred_neg_index]
            subclass_names_pr=subclass_names_pr[XXpred_neg_index]
            OBJID_tr = OBJID_tr[XX_neg_index]
            OBJID_pr = OBJID_pr[XXpred_neg_index]
            RA_tr,DEC_tr = RA_tr[XX_neg_index],DEC_tr[XX_neg_index]
            RA_pr,DEC_pr = RA_pr[XXpred_neg_index],DEC_pr[XXpred_neg_index]

        return XX,XXpredict,classnames_tr,classnames_pr,subclass_tr,subclass_names_tr,subclass_pr,subclass_names_pr,OBJID_tr,OBJID_pr,RA_tr,DEC_tr,RA_pr,DEC_pr
    else:
        return XX,XXpredict,classnames_tr,classnames_pr,subclass_tr,subclass_names_tr,subclass_pr,subclass_names_pr,OBJID_tr,OBJID_pr,RA_tr,DEC_tr,RA_pr,DEC_pr

def weightinput(XX,classnames_tr,OBJID_tr,RA_tr,DEC_tr): # Weights num of objects in training set by class, settings defined in settings.weightimput
    if len(settings.weightinput) > 0:
        run_opts_log.info('')
        weightinput_log=logging.getLogger('weightinput')
        weightinput_log.info('Weighting input ...')
        totalnum=0
        finalsel=[]
        for i in range(len(settings.weightinput)):
            numobj=numpy.floor((settings.weightinput[i]/100)*settings.traindatanum)
            grouped = numpy.where(XX[:,-1] == i)
            weightinput_log.info(numobj)
            x = grouped[0][0:numobj]
            finalsel=numpy.append(finalsel,x)
            totalnum = totalnum + numobj
        objdiff=settings.traindatanum-totalnum
        if objdiff > 0:
            weightinput_log.info('Missing %s objects' %objdiff) # Only enters this loop if total num of obj =/= total desired num
            for i in range(numpy.int64(objdiff)):
                finalsel=numpy.append(finalsel,grouped[0][numobj+i])
                weightinput_log.info(XX[grouped[0][numobj+i]])
            weightinput_log.info('Added %s objects from last class' %objdiff) # Then just tacks on the difference from the last class
        weightinput_log.info('Total array length: %s' %(len(finalsel)))
        finalsel_sort=numpy.sort(numpy.int64(finalsel))
        XX=XX[numpy.int64(finalsel)]
        classnames_tr=classnames_tr[numpy.int64(finalsel_sort)]
        OBJID_tr = OBJID_tr[numpy.int64(finalsel_sort)] # THIS NEEDS TO BE TESTED/CHECKED
        RA_tr = RA_tr[numpy.int64(finalsel_sort)] # THIS NEEDS TO BE TESTED/CHECKED
        DEC_tr = DEC_tr[numpy.int64(finalsel_sort)] # THIS NEEDS TO BE TESTED/CHECKED

        return XX,classnames_tr,OBJID_tr,RA_tr,DEC_tr
    else:
        return XX,classnames_tr,OBJID_tr,RA_tr,DEC_tr

# Find and exclude unclassified objects (subclass)
def find_only_classified(traindata,preddata):
    if settings.find_only_classified == 1:
        run_opts_log.info('')
        find_only_classified_log=logging.getLogger('find_only_classified')
        find_only_classified_log.info('Finding and excluding objects without subclass (training data)')
        find_only_classified_log.info('------------')
        subclassid_tr=numpy.array(list(traindata['SPEC_SUBCLASS_ID']))        
        noclass_tr = subclassid_tr== 0
        #find_only_classified_log.info('Was working with %s objects' %len(opendata[1].data))
        traindata=traindata[~noclass_tr]
        #find_only_classified_log.info('Now working with %s objects' %len(opendata[1].data))
        del noclass_tr,subclassid_tr
        
        find_only_classified_log.info('Finding and excluding objects without subclass (predict data)')
        find_only_classified_log.info('------------')
        subclassid_pr=numpy.array(list(preddata['SPEC_SUBCLASS_ID']))        
        noclass_pr = subclassid_pr == 0
        #find_only_classified_log.info('Was working with %s objects' %len(opendata[1].data))
        preddata=preddata[~noclass_pr]
        #find_only_classified_log.info('Now working with %s objects' %len(opendata[1].data))
        del noclass_pr,subclassid_pr
        return traindata,preddata
    else:
        return traindata,preddata

# Calculate all possible colours and append them to train and predict sets
def calculate_colours(filt_train,filt_predict,n_filt):
    if settings.calculate_colours == 1:
        run_opts_log.info('')
        calculate_colours_log=logging.getLogger('calculating_colours')
        calculate_colours_log.info('Calculating colours ...')
#        n_filt = filt_train.shape[1]
#        print(n_filt)
        combs = list(it.combinations(range(n_filt),2))
#        print(combs)
        colours_all_train=[]
        for i in range(len(combs)):
            colours=filt_train[:,combs[i][0]]-filt_train[:,combs[i][1]]
            colours_all_train.append(colours)
        
        colours_all_predict=[]
        for i in range(len(combs)):
            colours=filt_predict[:,combs[i][0]]-filt_predict[:,combs[i][1]]
            colours_all_predict.append(colours)
        filt_train=numpy.column_stack((filt_train,numpy.transpose(colours_all_train)))
        filt_predict=numpy.column_stack((filt_predict,numpy.transpose(colours_all_predict)))
        del colours, colours_all_train, colours_all_predict
        return filt_train,filt_predict,combs
    else:
        combs=0
        return filt_train,filt_predict,combs

def use_filt_colours(filt_train,filt_predict,j,n_filt): # This reads in which colours to use. Defined in settings.
    if settings.calculate_colours == 1:
        run_opts_log.info('')
        use_filt_colours_log=logging.getLogger('use_filt_colours')
        use_filt_colours_log.info('Selecting colours to use from settings.usecolours')
        x=[]
        for i in range(len(settings.use_colours[j])):
            filt=filt_train[:,(settings.use_colours[j][i]+n_filt)]
            x.append(filt)
        filt_train=numpy.column_stack((filt_train[:,0:n_filt],numpy.transpose(x)))
        x=[]
        for i in range(len(settings.use_colours[j])):
            filt=filt_predict[:,(settings.use_colours[j][i]+n_filt)]
            x.append(filt)
        filt_predict=numpy.column_stack((filt_predict[:,0:n_filt],numpy.transpose(x)))
        n_colour=(len(x))
        del x, filt
        return filt_train,filt_predict,n_colour
    else:
        n_colour=0
        return filt_train,filt_predict,n_colour

def diagnostics(x, state): # This is quite a 'free' function that is meant to output diagnostic information.
    if settings.diagnostics == 1:
        run_opts_log.info('')
        diagnostics_log=logging.getLogger('diagnostics')
        if state == 'inputdata': # This switch tells the function which part the main code is in. Provides a breakdown of classes going to the MLA
            diagnostics_log.info('Running input data diagnostics...')
            trainnum=settings.traindatanum
            prednum=settings.predictdatanum
            uniquetarget_tr=numpy.unique(x[2],return_index=True,return_counts=True)
            unique_IDS_tr=x[0][uniquetarget_tr[1]]
            uniquetarget_pr=numpy.unique(x[3],return_index=True,return_counts=True)
            unique_IDS_pr=x[1][uniquetarget_pr[1]]
            diagnostics_log.info('There are %s missing target types in training set' %(len(unique_IDS_pr)-len(unique_IDS_tr)))
            diagnostics_log.info('------------') 
            diagnostics_log.info('TRAINING SET')
            diagnostics_log.info('Class Number        Class Name, No. Objects, Percent of Total')
            for i in range(len(unique_IDS_tr)):
                diagnostics_log.info('%4s %25s %10s %5s' %(unique_IDS_tr[i],uniquetarget_tr[0][i],uniquetarget_tr[2][i]\
                ,round(((uniquetarget_tr[2][i]/trainnum)*100),3)))
            diagnostics_log.info('------------') 
            diagnostics_log.info('PREDICTION SET')
            diagnostics_log.info('Class Number        Class Name, No. Objects, Percent of Total')
            for i in range(len(unique_IDS_pr)):
                diagnostics_log.info('%4s %25s %10s %5s' %(unique_IDS_pr[i],uniquetarget_pr[0][i],uniquetarget_pr[2][i]\
                ,round(((uniquetarget_pr[2][i]/prednum)*100),3)))
            diagnostics_log.info('------------') 

            return unique_IDS_tr, unique_IDS_pr,uniquetarget_tr,uniquetarget_pr

        if state == 'result': # At the end, provide the same type of class breakdown, but this time for results
            result, yypredict, unique_IDS_tr, unique_IDS_pr,uniquetarget_tr,uniquetarget_pr = x[0],x[1],x[2],x[3],x[4],x[5]
            diagnostics_log.info('Class Number        Class Name, No. correct, No. true, Percent correct')
            for i in range(len(unique_IDS_pr)):
                findalltrue = yypredict == unique_IDS_pr[i]
                grouptrue = yypredict[findalltrue]
                grouppred = result[findalltrue]
                numright_group = sum(grouptrue==grouppred)
                diagnostics_log.info('%4s %21s %12s %15s %5s' %(unique_IDS_pr[i],uniquetarget_pr[0][i],numright_group, len(grouptrue),round(((numright_group/len(grouptrue))*100),3)))
    else:
        return