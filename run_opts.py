# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 15:53:16 2016

@author: moricex
"""
import settings
import numpy
import itertools as it
import logging
import os
run_opts_log=logging.getLogger('run_opts') # Set up overall logger for file
#from minepy import MINE
from scipy.stats import pearsonr
from sklearn.metrics import mutual_info_score
from collections import defaultdict
from spark_sklearn import GridSearchCV
from time import time
from sklearn.metrics import classification_report
import numpy.ma as ma
from operator import itemgetter

from pyspark import SparkContext,SparkConf

conf = SparkConf().setAppName("App")
conf = (conf.setMaster('local[*]')
        .set('spark.executor.memory', '4G')
        .set('spark.driver.memory', '45G')
        .set('spark.driver.maxResultSize', '10G'))
sc = SparkContext(conf=conf)

# This checks all the mags in the whole catalogue are positive.
# It cuts ones that aren't
def checkmagspos_old(XX,XXpredict,specz_tr,specz_pr,classnames_tr,classnames_pr,subclass_tr,subclass_names_tr,subclass_pr,subclass_names_pr,OBJID_tr,OBJID_pr,SPECOBJID_pr,RA_tr,DEC_tr,RA_pr,DEC_pr,filtstats\
,objc_type_tr,objc_type_tr_u,objc_type_tr_g,objc_type_tr_r,objc_type_tr_i,objc_type_tr_z,objc_type_pr,objc_type_pr_u,objc_type_pr_g,objc_type_pr_r,objc_type_pr_i,objc_type_pr_z,dered_tr_r,dered_pr_r):
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
            negmagsXX = XX[:,bottom:n] < 5
            negmagsXXpred = XXpredict[:,bottom:n] < 5
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
            SPECOBJID_pr =SPECOBJID_pr[XXpred_neg_index]
            RA_tr,DEC_tr = RA_tr[XX_neg_index],DEC_tr[XX_neg_index]
            RA_pr,DEC_pr = RA_pr[XXpred_neg_index],DEC_pr[XXpred_neg_index]
            specz_tr,specz_pr = specz_tr[XX_neg_index],specz_pr[XXpred_neg_index]
            objc_type_tr,objc_type_tr_u,objc_type_tr_g,objc_type_tr_r,objc_type_tr_i,objc_type_tr_z,objc_type_pr,objc_type_pr_u,objc_type_pr_g,objc_type_pr_r,objc_type_pr_i,objc_type_pr_z\
            = objc_type_tr[XX_neg_index],objc_type_tr_u[XX_neg_index],objc_type_tr_g[XX_neg_index],objc_type_tr_r[XX_neg_index],objc_type_tr_i[XX_neg_index],objc_type_tr_z[XX_neg_index]\
            ,objc_type_pr[XXpred_neg_index],objc_type_pr_u[XXpred_neg_index],objc_type_pr_g[XXpred_neg_index],objc_type_pr_r[XXpred_neg_index],objc_type_pr_i[XXpred_neg_index],objc_type_pr_z[XXpred_neg_index]
            dered_tr_r=dered_tr_r[XX_neg_index]
            dered_pr_r=dered_pr_r[XXpred_neg_index]
            
        return XX,XXpredict,specz_tr,specz_pr,classnames_tr,classnames_pr,subclass_tr,subclass_names_tr,subclass_pr,subclass_names_pr,OBJID_tr,OBJID_pr,SPECOBJID_pr,RA_tr,DEC_tr,RA_pr,DEC_pr\
        ,objc_type_tr,objc_type_tr_u,objc_type_tr_g,objc_type_tr_r,objc_type_tr_i,objc_type_tr_z,objc_type_pr,objc_type_pr_u,objc_type_pr_g,objc_type_pr_r,objc_type_pr_i,objc_type_pr_z,dered_tr_r,dered_pr_r
    else:
        return XX,XXpredict,specz_tr,specz_pr,classnames_tr,classnames_pr,subclass_tr,subclass_names_tr,subclass_pr,subclass_names_pr,OBJID_tr,OBJID_pr,SPECOBJID_pr,RA_tr,DEC_tr,RA_pr,DEC_pr\
        ,objc_type_tr,objc_type_tr_u,objc_type_tr_g,objc_type_tr_r,objc_type_tr_i,objc_type_tr_z,objc_type_pr,objc_type_pr_u,objc_type_pr_g,objc_type_pr_r,objc_type_pr_i,objc_type_pr_z,dered_tr_r,dered_pr_r

def checkmagspos(XX,XXpredict,specz_tr,specz_pr,classnames_tr,classnames_pr,subclass_tr,subclass_names_tr,subclass_pr,subclass_names_pr,OBJID_tr,OBJID_pr,SPECOBJID_pr,RA_tr,DEC_tr,RA_pr,DEC_pr,filtstats\
,objc_type_tr,objc_type_tr_u,objc_type_tr_g,objc_type_tr_r,objc_type_tr_i,objc_type_tr_z,objc_type_pr,objc_type_pr_u,objc_type_pr_g,objc_type_pr_r,objc_type_pr_i,objc_type_pr_z,dered_tr_r,dered_pr_r):
    if settings.checkmagspos == 1: # If set to check for neg mags
        run_opts_log.info('')
        checkmagspos_log=logging.getLogger('checkmagspos')
        checkmagspos_log.info('Checking mags aren''t below 0 ...')
        checkmagspos_log.info('------------')
#        checkmagspos_log.info(len(XX))
#        n=bottom+filtstats[i][0]
        checkmagspos_log.info('Checking mags in XX')
        negmagsXX=numpy.where((XX <-50) | (XX>60))
        negmagsXX2=numpy.where(XX[:,0:len(settings.filters)] <5)
        negmagsXX=numpy.concatenate((negmagsXX[0],negmagsXX2[0]))
        negmagsXXind = [x in numpy.unique(negmagsXX) for x in range(len(XX))]
        XX_neg_index= ~numpy.array(negmagsXXind)

        negmagsXXpred=numpy.where((XXpredict <-50) | (XXpredict>60))
        negmagsXXpred2=numpy.where(XXpredict[:,0:len(settings.filters)] <5)
        negmagsXXpred=numpy.concatenate((negmagsXXpred[0],negmagsXXpred2[0]))
        negmagsXXpredind = [x in numpy.unique(negmagsXXpred) for x in range(len(XXpredict))]
        XXpred_neg_index= ~numpy.array(negmagsXXpredind)        
        
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
        SPECOBJID_pr =SPECOBJID_pr[XXpred_neg_index]
        RA_tr,DEC_tr = RA_tr[XX_neg_index],DEC_tr[XX_neg_index]
        RA_pr,DEC_pr = RA_pr[XXpred_neg_index],DEC_pr[XXpred_neg_index]
        specz_tr,specz_pr = specz_tr[XX_neg_index],specz_pr[XXpred_neg_index]
        objc_type_tr,objc_type_tr_u,objc_type_tr_g,objc_type_tr_r,objc_type_tr_i,objc_type_tr_z,objc_type_pr,objc_type_pr_u,objc_type_pr_g,objc_type_pr_r,objc_type_pr_i,objc_type_pr_z\
        = objc_type_tr[XX_neg_index],objc_type_tr_u[XX_neg_index],objc_type_tr_g[XX_neg_index],objc_type_tr_r[XX_neg_index],objc_type_tr_i[XX_neg_index],objc_type_tr_z[XX_neg_index]\
        ,objc_type_pr[XXpred_neg_index],objc_type_pr_u[XXpred_neg_index],objc_type_pr_g[XXpred_neg_index],objc_type_pr_r[XXpred_neg_index],objc_type_pr_i[XXpred_neg_index],objc_type_pr_z[XXpred_neg_index]
        dered_tr_r=dered_tr_r[XX_neg_index]
        dered_pr_r=dered_pr_r[XXpred_neg_index]
            
        return XX,XXpredict,specz_tr,specz_pr,classnames_tr,classnames_pr,subclass_tr,subclass_names_tr,subclass_pr,subclass_names_pr,OBJID_tr,OBJID_pr,SPECOBJID_pr,RA_tr,DEC_tr,RA_pr,DEC_pr\
        ,objc_type_tr,objc_type_tr_u,objc_type_tr_g,objc_type_tr_r,objc_type_tr_i,objc_type_tr_z,objc_type_pr,objc_type_pr_u,objc_type_pr_g,objc_type_pr_r,objc_type_pr_i,objc_type_pr_z,dered_tr_r,dered_pr_r
    else:
        return XX,XXpredict,specz_tr,specz_pr,classnames_tr,classnames_pr,subclass_tr,subclass_names_tr,subclass_pr,subclass_names_pr,OBJID_tr,OBJID_pr,SPECOBJID_pr,RA_tr,DEC_tr,RA_pr,DEC_pr\
        ,objc_type_tr,objc_type_tr_u,objc_type_tr_g,objc_type_tr_r,objc_type_tr_i,objc_type_tr_z,objc_type_pr,objc_type_pr_u,objc_type_pr_g,objc_type_pr_r,objc_type_pr_i,objc_type_pr_z,dered_tr_r,dered_pr_r

def weightinput(XX,classnames_tr,OBJID_tr,RA_tr,DEC_tr,specz_tr): # Weights num of objects in training set by class, settings defined in settings.weightimput
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
        specz_tr = specz_tr[numpy.int64(finalsel_sort)] # THIS NEEDS TO BE TESTED/CHECKED


        return XX,classnames_tr,OBJID_tr,RA_tr,DEC_tr,specz_tr
    else:
        return XX,classnames_tr,OBJID_tr,RA_tr,DEC_tr,specz_tr

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
def calculate_colours(filt_train,filt_predict,n_filt,filt_names,j):
    if settings.calculate_colours == 1:
        run_opts_log.info('')
        calculate_colours_log=logging.getLogger('calculating_colours')
        calculate_colours_log.info('Calculating colours ...')
#        n_filt = filt_train.shape[1]
#        print(n_filt)
        col_names=[]
        combs = list(it.combinations(range(n_filt),2))
#        print(combs)
        colours_all_train=[]
        for i in range(len(combs)):
            colours=filt_train[:,combs[i][0]]-filt_train[:,combs[i][1]]
            colours_all_train.append(colours)
            if settings.calculate_cross_colours==0:
                col_names.append('%s - %s' %(settings.filters[j][combs[i][0]],settings.filters[j][combs[i][1]]))
            else:
                col_names.append('%s - %s' %(settings.filters[combs[i][0]],settings.filters[combs[i][1]]))
        colours_all_predict=[]
        for i in range(len(combs)):
            colours=filt_predict[:,combs[i][0]]-filt_predict[:,combs[i][1]]
            colours_all_predict.append(colours)
        
        filt_train=numpy.column_stack((filt_train,numpy.transpose(colours_all_train)))
        filt_predict=numpy.column_stack((filt_predict,numpy.transpose(colours_all_predict)))
        del colours, colours_all_train, colours_all_predict
        return filt_train,filt_predict,combs,filt_names,col_names
    else:
        combs=0
        return filt_train,filt_predict,combs,filt_names,col_names

def use_filt_colours(filt_train,filt_predict,j,n_filt,col_names_j): # This reads in which colours to use. Defined in settings.
    if settings.calculate_colours == 1:
        run_opts_log.info('')
        use_filt_colours_log=logging.getLogger('use_filt_colours')
        use_filt_colours_log.info('Selecting colours to use from settings.usecolours')
        x=[]
        cut_col_names=[]
        for i in range(len(settings.use_colours[j])):
            filt=filt_train[:,(settings.use_colours[j][i]+n_filt)]
            cut_col_names.append(col_names_j[settings.use_colours[j][i]])
            x.append(filt)
        filt_train=numpy.column_stack((filt_train[:,0:n_filt],numpy.transpose(x)))
        x=[]
        for i in range(len(settings.use_colours[j])):
            filt=filt_predict[:,(settings.use_colours[j][i]+n_filt)]
            x.append(filt)
        filt_predict=numpy.column_stack((filt_predict[:,0:n_filt],numpy.transpose(x)))
        n_colour=(len(x))
        del x, filt
        return filt_train,filt_predict,n_colour,cut_col_names
    else:
        n_colour=0
        return filt_train,filt_predict,n_colour,cut_col_names

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
            if settings.make_binary==1:
                uniquetarget_tr[0][1]='PointS'
                uniquetarget_tr=[uniquetarget_tr[0][0:2]]
                unique_IDS_tr = unique_IDS_tr[0:2]
                uniquetarget_pr[0][1]='PointS'
                uniquetarget_pr=[uniquetarget_pr[0][0:2]]
                unique_IDS_pr = unique_IDS_pr[0:2]
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

def compute_mic(XX):
    mic_all=[]
    combs=[]
    if (settings.compute_mic ==1)|(settings.compute_contribution_mic==1):
        compute_mic_log=logging.getLogger('compute_mic')
        compute_mic_log.info('Computing MIC ...')
        m = MINE()
        combs = list(it.combinations(range(len(XX[0])),2))
        for i in range(len(combs)):
            m.compute_score(XX[:,combs[i][0]],XX[:,combs[i][1]])
            mic_all.append(m.mic())
    return combs, mic_all

def compute_pearson(XX):
    pearson_all=[]
    combs=[]
    if settings.compute_pearson ==1:
        compute_pearson_log=logging.getLogger('compute_pearson')
        compute_pearson_log.info('Computing pearson coeffs ...')
        combs = list(it.combinations(range(len(XX[0])),2))
        for i in range(len(combs)):
            x=pearsonr(XX[:,combs[i][0]],XX[:,combs[i][1]])[0]
            pearson_all.append(x)
    return combs, pearson_all

def calc_MI(x, y, bins=None):
    if bins is None:
        bins = int(numpy.sqrt(len(x)/5))# 25 # bins= sqrt(n/5) so have 5 points for every cell of hist
    c_xy = numpy.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi

class MutualInformation:
    """Determine MI using scikit-learn"""
    def __init__(self, X, Y):
        self.X, self.Y = X, Y
    def mutual_information_2d(self, bins=None):
        """normalised mutual information score alpha
        http://scikit-learn.org/stable/modules/generated/sklearn.metrics.normalized_mutual_info_score.html
        """
        return calc_MI(self.X, self.Y, bins=bins) / numpy.sqrt(calc_MI(self.X, self.X, bins=bins) * calc_MI(self.Y, self.Y, bins=bins))

def recursive_defaultdict():
    return defaultdict(recursive_defaultdict)

def setpath(d, p, k):
    if len(p) == 1:
        d[p[0]] = k
    else:
        setpath(d[p[0]], p[1:], k)

def calc_MINT(XX,XXpredict,yy): # ALSO CALCULATES MI AND SAVES THEM. MI_XX can use train+predict, MI_YY can only use train.
    MI_XX = {'correlation_results':{}}
    MI_XY = {'correlation_results':{}}
    trainpred=[]
    combsXX,combsXY = [],[]
    res={}
    S = settings.MINT_n_feat
    trainnum=settings.traindatanum
    dirlist=os.listdir('/users/moricex/ML_RF/MINT_res')
    MINT_res_name='MINT_res_ntrain_%s_nMINT_%s.npy' %(trainnum,S)
    if settings.calc_MINT == 1:
        if MINT_res_name not in dirlist:
            trainpred = numpy.row_stack((XX,XXpredict))
            combsXX =  combsXX= list(it.product(list(range(len(XX[0][:-1]))),list(range(len((XX[0][:-1]))))))
            combsXY = list(it.product(list(range(len(XX[0][:-1]))),numpy.int64(list(numpy.unique(yy)))))
            MI_XX['columns'] = list(range(len(XX[0][:-1])))
            MI_XX['columns2'] = list(range(len(XX[0][:-1])))
            MI_XY['columns'] = numpy.int64(list(numpy.unique(yy)))
            MI_XY['columns2'] = list(range(len(XX[0][:-1])))
            
            # SET UP DICT
            for i in range(len(combsXX)):
                MI_XX['correlation_results'][combsXX[i][0]] = {}
                for j in range(len(combsXX)):
                    MI_XX['correlation_results'][combsXX[i][0]][combsXX[j][1]] = {}
            # CALC
            for i in range(len(combsXX)):
                XX_mi = MutualInformation(trainpred.T[combsXX[i][0]],trainpred.T[combsXX[i][1]]).mutual_information_2d()
                MI_XX['correlation_results'][combsXX[i][0]][combsXX[i][1]]['MI'] = XX_mi
            
            # SET UP DICT
            for i in range(len(combsXY)):
                MI_XY['correlation_results'] = {}
                for j in range(len(combsXY)):
                    MI_XY['correlation_results'][combsXY[j][0]] = {}
            # CALC
            for i in range(len(combsXY)):
                XY_mi = MutualInformation(XX.T[combsXY[i][0]],yy).mutual_information_2d()
                MI_XY['correlation_results'][combsXY[i][0]]['MI'] = XY_mi
            
            #MINT START
            xfeats_ = numpy.array(MI_XY['columns2']) # This would be [0:49]
            xfeats_ = [i for i in xfeats_ if (i in MI_XX['columns']) and (i in MI_XX['columns2'])] # I'm thinking ['columns'] and ['columns2'] would be [0:49] for me
            
            res = {}
    #        for yfeat in MI_XY['columns']: # And this would be [0:2]. So for each class ...
                
            print('before', len(xfeats_))
            xfeats = numpy.array([i for i in xfeats_ if (i in MI_XX['columns']) and (i in MI_XX['columns2'])]) # only find ones you want to compare
            
            print('after', len(xfeats_))
            MIXY = numpy.array([MI_XY['correlation_results'][j]['MI'] for j in xfeats]) # Select MIXY for particular class 
            indF = numpy.isfinite(MIXY) == True # Check if finite
            MIXY = MIXY[indF] # Apply finite cut
            xfeats = xfeats[indF] # Apply to xfeats array too
            inds_ = numpy.argsort(MIXY)[::-1] # Return indicies that would sort array then flip reverse it
            sort_MIXY = MIXY[inds_] # Sort the MIXY array to most important feature first? Though it's never used ...
            sort_xfeats = xfeats[inds_] # Sort the xfeats array
            global_Phi = -100
            
            for feature1 in xfeats: # Starting with the top feature (one that is most strongly correlated with class in question) ...
                feature_x = [feature1] # Index of feature/s in question
                    
                for S_ in numpy.arange(S - 1) + 2: # for selected features 2 to S (in the def case, that's 10) | Now S_ is 3, and feature_x has 2 features in it
                    
                    feats =  [f1 for f1 in xfeats if f1 not in feature_x] # select all features but one/s in question (feature_x) |
                    Phi_best = -100
                    for feat2 in feats: # For every element in the array without the feature/s in question |
                        all_feats = feature_x + [feat2] # all_feats array contains the index of the feature/s in question and the index of the rest
                        Phi = 1.0 / S_ * numpy.sum([MI_XY['correlation_results'][j]['MI'] for j in all_feats]) - 1.0 / (S_ * S_) * (numpy.sum([MI_XX['correlation_results'][j][k]['MI'] for j in all_feats for k in all_feats]))
             # Calculate Phi = 1 / num_feats_in_question * sum(MIXY[class][...])...
                        if Phi > Phi_best:
                            Phi_best = Phi
                            best_new_feat = feat2 # If it finds a good feature ...|
        
                    feature_x.append(best_new_feat) # Add it to the list of features in question | Now go to next iteration of S_ on line 81 || Once S_ gets to 10 ...
                if Phi_best > global_Phi: # | If this beats feature 0, this set of features is best, switch to this one.
                    global_Phi = Phi_best
                    global_feats = feature_x
                    print('global_Phi phi', global_Phi)
                    print(global_feats)
        
            res = {'best_phi': global_Phi, 'best_feats': global_feats}
            numpy.save('/users/moricex/ML_RF/MINT_res/'+MINT_res_name,res)
        else:
            loaded=numpy.load('/users/moricex/ML_RF/MINT_res/'+MINT_res_name)
            res=loaded.tolist()
    return res

def gridsearch(XX,XXpredict,yy,yypredict,clf):
#    tuned_parameters=settings.param_grid
    param_grid=settings.param_grid  
    print("Gridsearch start")
    def report(grid_scores, n_top=3):
    	top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    	for i, score in enumerate(top_scores):
            print("Model with rank: {0}".format(i + 1))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(score.mean_validation_score, numpy.std(score.cv_validation_scores)))
            print("Parameters: {0}".format(score.parameters))
            print("")
    grid_search = GridSearchCV(sc, clf, param_grid=param_grid,cv=10,n_jobs=-1,verbose=1)
    start = time()
    grid_search.fit(XX, yy)
    print("GridSearchCV took {:.2f} seconds for {:d} candidate settings.".format(time() - start, len(grid_search.grid_scores_)))
    report(grid_search.grid_scores_)
    return grid_search
#    X_train, X_test, y_train, y_test = XX,XXpredict,yy,yypredict
#    
#    # Set the parameters by cross-validation
#
#    scores = ['precision', 'recall']
#    
#    for score in scores:
#        print("# Tuning hyper-parameters for %s" % score)
#        print()
#    
#        clf = GridSearchCV(clf, tuned_parameters, cv=5,
#                           scoring='%s_macro' % score)
#        clf.fit(X_train, y_train)
#    
#        print("Best parameters set found on development set:")
#        print()
#        print(clf.best_params_)
#        print()
#        print("Grid scores on development set:")
#        print()
#        means = clf.cv_results_['mean_test_score']
#        stds = clf.cv_results_['std_test_score']
#        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#            print("%0.3f (+/-%0.03f) for %r"
#                  % (mean, std * 2, params))
#        print()
#    
#        print("Detailed classification report:")
#        print()
#        print("The model is trained on the full development set.")
#        print("The scores are computed on the full evaluation set.")
#        print()
#        y_true, y_pred = y_test, clf.predict(X_test)
#        print(classification_report(y_true, y_pred))
#        print()
