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
import shutil
import requests
import treeinterpreter as ti
from sklearn import metrics
import markup
from markup import oneliner as e
from sklearn import tree


temp_train='./temp_train.csv' # Define temp files for pyspark
temp_pred='./temp_pred.csv'

os.chdir(settings.programpath) # Change directory
cwd=os.getcwd()
dirs=os.listdir(cwd)

logging.basicConfig(level=logging.INFO,\
                    format='%(asctime)s %(name)-20s %(levelname)-6s %(message)s',\
                    datefmt='%d-%m-%y %H:%M',\
                    filename=settings.log_outfile,\
                    filemode='w')
# define a Handler which writes INFO messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter) # tell the handler to use this format
logger=logging.getLogger('') # add the handler to the root logger
logger.addHandler(console)

if 'plots' not in dirs: # Create plots directory  if it doesn't exist
    os.mkdir('plots')

def get_function(function_string):
    import importlib
    module, function = function_string.rsplit('.', 1)
    module = importlib.import_module(module)
    function = getattr(module, function)
    return function

traindatanum=settings.traindatanum # Number to train and predict
predictdatanum=settings.predictdatanum

if 'clf' in locals(): # Clear the last fit if there was one
    del clf

logger.info('Program start')
logger.info('------------')
logger.info('CWD is %s' %cwd)
logger.info('Loading data for preprocessing')
logger.info('------------')

if 'traindata' in locals(): # Check if data is loaded, else load it (Might remove this)
    logger.info('Data already loaded, skipping')
    logger.info('------------')
else:
    traindata=fits.open(settings.trainpath)
    traindata=traindata[1].data
    preddata=fits.open(settings.predpath)
    preddata=preddata[1].data

# Extra options before running
traindata, preddata = run_opts.find_only_classified(traindata,preddata) # Find and exclude unclassified objects (subclass)

filt_train_all= {} # Set up arrays
filt_predict_all = {}
filt_names={}
col_names={}
combs={}
feat_names=[]

for j in range(len(settings.filters)): # For each filter set
    n_filt=len(settings.filters[j]) # Get the number of filters
    filt_train=[] # Set up filter array
    for i in range(numpy.size(settings.filters[j])): # For each filter i in filter set j 
        filt_train.append(traindata[settings.filters[j][i]]) # Append filter to filter array
    filt_names[j]= settings.filters[j]
    filt_train=numpy.transpose(filt_train)
#    filt_names=numpy.transpose(filt_names)
    filt_predict=[]
    for i in range(numpy.size(settings.filters[j])): # Do same for prediction set
        filt_predict.append(preddata[settings.filters[j][i]])
    filt_predict=numpy.transpose(filt_predict)
    
    filt_train,filt_predict,combs[j],filt_names,col_names_j = run_opts.calculate_colours(filt_train,filt_predict,n_filt,filt_names,j) # Section that calculates all possible colours
    
    filt_train,filt_predict,n_colour,col_names_j = run_opts.use_filt_colours(filt_train,filt_predict,j,n_filt,col_names_j) # Section that checks use_colours and cuts colours accordingly
    col_names[j]=col_names_j
    
    filt_train_all[j]=filt_train,n_filt,n_colour # Create list of filter sets, with the num of filts and colours
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

for j in range(len(settings.filters)):
    feat_names = feat_names+filt_names[j]+col_names[j]
feat_names = feat_names+settings.othertrain
# Stack arrays to feed to MLA
XX=numpy.array(filt_train_all[0][0])
if len(filt_train_all) >= 1:
    for i in range(1,len(filt_train_all)):
        XX=numpy.column_stack((XX,numpy.array(filt_train_all[i][0])))
for i in range(len(settings.othertrain)): # Tack on other training features (not mags, like redshift)
    XX = numpy.column_stack((XX,traindata[settings.othertrain[i]]))

# Other variables to carry through cuts
classnames_tr=traindata[settings.predict[:-3]] # Get classnames
subclass_tr=traindata['SPEC_SUBCLASS_ID']
subclass_names_tr=traindata['SPEC_SUBCLASS']
OBJID_tr = traindata['OBJID']
RA_tr,DEC_tr = traindata['RA'],traindata['DEC']
specz_tr = traindata['SPECZ']

XX=numpy.column_stack((XX,traindata[settings.predict])) # Stack training data for MLA, tack on true answers

# Do the same for predict data
XXpredict=numpy.array(filt_predict_all[0][0])
if len(filt_predict_all) > 1:
    for i in range(1,len(filt_predict_all)):
        XXpredict=numpy.column_stack((XXpredict,filt_predict_all[i][0]))
for i in range(len(settings.othertrain)): # Tack on other prediction features (not mags, like redshift)
    XXpredict = numpy.column_stack((XXpredict,preddata[settings.othertrain[i]]))
classnames_pr=preddata[settings.predict[:-3]]
subclass_pr = preddata['SPEC_SUBCLASS_ID']
subclass_names_pr = preddata['SPEC_SUBCLASS']
OBJID_pr = preddata['OBJID']
RA_pr,DEC_pr = preddata['RA'],preddata['DEC']
specz_pr = preddata['SPECZ']

XXpredict=numpy.column_stack((XXpredict,preddata[settings.predict])) # Stack training data for MLA, tack on true answers so can evaluate after

# Filter out negative magnitudes
# THIS MUST BE DONE LAST IN THIS PROCESSING PART.
XX,XXpredict,specz_tr,specz_pr,classnames_tr,classnames_pr,subclass_tr,subclass_names_tr,subclass_pr,subclass_names_pr,OBJID_tr,OBJID_pr,RA_tr,DEC_tr,RA_pr,DEC_pr\
 = run_opts.checkmagspos(XX,XXpredict,specz_tr,specz_pr,classnames_tr,classnames_pr,subclass_tr,subclass_names_tr,subclass_pr,subclass_names_pr,OBJID_tr,OBJID_pr,RA_tr,DEC_tr,RA_pr,DEC_pr,filtstats)

XX,classnames_tr,OBJID_tr,RA_tr,DEC_tr,specz_tr = run_opts.weightinput(XX,classnames_tr,OBJID_tr,RA_tr,DEC_tr,specz_tr) # Weight training set? - specified in settings

XX = XX[0:traindatanum] # Cut whole training array down to size specified in settings
XXpredict=XXpredict[0:predictdatanum]
classnames_tr=classnames_tr[0:traindatanum] # Do same for classnames
classnames_pr=classnames_pr[0:predictdatanum]
OBJID_tr = OBJID_tr[0:traindatanum]
OBJID_pr = OBJID_pr[0:predictdatanum]
RA_tr,DEC_tr = RA_tr[0:traindatanum],DEC_tr[0:traindatanum]
RA_pr,DEC_pr = RA_pr[0:predictdatanum],DEC_pr[0:predictdatanum]
specz_tr,specz_pr = specz_tr[0:traindatanum],specz_pr[0:predictdatanum]

# Cuts for doublesubrun
subclass_tr = subclass_tr[0:traindatanum]
subclass_names_tr = subclass_names_tr[0:traindatanum]
subclass_pr = subclass_pr[0:predictdatanum]
subclass_names_pr = subclass_names_pr[0:predictdatanum]


del traindata,preddata,filt_train,filt_predict,filt_train_all,filt_predict_all # Clean up

unique_IDS_tr, unique_IDS_pr,uniquetarget_tr,uniquetarget_pr = \
run_opts.diagnostics([XX[:,-1],XXpredict[:,-1],classnames_tr,classnames_pr],'inputdata') # Total breakdown of types going in

yy = XX[:,-1] # Training answers
yypredict = XXpredict[:,-1] # Prediction answers

if settings.one_vs_all == 1: # target is unique_IDs_tr[i] in loop
    XX_one_vs_all,XXpredict_one_vs_all,yy_one_vs_all,yypredict_one_vs_all = {},{},{},{}
    for i in range(len(unique_IDS_tr)):
        yy_orig = yy
        yypredict_orig = yypredict
        yy_out = [numpy.float32(99) if x!=unique_IDS_tr[i] else x for x in yy_orig]
        yypredict_out = [numpy.float32(99) if x!=unique_IDS_tr[i] else x for x in yypredict_orig]
#        classnames_tr_out = ['Other' if x!=numpy.unique(classnames_tr)[i] else x for x in classnames_tr]
#        classnames_pr_out = ['Other' if x!=numpy.unique(classnames_pr)[i] else x for x in classnames_pr]
#        yy_orig=yy_orig[yy_orig != unique_IDS_tr[i]] = 99
#        yypredict_orig[yypredict_orig != unique_IDS_tr[i]] = 99
        XX_stack = numpy.column_stack((XX[:,:-1],yy_out))
        XXpredict_stack = numpy.column_stack((XXpredict[:,:-1],yypredict_out))
        XX_one_vs_all[i] = XX_stack
        XXpredict_one_vs_all[i] = XXpredict_stack
        yy_one_vs_all[i] = yy_out
        yypredict_one_vs_all[i] = yypredict_out
#        classnames_tr_one_vs_all[i]=classnames_tr_out
#        classnames_pr_one_vs_all[i]=classnames_pr_out

def run_MLA(XX,XXpredict,yy,yypredict,unique_IDS_tr,unique_IDS_pr,uniquetarget_tr,uniquetarget_pr,n_feat,ind_run_name,n_run):
    logger.info('Starting MLA run')
    logger.info('------------')
    if settings.pyspark_on == 1:                # Use pyspark or not? Pyspark makes cross node (HPC) calculation possible.
        from pyspark import SparkContext        # It's slower, manages resources between nodes using HTTP. 
        from pyspark.sql import SQLContext      # So far, it does not include feature importance outputs.
        from pyspark.ml import Pipeline         # I would have to program feature importances myself. May be time consuming.
        from pyspark.ml.feature import VectorAssembler
        from pyspark.ml.classification import RandomForestClassifier
        from pyspark.ml.feature import StringIndexer, VectorIndexer
        from pyspark.ml.evaluation import MulticlassClassificationEvaluator
        # pyspark go
        
        if settings.pyspark_remake_csv == 1: # Making the csv files for the pyspark MLA to read in is time consuming, turn off the file generation?
            logger.info('Remaking csvs for pysparks...')
            numpy.savetxt(temp_train, XX, delimiter=",")
            logger.info('Training csv saved')
            numpy.savetxt(temp_pred, XXpredict, delimiter=",")
            logger.info('Predict csv saved')
        sc = SparkContext(appName="ML_RF") # Initiate spark
        
        sclogger=sc._jvm.org.apache.log4j # Initiate spark logging
        sclogger.LogManager.getLogger("org").setLevel(sclogger.Level.ERROR)
        sclogger.LogManager.getLogger("akka").setLevel(sclogger.Level.ERROR)
        sqlContext=SQLContext(sc)
        # Read in data
        data_tr = sqlContext.read.format("com.databricks.spark.csv").options(header='false',inferSchema='true').load(temp_train)
        data_pr = sqlContext.read.format("com.databricks.spark.csv").options(header='false',inferSchema='true').load(temp_pred)
        data_tr=data_tr.withColumnRenamed(data_tr.columns[-1],"label") # rename last column (answers), to label
        data_pr=data_pr.withColumnRenamed(data_pr.columns[-1],"label")
        
        assembler=VectorAssembler(inputCols=data_tr.columns[:-1],outputCol="features")
        reduced=assembler.transform(data_tr.select('*')) # Assemble feature vectos for spark MLA
        
        assembler_pr=VectorAssembler(inputCols=data_pr.columns[:-1],outputCol="features")
        reduced_pr=assembler_pr.transform(data_pr.select('*'))
        
        labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(reduced) # Index vectors        
        featureIndexer =VectorIndexer(inputCol="features", outputCol="indexedFeatures").fit(reduced)
        # Initiate MLA alg
        rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures",numTrees=100,maxDepth=5,maxBins=200)
        
        pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf]) # Set up fitting pipeline
        start, end=[],[] # Timer
        logger.info('Fit start')
        logger.info('------------')
        start = time.time()
        model=pipeline.fit(reduced) # Fit
        end = time.time()
        logger.info('Fit ended in %s seconds' %(end-start))
        logger.info('------------')
        start, end=[],[]
        logger.info('Predict start')
        logger.info('------------')
        start = time.time()
        predictions = model.transform(reduced_pr) # Predict
        evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel",predictionCol="prediction",metricName="precision")
        accuracy = evaluator.evaluate(predictions)
        logger.info("Test Error = %g" %(1.0-accuracy))
        logger.info('------------')
        logger.info('Pulling results ...')
        yypredict=numpy.array(predictions.select("indexedLabel").collect()) # Pulls all results into numpy arrays to continue program
        yypredict=yypredict[:,0]
        result=numpy.array(predictions.select("prediction").collect())
        result=result[:,0]
        XXpredict=numpy.array(predictions.select("indexedFeatures").collect())
        XXpredict=XXpredict[:,0]
        probs=numpy.array(predictions.select("probability").collect())
        probs=probs[:,0]
        XXpredict=numpy.column_stack((XXpredict,yypredict))
        end=time.time()
        logger.info('Predict ended in %s seconds' %(end-start))
        logger.info('------------')
    
    else:
        # Run sklearn MLA switch
        MLA = get_function(settings.MLA) # Pulls in machine learning algorithm from settings
        clf = MLA().set_params(**settings.MLAset)
        logger.info('MLA settings') 
        logger.info(clf)
        logger.info('------------')    
        start, end=[],[] # Timer
        logger.info('Fit start')
        logger.info('------------')
        start = time.time()
        clf = clf.fit(XX[:,0:n_feat],yy) # XX is train array, yy is training answers
        end = time.time()
        logger.info('Fit ended in %s seconds' %(end-start))
        logger.info('------------')
        score = clf.score
        if 'OvsA' not in ind_run_name:
            if settings.output_all_trees == 1:
                i_tree = 0
                for tree_in_forest in clf.estimators_:
                    with open('plots/tree_' + str(i_tree) + '.dot', 'w') as my_file:
                        my_file = tree.export_graphviz(tree_in_forest, out_file = my_file,feature_names=feat_names,class_names=uniquetarget_tr[0], filled=True)
                    os.system('dot -Tpng plots/tree_%s.dot -o plots/tree_%s.png' %(i_tree,i_tree))
                    os.remove('plots/tree_%s.dot' %i_tree)
                    i_tree = i_tree + 1        
            else:
                with open('plots/tree_example.dot', 'w') as my_file:
                    my_file = tree.export_graphviz(clf.estimators_[0], out_file = my_file,feature_names=feat_names,class_names=uniquetarget_tr[0], filled=True)
                os.system('dot -Tpng plots/tree_example.dot -o plots/tree_example.png')
                os.remove('plots/tree_example.dot')
        start, end=[],[]
        # Split cats for RAM management
        numcats = numpy.int64((2*XXpredict.size*clf.n_jobs/1024/1024)/(clf.n_jobs*8))*10
        if numcats < 1:
            numcats = 1
        logger.info('Predict start')
        logger.info('------------')
        start = time.time()
        result,probs,bias,contributions=[],[],[],[]
        XXpredict_cats=numpy.array_split(XXpredict,numcats)
        logger.info('Splitting predict array into %s' %numcats)
        logger.info('------------')
        for i in range(len(XXpredict_cats)):
            logger.info('Predicting cat %s/%s' %(i,len(XXpredict_cats)))
            result.extend(clf.predict(XXpredict_cats[i][:,0:n_feat])) # XX is predict array.
            probs.extend(clf.predict_proba(XXpredict_cats[i][:,0:n_feat])) # Only take from 0:n_feat because answers are tacked on end
            if settings.get_contributions == 1:            
                tiresult = ti.predict(clf,XXpredict_cats[i][:,0:n_feat])
                contributions.extend(tiresult[2])
                bias = tiresult[1][0]
        feat_importance = clf.feature_importances_
        result=numpy.float32(result)
        probs=numpy.float32(probs)

        accuracy = metrics.accuracy_score(result,yypredict)
        recall = metrics.recall_score(result,yypredict,average=None)
        precision = metrics.precision_score(result,yypredict,average=None)
        score = metrics.f1_score(result, yypredict,average=None)
        
        end = time.time()
        logger.info('Predict ended in %s seconds' %(end-start))
        logger.info('------------')

    logger.info('Recall Score: %s' %recall)
    logger.info('Precision Score: %s' %precision)
    logger.info('Accuracy Score: %s' %accuracy)
    logger.info('F1 Score: %s' %score)
    percentage=(n/predictdatanum)*100
    
    run_opts.diagnostics([result,yypredict,unique_IDS_tr, unique_IDS_pr,uniquetarget_tr,uniquetarget_pr],'result')
#    stats=numpy.array([])
#    stats=numpy.column_stack((clf.n_estimators,traindatanum,predictdatanum,percentage))
    # SAVE
    if settings.saveresults == 1:
        logger.info('Saving results')
        logger.info('------------')

        numpy.savetxt(settings.result_outfile+('_%s' %ind_run_name)+'.txt',numpy.column_stack((yypredict,result)),header="True_target Predicted_target")
        numpy.savetxt(settings.prob_outfile+('_%s' %ind_run_name)+'.txt',probs)
        numpy.savetxt(settings.feat_outfile+('_%s' %ind_run_name)+'.txt',feat_importance)
        numpy.savetxt(settings.stats_outfile+('_%s' %ind_run_name)+'.txt',numpy.column_stack((clf.n_estimators,traindatanum,predictdatanum,percentage,clf.max_depth)),header="n_est traindatanum predictdatanum percentage max_depth",fmt="%s")
    
    return result,feat_importance,probs,bias,contributions,accuracy,recall,precision,score,clf

for n in range(0,settings.n_runs):
    logging.info('%s/%s runs' %(n,settings.n_runs))
    if settings.one_vs_all == 1:
        one_vs_all_results = {}
        tree_was_on = 0
        if settings.output_all_trees == 1:
            tree_was_on = 1
            settings.output_all_trees = 0
        for i in range(len(unique_IDS_tr)):
            ind_run_name = 'OvsA_%s_%s' %(uniquetarget_tr[0][i],n)
            unique_IDs_tr_loop=[unique_IDS_tr[i],numpy.float32(99)]
            unique_IDs_pr_loop=[unique_IDS_pr[i],numpy.float32(99)]
            uniquetarget_tr_loop=[[uniquetarget_tr[0][i],'Other']]
            uniquetarget_pr_loop=[[uniquetarget_pr[0][i],'Other']]
            result,feat_importance,probs,bias,contributions,accuracy,recall,precision,score,clf = run_MLA(XX_one_vs_all[i],XXpredict_one_vs_all[i],numpy.array(yy_one_vs_all[i]),numpy.array(yypredict_one_vs_all[i]),unique_IDs_tr_loop,unique_IDs_pr_loop,uniquetarget_tr_loop,uniquetarget_pr_loop,n_feat,ind_run_name,n)
            one_vs_all_results[i] = {'class_ID' : unique_IDS_tr[i],'result' : result,'feat_importance' : feat_importance,'uniquetarget_tr_loop' : uniquetarget_tr_loop}
        plots_feat_per_class_outname = plots.plot_feat_per_class(one_vs_all_results,feat_names,n)
    #    if len(settings.othertrain) > 0:    
    #        plots.plot_feat_per_class_oth(one_vs_all_results,n_filt,n_colours)
        if tree_was_on == 1:
            settings.output_tree = 1
    
    if settings.actually_run == 1:# If it is set to actually run in settings
        ind_run_name = 'standard_%s' %n
        result,feat_importance,probs,bias,contributions,accuracy,recall,precision,score,clf = run_MLA(XX,XXpredict,yy,yypredict,unique_IDS_tr,unique_IDS_pr,uniquetarget_tr,uniquetarget_pr,n_feat,ind_run_name,n)

        # PLOTS
        logger.info('Plotting ...')
        plots.plot_subclasshist(XX,XXpredict,classnames_tr,classnames_pr) # Plot a histogram of the subclasses in the data
        plots_bandvprob_outnames = plots.plot_bandvprob(XXpredict,probs,filtstats,numpy.shape(probs)[1]) # Plot band vs probability.
        plots_colourvprob_outnames = plots.plot_colourvprob(XXpredict,probs,filtstats,numpy.shape(probs)[1],combs) # Plot colour vs probability
        plots_feat_outname = plots.plot_feat(feat_importance,feat_names,n)

    if settings.double_sub_run == 1:
        XX = numpy.column_stack((XX,subclass_tr))
        XXpredict = numpy.column_stack((XXpredict[:,:-1],result))
        n_feat=n_feat+1
        yy=subclass_tr
        yypredict=subclass_pr
        logger.info('Starting *SECOND* MLA run')
        ind_run_name = 'DSR_%s' %n
        unique_IDS_tr, unique_IDS_pr,uniquetarget_tr,uniquetarget_pr = \
        run_opts.diagnostics([XX[:,-1],yypredict,subclass_names_tr,subclass_names_pr],'inputdata') # Total breakdown of types going in
        settings.MLA = settings.MLA(n_estimators=100,n_jobs=16,bootstrap=True,verbose=True) 
        result2,feat_importance2,probs2,bias2,contributions2,accuracy2,recall2,precision2,score2,clf2 = run_MLA(XX,XXpredict,yy,yypredict,unique_IDS_tr,unique_IDS_pr,uniquetarget_tr,uniquetarget_pr,n_feat,ind_run_name,n)

if settings.get_images == 1:
    image_IDs = {}
    logging.getLogger("requests").setLevel(logging.WARNING)
    for i in range(len(unique_IDS_pr)):
        # create masks
        yymask = yypredict == i
        index_loop = numpy.where(yymask)
        OBJID_pr_loop = OBJID_pr[yymask]
        result_loop = result[yymask]
        yypredict_loop = yypredict[yymask]
        probs_loop = probs[yymask]
        RA_pr_loop = RA_pr[yymask]
        DEC_pr_loop = DEC_pr[yymask]
        specz_pr_loop = specz_pr[yymask]
        
        good_mask = (result_loop == yypredict_loop) & (probs_loop[:,i] > .9)
        ok_mask = (probs_loop[:,i] > .45) & (probs_loop[:,i] < 0.55)
        bad_mask = probs_loop[:,i] < 0.1
        
        image_IDs[i] = {'class' : unique_IDS_pr[i], 'good_ID' : OBJID_pr_loop[good_mask], 'good_RA' : RA_pr_loop[good_mask]\
        , 'good_DEC' : DEC_pr_loop[good_mask], 'good_specz' : specz_pr_loop[good_mask], 'good_result' : result_loop[good_mask],'good_probs' : probs_loop[good_mask],'good_index' : index_loop[0][good_mask],'good_true_class' : yypredict_loop[good_mask], 'ok_ID' : OBJID_pr_loop[ok_mask], 'ok_RA' : RA_pr_loop[ok_mask]\
        , 'ok_DEC' : DEC_pr_loop[ok_mask], 'ok_specz' : specz_pr_loop[ok_mask],'ok_result' : result_loop[ok_mask],'ok_probs' : probs_loop[ok_mask],'ok_index' : index_loop[0][ok_mask],'ok_true_class' : yypredict_loop[ok_mask], 'bad_ID' : OBJID_pr_loop[bad_mask], 'bad_RA' : RA_pr_loop[bad_mask], 'bad_DEC' : DEC_pr_loop[bad_mask], 'bad_specz' : specz_pr_loop[bad_mask], 'bad_result' : result_loop[bad_mask], 'bad_probs' : probs_loop[bad_mask],'bad_index' : index_loop[0][bad_mask],'bad_true_class' : yypredict_loop[bad_mask]}

    num_max_images = 10
    for i in range(len(unique_IDS_pr)):
        url_list,url_objid_list,tiresult_list=[],[],[]
        if len(image_IDs[i]['good_ID']) > num_max_images:
            top_good = num_max_images
        else:
            top_good = len(image_IDs[i]['good_ID'])
        for j in range(0,top_good):   
            img_ID_good = image_IDs[i]['good_ID'][j]
            img_RA_good = image_IDs[i]['good_RA'][j]
            img_DEC_good = image_IDs[i]['good_DEC'][j]
            img_index_good = image_IDs[i]['good_index'][j]
            tiresult = ti.predict(clf,XXpredict[:,0:n_feat][img_index_good].reshape(1,-1))
            tiresult_list.append(tiresult[2])
            url_objid_list.append('http://skyserver.sdss.org/dr12/en/tools/explore/Summary.aspx?id=%s' %img_ID_good)
            url_list.append('http://skyserver.sdss.org/SkyserverWS/dr12/ImgCutout/getjpeg?TaskName=Skyserver.Explore.Image&ra=%s&dec=%s&scale=0.2&width=200&height=200&opt=G' %(img_RA_good,img_DEC_good))
        image_IDs[i].update({'good_url':url_list, 'good_contributions' : tiresult_list, 'good_url_objid' : url_objid_list})
        url_list,url_objid_list,tiresult_list=[],[],[]
        if len(image_IDs[i]['ok_ID']) > num_max_images:
            top_ok = num_max_images
        else:
            top_ok = len(image_IDs[i]['ok_ID'])
        for j in range(0,top_ok):  
            img_ID_ok = image_IDs[i]['ok_ID'][j]
            img_RA_ok =  image_IDs[i]['ok_RA'][j]
            img_DEC_ok = image_IDs[i]['ok_DEC'][j]
            img_index_ok = image_IDs[i]['ok_index'][j]
            tiresult = ti.predict(clf,XXpredict[:,0:n_feat][img_index_ok].reshape(1,-1))
            tiresult_list.append(tiresult[2])
            url_objid_list.append('http://skyserver.sdss.org/dr12/en/tools/explore/Summary.aspx?id=%s' %img_ID_ok)
            url_list.append('http://skyserver.sdss.org/SkyserverWS/dr12/ImgCutout/getjpeg?TaskName=Skyserver.Explore.Image&ra=%s&dec=%s&scale=0.2&width=200&height=200&opt=G' %(img_RA_ok,img_DEC_ok))
        image_IDs[i].update({'ok_url':url_list,'ok_contributions' : tiresult_list, 'ok_url_objid' : url_objid_list})
        url_list,url_objid_list,tiresult_list=[],[],[]
        if len(image_IDs[i]['bad_ID']) > num_max_images:
            top_bad = num_max_images
        else:
            top_bad = len(image_IDs[i]['bad_ID'])
        for j in range(0,top_bad):
            img_ID_bad = image_IDs[i]['bad_ID'][j]
            img_RA_bad =  image_IDs[i]['bad_RA'][j]
            img_DEC_bad = image_IDs[i]['bad_DEC'][j]
            img_index_bad = image_IDs[i]['bad_index'][j]
            tiresult = ti.predict(clf,XXpredict[:,0:n_feat][img_index_bad].reshape(1,-1))
            tiresult_list.append(tiresult[2])
            url_objid_list.append('http://skyserver.sdss.org/dr12/en/tools/explore/Summary.aspx?id=%s' %img_ID_bad)
            url_list.append('http://skyserver.sdss.org/SkyserverWS/dr12/ImgCutout/getjpeg?TaskName=Skyserver.Explore.Image&ra=%s&dec=%s&scale=0.2&width=200&height=200&opt=G' %(img_RA_bad,img_DEC_bad))
        image_IDs[i].update({'bad_url':url_list,'bad_contributions' : tiresult_list, 'bad_url_objid' : url_objid_list})

# Make results html
os.chdir(settings.programpath)
html_title='Results for run: %s' %ind_run_name
page = markup.page()
page.init(title=html_title)
page.p(page.h3("Results for run: %s" %ind_run_name))
page.a( "Home",href="index.html")
page.a( "Example Tree",href="trees.html")
page.a( "Plots",href="plots.html")
page.a( "Images",href="images.html")
page.p( "Accuracy: %s" %accuracy)
page.p("")

page.table(border=1)
page.tr(),page.th(""),page.th(uniquetarget_tr[0][0]),page.th(uniquetarget_tr[0][1]),page.th(uniquetarget_tr[0][2]),page.tr.close()
page.tr(),page.td(),page.b("Recall"),page.td.close(),page.td(round(recall[0],5)),page.td(round(recall[1],5)),page.td(round(recall[2],5)),page.tr.close()
page.tr(),page.td(),page.b("Precision"),page.td.close(),page.td(round(precision[0],5)),page.td(round(precision[1],5)),page.td(round(precision[2],5)),page.tr.close()
page.tr(),page.td(),page.b("F1 Score"),page.td.close(),page.td(round(score[0],5)),page.td(round(score[1],5)),page.td(round(score[2],5)),page.tr.close()
page.table.close()

# Write out settings
html_settings=("Number of training objects: %s" %settings.traindatanum,"Number of prediction objects: %s" %settings.predictdatanum\
,"","Features","    Filters: %s" %settings.filters, "    Colours: %s" %col_names, "    Other: %s" %settings.othertrain)
page.p(html_settings)
# Links to plots and images

# Save html
html_file= open("index.html","w")
html_file.write(page())
html_file.close()

# Create tree page
page_tree = markup.page()
page_tree.init(title=html_title+" Example Tree")
page_tree.p(page_tree.h3("Results for run: %s Example Tree" %ind_run_name))
page_tree.a( "Home",href="index.html")
page_tree.a( "Example Tree",href="trees.html")
page_tree.a( "Plots",href="plots.html")
page_tree.a( "Images",href="images.html")
page_tree.p("Example Tree")
page_tree.img(src="plots/tree_example.png")

html_file= open("trees.html","w")
html_file.write(page_tree())
html_file.close()

# Create pages for plots
page_plots = markup.page()
page_plots.init(title=html_title+" Plots")
page_plots.p(page_plots.h3("Results for run: %s Plots" %ind_run_name))
page_plots.a( "Home",href="index.html")
page_plots.a( "Example Tree",href="trees.html")
page_plots.a( "Plots",href="plots.html")
page_plots.a( "Images",href="images.html")
page_plots.p("Overall Feature Importance")
page_plots.img(src=plots_feat_outname)
page_plots.p("")
page_plots.p("Feature importance per class")
page_plots.img(src=plots_feat_per_class_outname)
allfiltplots= [s for s in plots_bandvprob_outnames if 'allfilt' in s]
for i in range(len(allfiltplots)):
    page_plots.p(["",allfiltplots[i]])
    page_plots.img(src=allfiltplots[i])

html_file= open("plots.html","w")
html_file.write(page_plots())
html_file.close()
# Create pages for images
page_images = markup.page()
page_images.init(title=html_title+" Images")
page_images.p(page_images.h3("Results for run: %s Images" %ind_run_name))
page_images.a( "Home",href="index.html")
page_images.a( "Example Tree",href="trees.html")
page_images.a( "Plots",href="plots.html")
page_images.a( "Images",href="images.html")
page_images.p("")

page_images.table(border=1)
page_images.tr(),page_images.td(),page_images.a( e.img( src=image_IDs[0]['good_url'][0]), href=image_IDs[0]['good_url_objid'][0]),page_images.td.close(),page_images.tr.close()
page_images.tr(),page_images.td(),page_images.b('Class'),page_images.td.close(),page_images.td(str(uniquetarget_tr[0][image_IDs[0]['class']])),page_images.tr.close()
page_images.tr(),page_images.td(),page_images.b('Predicted Class'),page_images.td.close(),page_images.td(str(uniquetarget_tr[0][image_IDs[0]['good_result'][0]])),page_images.tr.close()
page_images.tr(),page_images.td(),page_images.b('ObjID'),page_images.td.close(),page_images.td(str(image_IDs[0]['good_ID'][0])),page_images.tr.close()
page_images.tr(),page_images.td(),page_images.b('Redshift'),page_images.td.close(),page_images.td(str(image_IDs[0]['good_specz'][0])),page_images.tr.close()
page_images.table.close()

page_images.table(border=1)
page_images.tr(),page_images.th(""),page_images.th(uniquetarget_tr[0][0]),page_images.th(uniquetarget_tr[0][1]),page_images.th(uniquetarget_tr[0][2]),page_images.tr.close()
page_images.tr(),page_images.td(),page_images.b("Probability"),page_images.td.close(),page_images.td(str(image_IDs[0]['good_probs'][0][0])),page_images.td(str(image_IDs[0]['good_probs'][0][1])),page_images.td(str(image_IDs[0]['good_probs'][0][2])),page_images.tr.close()
page_images.table.close()

page_images.table(border=1)
page_images.tr(),page_images.th(""),page_images.th(uniquetarget_tr[0][0]),page_images.th(uniquetarget_tr[0][1]),page_images.th(uniquetarget_tr[0][2]),page_images.tr.close()
for i in range(len(feat_names)):
    page_images.tr()
    page_images.td(feat_names[i]),page_images.td(round(image_IDs[0]['good_contributions'][0][0][:,0][i],5)),page_images.td(round(image_IDs[0]['good_contributions'][0][0][:,1][i],5)),page_images.td(round(image_IDs[0]['good_contributions'][0][0][:,2][i],5))
    page_images.tr.close()

html_file= open("images.html","w")
html_file.write(page_images())
html_file.close()

logger.removeHandler(console)
#http://skyserver.sdss.org/dr12/en/tools/explore/Summary.aspx?id=1237655129301975515