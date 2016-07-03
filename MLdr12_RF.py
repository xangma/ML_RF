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

temp_train='./temp_train.csv' # Define temp files for pyspark
temp_pred='./temp_pred.csv'

os.chdir(settings.programpath) # Change directory
cwd=os.getcwd()
dirs=os.listdir(cwd)

logging.basicConfig(level=logging.INFO,\
                    format='%(asctime)s %(name)-20s %(levelname)-6s %(message)s',\
                    datefmt='%d-%m-%y %H:%M',\
                    filename='ML_RF.log',\
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
logger.info(settings.programpath)
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
combs={}

for j in range(len(settings.filters)): # For each filter set
    n_filt=len(settings.filters[j]) # Get the number of filters
    filt_train=[] # Set up filter array
    for i in range(numpy.size(settings.filters[j])): # For each filter i in filter set j 
        filt_train.append(traindata[settings.filters[j][i]]) # Append filter to filter array
    filt_train=numpy.transpose(filt_train)
    filt_predict=[]
    for i in range(numpy.size(settings.filters[j])): # Do same for prediction set
        filt_predict.append(preddata[settings.filters[j][i]])
    filt_predict=numpy.transpose(filt_predict)
    
    filt_train,filt_predict,combs[j] = run_opts.calculate_colours(filt_train,filt_predict,n_filt) # Section that calculates all possible colours
    filt_train,filt_predict,n_colour = run_opts.use_filt_colours(filt_train,filt_predict,j,n_filt) # Section that checks use_colours and cuts colours accordingly

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
    
# Stack arrays to feed to MLA
XX=numpy.array(filt_train_all[0][0])
if len(filt_train_all) >= 1:
    for i in range(1,len(filt_train_all)):
        XX=numpy.column_stack((XX,numpy.array(filt_train_all[i][0])))
for i in range(len(settings.othertrain)): # Tack on other training features (not mags, like redshift)
    XX = numpy.column_stack((XX,traindata[settings.othertrain[i]]))
classnames_tr=traindata[settings.predict[:-3]] # Get classnames
subclass_tr=traindata['SPEC_SUBCLASS_ID']
subclass_names_tr=traindata['SPEC_SUBCLASS']
XX=numpy.column_stack((XX,traindata[settings.predict])) # Stack training data for MLA, tack on true answers

XXpredict=numpy.array(filt_predict_all[0][0])
if len(filt_predict_all) > 1:
    for i in range(1,len(filt_predict_all)):
        XXpredict=numpy.column_stack((XXpredict,filt_predict_all[i][0]))
for i in range(len(settings.othertrain)): # Tack on other prediction features (not mags, like redshift)
    XXpredict = numpy.column_stack((XXpredict,preddata[settings.othertrain[i]]))
classnames_pr=preddata[settings.predict[:-3]]
subclass_pr = preddata['SPEC_SUBCLASS_ID']
subclass_names_pr = preddata['SPEC_SUBCLASS']
XXpredict=numpy.column_stack((XXpredict,preddata[settings.predict])) # Stack training data for MLA, tack on true answers so can evaluate after

# Filter out negative magnitudes
# THIS MUST BE DONE LAST IN THIS PROCESSING PART.
XX,XXpredict,classnames_tr,classnames_pr,subclass_tr,subclass_names_tr,subclass_pr,subclass_names_pr\
 = run_opts.checkmagspos(XX,XXpredict,classnames_tr,classnames_pr,subclass_tr,subclass_names_tr,subclass_pr,subclass_names_pr,filtstats)

XX,classnames_tr = run_opts.weightinput(XX,classnames_tr) # Weight training set? - specified in settings

XX = XX[0:traindatanum] # Cut whole training array down to size specified in settings
XXpredict=XXpredict[0:predictdatanum]
classnames_tr=classnames_tr[0:traindatanum] # Do same for classnames
classnames_pr=classnames_pr[0:predictdatanum]

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

def run_MLA(XX,XXpredict,yy,yypredict,n_feat):
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
    return result
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

if settings.actually_run == 1:# If it is set to actually run in settings
    result=run_MLA(XX,XXpredict,yy,yypredict,n_feat)

if settings.double_sub_run == 1:
    XX = numpy.column_stack((XX,subclass_tr))
    XXpredict = numpy.column_stack((XXpredict[:,:-1],result))
    n_feat=n_feat+1
    yy=subclass_tr
    yypredict=subclass_pr
    logger.info('Starting *SECOND* MLA run')
    unique_IDS_tr, unique_IDS_pr,uniquetarget_tr,uniquetarget_pr = \
    run_opts.diagnostics([XX[:,-1],yypredict,subclass_names_tr,subclass_names_pr],'inputdata') # Total breakdown of types going in
    settings.MLA = settings.MLA(n_estimators=100,n_jobs=16,bootstrap=True,verbose=True) 
    result2 = run_MLA(XX,XXpredict,yy,yypredict,n_feat)

logger.removeHandler(console)
