# -*- coding: utf-8 -*-
#[PROGRAM OPTIONS]
programpath='/users/moricex/ML_RF/'                                         # Root path to program 
trainpath='/users/moricex/DR12photodata/specPhotoDR12v3_hoyleb_extcorr_train.fit'       # Input training data
predpath='/users/moricex/DR12photodata/specPhotoDR12v3_hoyleb_extcorr_predict.fit'      # Input prediction data
filters=[['DERED_U','DERED_G','DERED_R','DERED_I','DERED_Z']\
,['PSFMAG_U','PSFMAG_G','PSFMAG_R','PSFMAG_I','PSFMAG_Z']\
,['FIBERMAG_U','FIBERMAG_G','FIBERMAG_R','FIBERMAG_I','FIBERMAG_Z']]        # Filter list as it is in fits file
othertrain=['EXPRAD_U','EXPRAD_G','EXPRAD_R','EXPRAD_I','EXPRAD_Z']#['SPEC_CLASS_ID']#['SPECZ']                                   # Other features to give the MLA
predict = 'SPEC_CLASS_ID'                                                   # Feature to predict


double_sub_run = 0
one_vs_all = 1                                                              # WARNING. Takes as many runs as there are classes in training set

pyspark_on=0								    # Use pyspark instead of sklearn
pyspark_remake_csv=0							    # Remake csv files for pyspark? (If you know the settings are the same, don't rebuild)

saveresults=1                                                               # Save results or not? 
resultsstack_save=0                                                         # Save all results? WARNING: Very large table (>1gb probably)                                                         
outfile = 'ML_RF_resultsstack'                                          # Filename for results
feat_outfile = 'ML_RF_feat_'                                  # Filename for feature importance results
result_outfile = 'ML_RF_results'
prob_outfile = 'ML_RF_probs'
log_outfile='ML_RF_logfile'						    # Name of output logfile
stats_outfile='ML_RF_stats'

output_tree = 0
get_contributions = 0

traindatanum=2500                                                           # Number of objects to train on
predictdatanum=100000                                                       # Number of objects to predict
weightinput=[]#[34,33,33]                                                   # Weights number of objects in each class. Value is percentage.

diagnostics=1
# MLA settings
MLA = 'sklearn.ensemble.RandomForestClassifier'                             # Which MLA to load
MLAset = {'n_estimators': 256, 'n_jobs': 4,'bootstrap':True,'verbose':True,'max_depth':None}         # MLA settings
actually_run=1   
n_runs = 10                                                                 # Actually run the MLA

# RUN OPTS
checkmagspos=1                                                              # Checks filter mags are positive. Keep this on
find_only_classified=0                                                      # MUST RESET PYTHON IF CHANGED
find_all_classes=0                                                          # Finds all subclasses and stores them in a variable

calculate_colours=1                                                         # Give MLA colours?
    #This goes as:
    # u-g, u-r, u-i, u-z, g-r, g-i, g-z, r-i, r-z, i-z
    # for all filters
use_colours=[[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9]]

# PLOTS
plotsubclasshist=0                                                          # Plot hist of subclasses (for subclass, not classes!)
plotbandvprob=0								    # Plot hist of filter band vs prob for each class
plotcolourvprob=0    
plotfeatimp = 1                                                       # Plot hist of colour bands vs prob for each class (for class, not subclass)
get_images=0