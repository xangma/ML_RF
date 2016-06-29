# -*- coding: utf-8 -*-

def get_function(function_string):
    import importlib
    module, function = function_string.rsplit('.', 1)
    module = importlib.import_module(module)
    function = getattr(module, function)
    return function

# PROGRAM OPTIONS
programpath='/users/moricex/ML/v1.0.4/'                                     # Root path to program 
trainpath='../specPhotoDR12v3_hoyleb_extcorr_train.fit'                     # Input training data
predpath='../specPhotoDR12v3_hoyleb_extcorr_predict.fit'   
                 # Input prediction data
filters=[['DERED_U','DERED_G','DERED_R','DERED_I','DERED_Z']\
,['PSFMAG_U','PSFMAG_G','PSFMAG_R','PSFMAG_I','PSFMAG_Z']\
,['FIBERMAG_U','FIBERMAG_G','FIBERMAG_R','FIBERMAG_I','FIBERMAG_Z']]        # Filter list as it is in fits file
othertrain=[]#['SPECZ']                                                     # Other features to give the MLA
predict = 'SPEC_CLASS_ID'                                                   # Feature to predict

saveresults=0                                                               # Save results or not?
outfile = 'ML_RF_results.txt'                                               # Filename for results
feat_outfile = 'ML_RF_feat_importance.txt'                                  # Filename for feature importance results
logfile_out='ML_RF_logfile.txt'
traindatanum=10000                                                          # Number of objects to train on
predictdatanum=500000                                                       # Number of objects to predict
weightinput=[]#[34,33,33]

diagnostics=1
# MLA settings
MLA = get_function('sklearn.ensemble.RandomForestClassifier')               # Which MLA to load
MLA = MLA(n_estimators=50,n_jobs=6,bootstrap=True,verbose=True)             # MLA settings
actually_run=0                                                              # Actually run the MLA

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
plotsubclasshist=0                                                          # for subclass, not class
plotbandvprob=0
plotcolourvprob=0                                                           # for class, not subclass
