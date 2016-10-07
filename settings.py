# -*- coding: utf-8 -*-
#[PROGRAM OPTIONS]
programpath='/users/moricex/ML_RF/'                                         # Root path to program 
trainpath='/users/moricex/DR12photodata/specPhotoDR12v3_hoyleb_extcorr_train_wtype.fit'       # Input training data
predpath='/users/moricex/DR12photodata/specPhotoDR12v3_hoyleb_extcorr_predict_wtype.fit'      # Input prediction data
filters=[['DERED_U','DERED_G','DERED_R','DERED_I','DERED_Z']\
,['PSFMAG_U','PSFMAG_G','PSFMAG_R','PSFMAG_I','PSFMAG_Z']\
,['FIBERMAG_U','FIBERMAG_G','FIBERMAG_R','FIBERMAG_I','FIBERMAG_Z']\
,['CMODELMAG_U_EXT','CMODELMAG_G_EXT','CMODELMAG_R_EXT','CMODELMAG_I_EXT','CMODELMAG_Z_EXT']]        # Filter list as it is in fits file
othertrain=['EXPRAD_U','EXPRAD_G','EXPRAD_R','EXPRAD_I','EXPRAD_Z']#['SPEC_CLASS_ID']#['SPECZ']                                   # Other features to give the MLA
predict = 'SPEC_CLASS_ID'                                                   # Feature to predict

onlyuse=['PSFMAG_U - CMODELMAG_U_EXT','PSFMAG_G - CMODELMAG_G_EXT','PSFMAG_R - CMODELMAG_R_EXT','PSFMAG_I - CMODELMAG_I_EXT','PSFMAG_Z - CMODELMAG_Z_EXT']

double_sub_run = 0
one_vs_all = 0                                                              # WARNING. Takes as many runs as there are classes in training set

pyspark_on=0								    # Use pyspark instead of sklearn
pyspark_remake_csv=0							    # Remake csv files for pyspark? (If you know the settings are the same, don't rebuild)

make_binary = 1

saveresults=1                                                               # Save results or not? 
feat_outfile = 'ML_RF_feat'                                  # Filename for feature importance results
result_outfile = 'ML_RF_results'                                          # Filename for results
prob_outfile = 'ML_RF_probs'
log_outfile='ML_RF_logfile'						    # Name of output logfile
stats_outfile='ML_RF_stats'
featnames_outfile='ML_RF_featnames'
scores_outfile='ML_RF_scores'

# THESE TAKE CONSIDERABLY MORE TIME (and also save automatically)
output_all_trees = 0
get_contributions = 0
get_perfect_contributions=0
compute_contribution_mic=0                                                  # Done post ML fit (to calc and save new contributions)

#Preprocessing features analysis functions
compute_mic=0                                                               # Done pre-processing WARNING - takes some time. Computes Maximal Information Coefficients between all features on training set
compute_pearson=0

# WARNING: MINT saves results to save computing time. This is hardcoded in run_opts.
# WARNING: This means if your input catalogue/or targetted classes changes for any reason, these must be deleted and recalculated.
calc_MINT = 1 # Does not work in OvsA
MINT_n_feat=4

compute_mifs=0 # Does not work in OvsA
mifs_types=['JMI','JMIM','MRMR']
mifs_n_feat=10 

traindatanum=2500                                                           # Number of objects to train on
predictdatanum=50000                                                        # Number of objects to predict
weightinput=[]#[34,33,33]                                                   # Weights number of objects in each class. Value is percentage.
cut_outliers=0

diagnostics=1
# MLA settings
MLA = 'sklearn.ensemble.RandomForestClassifier'                             # Which MLA to load
MLAset = {'n_estimators': 256, 'n_jobs': 8,'bootstrap':True,'verbose':True,'max_depth':None}         # MLA settings
#MLA = 'sklearn.ensemble.AdaBoostClassifier'
#MLAset = {'n_estimators':1024,'learning_rate':1}
actually_run=1                                                              # Actually run the MLA
n_runs = 1     
                                                           
# RUN OPTS
checkmagspos=1                                                              # Checks filter mags are positive. Keep this on
find_only_classified=0                                                      # MUST RESET PYTHON IF CHANGED
find_all_classes=0                                                          # Finds all subclasses and stores them in a variable

calculate_colours=1                                                         # Give MLA colours?
    #This goes as:
    # u-g, u-r, u-i, u-z, g-r, g-i, g-z, r-i, r-z, i-z
    # for all filters
use_colours=[[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9]]
calculate_cross_colours=1

# PLOTS
plotsubclasshist=0                                                          # Plot hist of subclasses (for subclass, not classes!)
plotbandvprob = 0							    # Plot hist of filter band vs prob for each class
plotcolourvprob = 0
plotfeatimp = 0                                                       # Plot hist of colour bands vs prob for each class (for class, not subclass)
plot_col_rad = 0                                                    # Doesn't work with MINT or mifs ...
plot_col_cont = 0
plot_col_cont_true = 0
plot_mic = 0
plot_pearson=0
plot_mic_cont=0
plot_decision_boundaries_MINT = 0                                                 # Only works with MINT so far
plot_decision_boundaries = 0
plot_decision_boundaries_DT = 0
plot_depth_acc = 0

get_images=0
html_on=0

objc_type_cuts=0