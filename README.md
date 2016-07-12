# ML_RF
Machine Learning Random Forest Classification for DR12.

This code performs star/galaxy/QSO separation using a training set with known targets.

####MLdr12_RF.py 
This is the main file. Run this to start program.

####plots.py 
This deals with plots (enabled in settings.py)

####run_opts.py 
These are optional functions that the main code dips out to. Most can be enabled or disabled in settings.py

####settings.py
Contains all settings for program. Is quite particular so be careful. Can swap out filters/colours as needed.

6/7/16

- Added run_sciama_plots.py. This uses the outputs of run_sciama.py (reading the directory structure) and allows the program to:
 - Create plots of multiple runs, which will give me a view on how the program performs (scaling through number of training objects and number of trees/estimators).
 - Obtain a (crude) representation of feature importances (using sklearn) from multiple runs.

3/7/16
- Added run_sciama.py. This so far creates a folder structure, copies up the code, and submits it to sciama. It is iterated to submit multiple jobs.
- Started run_sciama_plots.py. This file is able to read the directory structure created by run_sciama.py. This means plots will be able to be created from multiple runs.

2/7/16
- Added 'double_sub_run' in settings. If set to do CLASS separation (star/gal/QSO), this function then uses the predicted results of the CLASS run, appends them to the training data, then tries to predict subclass. Don't really know how useful this will be, but it's fun to include it. Shall investigate further when pipeline to sciama is finished.

1/7/16
- Comments!

30/6/16
 - Added pyspark functionality. Upon running this on an HPC cluster, I saw that sharing resources across nodes is difficult. Pyspark allows this to happen. Unfortunately, there is no way (none that I could see anyway in the current code), of assessing feature importances. It is for this reason I will be pausing development of the pyspark section of the code ... the other option would be to code it myself. I feel this wouldn't be the best use of my time, considering others on the pyspark team seem to be working on this.
 
# TO DO

- Priority: Do a one class vs. all run for each class to get feature importances per class. After that's done, the most important feature per class (say, r-i colour) could be binned along the x-axis maybe? I have to think more about this bit ...
- Add cross colours.
- Add other (all) features.
- Investigate feature importance as a function of redshift?
