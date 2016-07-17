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

17/7/16
- Added support for treeinterpreter! This currently makes it quite slow, might need to do something about that soon. Have turned the number of objects to be predicted down in the meantime. This function gives the contributions of each feature for each object for each class. Now we can see which feature helped decide each object individually!

13/7/16
- Added option to get images of each class, ones where the MLA got it right, nearly got it right, and got it really wrong. Places them in a temp folder and labels with MLA guess (good is prob > 0.9, ok is between 0.45 and 0.55, and bad is < 0.1), class, OBJID, and SPECZ.

12/7/16
- Added option to assess feature importances of each class compared to the others. Shows a crude plot.

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

- [ ] Add cross colours.
- [ ] Add other (all) features.
- [ ] Investigate feature importance as a function of redshift?

#### DONE
- [x] Find a selection of all types of results - great results where the MLA is certain, borderline results where the MLA can't decide, and really awful results where the MLA has gotten it completely wrong. Find the obj_IDs and find images of them on skyserver to see if I can come up with a reason as to why they certain answers are certain, and awful results are awful (maybe contamination, bad images etc.)
- [x] Do a one class vs. all run for each class to get feature importances per class. After that's done, the most important feature per class (say, r-i colour) could be binned along the x-axis maybe? I have to think more about this bit ...
- [x] Create a way of submitting jobs to the HPC cluster where I can iterate through settings.
- [x] Create plots of all these runs, which will give me a view on how the program performs.
- [x] Focus on representing feature importances (using sklearn) from multiple runs.
