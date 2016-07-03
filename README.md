# ML_RF
Machine Learning Random Forest Classification for DR12.

This code performs star/galaxy/QSO separation using a training set with known targets.

MLdr12_RF.py 
This is the main file. Run this to start program.

plots.py 
This deals with plots (enabled in settings.py)

run_opts.py 
These are optional functions that the main code dips out to. Most can be enabled or disabled in settings.py

settings.py
Contains all settings for program. Is quite particular so be careful. Can swap out filters/colours as needed.


3/7/16
- Added run_sciama.py. This so far creates a folder structure, copies up the code, and submits it to sciama. It needs to be iterated to submit multiple jobs.

30/6/16
 - Added pyspark functionality. Upon running this on an HPC cluster, I saw that sharing resources across nodes is difficult. Pyspark allows this to happen. Unfortunately, there is no way (none that I could see anyway in the current code), of assessing feature importances. It is for this reason I will be pausing development of the pyspark section of the code ... the other option would be to code it myself. I feel this wouldn't be the best use of my time, considering others on the pyspark team seem to be working on this.
 
# TO DO

- Add cross colours.
- Add other (all) features.
- Create a way of submitting jobs to the HPC cluster where I can iterate through settings. (Half done 3/7/16)
- Create plots of all these runs, which will give me a view on how the program performs.
- Focus on representing feature importances (using sklearn) from multiple runs.
