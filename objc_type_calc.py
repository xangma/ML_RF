# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 12:03:42 2016

@author: moricex
"""

import astropy.io.fits as fits
import os
import settings
import numpy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

os.chdir(settings.programpath) # Change directory
cwd=os.getcwd()
dirs=os.listdir(cwd)

traindata=fits.open(settings.trainpath)
traindata=traindata[1].data
preddata=fits.open(settings.predpath)
preddata=preddata[1].data

typenames=['type_u','type_g','type_r','type_i','type_z']
psfmagnames=['PSFMAG_U','PSFMAG_G','PSFMAG_R','PSFMAG_I','PSFMAG_Z']
psfmagerrnames=['PSFMAGERR_U','PSFMAGERR_G','PSFMAGERR_R','PSFMAGERR_I','PSFMAGERR_Z']
cmodelmagnames=['CMODELMAG_U','CMODELMAG_G','CMODELMAG_R','CMODELMAG_I','CMODELMAG_Z']
cmodelmagerrnames=['CMODELMAGERR_U','CMODELMAGERR_G','CMODELMAGERR_R','CMODELMAGERR_I','CMODELMAGERR_Z']
extnames=['EXTINCTION_U','EXTINCTION_G','EXTINCTION_R','EXTINCTION_I','EXTINCTION_Z']
x=preddata
templ_type=x['SPEC_CLASS_ID']
x=x[x['clean']==1] # CUT CLEAN PHOTOMETRY. TAKE CARE HERE. REDUCES ARRAY BY 10%
#MAKE BINARY
stars_train = templ_type == 2
QSO_train = templ_type == 1
PS_indexes = stars_train+QSO_train
templ_type[PS_indexes] = 1

#def calc_objc_type(x,name):
type_arr,psfmags,cmodelmags,mytype,ext,match=[],[],[],[],[],[]
psfmagerrs,cmodelmagerrs=[],[]
mask={}
for i in range(len(typenames)):
    type_arr.append(x[typenames[i]])
for i in range(len(psfmagnames)):
    psfmags.append(x[psfmagnames[i]])
for i in range(len(cmodelmagnames)):
    cmodelmags.append(x[cmodelmagnames[i]])
for i in range(len(extnames)):
    ext.append(x[extnames[i]])
for i in range(len(psfmagerrnames)):
    psfmagerrs.append(x[psfmagerrnames[i]])
for i in range(len(cmodelmagerrnames)):
    cmodelmagerrs.append(x[cmodelmagerrnames[i]])
#type_arr.append(x['type_r'])
for i in range(len(psfmagnames)):
    mytype.append(psfmags[i]-(cmodelmags[i]-ext[i])>=0.145)
    mytype[i]=numpy.where(mytype[i],3,6)
    print(sum(type_arr[i]==mytype[i]))
    match.append(type_arr[i]==mytype[i])
    match[i] = match[i].astype(int)
output=numpy.column_stack((mytype[2],match[2]))
output=output.T
numpy.save('mytype_res_predict2',output)
#Filter out negatives
#mask=[]
#for i in range(len(psfmagnames)):
#    mask.append((psfmags[i]>0) & (psfmags[i]>5))


mytypeT=numpy.transpose(mytype)
psfmagsT=numpy.transpose(psfmags)
cmodelmagsT=numpy.transpose(cmodelmags)
extT=numpy.transpose(ext)
#    mytype_all=[]
#    for i in range(len(psfmagsT)):
#        mytype_all.append((sum(psfmagsT[i])-sum(cmodelmagsT[i]-extT[i]))>=0.145)
#        mytype_all[i]=numpy.where(mytype_all[i],3,6)
#    return mask,psfmagsT,cmodelmagsT,extT,mytype,type_arr

    #print(sum(type_arr[-1]==numpy.transpose(mytype_all)))
    #NOTAVOTE
#    mytype_all=[]
#    for i in range(len(mytypeT)):
#        if (sum(mytypeT[i]==3)>2) ==True:
#            mytype_all.append(3)
#        elif (sum(mytypeT[i]==6)>2) ==True:
#            mytype_all.append(6)
#    print(sum(mytype_all==type_arr[-1]))
#    matches=mytype_all==type_arr[-1]
#    mytype_res=numpy.vstack((mytype_all,matches))
#    numpy.save('mytype_res_'+name,mytype_res)
    
#mask,psfmagsT,cmodelmagsT,extT,mytype,type_arr = calc_objc_type(traindata,'train')
#mask,psfmagsT,cmodelmagsT,extT,mytype,type_arr = calc_objc_type(preddata,'predict')

#gals_match=[]
#ps_match=[]
#gals_objc_u=traindata['TYPE_U']==3
#ps_objc_u =  traindata['TYPE_U']==6
#gals_objc_g=traindata['TYPE_G']==3
#ps_objc_g =  traindata['TYPE_G']==6
#gals_objc_r=traindata['TYPE_R']==3
#ps_objc_r =  traindata['TYPE_R']==6
#gals_objc_i=traindata['TYPE_I']==3
#ps_objc_i =  traindata['TYPE_I']==6
#gals_objc_z=traindata['TYPE_Z']==3
#ps_objc_z =  traindata['TYPE_Z']==6
#gals_spec= traindata['SPEC_CLASS_ID']==0
#ps_spec = traindata['SPEC_CLASS_ID']>0
#
#gals_objc_r_MINE=mytype[2]==3
#ps_objc_r_MINE=mytype[2]==6
##gals_objc_r_MINE=gals_objc_r_MINE[mask[2]]
##ps_objc_r_MINE=ps_objc_r_MINE[mask[2]]
#
#colourR=psfmagsT[:,2]-(cmodelmagsT[:,2]-extT[:,2])
#plt.figure()
#plt.scatter(colourR[ps_objc_r_MINE[0:2500]][0:2500],psfmagsT[:,2][ps_objc_r_MINE[0:2500]][0:2500],color='blue',s=3)
#plt.scatter(colourR[gals_objc_r_MINE[0:2500]][0:2500],psfmagsT[:,2][gals_objc_r_MINE[0:2500]][0:2500],color='red',s=3)
#ps_patch = mpatches.Patch(color='blue', label='Point Sources')
#gals_patch = mpatches.Patch(color='red', label='Galaxies')
#plt.legend(handles=[gals_patch,ps_patch])
#plt.title('SDSS Frames colour cut',fontsize=18)
#plt.legend()
#plt.axvline(x=0.145,linewidth=2,color='black')
#plt.ylabel('Psfmag_r',fontsize=15)
#plt.xlabel('Psfmag_r - Cmodelmag_r',fontsize=15)
#plt.tight_layout()
#plt.ylim(10,25)
#plt.xlim(-1,4)
#plt.show()
#plt.savefig('FRAMES_R_BAND_CUT.png')