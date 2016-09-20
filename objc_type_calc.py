# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 12:03:42 2016

@author: moricex
"""

import astropy.io.fits as fits
import os
import settings
import numpy
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

def calc_objc_type(x,name):
    type_arr,psfmags,cmodelmags,mytype,ext=[],[],[],[],[]
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
    type_arr.append(x['type'])
    for i in range(len(psfmagnames)):
        mytype.append(psfmags[i]-(cmodelmags[i]-ext[i])>=0.145)
        mytype[i]=numpy.where(mytype[i],3,6)
        print(sum(type_arr[i]==mytype[i]))
    
    #Filter out negatives
    mask=[]
    for i in range(len(psfmagnames)):
        mask.append(psfmags[i]>0)
    
    
    mytypeT=numpy.transpose(mytype)
    psfmagsT=numpy.transpose(psfmags)
    cmodelmagsT=numpy.transpose(cmodelmags)
    extT=numpy.transpose(ext)
    #mytype_all=[]
    #for i in range(len(psfmagsT)):
    #    mytype_all.append((sum(psfmagsT[i])-sum(cmodelmagsT[i]-extT[i]))>=0.145)
    #    mytype_all[i]=numpy.where(mytype_all[i],3,6)
    #print(sum(type_arr[-1]==numpy.transpose(mytype_all)))
    #NOTAVOTE
    mytype_all=[]
    for i in range(len(mytypeT)):
        if (sum(mytypeT[i]==3)>2) ==True:
            mytype_all.append(3)
        elif (sum(mytypeT[i]==6)>2) ==True:
            mytype_all.append(6)
    print(sum(mytype_all==type_arr[-1]))
    matches=mytype_all==type_arr[-1]
    mytype_res=numpy.vstack((mytype_all,matches))
    numpy.save('mytype_res_'+name,mytype_res)
    
calc_objc_type(traindata,'train')
calc_objc_type(preddata,'predict')