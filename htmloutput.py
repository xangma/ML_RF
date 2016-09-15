# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 10:16:27 2016

@author: moricex
"""
import os
import settings
import markup
from markup import oneliner as e

# HTML OUTPUT
def htmloutput(ind_run_name,accuracy,uniquetarget_tr,recall,precision,score,clf,col_names,plots_col_cont_outnames\
,plots_col_cont_true_outnames,plots_col_rad_outnames,plots_bandvprob_outnames,plots_feat_outname\
,plots_feat_per_class_outname,plots_colourvprob_outnames,image_IDs,feat_names,plots_mic_outnames,plots_pearson_outnames\
,plots_mic_contributions_outnames,results_dict):
        # Make index html
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
    
    #TABLES
    for j in range(len(results_dict)):
        page.table(border=1)
        page.tr(),page.th(results_dict[j]['run_name'])
        for i in range(len(results_dict[j]['uniquetarget_tr'])):
            page.th(results_dict[j]['uniquetarget_tr'][0][i])
        page.tr.close()
        page.tr(),page.td(),page.b("Recall"),page.td.close()
        for i in range(len(results_dict[j]['recall'])):
            page.td(round(results_dict[j]['recall'][i],5))
        page.tr.close()
        page.tr(),page.td(),page.b("Precision"),page.td.close()
        for i in range(len(results_dict[j]['precision'])):
            page.td(round(results_dict[j]['precision'][i],5))
        page.tr.close()
        page.tr(),page.td(),page.b("F1 Score"),page.td.close()
        for i in range(len(results_dict[j]['score'])):
            page.td(round(results_dict[j]['score'][i],5))
        page.tr.close()
        page.table.close()
    
    # Write out settings
    html_settings=("Number of training objects: %s" %settings.traindatanum,"Number of prediction objects: %s" %settings.predictdatanum\
    ,"","Random Forest Settings:",clf\
    ,"","Features:","    Filters: %s" %settings.filters, "    Colours: %s" %col_names, "    Other: %s" %settings.othertrain)
    page.p(html_settings)
    
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
    page_plots.p("")
    
    page_plots_col_cont = markup.page()
    page_plots_col_cont.init(title=html_title+" Plots_col_cont")
    page_plots_col_cont.p(page_plots_col_cont.h3("Results for run: %s Plots_col_cont" %ind_run_name))
    page_plots_col_cont.a( "Home",href="index.html")
    page_plots_col_cont.a( "Example Tree",href="trees.html")
    page_plots_col_cont.a( "Plots",href="plots.html")
    page_plots_col_cont.a( "Images",href="images.html")
    page_plots_col_cont.p("")
    page_plots_col_cont.a( "plot_col_cont",href="plots_col_cont.html")
    page_plots_col_cont.a( "plot_col_rad",href="plots_col_rad.html")
    page_plots_col_cont.p("")
    
    for i in range(len(plots_col_cont_outnames)):
        page_plots_col_cont.div(style='width: 2000px; height: 600px;',id='cc%s' %i)
        page_plots_col_cont.p(["",plots_col_cont_outnames[i]]),page_plots_col_cont.p(["",plots_col_cont_true_outnames[i]])
        page_plots_col_cont.img(src=plots_col_cont_outnames[i]),page_plots_col_cont.img(src=plots_col_cont_true_outnames[i])
        page_plots_col_cont.div.close()
    
    page_plots_col_rad = markup.page()
    page_plots_col_rad.init(title=html_title+" Plots_col_rad")
    page_plots_col_rad.p(page_plots_col_rad.h3("Results for run: %s Plots_col_rad" %ind_run_name))
    page_plots_col_rad.a( "Home",href="index.html")
    page_plots_col_rad.a( "Example Tree",href="trees.html")
    page_plots_col_rad.a( "Plots",href="plots.html")
    page_plots_col_rad.a( "Images",href="images.html")
    page_plots_col_rad.p("")
    page_plots_col_rad.a( "plot_col_cont",href="plots_col_cont.html")
    page_plots_col_rad.a( "plot_col_rad",href="plots_col_rad.html")
    page_plots_col_rad.p("")
    
    for i in range(len(plots_col_rad_outnames)):
        page_plots_col_rad.p(["",plots_col_rad_outnames[i]])
        page_plots_col_rad.img(src=plots_col_rad_outnames[i])
    
    page_plots.a( "plot_col_cont",href="plots_col_cont.html")
    page_plots.a( "plot_col_rad",href="plots_col_rad.html")
    page_plots.p("")
    page_plots.p("Overall Feature Importance")
    page_plots.img(src=plots_feat_outname)
    page_plots.p("")
    if settings.one_vs_all == 1:
        page_plots.p("Feature importance per class")
        page_plots.img(src=plots_feat_per_class_outname)
    if (settings.plot_mic ==1 and settings.plot_pearson == 1 and settings.plot_mic_cont == 1):
        page_plots.p("Maximal Information Coefficients")    
        for i in range(len(plots_mic_outnames)):
            page_plots.div(style='width: 2000px; height: 600px;')
            page_plots.p(plots_mic_outnames[i]),page_plots.p(plots_pearson_outnames[i])
            page_plots.img(src=plots_mic_outnames[i]),page_plots.img(src=plots_pearson_outnames[i])
            page_plots.div.close()
        
        for i in range(len(plots_mic_contributions_outnames)):
            page_plots.p("MIC of Contribution to P(%s) for each feature"%uniquetarget_tr[0][i])
            page_plots.img(src=plots_mic_contributions_outnames[i])
    
    allfiltplots= [s for s in plots_bandvprob_outnames if 'allfilt' in s]
    for i in range(len(allfiltplots)):
        page_plots.p(["",allfiltplots[i]])
        page_plots.img(src=allfiltplots[i])
    
    allfiltplots_cols= [s for s in plots_colourvprob_outnames if 'allfilt' in s]
    for i in range(len(allfiltplots_cols)):
        page_plots.p(["",allfiltplots_cols[i]])
        page_plots.img(src=allfiltplots_cols[i])
    
    html_file= open("plots.html","w")
    html_file.write(page_plots())
    html_file.close()
    html_file= open("plots_col_rad.html","w")
    html_file.write(page_plots_col_rad())
    html_file.close()
    html_file= open("plots_col_cont.html","w")
    html_file.write(page_plots_col_cont())
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
    
    for k in range(len(image_IDs)):
        for j in range(len(image_IDs[k]['good_url'])):
            page_images.p("")
            page_images.table(border=1)
            page_images.tr(),page_images.td(),page_images.a( e.img( src=image_IDs[k]['good_url'][j]), href=image_IDs[k]['good_url_objid'][j]),page_images.td.close(),page_images.td(),page_images.a( e.img( src=image_IDs[k]['good_spectra'][j]),width=200,height=143,href=image_IDs[k]['good_url_objid'][j]),page_images.td.close(),page_images.tr.close()
            page_images.tr(),page_images.td(),page_images.b('Class'),page_images.td.close(),page_images.td(str(uniquetarget_tr[0][image_IDs[k]['class']])),page_images.tr.close()
            page_images.tr(),page_images.td(),page_images.b('Predicted Class'),page_images.td.close(),page_images.td(str(uniquetarget_tr[0][image_IDs[k]['good_result'][j]])),page_images.tr.close()
            page_images.tr(),page_images.td(),page_images.b('ObjID'),page_images.td.close(),page_images.td(str(image_IDs[k]['good_ID'][j])),page_images.tr.close()
            page_images.tr(),page_images.td(),page_images.b('Redshift'),page_images.td.close(),page_images.td(str(image_IDs[k]['good_specz'][j])),page_images.tr.close()
            page_images.table.close()
            
            page_images.table(border=1)
            page_images.tr(),page_images.th("")
            for num in range(len(uniquetarget_tr)):
                page_images.th(uniquetarget_tr[0][num])
            page_images.tr.close()
            page_images.tr(),page_images.td(),page_images.b("Probability"),page_images.td.close(),page_images.td(str(image_IDs[k]['good_probs'][j][0])),page_images.td(str(image_IDs[k]['good_probs'][j][1])),page_images.td(str(image_IDs[k]['good_probs'][j][2])),page_images.tr.close()
            page_images.tr(),page_images.td(),page_images.b("Bias"),page_images.td.close(),page_images.td(str(image_IDs[k]['good_tiresult'][j][1][0][0])),page_images.td(str(image_IDs[k]['good_tiresult'][j][1][0][1])),page_images.td(str(image_IDs[k]['good_tiresult'][j][1][0][2])),page_images.tr.close()
            page_images.table.close()
            page_images.p("Contributions to Probability")
            page_images.table(border=1)
            page_images.tr(),page_images.th(""),page_images.td("Values")
            for num in range(len(uniquetarget_tr)):
                page_images.th(uniquetarget_tr[0][num])
            page_images.tr.close()
            for i in range(len(feat_names)):
                page_images.tr()
                page_images.td(feat_names[i]),page_images.td(str(image_IDs[k]['good_values'][j][i])),page_images.td(round(image_IDs[k]['good_tiresult'][j][2][0][:,0][i],5)),page_images.td(round(image_IDs[k]['good_tiresult'][j][2][0][:,1][i],5)),page_images.td(round(image_IDs[k]['good_tiresult'][j][2][0][:,2][i],5))
                page_images.tr.close()
    
    for k in range(len(image_IDs)):
        for j in range(len(image_IDs[k]['ok_url'])):
            page_images.p("")
            page_images.table(border=1)
            page_images.tr(),page_images.td(),page_images.a( e.img( src=image_IDs[k]['ok_url'][j]), href=image_IDs[k]['ok_url_objid'][j]),page_images.td.close(),page_images.td(),page_images.a( e.img( src=image_IDs[k]['ok_spectra'][j]),width=200,height=143, href=image_IDs[k]['ok_url_objid'][j]),page_images.td.close(),page_images.tr.close()
            page_images.tr(),page_images.td(),page_images.b('Class'),page_images.td.close(),page_images.td(str(uniquetarget_tr[0][image_IDs[k]['class']])),page_images.tr.close()
            page_images.tr(),page_images.td(),page_images.b('Predicted Class'),page_images.td.close(),page_images.td(str(uniquetarget_tr[0][image_IDs[k]['ok_result'][j]])),page_images.tr.close()
            page_images.tr(),page_images.td(),page_images.b('ObjID'),page_images.td.close(),page_images.td(str(image_IDs[k]['ok_ID'][j])),page_images.tr.close()
            page_images.tr(),page_images.td(),page_images.b('Redshift'),page_images.td.close(),page_images.td(str(image_IDs[k]['ok_specz'][j])),page_images.tr.close()
            page_images.table.close()
            
            page_images.table(border=1)
            page_images.tr(),page_images.th(""),page_images.th(uniquetarget_tr[0][0]),page_images.th(uniquetarget_tr[0][1]),page_images.th(uniquetarget_tr[0][2]),page_images.tr.close()
            page_images.tr(),page_images.td(),page_images.b("Probability"),page_images.td.close(),page_images.td(str(image_IDs[k]['ok_probs'][j][0])),page_images.td(str(image_IDs[k]['ok_probs'][j][1])),page_images.td(str(image_IDs[k]['ok_probs'][j][2])),page_images.tr.close()
            page_images.tr(),page_images.td(),page_images.b("Bias"),page_images.td.close(),page_images.td(str(image_IDs[k]['ok_tiresult'][j][1][0][0])),page_images.td(str(image_IDs[k]['ok_tiresult'][j][1][0][1])),page_images.td(str(image_IDs[k]['ok_tiresult'][j][1][0][2])),page_images.tr.close()
            page_images.table.close()
            page_images.p("Contributions to Probability")       
            page_images.table(border=1)
            page_images.tr(),page_images.th(""),page_images.td("Values"),page_images.th(uniquetarget_tr[0][0]),page_images.th(uniquetarget_tr[0][1]),page_images.th(uniquetarget_tr[0][2]),page_images.tr.close()
            for i in range(len(feat_names)):
                page_images.tr()
                page_images.td(feat_names[i]),page_images.td(str(image_IDs[k]['ok_values'][j][i])),page_images.td(round(image_IDs[k]['ok_tiresult'][j][2][0][:,0][i],5)),page_images.td(round(image_IDs[k]['ok_tiresult'][j][2][0][:,1][i],5)),page_images.td(round(image_IDs[k]['ok_tiresult'][j][2][0][:,2][i],5))
                page_images.tr.close()
    
    for k in range(len(image_IDs)):
        for j in range(len(image_IDs[k]['bad_url'])):
            page_images.p("")
            page_images.table(border=1)
            page_images.tr(),page_images.td(),page_images.a( e.img( src=image_IDs[k]['bad_url'][j]), href=image_IDs[k]['bad_url_objid'][j]),page_images.td.close(),page_images.td(),page_images.a( e.img( src=image_IDs[k]['bad_spectra'][j]),width=200,height=143, href=image_IDs[k]['bad_url_objid'][j]),page_images.td.close(),page_images.tr.close()
            page_images.tr(),page_images.td(),page_images.b('Class'),page_images.td.close(),page_images.td(str(uniquetarget_tr[0][image_IDs[k]['class']])),page_images.tr.close()
            page_images.tr(),page_images.td(),page_images.b('Predicted Class'),page_images.td.close(),page_images.td(str(uniquetarget_tr[0][image_IDs[k]['bad_result'][j]])),page_images.tr.close()
            page_images.tr(),page_images.td(),page_images.b('ObjID'),page_images.td.close(),page_images.td(str(image_IDs[k]['bad_ID'][j])),page_images.tr.close()
            page_images.tr(),page_images.td(),page_images.b('Redshift'),page_images.td.close(),page_images.td(str(image_IDs[k]['bad_specz'][j])),page_images.tr.close()
            page_images.table.close()
            
            page_images.table(border=1)
            page_images.tr(),page_images.th(""),page_images.th(uniquetarget_tr[0][0]),page_images.th(uniquetarget_tr[0][1]),page_images.th(uniquetarget_tr[0][2]),page_images.tr.close()
            page_images.tr(),page_images.td(),page_images.b("Probability"),page_images.td.close(),page_images.td(str(image_IDs[k]['bad_probs'][j][0])),page_images.td(str(image_IDs[k]['bad_probs'][j][1])),page_images.td(str(image_IDs[k]['bad_probs'][j][2])),page_images.tr.close()
            page_images.tr(),page_images.td(),page_images.b("Bias"),page_images.td.close(),page_images.td(str(image_IDs[k]['bad_tiresult'][j][1][0][0])),page_images.td(str(image_IDs[k]['bad_tiresult'][j][1][0][1])),page_images.td(str(image_IDs[k]['bad_tiresult'][j][1][0][2])),page_images.tr.close()
            page_images.table.close()
            page_images.p("Contributions to Probability")
            page_images.table(border=1)
            page_images.tr(),page_images.th(""),page_images.td("Values"),page_images.th(uniquetarget_tr[0][0]),page_images.th(uniquetarget_tr[0][1]),page_images.th(uniquetarget_tr[0][2]),page_images.tr.close()
            for i in range(len(feat_names)):
                page_images.tr()
                page_images.td(feat_names[i]),page_images.td(str(image_IDs[k]['bad_values'][j][i])),page_images.td(round(image_IDs[k]['bad_tiresult'][j][2][0][:,0][i],5)),page_images.td(round(image_IDs[k]['bad_tiresult'][j][2][0][:,1][i],5)),page_images.td(round(image_IDs[k]['bad_tiresult'][j][2][0][:,2][i],5))
                page_images.tr.close()
    
    html_file= open("images.html","w")
    html_file.write(page_images())
    html_file.close()