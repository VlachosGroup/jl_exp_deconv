# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 23:28:29 2019

@author: lansf
"""
import numpy as np
from jl_exp_deconv import IR_Results
from jl_exp_deconv.plotting_tools import set_figure_settings
from ast import literal_eval

set_figure_settings('paper')
frequency_range = np.linspace(850,1850,num=501,endpoint=True)
pure_data_path = input('Please enter the directory to the pure-species data files: ')
mixture_data_path = input('Please enter the directory to the mixture data: ')
output_folder = input('Please enter the directory to the save the data: ')
True_or_False = input('Does the mixture data contain known concentrations?\
enter "True" or "False" without quotes. If True,\
a parity plot is made. If False, the data is considred reaction data.: ')
contains_concentrations = literal_eval(True_or_False)
deconv = IR_Results(4, frequency_range, pure_data_path)
deconv.set_mixture_data(mixture_data_path, contains_concentrations=contains_concentrations)
if contains_concentrations == True:
    deconv.get_mixture_figures(output_folder)
    data_to_save = np.concatenate((np.array(deconv.MIXTURE_FILES).reshape(-1,1),deconv.get_predictions(deconv.MIXTURE_STANDARDIZED)),axis=1)
    Titles = np.concatenate((np.array(['File_name']),deconv.PURE_NAMES))
    new_data_to_save = np.concatenate((Titles.reshape(1,-1),data_to_save),axis=0)
    np.savetxt(output_folder+'/Model_Validation.csv',new_data_to_save,delimiter=',',fmt="%s")
else:
    deconv.get_reaction_figures(output_folder)
    predictions = deconv.get_predictions(deconv.MIXTURE_STANDARDIZED)
    deconvoluted_spectra = deconv.get_deconvoluted_spectra(deconv.MIXTURE_STANDARDIZED)
    if type(predictions) != list:
        predictions = [predictions]
        deconvoluted_spectra = [deconvoluted_spectra]
        mixed_spectra = [deconv.MIXTURE_STANDARDIZED]
    else:
        mixed_spectra = deconv.MIXTURE_STANDARDIZED
    for count, array_info in enumerate(deconv.MIXTURE_INFO):
        data_to_save = np.concatenate((array_info.reshape(-1,1),predictions[count]),axis=1)
        Titles = np.concatenate((np.array(['Time']),deconv.PURE_NAMES))
        new_data_to_save = np.concatenate((Titles.reshape(1,-1),data_to_save),axis=0)
        np.savetxt(output_folder+'/concentration_data_v'+str(count)+'.csv',new_data_to_save,delimiter=',',fmt="%s")
        for count2, time in enumerate(array_info.reshape(-1,1)):
            np.savetxt(output_folder+'/'+'deconvolution_v'+str(count)+'_t'+str(count2)+'_deconv'+'.csv'\
                       ,np.concatenate((frequency_range.reshape(-1,1)\
                        ,mixed_spectra[count][count2].reshape(-1,1)\
                         ,deconvoluted_spectra[count][count2].T),axis=1)\
                          ,delimiter=',',header='Frequency,Mixed_Spectra,'+'_Deconvoluted,'.join(deconv.PURE_NAMES),fmt="%s")
                       
