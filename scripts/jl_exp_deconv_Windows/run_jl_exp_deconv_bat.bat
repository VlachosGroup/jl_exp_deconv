@echo off
echo # -*- coding: utf-8 -*->temp.py
echo ^""">>temp.py
echo Created on Thu Oct  3 23:28:29 2019>>temp.py
echo.>>temp.py
echo @author: lansf>>temp.py
echo ^""">>temp.py
echo import numpy as np>>temp.py
echo from jl_exp_deconv import get_defaults>>temp.py
echo from jl_exp_deconv import IR_Results>>temp.py
echo from jl_exp_deconv.plotting_tools import set_figure_settings>>temp.py
echo from ast import literal_eval>>temp.py
echo.>>temp.py
echo set_figure_settings('paper')>>temp.py
echo #frequency_range = np.linspace(850,1850,num=501,endpoint=True)>>temp.py
echo frequency_range, pure_data_path, mixture_data_path_default, reaction_data_path_default = get_defaults()>>temp.py
echo pca_to_keep = ^4>>temp.py
echo use_your_own = input('Do you want to use your own pure-data? Responds "yes" or "no" without quotes. \>>temp.py
echo Respond no if you want to use the default pure data to train the model.: ')>>temp.py
echo if use_your_own.lower() in ['yes', 'y']:>>temp.py
echo     pure_data_path = input('Please enter the directory to the pure-species data file: ')>>temp.py
echo     frequency_start = input('Please enter the lowest frequency to consider: ')>>temp.py
echo     frequency_end = input('Please enter the highest frequency to consider: ')>>temp.py
echo     pca_to_keep = input('Please enter the number of principal componets in the spectra to keep. \>>temp.py
echo     A good starting number is the number of pure-components: ')>>temp.py
echo     frequency_range = np.linspace(float(frequency_start),float(frequency_end),num=501,endpoint=True)>>temp.py
echo mixture_data_path = input('Please enter the directory to the mixture data: ')>>temp.py
echo output_folder = input('Please enter the directory to the save the data: ')>>temp.py
echo True_or_False = input('Does the mixture data contain known concentrations? \>>temp.py
echo Enter "True" or "False" without quotes. If True,\>>temp.py
echo a parity plot is made. If False, the data is considred reaction data.: ')>>temp.py
echo if True_or_False.lower() in ['yes', 'y']:>>temp.py
echo     True_or_False = 'True'>>temp.py
echo elif True_or_False.lower() in ['no', 'n']:>>temp.py
echo     True_or_False = 'False'>>temp.py
echo contains_concentrations = literal_eval(True_or_False)>>temp.py
echo deconv = IR_Results(int(pca_to_keep), frequency_range, pure_data_path)>>temp.py
echo deconv.set_mixture_data(mixture_data_path, contains_concentrations=contains_concentrations)>>temp.py
echo if contains_concentrations == True:>>temp.py
echo     deconv.get_mixture_figures(output_folder)>>temp.py
echo     deconv.save_parity_data(output_folder)>>temp.py
echo     deconv.save_deconvoluted_spectra(output_folder)>>temp.py
echo else:>>temp.py
echo     deconv.get_reaction_figures(output_folder)>>temp.py
echo     deconv.save_reaction_data(output_folder)>>temp.py


TIMEOUT 1
python temp.py
pause