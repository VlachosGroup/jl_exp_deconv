# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 23:28:29 2019

@author: lansf
"""
from jl_exp_deconv import IR_Results
from jl_exp_deconv import get_defaults
from jl_exp_deconv.plotting_tools import set_figure_settings
from ast import literal_eval

set_figure_settings('paper')
frequency_range, pure_data_path, mixture_data_path, reaction_data_path = get_defaults()
pure_data_path = input('Please enter the directory to the pure-species data files: ')
mixture_data_path = input('Please enter the directory to the mixture data: ')
output_folder = input('Please enter the directory to the save the data: ')
True_or_False = input('Does the mixture data contain known concentrations?\
                                enter "True" or "False" without quotes. If True,\
                                a parity plot is made. If False, the data is considred\
                                reaction data.: ')
contains_concentrations = literal_eval(True_or_False)
deconv = IR_Results(4, frequency_range, pure_data_path)
deconv.set_mixture_data(mixture_data_path, contains_concentrations=contains_concentrations)
if contains_concentrations == True:
    deconv.get_mixture_figures(output_folder)
else:
    deconv.get_reaction_figures(output_folder)
