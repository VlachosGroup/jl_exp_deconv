# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 23:28:29 2019

@author: lansf
"""
from __future__ import absolute_import, division, print_function
import os
from jl_exp_deconv import IR_Results
from jl_exp_deconv import get_defaults
from jl_exp_deconv.plotting_tools import set_figure_settings
set_figure_settings('paper')
frequency_range, pure_data_path, mixture_data_path = get_defaults()
deconv = IR_Results(4,frequency_range,pure_data_path)
Figure_folder=os.path.join(os.path.expanduser("~"),'Downloads')
mixture_data_path_file = os.path.join(mixture_data_path,os.listdir(mixture_data_path)[4])
deconv.get_mixture_data(mixture_data_path)
print(deconv.get_predictions(deconv.MIXTURE_STANDARDIZED[0]))
deconv.get_mixture_figures(Figure_folder)