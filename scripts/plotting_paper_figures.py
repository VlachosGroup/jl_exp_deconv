# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 23:28:29 2019

@author: lansf
"""
from __future__ import absolute_import, division, print_function
import os
from pquad import IR_Results
from pquad import get_defaults
from pquad.plotting_tools import set_figure_settings
set_figure_settings('paper')
(frequency_range, pure_data_path, mixture_training_data
            , mixture_data_path, reaction_data_path) = get_defaults()
deconv = IR_Results(frequency_range, pure_data_path
                    , training_data_type='pure', regression_method='PLS'
                    , NUM_PCs=4)
Figure_folder=os.path.join(os.path.expanduser("~"), 'Downloads')
deconv.set_mixture_data(mixture_data_path)
deconv.get_mixture_figures(Figure_folder)

#deconv.set_mixture_data(reaction_data_path, contains_concentrations=False)
#x=deconv.get_predictions(deconv.MIXTURE_STANDARDIZED)
#deconv.get_reaction_figures(figure_directory=Figure_folder)
