#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  06 17:14:20 2020

@author: SEMBLANET Tom
"""

import pickle
import os
from datetime import datetime
from configparser import ConfigParser

from src.optimal_control import utils
from src.optimal_control.transcription import Transcription
from src.optimal_control.solver import IPOPT


class Optimization:
    """ `Optimization` class is the global structure that manages the optimization of an
            optimal control problem namely :
                    - The user-defined optimal control problem
                    - The transcription algorithm
                    - The NLP solver (IPOPT or SNOPT)

            Parameters
            ----------
            problem : Problem
                    Optimal-control problem defined by the user
            kwargs : dict - optional
                    Transcription and Optimization options dictionnary

            Attributes
            ----------
            problem : Problem
                    Optimal-control problem defined by the user
            transcription : Transcription
                    Transcription object
            solver : Solver
                    Solver object
            options : dict
                    Transcription and Optimization options dictionnary
            results : dict
                    Dictionnary containing the optimization results

    """

    def __init__(self, problem, **kwargs):
        """ Initizalization of the `Optimization` class """

        # Initialization of the `options` dictionnary
        self.options = self.set_options(**kwargs)

        # Storage of the `problem` object
        self.problem = problem
        self.problem.setup(transcription_method=self.options['tr_method'])

        # Initialization of the `transcription` object
        self.transcription = Transcription(
            problem=self.problem, options=self.options)

        # Initialization of the `solver` object
        self.solver = IPOPT(transcription=self.transcription) 

    def run(self):
        """ Launches the optimization of the transcribed optimal control problem using the optimizer
                choosen by the user (either IPOPT or SNOPT) """

        # Launch of the optimization process
        opt_st, opt_ct, opt_col_ct, opt_pr, opt_tm = self.solver.launch()

        # Once optimization is finished, we post-process the results
        self.post_process(opt_st, opt_ct, opt_col_ct, opt_pr, opt_tm)

    def post_process(self, opt_st, opt_ct, opt_col_ct, opt_pr, opt_tm):
        """ Saves the optimization results and manages the post-process

                Parameters
                ----------
                opt_st : ndarray
                        States matrix returned by the optimizer
                opt_ct : ndarray
                        Controls matrix returned by the optimizer
                opt_col_ct : ndarray
                        Controls at collocation points matrix return by the optimizer
                opt_pr : array
                        Free parameters array returned by the optimizer
                opt_tm : array
                        Time grid array return by the optimizer

        """

        # Initialization of a dictionnary containing all the results
        self.results = dict()

        # Storage of the optimal states, controls, free parameters and time
        self.results['opt_st'] = opt_st
        self.results['opt_ct'] = opt_ct
        self.results['opt_col_ct'] = opt_col_ct
        self.results['opt_pr'] = opt_pr
        self.results['opt_tm'] = opt_tm

        # Storage of the transcription and solver options
        self.results['options'] = self.options

        if self.options['plot_results']:
            self.plot_results()

        # Runs an explicit integration
        if self.options['explicit_integration']:
            self.explicit_integration()

        # Pickles the results
        if self.options['pickle_results']:
            self.pickle_results()

    def explicit_integration(self):
        """ Manages the explicit integration and stores the results
            in the `results` dictionnary """

        if self.options['tr_method'] in ['trapezoidal', 'hermite-simpson']:
            # If collocation method is used, we have to build the control(s) interpolation
            # function(s)
            self.transcription.tr_method.build_interpolation_func(
                self.results['opt_tm'], self.results['opt_ct'], self.results['opt_col_ct'])

        self.results['int_st'] = utils.explicit_integration(
            self.results['opt_tm'], self.results['opt_st'][:, 0], self.results['opt_ct'], \
            self.results['opt_col_ct'], self.results['opt_pr'], self.problem.dynamics, \
            self.transcription.tr_method.interpolate_ctrl)

    def pickle_results(self):
        """ Pickles the optimization results and stores the file in the current folder """

        # Storage of the pickle file
        date = datetime.now().strftime("%d:%m:%Y_%H:%M:%S")
        with open(self.transcription.options['name'] + "_" + str(date), 'wb') as file:
            pickle.dump(self.results, file)

    def plot_results(self):
        """ Plotting of the results """
        utils.classic_display(
            self.results['opt_tm'], self.results['opt_st'], self.results['opt_ct'])

    @staticmethod
    def set_options(**kwargs):
        """ Setting the transcription and optimization parameters

                Parameters
                ----------
                kwargs : dict - optional
                   Transcription and Optimization options dictionnary.
                   if not provided default options from `default.ini` file will be loaded

                Returns
                -------
                options : dict
                        Full Transcription and Optimization options dictionnary

        """

        options = dict()

        config = ConfigParser()

        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, "../init", 'default.ini')

        config.read(filename)

        options['name'] = kwargs['name'] if 'name' in kwargs.keys(
        ) else config.get('Generals', 'name')

        options['linear_solver'] = kwargs['linear_solver'] if 'linear_solver' in kwargs.keys(
        ) else config.get('Transcription', 'linear_solver')
        options['tr_method'] = kwargs['tr_method'] if 'tr_method' in kwargs.keys(
        ) else config.get('Transcription', 'tr_method')
        options['solver'] = kwargs['solver'] if 'solver' in kwargs.keys(
        ) else config.get('Transcription', 'solver')

        options['plot_results'] = kwargs['plot_results'] if 'plot_results' in kwargs.keys(
        ) else config.getboolean('Post_Process', 'plot_results')
        options['pickle_results'] = kwargs['pickle_results'] if 'pickle_results' in kwargs.keys(
        ) else config.getboolean('Post_Process', 'pickle_results')
        options['plot_physical_err'] = kwargs['plot_physical_err'] if 'plot_physical_err' in \
        kwargs.keys() else config.getboolean('Post_Process', 'plot_physical_err')
        options['explicit_integration'] = kwargs['explicit_integration'] if 'explicit_integration' \
        in kwargs.keys() else config.getboolean('Post_Process', 'explicit_integration')

        return options
