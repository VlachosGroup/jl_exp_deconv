from __future__ import absolute_import, division, print_function
import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pkg_resources

if __name__ == '__main__':
    import error_metrics
else:
    from . import error_metrics

#default values
data_path = pkg_resources.resource_filename(__name__, 'data/')

def get_defaults():
    """
    Returns default frequencies to project intensities onto as well as default
    paths for locations of the pure and mixture spectroscopic data.
    
    Returns
    -------
    frequency_range: numpy.ndarray
        Frequencies over which to project the intensities.
    
    pure_data_path : str
        Directory location where pure-component spectra are stored.
        
    mixture_data_path : str
        Directory location where mixed-component spectra are stored.
    
    """
    pure_data_path = os.path.join(data_path, 'pure_components/')
    mixture_data_path = os.path.join(data_path, 'mixed_components/')
    reaction_data_path = os.path.join(data_path, 'reaction/')
    frequency_range = np.linspace(850,1850,num=501,endpoint=True)
    return frequency_range, pure_data_path, mixture_data_path, reaction_data_path

class IR_DECONV:
    """Class for generating functions used to to deconvolute spectra"""
    def __init__(self, frequency_range, pure_data_path):
        """ 
        Parameters
        ----------
        frequency_range : numpy.narray
            Frequencies over which to project the intensities.

        pure_data_path : str
            Directory location where pure component spectra are stored.
        
        Attributes
        ----------
        NUM_TARGETS : int
            Number of different pure commponent species.
        PURE_DATA : list[numpy.ndarray]
            Original values of the of the experimental pure-component spectra.
            There is a separate numpy.ndarray for each pure-component. Each 
            array has shape $(m+1)$x$n$ where $m$ is the number of spectra for a
            pure component and $n$ is the number of discrete intensities
            sampled by the spectrometer. nump.ndarray[0] corresponds to the
            frequencies over which the intensities are measured.
            
        PURE_CONCENTRATIONS : list[numpy.ndarray]
            Concentrations (M) for each pure-component solution measured. There
            is a separate numpy.ndarry for each experimental pure-component
            and each array is of length $m$.
            
        PURE_FILES : list[str]
            Location to each file in pure_data_path.
            
        FREQUENCY_RANGE : numpy.ndarray
            Numpy array of frequencies to project each spectra onto.
            
        NUM_PURE_SPECTRA : list[int]
            Number of spectra for each pure-component.
            
        PURE_STANDARDIZED : list[numpy.ndarray]
            List containing standardized sets of pure spectra where each set
            of spectra is represented by a $m$x$n$ array where $m$
            is the the number of spectra for each pure-componet species and $n$
            is the length of FREQUENCY_RANGE.
            
        PURE_NAMES : list[str]
            List of pure species names.
        """
        PURE_CONCENTRATIONS = []
        PURE_DATA = []
        if os.path.isdir(pure_data_path) == True:
            PURE_FILES = os.listdir(pure_data_path)
        elif os.path.isfile(pure_data_path) == True:
            PURE_FILES = [os.path.basename(pure_data_path)]
        for component in PURE_FILES:
            if os.path.isdir(pure_data_path) == True:
                file_path = os.path.join(pure_data_path,component)
            elif os.path.isfile(pure_data_path) == True:
                file_path = pure_data_path
            data = np.loadtxt(file_path, delimiter=',', skiprows=1).T
            PURE_DATA.append(data)
            concentrations = np.genfromtxt(file_path, delimiter=','\
                                  , skip_header=0,usecols=np.arange(1,data.shape[0]),max_rows=1,dtype=float)
            PURE_CONCENTRATIONS.append(concentrations)
        NUM_PURE_SPECTRA = [len(i) for i in PURE_CONCENTRATIONS]
        PURE_NAMES = np.char.replace(PURE_FILES,'_DATA.csv','').tolist()
        self.NUM_TARGETS = len(PURE_FILES)
        self.PURE_DATA = PURE_DATA
        self.PURE_CONCENTRATIONS = PURE_CONCENTRATIONS
        self.PURE_FILES = PURE_FILES
        self.PURE_NAMES = PURE_NAMES
        self.FREQUENCY_RANGE = frequency_range
        self.NUM_PURE_SPECTRA = NUM_PURE_SPECTRA
        self.PURE_STANDARDIZED = self.get_standardized_spectra(PURE_DATA)
    
    def _get_pure_single_spectra(self):
        """
        Returns the pure spectra and concentrations in a format X and y, respectively,
        where X is all spectra and y is the corresponding concentration vectors.    
           
        Returns
        -------
        X : numpy.ndarray
            Array of concatenated pure-component spectra of dimensions $m$x$n$ 
            where $m$ is the number of spectra and $n$ is the number of
            discrete frequencies
        
        y : numpy.ndarray
            Array of concatenated pure-component concentrations of dimensions
            $m$x$n$ where $m$ is the number of spectra and $n$ the number of
            pure-component species.
        """
        NUM_TARGETS = self.NUM_TARGETS
        FREQUENCY_RANGE = self.FREQUENCY_RANGE
        PURE_STANDARDIZED = self.PURE_STANDARDIZED
        PURE_CONCENTRATIONS = self.PURE_CONCENTRATIONS
        NUM_PURE_SPECTRA = self.NUM_PURE_SPECTRA
        X = np.zeros((np.sum(NUM_PURE_SPECTRA),FREQUENCY_RANGE.size))
        y = np.zeros((np.sum(NUM_PURE_SPECTRA),NUM_TARGETS))
        for i in range(NUM_TARGETS):
            y[np.sum(NUM_PURE_SPECTRA[0:i],dtype='int'):np.sum(NUM_PURE_SPECTRA[0:i+1],dtype='int'),i] = PURE_CONCENTRATIONS[i]
            X[np.sum(NUM_PURE_SPECTRA[0:i],dtype='int'):np.sum(NUM_PURE_SPECTRA[0:i+1],dtype='int')] = PURE_STANDARDIZED[i]
        return X, y
               
    def get_standardized_spectra(self, DATA):
        """
        Returns standardize spectra by projecting the intensities onto the same
        set of frequencies.
        
        Parameters
        ----------
        DATA : list[numpy.ndarray]
            Spectra to be standardized. Must contain the experimental
            frequencies of the spectra to be standardized as the first entry
            of each ndarry and all following entries are the spectra.
            Each ndarray should there for be of shape (>1,n) where n is the
            number of frequencies sampled by the spectrometer.
           
        Returns
        -------
        STANDARDIZED_SPECTRA : list[numpy.ndarray]
            List containing standardized sets of spectra in DATA projected
            onto FREQUENCY_RANGE.
        """
        FREQUENCY_RANGE = self.FREQUENCY_RANGE
        if len(DATA[0].shape) == 1:
            DATA = [np.copy(DATA)]
        NUM_SPECTRA = [len(i)-1 for i in DATA]
        STANDARDIZED_SPECTRA = []
        for i in range(len(DATA)):
            if NUM_SPECTRA[i] == 1:
                STANDARDIZED_SPECTRA.append(np.zeros(FREQUENCY_RANGE.size))
                STANDARDIZED_SPECTRA[i] = np.interp(FREQUENCY_RANGE, DATA[i][0], DATA[i][1], left=None, right=None, period=None)
            else:
                STANDARDIZED_SPECTRA.append(np.zeros((NUM_SPECTRA[i],FREQUENCY_RANGE.size)))
                for ii in range(NUM_SPECTRA[i]):
                    STANDARDIZED_SPECTRA[i][ii] = np.interp(FREQUENCY_RANGE, DATA[i][0], DATA[i][ii+1], left=None, right=None, period=None)
        return STANDARDIZED_SPECTRA

    def _get_concentrations_2_pure_spectra(self):
        """
        Returns regressed parameters for computing pure component spectra 
        from individual concentrations
                  
        Returns
        -------
        CONCENTRATIONS_2_PURE_SPECTRA : list[numpy.ndarray]
            List of parameters to estimate pure-spectra given its concentration.
        """
        NUM_TARGETS = self.NUM_TARGETS
        PURE_SPECTRA = self.PURE_STANDARDIZED
        PURE_CONCENTRATIONS = self.PURE_CONCENTRATIONS
        _get_concentration_coefficients = self._get_concentration_coefficients
        CONCENTRATIONS_2_PURE_SPECTRA = []
        CONCENTRATION_COEFFICIENTS = []
        for i in range(NUM_TARGETS):
            concentration_coefficients = _get_concentration_coefficients(PURE_CONCENTRATIONS[i])
            CONCENTRATION_COEFFICIENTS.append(concentration_coefficients)
            concentrations_2_pure_spectra, res, rank, s = np.linalg.lstsq(concentration_coefficients, PURE_SPECTRA[i], rcond=None)
            CONCENTRATIONS_2_PURE_SPECTRA.append(concentrations_2_pure_spectra)
        return CONCENTRATIONS_2_PURE_SPECTRA
               
    def _get_PC_loadings(self,NUM_PCs):
        """
        Returns principal component loadings after performing SVD on the
        matrix of pure spectra where $pure-single_spectra = USV^T$
        
        Parameters
        ----------
        NUM_PCs : int
            The number of principal components of the spectra to keep.
            
        Attributes
        ----------
        TOTAL_EXPLAINED_VARIANCE : numpy.ndarray
            Total explained variance by the $n$ principal components where
            $n=NUM_PCs$
                  
        Returns
        -------
        PC_loadings : numpy.ndarray
            The first loadings of the first $N$ principal components where $N$
            is equal to NUM_PCs. $PC_loadings = V$ 
        
        """
        _get_pure_single_spectra = self._get_pure_single_spectra
        pure_spectra, concentrations  = _get_pure_single_spectra()
        U, S, V = np.linalg.svd(pure_spectra, full_matrices=False)
        PC_loadings = V[:NUM_PCs]
        self.TOTAL_EXPLAINED_VARIANCE = np.sum(S[:NUM_PCs]**2)/np.sum(S**2)
        return PC_loadings
    
    def _get_PCs_and_regressors(self,NUM_PCs):
        """
        Returns principal component loadings of the spectra as well as the
        matrix that multiplies the principal components of a given mixed
        spectra to return.
                  
        Parameters
        ----------
        NUM_PCs : int
            The number of principal components of the spectra to keep.
        
        Returns
        -------
        PC_loadings : numpy.ndarray
            The first loadings of the first $N$ principal components where $N$
            is equal to the number of pure-component species on which model is
            trained.
            
        PCs_2_concentrations : numpy.ndarray
            Regressed matrix to compute concentrations given the principal
            components of a mixed spectra.
        
        """
        _get_pure_single_spectra = self._get_pure_single_spectra
        _get_PC_loadings = self._get_PC_loadings
        pure_spectra, concentrations  = _get_pure_single_spectra()
        pca_loadings = _get_PC_loadings(NUM_PCs)
        PCs = np.dot(pure_spectra,pca_loadings.T)
        PCs_2_concentrations, res, rank, s = np.linalg.lstsq(PCs,concentrations,rcond=None)
        return pca_loadings, PCs_2_concentrations
    
    def _get_concentration_coefficients(self,concentrations):
        """
        Get coefficients used in computing the individual spectra given 
        their pure component conentrations. Also used in regressing parameters
        for computing pure component spectra.
        
        Parameters
        ----------
        concentrations : float, np.ndarray, or list
            The concentration(s) whose pure-component spectra must be computed
        
        Returns
        -------
        concentration_coefficients : numpy.ndarray
            set of coefficients for computing pure-component spectra given
            the concentration of that pure-component
        
        """
        concentration_coefficients = np.concatenate((np.ones_like(concentrations).reshape(-1,1), concentrations.reshape(-1,1), concentrations.reshape(-1,1)**2, concentrations.reshape(-1,1)**3),axis=1)            
        return concentration_coefficients
            

class IR_Results(IR_DECONV):
    """Class for deconvoluting experimental spectra whose intensity increaeses
       monotonically with concentration."""
    def __init__(self, NUM_PCs, frequency_range, pure_data_path):
        """ 
        Parameters
        ----------
        NUM_PCs : int
            The number of principal components of the spectra to keep.
        
        frequency_range : numpy.narray
            Frequencies over which to project the intensities.

        pure_data_path : str
            Directory location where pure component spectra are stored.
        
        Attributes
        ----------
        pca_loadings : numpy.ndarray
            The first loadings of the first $N$ principal components where $N$
            is equal to the number of pure-component species on which model is
            trained.
            
        PCs_2_concentrations : numpy.ndarray
            Regressed matrix to compute concentrations given the principal
            components of a mixed spectra.
        """
        IR_DECONV.__init__(self, frequency_range, pure_data_path)
        self.pca_loadings, self.PCs_2_concentrations = self._get_PCs_and_regressors(NUM_PCs)
        
    def set_mixture_data(self, mixture_data_path, contains_concentrations=True):
        """
        Instantiates the mixture data that needs to deconvoluted from the file(s)
        in mixture_data_path. A path to a folder of files, a single file, or
        a numpy array can be passed.
                  
        Parameters
        ----------
            
        mixture_data_path : str or numpy.array
            Directory or file where mixture data is stored. A numpy.array is
            also accepted.
            
        Attributes
        ----------
        PURE_DATA_IN_MIXTURE : list[numpy.ndarray]
            List containing unstandardized single-spectra data. Read from the
            mixture files by numpy.load.txt. Instantiated if
            contains_concentrations=True.
            
        MIXTURE_DATA : list[numpy.ndarray]
            List containing unstandardized mixture spectra directly from .csv 
            files. Read by numpy.loadtxt.
            
        MIXTURE_CONCENTRATIONS : list[numpy.ndarray]
            List containing pure-component concentrations. Instantiated if
            contains_concentrations=True.
            
        MIXTURE_FILES : list[str]
        List containing file(s) in mixture_data_path.
        
        PURE_DATA_IN_MIXTURE_STANDARDIZED : list[numpy.ndarray]
            List containing standardized pure-component spectra projected onto
            FREQUENCY_RANGE. Instantiated if contains_concentrations=True.
            
        MIXTURE_STANDARDIZED : numpy.ndarray
            List containing standardized mixture spectra projected onto
            FREQUENCY_RANGE. If data is from mixture(s) with concentration data
            then it is a 2-D array. If data is from reaction data it is either
            an array of objects which obejct is a numpy or it is a 3-D array if
            all reaction files have the same number of spectra.
            
        NUM_MIXED : int
            Length of MIXTURE_FILES
            
        MIXTURE_INFO : list[str]
            Data from the first row of the mixture.csv files if
            contains_concentrations=False
        
            
        """
        PURE_FILES = self.PURE_FILES
        NUM_TARGETS = self.NUM_TARGETS
        MIXTURE_DATA = []
        if contains_concentrations == True:
            MIXTURE_CONCENTRATIONS = []
            PURE_DATA_IN_MIXTURE = []
        else:
            MIXTURE_INFO = []
        if type(mixture_data_path) == str:
            if os.path.isdir(mixture_data_path) == True:
                MIXTURE_FILES = os.listdir(mixture_data_path)
            elif os.path.isfile(mixture_data_path) == True:
                MIXTURE_FILES = [os.path.basename(mixture_data_path)]
            else:
                try:
                    open(mixture_data_path)
                except:
                    print('mixture_data_path is not a valide file or directory')
                    raise
            for file in MIXTURE_FILES:
                if os.path.isdir(mixture_data_path) == True:
                    file_path = os.path.join(mixture_data_path,file)
                elif os.path.isfile(mixture_data_path) == True:
                    file_path = mixture_data_path
                if contains_concentrations == True:
                    index_list = np.zeros(len(PURE_FILES),dtype=int)
                    component = np.genfromtxt(file_path, delimiter=','\
                                          , skip_header=0,usecols=np.arange(2,2+NUM_TARGETS),max_rows=1\
                                          ,autostrip=True,dtype=str,replace_space='_')
                    for i in range(component.size):
                        component[i] = component[i].replace(' ','_')
                        for count, ii in enumerate(PURE_FILES):
                            if component[i].lower() in ii.lower():
                                index_list[count] = i
                    individual_spectra = np.loadtxt(file_path, delimiter=',', skiprows=2,usecols=[0]+np.arange(2,2+NUM_TARGETS).tolist()).T
                    PURE_DATA_IN_MIXTURE.append(np.concatenate((np.array([individual_spectra[0]]),individual_spectra[1:][index_list]),axis=0))
                    concentration = np.genfromtxt(file_path, delimiter=','\
                                          , skip_header=1,usecols=np.arange(2,2+NUM_TARGETS),max_rows=1\
                                          ,dtype=float)
                    MIXTURE_CONCENTRATIONS.append(concentration[index_list])
                    data = np.loadtxt(file_path, delimiter=',', skiprows=2,usecols=[0,1]).T
                else:
                    data = np.loadtxt(file_path, delimiter=',', skiprows=1).T
                    try:
                        mixture_info = np.loadtxt(file_path, delimiter=',', skiprows=0,max_rows=1,usecols=np.arange(1,len(data)),dtype=float)
                    except ValueError:
                        mixture_info = np.loadtxt(file_path, delimiter=',', skiprows=0,max_rows=1,usecols=np.arange(1,len(data)),dtype=str)
                    MIXTURE_INFO.append(mixture_info.reshape(-1))
                MIXTURE_DATA.append(data)
        else:
            MIXTURE_DATA = mixture_data_path
        MIXTURE_STANDARDIZED = self.get_standardized_spectra(MIXTURE_DATA)
        self.MIXTURE_DATA = MIXTURE_DATA
        self.NUM_MIXED = len(MIXTURE_STANDARDIZED)
        self.MIXTURE_STANDARDIZED = np.copy(MIXTURE_STANDARDIZED) 
        if type(mixture_data_path) == str:
            self.MIXTURE_FILES = MIXTURE_FILES
            if contains_concentrations == True:
                self.PURE_DATA_IN_MIXTURE = PURE_DATA_IN_MIXTURE
                self.MIXTURE_CONCENTRATIONS = MIXTURE_CONCENTRATIONS
                self.PURE_IN_MIXTURE_STANDARDIZED = self.get_standardized_spectra(PURE_DATA_IN_MIXTURE)
            else:
                self.MIXTURE_INFO = MIXTURE_INFO
            
    def get_mixture_figures(self, figure_directory):
        """
        Returns parity plot and visualizes input data for the case the case
        where concentrations in the mixture are known.
                  
        Parameters
        ----------
        figure_directory : str
            Directory where figures should be saved. In place of a directory,
            keywords 'fit' or 'print' can also be provided. These do not save
            the images but sends them to the gui; 'fit' adjusts the figure
            size to the screen it is being viewed on.
                        
        Returns
        -------
        Figures describing the data and results of the model.     
        """
        try:
            self.MIXTURE_CONCENTRATIONS is not None
        except:
            print('You must run IR_Results.set_mixture_data with\
                  conctains_concentrations=True before running this method')
            return
        self._visualize_data(figure_directory)
        self.plot_parity_plot(figure_directory)
        self.plot_deconvoluted_spectra(figure_directory)
        
    def get_reaction_figures(self, figure_directory='fit'):
        """
        Returns parity plot and visualizes input data for the case the case
        where concentrations in the mixture are unknown.
                  
        Parameters
        ----------
        figure_directory : str
            Directory where figures should be saved. In place of a directory,
            keywords 'fit' or 'print' can also be provided. These do not save
            the images but sends them to the gui; 'fit' adjusts the figure
            size to the screen it is being viewed on.
                        
        Returns
        -------
        Figures describing the data and results of the model.     
        """
        try:
            self.MIXTURE_INFO is not None
        except:
            print('You must run IR_Results.set_mixture_data with\
                  conctains_concentrations=False before running this method')
            raise
        NUM_TARGETS = self.NUM_TARGETS
        PURE_NAMES = self.PURE_NAMES
        MIXTURE_INFO = self.MIXTURE_INFO
        predictions = self.get_predictions(self.MIXTURE_STANDARDIZED)
        errors = self.get_95PI(self.MIXTURE_STANDARDIZED)
        if type(predictions) is not list:
            predictions = [predictions]
            errors = [errors]
        for ii in range(len(predictions)):
            
            Markers = ['o','s','D','^']
            Colors = ['orange','g','b','r']
            linestyle = ['-','--',':','-.']
            if figure_directory=='fit':
                plt.figure(ii+1)
            else:
                plt.figure(ii+1, figsize=(7.2,5),dpi=400)
            for i in range(NUM_TARGETS):
                plt.plot(MIXTURE_INFO[ii], predictions[ii][:,i],marker=Markers[i],color=Colors[i],linestyle=linestyle[i])
                plt.errorbar(MIXTURE_INFO[ii], predictions[ii][:,i], yerr=errors[ii][:,i], xerr=None, fmt='none', ecolor='k',elinewidth=1,capsize=3)
                plt.errorbar(MIXTURE_INFO[ii], predictions[ii][:,i], yerr=errors[ii][:,i], xerr=None, fmt='none', ecolor='k', barsabove=True,elinewidth=1,capsize=3)
            plt.legend([i.replace('_',' ') for i in PURE_NAMES])
            plt.xlabel('Time')
            plt.ylabel('Predicted Concentration')
            plt.tight_layout()
            if figure_directory in ['print','fit']:
                plt.show()
            else:
                plt.savefig(figure_directory+'/Model_Validation_v'+str(ii)+'.png', format='png')
                plt.close()
    
    def save_reaction_data(self, figure_directory='fit'):
        """
        Returns parity plot and visualizes input data for the case the case
        where concentrations in the mixture are unknown.
                  
        Parameters
        ----------
        figure_directory : str
            Directory where figures should be saved.
                        
        Returns
        -------
        Saves data describing predicted concentrations.     
        """
        try:
            self.MIXTURE_INFO is not None
        except:
            print('You must run IR_Results.set_mixture_data with\
                  conctains_concentrations=False before running this method')
            raise
        FREQUENCY_RANGE = self.FREQUENCY_RANGE
        PURE_NAMES = self.PURE_NAMES
        MIXTURE_INFO = self.MIXTURE_INFO
        MIXTURE_STANDARDIZED = self.MIXTURE_STANDARDIZED
        predictions = self.get_predictions(MIXTURE_STANDARDIZED)
        errors = self.get_95PI(MIXTURE_STANDARDIZED)
        deconvoluted_spectra = self.get_deconvoluted_spectra(MIXTURE_STANDARDIZED)
        if type(predictions) != list:
            predictions = [predictions]
            errors = [errors]
            deconvoluted_spectra = [deconvoluted_spectra]
            mixed_spectra = [MIXTURE_STANDARDIZED]
        else:
            mixed_spectra = MIXTURE_STANDARDIZED
        for count, array_info in enumerate(MIXTURE_INFO):
            data_to_save = np.concatenate((array_info.reshape((-1,1)),predictions[count],errors[count]),axis=1)
            Titles = np.concatenate((np.array(['Time']), PURE_NAMES, [i+'_errors' for i in PURE_NAMES]))
            new_data_to_save = np.concatenate((Titles.reshape(1,-1),data_to_save),axis=0)
            np.savetxt(figure_directory+'/concentration_data_v'+str(count)+'.csv',new_data_to_save,delimiter=',',fmt="%s")
            for count2, time in enumerate(array_info.reshape(-1,1)):
                data_to_save = np.concatenate((FREQUENCY_RANGE.reshape((-1,1)),mixed_spectra[count][count2].reshape((-1,1))\
                ,deconvoluted_spectra[count][count2].T),axis=1)
                Titles = np.concatenate((np.array(['Frequency','Mixed_Spectra']),PURE_NAMES))
                new_data_to_save = np.concatenate((Titles.reshape(1,-1),data_to_save),axis=0)
                np.savetxt(figure_directory+'/'+'deconvolution_v'+str(count)+'_t'+str(count2)+'_deconv'+'.csv'\
                              ,new_data_to_save,delimiter=',',fmt="%s")
        
    def _visualize_data(self,figure_directory):
        """
        Plots visualization of the input data.
                  
        Parameters
        ----------
        figure_directory : str
            Directory where figures should be saved. In place of a directory,
            keywords 'fit' or 'print' can also be provided. These do not save
            the images but sends them to the gui; 'fit' adjusts the figure
            size to the screen it is being viewed on.
                        
        Returns
        -------
        Figures describing the data and results of the model.     
        """
        NUM_TARGETS = self.NUM_TARGETS
        PURE_CONCENTRATIONS = self.PURE_CONCENTRATIONS
        PURE_NAMES = self.PURE_NAMES
        PURE_STANDARDIZED = self.PURE_STANDARDIZED
        PURE_IN_MIXTURE_STANDARDIZED = self.PURE_IN_MIXTURE_STANDARDIZED
        MIXTURE_STANDARDIZED = self.MIXTURE_STANDARDIZED
        _get_concentration_coefficients = self._get_concentration_coefficients
        _get_concentrations_2_pure_spectra = self._get_concentrations_2_pure_spectra
        PURE_IN_MIX_SUMMED = np.array([np.sum(i,axis=0) for i in PURE_IN_MIXTURE_STANDARDIZED])
        CONCENTRATION_COEFFICIENTS = []
        for i in range(NUM_TARGETS):
            concentration_coefficients = _get_concentration_coefficients(PURE_CONCENTRATIONS[i])
            CONCENTRATION_COEFFICIENTS.append(concentration_coefficients)
        CONCENTRATIONS_2_PURE_SPECTRA = _get_concentrations_2_pure_spectra()
        pure_flattend = PURE_IN_MIX_SUMMED.flatten()
        markers = ['o','s','D','^']
        if figure_directory=='fit':
            plt.figure(0)
        else:
            plt.figure(0, figsize=(7.2,5),dpi=400)
        #print('Comparing regression to pure species spectra')
        for i in range(NUM_TARGETS):
            fit_values = np.dot(CONCENTRATION_COEFFICIENTS[i], CONCENTRATIONS_2_PURE_SPECTRA[i]).flatten()
            plt.plot(PURE_STANDARDIZED[i].flatten()\
                     ,fit_values\
                     ,markers[i])   
        plt.plot((np.min(PURE_STANDARDIZED),np.max(PURE_STANDARDIZED)),(np.min(PURE_STANDARDIZED),np.max(PURE_STANDARDIZED)),'k',zorder=0)
        plt.legend([PURE_NAMES[i].replace('_',' ') for i in range(NUM_TARGETS)]+['Parity'])
        plt.xlabel('Experimental Pure Component Intensities')
        plt.ylabel('Regressed Intensities')
        if figure_directory in ['print','fit']:
            plt.show()
        else:
            plt.savefig(figure_directory+'/Regressed_vs_Experimental.png', format='png')
            plt.close()
        plt.figure(0, figsize=(7.2,5),dpi=400)
        plt.plot((np.min(MIXTURE_STANDARDIZED),np.max(MIXTURE_STANDARDIZED)),(np.min(MIXTURE_STANDARDIZED),np.max(MIXTURE_STANDARDIZED)),'k',zorder=0)
        plt.plot(MIXTURE_STANDARDIZED.flatten(),pure_flattend,'o')
        plt.xlabel('Mixture Intensities')
        plt.ylabel('Summed Pure\n Component Intensities')
        if figure_directory == 'print':
            plt.show()
        else:
            plt.savefig(figure_directory+'/Summed_vs_Mixed.png', format='png')
            plt.close()
        
    def get_deconvoluted_spectra(self, spectra):
        """
        Given standardized mixed spectra return the corresponding deconvolute
        and return the corresponding pure-component spectra
                  
        Parameters
        ----------
        spectra : numpy.ndarray
            Standardized mixed spectra. Usually this is 
            IR_Results.MIXTURE_STANDARDIZED.
                        
        Returns
        -------
        reordered_spectra : list[numpy.ndarray] or list[list[numpy.ndarray]]
            Pure-component spectra that make up the mixed spectra. If multiple
            mixed spectra are provided the numpy arrays are ordered
            alphabetically by IR_Results.MIXTURE_FILES. Each numpy array is ordered
            alphabetically by IR_Results.PURE_FILES.
        """
        spectra = np.copy(spectra)
        if len(spectra.shape)==1 and len(spectra[0].shape)==0:
            spectra = spectra.reshape(1,-1)
        elif len(spectra.shape)==1 or len(spectra.shape)==3:
            output_list = []
            for value in spectra:
                output_list.append(self.get_deconvoluted_spectra(value))
            return output_list
        NUM_TARGETS = self.NUM_TARGETS
        FREQUENCY_RANGE = self.FREQUENCY_RANGE
        _get_concentrations_2_pure_spectra = self._get_concentrations_2_pure_spectra
        _get_concentration_coefficients = self._get_concentration_coefficients
        get_predictions = self.get_predictions
        CONCENTRATIONS_2_PURE_SPECTRA = _get_concentrations_2_pure_spectra()
        predictions = get_predictions(spectra)
        deconvoluted_spectra = []
        reordered_spectra = []
        for i in range(NUM_TARGETS):
            concentration_coefficients = _get_concentration_coefficients(predictions[:,i])
            deconvoluted_spectra.append(np.dot(concentration_coefficients,CONCENTRATIONS_2_PURE_SPECTRA[i]))
        for i in range(spectra.shape[0]):
            reordered_spectra_i = np.zeros((NUM_TARGETS, FREQUENCY_RANGE.size))
            for ii in range(NUM_TARGETS):
                reordered_spectra_i[ii] = deconvoluted_spectra[ii][i]
            reordered_spectra.append(reordered_spectra_i)
        return reordered_spectra
           
    def plot_parity_plot(self,figure_directory='print'):
        """
        Returns parity plot for the case the case where concentrations in the 
        mixture are known.
                  
        Parameters
        ----------
        figure_directory : str
            Directory where figures should be saved. In place of a directory,
            keywords 'fit' or 'print' can also be provided. These do not save
            the images but sends them to the gui; 'fit' adjusts the figure
            size to the screen it is being viewed on.
                        
        Returns
        -------
        Parity Plot of predicted vs. actual concentrations.     
        """
        if self.MIXTURE_CONCENTRATIONS is None:
            print('You must run IR_Results.set_mixture_data with\
                  conctains_concentrations=True before running this method')
            raise
        NUM_TARGETS = self.NUM_TARGETS
        PURE_NAMES = self.PURE_NAMES
        MIXED_SPECTRA = self.MIXTURE_STANDARDIZED
        predictions = self.get_predictions(MIXED_SPECTRA)
        errors = self.get_95PI(MIXED_SPECTRA)
        True_value = np.array(self.MIXTURE_CONCENTRATIONS)
        Markers = ['o','s','D','^']
        Colors = ['orange','g','b','r']
        if figure_directory=='fit':
            plt.figure(0)
        else:
            plt.figure(0, figsize=(7.2,5),dpi=400)
        for i in range(NUM_TARGETS):
            plt.plot(True_value[:,i], predictions[:,i],marker=Markers[i],color=Colors[i],linestyle='None')
            plt.errorbar(True_value[:,i], predictions[:,i], yerr=errors[:,i], xerr=None, fmt='none', ecolor='k',elinewidth=1,capsize=3)
            plt.errorbar(True_value[:,i], predictions[:,i], yerr=errors[:,i], xerr=None, fmt='none', ecolor='k', barsabove=True,elinewidth=1,capsize=3)
        plt.plot((np.min(True_value),np.max(True_value)),(np.min(True_value),np.max(True_value)),'k',zorder=0)
        plt.legend([i.replace('_',' ') for i in PURE_NAMES])
        plt.xlabel('Exeprimentally Measured Concentration')
        plt.ylabel('Predicted Concentration')
        
        y_true = np.array(self.MIXTURE_CONCENTRATIONS).flatten()
        y_pred = self.get_predictions(self.MIXTURE_STANDARDIZED).flatten()
        print('R2 of mixed prediction: ' + str(error_metrics.get_r2(y_true, y_pred)))
        print('RMSE of mixed prediction: ' + str(error_metrics.get_rmse(y_true, y_pred)))
        print('Max Error mixed prediction: ' + str(error_metrics.get_max_error(y_true, y_pred)))
        plt.tight_layout()
        if figure_directory in ['print','fit']:
            plt.show()
        else:
            plt.savefig(figure_directory+'/Model_Validation.png', format='png')
            plt.close()
            
    def save_parity_data(self,figure_directory):
        """
        saves parity data for the case the case where concentrations in the 
        mixture are known.
                  
        Parameters
        ----------
        figure_directory : str
            Directory where data should be saved.
                        
        Returns
        -------
        saves parity plot.     
        """
        if self.MIXTURE_CONCENTRATIONS is None:
            print('You must run IR_Results.set_mixture_data with\
                  conctains_concentrations=True before running this method')
            raise
        PURE_NAMES = self.PURE_NAMES
        MIXTURE_FILES = self.MIXTURE_FILES
        MIXED_SPECTRA = self.MIXTURE_STANDARDIZED
        predictions = self.get_predictions(MIXED_SPECTRA)
        errors = self.get_95PI(MIXED_SPECTRA)
        True_value = np.array(self.MIXTURE_CONCENTRATIONS)
        data_to_save = np.concatenate((np.array(MIXTURE_FILES).reshape(-1,1),predictions,True_value,errors),axis=1)
        Titles = np.concatenate((np.array(['File_name']),[i+'_pred' for i in PURE_NAMES]\
                                 ,[i+'_real' for i in PURE_NAMES],[i+'_errors' for i in PURE_NAMES]))
        new_data_to_save = np.concatenate((Titles.reshape(1,-1),data_to_save),axis=0)
        np.savetxt(figure_directory+'/Model_Validation.csv',new_data_to_save,delimiter=',',fmt="%s")
            
    def plot_deconvoluted_spectra(self,figure_directory='print'):
        """
        Returns deconvoluted spectra plot for when concentrations in the
        mixture are known.
                  
        Parameters
        ----------
        figure_directory : str
            Directory where figures should be saved. In place of a directory,
            keywords 'fit' or 'print' can also be provided. These do not save
            the images but sends them to the gui; 'fit' adjusts the figure
            size to the screen it is being viewed on.
                        
        Returns
        -------
        Comparison plot of deconvolated spectra vs the spectra for the
        pure-species in the mixture
        """
        try:
            self.MIXTURE_CONCENTRATIONS is not None
        except:
            print('You must run IR_Results.set_mixture_data with\
                  conctains_concentrations=True before running this method')
            raise
        MIXTURE_FILES = self.MIXTURE_FILES
        FREQUENCY_RANGE = self.FREQUENCY_RANGE
        NUM_MIXED = self.NUM_MIXED
        NUM_TARGETS = self.NUM_TARGETS
        PURE_NAMES = self.PURE_NAMES
        MIXED_SPECTRA = self.MIXTURE_STANDARDIZED
        PURE_IN_MIXTURE_STANDARDIZED = self.PURE_IN_MIXTURE_STANDARDIZED
        get_deconvoluted_spectra = self.get_deconvoluted_spectra
        deconvoluted_spectra = get_deconvoluted_spectra(MIXED_SPECTRA)
        Colors = ['orange','g','b','r']
        for i in range(NUM_MIXED):
            if figure_directory=='fit':
                plt.figure(i+1, figsize=(9.9,5))
            else:
                plt.figure(i+1, figsize=(9.9,5),dpi=400)
            plt.plot(FREQUENCY_RANGE,MIXED_SPECTRA[i],'k')
            for ii in range(NUM_TARGETS):
                plt.plot(FREQUENCY_RANGE,deconvoluted_spectra[i][ii],color=Colors[ii],linestyle = '-')
            plt.plot(FREQUENCY_RANGE,np.sum(PURE_IN_MIXTURE_STANDARDIZED[i],axis=0),'k:')
            for ii in range(NUM_TARGETS):
                plt.plot(FREQUENCY_RANGE,PURE_IN_MIXTURE_STANDARDIZED[i][ii],color=Colors[ii],linestyle = ':')
            plt.legend(['Mixture Spectra']+[i.replace('_',' ') +' - deconvoluted' for i in PURE_NAMES]\
                       +['Summed Pure Component Spectra']+[i.replace('_',' ') +' - pure' for i in PURE_NAMES],ncol=2)
            plt.xlabel('Frequency [cm$^{-1}$]')
            plt.ylabel('Intensity')
            plt.ylim([0, MIXED_SPECTRA[i].max()*1.75])
            #plt.title(MIXTURE_FILES[i][:-4])
            plt.tight_layout()
            if figure_directory in ['print','fit']:
                plt.show()
            else:
                plt.savefig(figure_directory+'/'+MIXTURE_FILES[i][:-4]+'.png', format='png')
                plt.close()
                
    def save_deconvoluted_spectra(self,figure_directory='print'):
        """
        Saves deconvoluted spectra for when concentrations in the
        mixture are known.
                  
        Parameters
        ----------
        figure_directory : str
            Directory where figures should be saved.
                        
        Returns
        -------
        Saves deconvoluted and pure spectra for the species.
        """
        try:
            self.MIXTURE_CONCENTRATIONS is not None
        except:
            print('You must run IR_Results.set_mixture_data with\
                  conctains_concentrations=True before running this method')
            raise
        MIXTURE_FILES = self.MIXTURE_FILES
        FREQUENCY_RANGE = self.FREQUENCY_RANGE
        NUM_MIXED = self.NUM_MIXED
        PURE_NAMES = self.PURE_NAMES
        MIXED_SPECTRA = self.MIXTURE_STANDARDIZED
        PURE_IN_MIXTURE_STANDARDIZED = self.PURE_IN_MIXTURE_STANDARDIZED
        get_deconvoluted_spectra = self.get_deconvoluted_spectra
        deconvoluted_spectra = get_deconvoluted_spectra(MIXED_SPECTRA)
        for i in range(NUM_MIXED):
            data_to_save = np.concatenate((FREQUENCY_RANGE.reshape((-1,1)),MIXED_SPECTRA[i].reshape((-1,1))\
                                           ,deconvoluted_spectra[i].T,PURE_IN_MIXTURE_STANDARDIZED[i].T),axis=1)
            Titles = np.concatenate((np.array(['Frequency','Mixed_Spectra'])\
                                     ,[ii+'_deconvoluted' for ii in PURE_NAMES]\
                                     ,[ii+'_pure_spectra' for ii in PURE_NAMES]))
            new_data_to_save = np.concatenate((Titles.reshape(1,-1),data_to_save),axis=0)
            np.savetxt(figure_directory+'/'+MIXTURE_FILES[i][:-4]+'_deconv'+'.csv'\
                       ,new_data_to_save,delimiter=',',fmt="%s")
    
    def get_predictions(self, spectra):
        """
        Returns predicted concentrations of the pure-component species that
        make up the mixed spectra.
                  
        Parameters
        ----------
        spectra : numpy.ndarray
            Standardized mixed spectra. Usually this is 
            IR_Results.MIXTURE_STANDARDIZED.
                        
        Returns
        -------
        predictions : numpy.ndarray or list[numpy.ndarray]
            Predicted concentrations
        """
        spectra = np.copy(spectra)
        if len(spectra.shape)==1 and len(spectra[0].shape)==0:
            spectra = spectra.reshape(1,-1)
        elif len(spectra.shape)==1 or len(spectra.shape)==3:
            output_list = []
            for value in spectra:
                output_list.append(self.get_predictions(value))
            return output_list
        PCs_2_concentrations = self.PCs_2_concentrations
        pca_loadings = self.pca_loadings
        PCs = np.dot(spectra,pca_loadings.T)  
        predictions =  np.dot(PCs,PCs_2_concentrations)
        return predictions
    
    def get_95PI(self, spectra):
        """
        Returns prediction interval of the predicted concentrations.
                  
        Parameters
        ----------
        spectra : numpy.ndarray
            Standardized mixed spectra. Usually this is 
            IR_Results.MIXTURE_STANDARDIZED.
                        
        Returns
        -------
        prediction_interval : numpy.ndarray or list[numpy.ndarray]
            Prediction interval at the 95% confidence level such that 95% of
            error bars of the predicted concentrations should overlap the
            parity line.
        """
        spectra = np.copy(spectra)
        if len(spectra.shape)==1 and len(spectra[0].shape)==0:
            spectra = spectra.reshape(1,-1)
        elif len(spectra.shape)==1 or len(spectra.shape)==3:
            output_list = []
            for value in spectra:
                output_list.append(self.get_95PI(value))
            return output_list
        pure_spectra, pure_concentrations  = self._get_pure_single_spectra()
        pca_loadings = self.pca_loadings
        y_fit = self.get_predictions(spectra=pure_spectra)
        NUM_TARGETS = self.NUM_TARGETS
        Xnew = np.dot(spectra,pca_loadings.T)
        Xfit = np.dot(pure_spectra,pca_loadings.T)
        var_yfit = np.zeros(NUM_TARGETS)
        var_ynew = np.zeros((spectra.shape[0],NUM_TARGETS))
        for i in range(NUM_TARGETS):
            var_yfit[i] = np.var(pure_concentrations[:,i]-y_fit[:,i],ddof=pca_loadings.shape[0])
            var_estimators = np.linalg.inv(np.dot(Xfit.T,Xfit))*var_yfit[i] 
            for ii in range(spectra.shape[0]):
                x1 = np.dot(Xnew[ii],var_estimators)
                x2 = np.dot(x1,Xnew[ii].reshape(-1,1))[0]
                var_ynew[ii][i] = var_yfit[i]+x2
        prediction_interval = stats.t.ppf(1-0.025,y_fit.shape[0]-pca_loadings.shape[0])*var_ynew**0.5
        return prediction_interval