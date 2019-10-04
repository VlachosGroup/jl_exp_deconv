from __future__ import absolute_import, division, print_function
import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pkg_resources
#if __name__ == '__main__':
#    from due import due, Doi
#else:
#    from .due import due, Doi

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
    frequency_range = np.linspace(850,1850,num=501,endpoint=True)
    return frequency_range, pure_data_path, mixture_data_path

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
        self.NUM_TARGETS = len(PURE_FILES)
        self.PURE_DATA = PURE_DATA
        self.PURE_CONCENTRATIONS = PURE_CONCENTRATIONS
        self.PURE_FILES = PURE_FILES
        self.FREQUENCY_RANGE = frequency_range
        self.NUM_PURE_SPECTRA = NUM_PURE_SPECTRA
        self.PURE_STANDARDIZED = self.standardize_spectra(PURE_DATA)
    
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
               
    def standardize_spectra(self, DATA):
        """
        Returns standardize spectra by projecting the intensities onto the same
        set of frequencies.
        
        Parameters
        ----------
        DATA : list[numpy.ndarray] or numpy.ndarray
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
        NUM_TARGETS = len(DATA)
        NUM_SPECTRA = [len(i)-1 for i in DATA]
        STANDARDIZED_SPECTRA = []
        for i in range(NUM_TARGETS):
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
        matrix of pure spectra.
        
        Parameters
        ----------
        NUM_PCs : int
            The number of principal components of the spectra to keep.
                  
        Returns
        -------
        PC_loadings : numpy.ndarray
            The first loadings of the first $N$ principal components where $N$
            is equal to NUM_PCs.
        
        """
        _get_pure_single_spectra = self._get_pure_single_spectra
        pure_spectra, concentrations  = _get_pure_single_spectra()
        U, S, V = np.linalg.svd(pure_spectra, full_matrices=False)
        PC_loadings = V[:NUM_PCs]
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
        PC_loadings : numpy.ndarray
            The first loadings of the first $N$ principal components where $N$
            is equal to the number of pure-component species on which model is
            trained.
            
        PCs_2_concentrations : numpy.ndarray
            Regressed matrix to compute concentrations given the principal
            components of a mixed spectra.
        """
        IR_DECONV.__init__(self, frequency_range, pure_data_path)
        self.pca_loadings, self.PCs_2_concentrations = self._get_PCs_and_regressors(NUM_PCs)
        
    def get_mixture_data(self, mixture_data_path):
        """
        Gets the mixture data need to deconvolute
                  
        Parameters
        ----------
            
        mixture_data_path : str
            Directory or file where mixture data is stored.
            
        Attributes
        ----------
        PURE_DATA_IN_MIXTURE : numpy.ndarray
            The first loadings of the first $N$ principal components where $N$
            is equal to the number of pure-component species on which model is
            trained.
            
        PCs_2_concentrations : numpy.ndarray
            Regressed matrix to compute concentrations given the principal
            components of a mixed spectra.
            
        self.PURE_DATA_IN_MIXTURE = PURE_DATA_IN_MIXTURE
        self.MIXTURE_DATA = MIXTURE_DATA
        self.MIXTURE_CONCENTRATIONS = MIXTURE_CONCENTRATIONS
        self.MIXTURE_FILES = MIXTURE_FILES
        self.MIXTURE_NAMES = MIXTURE_NAMES
        self.NUM_MIXED = len(MIXTURE_FILES)
        self.MIXTURE_STANDARDIZED = self.standardize_spectra(MIXTURE_DATA) 
        self.PURE_IN_MIXTURE_STANDARDIZED = self.standardize_spectra(PURE_DATA_IN_MIXTURE)
        
        Returns
        -------
        MIXTURE_STANDARDIZED : numpy.ndarray
            Standardized mixed spectra contained in file(s)
        """
        PURE_FILES = self.PURE_FILES
        NUM_TARGETS = self.NUM_TARGETS
        MIXTURE_CONCENTRATIONS = []
        MIXTURE_NAMES = []
        MIXTURE_DATA = []
        PURE_DATA_IN_MIXTURE = []
        if os.path.isdir(mixture_data_path) == True:
            MIXTURE_FILES = os.listdir(mixture_data_path)
        elif os.path.isfile(mixture_data_path) == True:
            MIXTURE_FILES = [os.path.basename(mixture_data_path)]
        for file in MIXTURE_FILES:
            if os.path.isdir(mixture_data_path) == True:
                file_path = os.path.join(mixture_data_path,file)
            elif os.path.isfile(mixture_data_path) == True:
                file_path = mixture_data_path
            index_list = np.zeros(len(PURE_FILES),dtype=int)
            component = np.genfromtxt(file_path, delimiter=','\
                                  , skip_header=0,usecols=np.arange(2,2+NUM_TARGETS),max_rows=1\
                                  ,autostrip=True,dtype=str,replace_space='_')
            for i in range(component.size):
                component[i] = component[i].replace(' ','_')
                for count, ii in enumerate(PURE_FILES):
                    if component[i] in ii:
                        index_list[count] = i
            MIXTURE_NAMES.append(component[index_list])
            individual_spectra = np.loadtxt(file_path, delimiter=',', skiprows=2,usecols=[0]+np.arange(2,2+NUM_TARGETS).tolist()).T
            PURE_DATA_IN_MIXTURE.append(np.concatenate((np.array([individual_spectra[0]]),individual_spectra[1:][index_list]),axis=0))
            concentration = np.genfromtxt(file_path, delimiter=','\
                                  , skip_header=1,usecols=np.arange(2,2+NUM_TARGETS),max_rows=1\
                                  ,dtype=float)
            MIXTURE_CONCENTRATIONS.append(concentration[index_list])
            data = np.loadtxt(file_path, delimiter=',', skiprows=2,usecols=[0,1]).T
            MIXTURE_DATA.append(data)
        MIXTURE_STANDARDIZED = self.standardize_spectra(MIXTURE_DATA)
        self.PURE_DATA_IN_MIXTURE = PURE_DATA_IN_MIXTURE
        self.MIXTURE_DATA = MIXTURE_DATA
        self.MIXTURE_CONCENTRATIONS = MIXTURE_CONCENTRATIONS
        self.MIXTURE_FILES = MIXTURE_FILES
        self.MIXTURE_NAMES = MIXTURE_NAMES
        self.NUM_MIXED = len(MIXTURE_FILES)
        self.MIXTURE_STANDARDIZED = np.copy(MIXTURE_STANDARDIZED) 
        self.PURE_IN_MIXTURE_STANDARDIZED = self.standardize_spectra(PURE_DATA_IN_MIXTURE)
        return MIXTURE_STANDARDIZED
    
    def get_mixture_figures(self, figure_directory):
        """
        Returns principal component loadings of the spectra as well as the
        matrix that multiplies the principal components of a given mixed
        spectra to return.
                  
        Parameters
        ----------
        figure_directory : str
            Directory where figures should be saved.
                        
        
        Returns
        -------
        Figures describing the data        
        """
        self._visualize_data(figure_directory)
        self._get_results(figure_directory)
        
    def _visualize_data(self,figure_directory):
        NUM_TARGETS = self.NUM_TARGETS
        PURE_CONCENTRATIONS = self.PURE_CONCENTRATIONS
        PURE_FILES = self.PURE_FILES
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
        plt.figure(0, figsize=(7.2,5),dpi=400)
        #print('Comparing regression to pure species spectra')
        for i in range(NUM_TARGETS):
            fit_values = np.dot(CONCENTRATION_COEFFICIENTS[i], CONCENTRATIONS_2_PURE_SPECTRA[i]).flatten()
            plt.plot(PURE_STANDARDIZED[i].flatten()\
                     ,fit_values\
                     ,markers[i])   
        plt.plot((np.min(PURE_STANDARDIZED),np.max(PURE_STANDARDIZED)),(np.min(PURE_STANDARDIZED),np.max(PURE_STANDARDIZED)),'k',zorder=0)
        plt.legend([PURE_FILES[i][0:-9].replace('_',' ') for i in range(NUM_TARGETS)]+['Parity'])
        plt.xlabel('Experimental Pure Component Intensities')
        plt.ylabel('Regressed Intensities')
        if figure_directory == 'print':
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
        
    def deconvolute_spectra(self, spectra):
        NUM_TARGETS = self.NUM_TARGETS
        FREQUENCY_RANGE = self.FREQUENCY_RANGE
        _get_concentrations_2_pure_spectra = self._get_concentrations_2_pure_spectra
        _get_concentration_coefficients = self._get_concentration_coefficients
        get_predictions = self.get_predictions
        CONCENTRATIONS_2_PURE_SPECTRA = _get_concentrations_2_pure_spectra()
        predictions = get_predictions(spectra)
        deconvoluted_spectra = []
        for i in range(NUM_TARGETS):
            concentration_coefficients = _get_concentration_coefficients(predictions[:,i])
            deconvoluted_spectra.append(np.dot(concentration_coefficients,CONCENTRATIONS_2_PURE_SPECTRA[i]))
        reordered_spectra = []
        for i in range(spectra.shape[0]):
            reordered_spectra_i = np.zeros((NUM_TARGETS, FREQUENCY_RANGE.size))
            for ii in range(NUM_TARGETS):
                reordered_spectra_i[ii] = deconvoluted_spectra[ii][i]
            reordered_spectra.append(reordered_spectra_i)
        return reordered_spectra
    
    def _get_results(self,figure_directory='print'):
        self.plot_parity_plot(figure_directory)
        self.plot_deconvoluted_spectra(figure_directory)
        
    def plot_parity_plot(self,figure_directory='print'):
        NUM_TARGETS = self.NUM_TARGETS
        MIXTURE_NAMES = self.MIXTURE_NAMES
        MIXED_SPECTRA = self.MIXTURE_STANDARDIZED
        predictions = self.get_predictions(MIXED_SPECTRA)
        errors = self.get_95PI(MIXED_SPECTRA)
        True_value = np.array(self.MIXTURE_CONCENTRATIONS)
        Markers = ['o','s','D','^']
        Colors = ['orange','g','b','r']
        plt.figure(0, figsize=(7.2,5),dpi=400)
        for i in range(NUM_TARGETS):
            plt.plot(True_value[:,i], predictions[:,i],marker=Markers[i],color=Colors[i],linestyle='None')
            plt.errorbar(True_value[:,i], predictions[:,i], yerr=errors[:,i], xerr=None, fmt='none', ecolor='k',elinewidth=1,capsize=3)
            plt.errorbar(True_value[:,i], predictions[:,i], yerr=errors[:,i], xerr=None, fmt='none', ecolor='k', barsabove=True,elinewidth=1,capsize=3)
        plt.plot((np.min(True_value),np.max(True_value)),(np.min(True_value),np.max(True_value)),'k',zorder=0)
        #plt.legend(list(MIXTURE_NAMES[0]))
        plt.legend([i.replace('_',' ') for i in MIXTURE_NAMES[0]])
        plt.xlabel('Exeprimentally Measured Concentration')
        plt.ylabel('Predicted Concentration')
        
        print('R2 of mixed prediction: ' + str(self._get_r2()))
        print('RMSE of mixed prediction: ' + str(self._get_rmse()))
        print('Max Error mixed prediction: ' + str(self._get_max_error()))
        plt.tight_layout()
        if figure_directory == 'print':
            plt.show()
        else:
            plt.savefig(figure_directory+'/Model_Validation.png', format='png')
            plt.close()
    def plot_deconvoluted_spectra(self,figure_directory='print'):
        MIXTURE_FILES = self.MIXTURE_FILES
        FREQUENCY_RANGE = self.FREQUENCY_RANGE
        NUM_MIXED = self.NUM_MIXED
        NUM_TARGETS = self.NUM_TARGETS
        MIXTURE_NAMES = self.MIXTURE_NAMES
        MIXED_SPECTRA = self.MIXTURE_STANDARDIZED
        deconvolute_spectra = self.deconvolute_spectra
        comparison_spectra = self.standardize_spectra(self.PURE_DATA_IN_MIXTURE)
        deconvoluted_spectra = deconvolute_spectra(MIXED_SPECTRA)
        Colors = ['orange','g','b','r']
        for i in range(NUM_MIXED):
            plt.figure(i+1, figsize=(9.9,5),dpi=400)
            plt.plot(FREQUENCY_RANGE,MIXED_SPECTRA[i],'k')
            for ii in range(NUM_TARGETS):
                plt.plot(FREQUENCY_RANGE,deconvoluted_spectra[i][ii],color=Colors[ii],linestyle = '-')
            plt.plot(FREQUENCY_RANGE,np.sum(comparison_spectra[i],axis=0),'k--')
            for ii in range(NUM_TARGETS):
                plt.plot(FREQUENCY_RANGE,comparison_spectra[i][ii],color=Colors[ii],linestyle = '--')
            plt.legend(['Mixture Spectra']+[i.replace('_',' ') +' - deconvoluted' for i in MIXTURE_NAMES[0]]\
                       +['Summed Pure Component Spectra']+[i.replace('_',' ') +' - pure' for i in MIXTURE_NAMES[0]],ncol=2)
            plt.xlabel('Frequency [cm$^{-1}$]')
            plt.ylabel('Intensity')
            plt.ylim([0, MIXED_SPECTRA[i].max()*1.75])
            #plt.title(MIXTURE_FILES[i][:-4])
            plt.tight_layout()
            if figure_directory == 'print':
                plt.show()
            else:
                plt.savefig(figure_directory+'/'+MIXTURE_FILES[i][:-4]+'.png', format='png')
                plt.close()
                np.savetxt(figure_directory+'/'+MIXTURE_FILES[i][:-4]+'.csv',np.concatenate((self.FREQUENCY_RANGE.reshape((-1,1)),MIXED_SPECTRA[i].reshape((-1,1))\
                       ,deconvoluted_spectra[i].T,comparison_spectra[i].T),axis=1)\
                       ,delimiter=',',header='Frequency,Mixed_Spectra,'+'_Deconvoluted,'.join(MIXTURE_NAMES[0])\
                       +'_Deconvoluted,'+'_PURE_SPECTRA,'.join(MIXTURE_NAMES[0])+'_PURE_SPECTRA')
            
    def _get_r2(self):
        y_pred = self.get_predictions(self.MIXTURE_STANDARDIZED).flatten()
        y_true = np.array(self.MIXTURE_CONCENTRATIONS).flatten()
        SStot = np.sum((y_true-y_true.mean())**2)
        SSres = np.sum((y_true-y_pred)**2)
        return 1 - SSres/SStot
    
    def _get_rmse(self):
        y_pred = self.get_predictions(self.MIXTURE_STANDARDIZED).flatten()
        y_true = np.array(self.MIXTURE_CONCENTRATIONS).flatten()
        SSres = np.mean((y_true-y_pred)**2)
        return SSres**0.5
    def _get_max_error(self):
        y_pred = self.get_predictions(self.MIXTURE_STANDARDIZED).flatten()
        y_true = np.array(self.MIXTURE_CONCENTRATIONS).flatten()
        return np.array(y_pred-y_true)[np.argmax(np.abs(y_pred-y_true))]
    
    def get_predictions(self, spectra):
        spectra = np.copy(spectra)
        PCs_2_concentrations = self.PCs_2_concentrations
        pca_loadings = self.pca_loadings
        PCs = np.dot(spectra,pca_loadings.T)  
        predictions =  np.dot(PCs,PCs_2_concentrations)
        return predictions
    
    def get_95PI(self, spectra):
        spectra = np.copy(spectra)
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
        
        return stats.t.ppf(1-0.025,y_fit.shape[0]-pca_loadings.shape[0])*var_ynew**0.5
