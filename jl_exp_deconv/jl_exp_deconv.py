from __future__ import absolute_import, division, print_function
import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pkg_resources
if __name__ == '__main__':
    from due import due, Doi
else:
    from .due import due, Doi

#__all__ = ["Model", "Fit", "opt_err_func", "transform_data", "cumgauss"]


# Use duecredit (duecredit.org) to provide a citation to relevant work to
# be cited. This does nothing, unless the user has duecredit installed,
# And calls this with duecredit (as in `python -m duecredit script.py`):
due.cite(Doi("10.1167/13.9.30"),
         description="Template project for small scientific Python projects",
         tags=["reference-implementation"],
         path='jl_exp_deconv')

#default values
data_path = pkg_resources.resource_filename(__name__, 'data/')
pure_data_path = os.path.join(data_path, 'pure_components/')
mixture_data_path = os.path.join(data_path, 'mixed_components/')
frequency_range = np.linspace(850,1850,num=501,endpoint=True)

class IR_DECONV:
    def __init__(self, frequency_range=frequency_range, pure_data_path=pure_data_path):
        """ Initialize a model object.

        Parameters
        ----------
        data : Pandas DataFrame
            Data from a subjective contrast judgement experiment

        func : callable, optional
            A function that relates x and y through a set of parameters.
            Default: :func:`cumgauss`
        """
        PURE_CONCENTRATIONS = []
        PURE_DATA = []
        PURE_FILES = os.listdir(pure_data_path)
        for component in PURE_FILES:
            concentration = np.genfromtxt(pure_data_path + component, delimiter=','\
                                  , skip_header=0,usecols=np.arange(1,8),max_rows=1,dtype=float)
            PURE_CONCENTRATIONS.append(concentration)
            data = np.loadtxt(pure_data_path + component, delimiter=',', skiprows=1).T
            PURE_DATA.append(data)
        NUM_CONCENTRATIONS = [len(i) for i in PURE_CONCENTRATIONS]
        self.PURE_DATA = PURE_DATA
        self.PURE_CONCENTRATIONS = PURE_CONCENTRATIONS
        self.PURE_FILES = PURE_FILES
        self.FREQUENCY_RANGE = frequency_range
        self.NUM_CONCENTRATIONS = NUM_CONCENTRATIONS
        self.PURE_STANDARDIZED = self.standardize_spectra(PURE_DATA) 
    
    def _get_pure_single_spectra(self):
        """
        The cumulative Gaussian at x, for the distribution with mean mu and
        standard deviation sigma.
    
        Parameters
        ----------
        x : float or array
           The values of x over which to evaluate the cumulative Gaussian function
    
        mu : float
           The mean parameter. Determines the x value at which the y value is 0.5
    
        sigma : float
           The variance parameter. Determines the slope of the curve at the point
           of Deflection
    
        Returns
        -------
    
        g : float or array
            The cumulative gaussian with mean $\\mu$ and variance $\\sigma$
            evaluated at all points in `x`.
    
        Notes
        -----
        Based on:
        http://en.wikipedia.org/wiki/Normal_distribution#Cumulative_distribution_function
    
        The cumulative Gaussian function is defined as:
    
        .. math::
    
            \\Phi(x) = \\frac{1}{2} [1 + erf(\\frac{x}{\\sqrt{2}})]
    
        Where, $erf$, the error function is defined as:
    
        .. math::
    
            erf(x) = \\frac{1}{\\sqrt{\\pi}} \\int_{-x}^{x} e^{t^2} dt
        """
        NUM_TARGETS = self.NUM_TARGETS
        FREQUENCY_RANGE = self.FREQUENCY_RANGE
        PURE_STANDARDIZED = self.PURE_STANDARDIZED
        PURE_CONCENTRATIONS = self.PURE_CONCENTRATIONS
        NUM_CONCENTRATIONS = self.NUM_CONCENTRATIONS
        X = np.zeros((np.sum(NUM_CONCENTRATIONS),FREQUENCY_RANGE.size))
        y = np.zeros((np.sum(NUM_CONCENTRATIONS),NUM_TARGETS))
        for i in range(NUM_TARGETS):
            y[np.sum(NUM_CONCENTRATIONS[0:i],dtype='int'):np.sum(NUM_CONCENTRATIONS[0:i+1],dtype='int'),i] = PURE_CONCENTRATIONS[i]
            X[np.sum(NUM_CONCENTRATIONS[0:i],dtype='int'):np.sum(NUM_CONCENTRATIONS[0:i+1],dtype='int')] = PURE_STANDARDIZED[i]
        return X, y
               
    def _standardize_spectra(self, DATA):
        """
        Standardize spectra to the same frequency discritization
        """
        if len(DATA[0].shape) == 1:
            DATA = [DATA]
        FREQUENCY_RANGE = self.FREQUENCY_RANGE
        NUM_TARGETS = len(DATA)
        NUM_CONCENTRATIONS = [len(i)-1 for i in DATA]
        STANDARDIZED_SPECTRA = []
        for i in range(NUM_TARGETS):
            if NUM_CONCENTRATIONS[i] == 1:
                STANDARDIZED_SPECTRA.append(np.zeros(FREQUENCY_RANGE.size))
                STANDARDIZED_SPECTRA[i] = np.interp(FREQUENCY_RANGE, DATA[i][0], DATA[i][1], left=None, right=None, period=None)
            else:
                STANDARDIZED_SPECTRA.append(np.zeros((NUM_CONCENTRATIONS[i],FREQUENCY_RANGE.size)))
                for ii in range(NUM_CONCENTRATIONS[i]):
                    STANDARDIZED_SPECTRA[i][ii] = np.interp(FREQUENCY_RANGE, DATA[i][0], DATA[i][ii+1], left=None, right=None, period=None)
        return STANDARDIZED_SPECTRA

    def _get_concentrations_2_pure_spectra(self):
        """
        Get regressed parameters for computing pure component spectra from individual concentration
        and coefficients F(concentration) which are used in the regression
        """
        NUM_TARGETS = self.NUM_TARGETS
        PURE_SPECTRA = self.PURE_STANDARDIZED
        PURE_CONCENTRATIONS = self.PURE_CONCENTRATIONS
        CONCENTRATIONS_2_PURE_SPECTRA = []
        CONCENTRATION_COEFFICIENTS = []
        for i in range(NUM_TARGETS):
            concentration_coefficients = self.get_concentration_coefficients(PURE_CONCENTRATIONS[i])
            CONCENTRATION_COEFFICIENTS.append(concentration_coefficients)
            concentrations_2_pure_spectra, res, rank, s = np.linalg.lstsq(concentration_coefficients, PURE_SPECTRA[i], rcond=None)
            CONCENTRATIONS_2_PURE_SPECTRA.append(concentrations_2_pure_spectra)
        return CONCENTRATIONS_2_PURE_SPECTRA
               
    def _get_PC_loadings(self):
        """
        Get regressed parameters for computing mixed spectra given individual conentrations.
        Used in regressing concentration given the mixed spectra.
        """
        NUM_TARGETS = self.NUM_TARGETS
        pure_spectra, concentrations  = self.get_pure_single_spectra()

        U, S, V = np.linalg.svd(pure_spectra, full_matrices=False)
        PC_loadings = V[:NUM_TARGETS]
        return PC_loadings
    
    def _get_PCs_2_concentrations(self):
        """
        Get regressed parameters for computing mixed spectra given individual conentrations.
        Used in regressing concentration given the mixed spectra.
        """
        pure_spectra, concentrations  = self._get_pure_single_spectra()
        pca_components = self._get_PC_loadings()
        PCs = np.dot(pure_spectra,pca_components.T)
        PCs_2_concentrations, res, rank, s = np.linalg.lstsq(PCs,concentrations,rcond=None)
        return PCs_2_concentrations
    
    def _get_concentration_coefficients(self,concentrations):
        """
        Get coefficients used in computing the individual spectra given their pure component conentrations.
        Also used in regressing paratmers for computing pure component spectra.
        """
        concentration_coefficients = np.concatenate((np.ones_like(concentrations).reshape(-1,1), concentrations.reshape(-1,1), concentrations.reshape(-1,1)**2, concentrations.reshape(-1,1)**3),axis=1)            
        return concentration_coefficients
            

class IR_Results(IR_DECONV):
    def __init__(self, dictionary=None, frequency_range = frequency_range\
        , pure_data_path=pure_data_path, mixture_data_path=mixture_data_path):
        IR_DECONV.__init__(self, frequency_range)
        PURE_FILES = self.PURE_FILES
        MIXTURE_CONCENTRATIONS = []
        MIXTURE_NAMES = []
        MIXTURE_DATA = []
        PURE_DATA_IN_MIXTURE = []
        MIXTURE_FILES = os.listdir(mixture_data_path)
        for file in MIXTURE_FILES:
            index_list = np.zeros(len(PURE_FILES),dtype=int)
            component = np.genfromtxt(mixture_data_path + file, delimiter=','\
                                  , skip_header=0,usecols=np.arange(2,6),max_rows=1\
                                  ,autostrip=True,dtype=str,replace_space='_')
            for i in range(component.size):
                component[i] = component[i].replace(' ','_')
                for count, ii in enumerate(PURE_FILES):
                    if component[i] in ii:
                        index_list[count] = i
            MIXTURE_NAMES.append(component[index_list])
            individual_spectra = np.loadtxt(mixture_data_path + file, delimiter=',', skiprows=2,usecols=[0]+np.arange(2,6).tolist()).T
            PURE_DATA_IN_MIXTURE.append(np.concatenate((np.array([individual_spectra[0]]),individual_spectra[1:][index_list]),axis=0))
            concentration = np.genfromtxt(mixture_data_path + file, delimiter=','\
                                  , skip_header=1,usecols=np.arange(2,6),max_rows=1\
                                  ,dtype=float)
            MIXTURE_CONCENTRATIONS.append(concentration[index_list])
            data = np.loadtxt(mixture_data_path + file, delimiter=',', skiprows=2,usecols=[0,1]).T
            MIXTURE_DATA.append(data)
        self.PURE_DATA_IN_MIXTURE = PURE_DATA_IN_MIXTURE
        self.MIXTURE_DATA = MIXTURE_DATA
        self.MIXTURE_CONCENTRATIONS = MIXTURE_CONCENTRATIONS
        self.MIXTURE_FILES = MIXTURE_FILES
        self.MIXTURE_NAMES = MIXTURE_NAMES
        self.NUM_TARGETS = len(PURE_FILES)
        self.NUM_MIXED = len(MIXTURE_FILES)
        self.MIXTURE_STANDARDIZED = self.standardize_spectra(MIXTURE_DATA) 
        self.PURE_IN_MIXTURE_STANDARDIZED = self.standardize_spectra(PURE_DATA_IN_MIXTURE)
        
    def visualize_data(self):
        NUM_TARGETS = self.NUM_TARGETS
        CONCENTRATION_COEFFICIENTS = self.CONCENTRATION_COEFFICIENTS
        CONCENTRATIONS_2_PURE_SPECTRA = self.CONCENTRATIONS_2_PURE_SPECTRA
        PURE_FILES = self.PURE_FILES
        PURE_STANDARDIZED = self.PURE_STANDARDIZED
        PURE_IN_MIX_SUMMED = np.array([np.sum(i,axis=0) for i in self.PURE_IN_MIXTURE_STANDARDIZED])
        MIX_STANDARDIZED = np.array(self.MIXTURE_STANDARDIZED)
        pure_flattend = PURE_IN_MIX_SUMMED.flatten()
        
        """
        #correction summed to mixed
        SUMMED_2_MIXED = self._regress_summed_2_mixed()
        num_coefficients = np.arange(NUM_TARGETS).sum()
        MIXTURE_CONCENTRATIONS = np.array(self.MIXTURE_CONCENTRATIONS)
        concentration_coefficients_shift_mix = np.zeros((MIXTURE_CONCENTRATIONS.shape[0],num_coefficients))
        for count, two_spectra in enumerate(itertools.combinations(range(NUM_TARGETS),2)):
            concentration_coefficients_shift_mix[:,count] = MIXTURE_CONCENTRATIONS[:,two_spectra[0]]*MIXTURE_CONCENTRATIONS[:,two_spectra[1]]
        pure_flattend += np.dot(concentration_coefficients_shift_mix,SUMMED_2_MIXED).flatten()
        """
        
        markers = ['o','s','D','^']
        plt.figure(0, figsize=(7.2,5),dpi=400)
        #print('Comparing regression to pure species spectra')
        for i in range(NUM_TARGETS):
            fit_values = np.dot(CONCENTRATION_COEFFICIENTS[i], CONCENTRATIONS_2_PURE_SPECTRA[i]).flatten()
            plt.plot(PURE_STANDARDIZED[i].flatten()\
                     ,fit_values\
                     ,markers[i])
            #print('std: ' + str(np.std(fit_values-PURE_STANDARDIZED[i].flatten())))
            #print('RMSE: ' + str(rmse(PURE_STANDARDIZED[i].flatten(),fit_values)))
            #print('mean: ' + str(np.mean(fit_values-PURE_STANDARDIZED[i].flatten())))    
        plt.plot((np.min(PURE_STANDARDIZED),np.max(PURE_STANDARDIZED)),(np.min(PURE_STANDARDIZED),np.max(PURE_STANDARDIZED)),'k',zorder=0)
        plt.legend([PURE_FILES[i][0:-9].replace('_',' ') for i in range(NUM_TARGETS)]+['Parity'])
        plt.xlabel('Experimental Pure Component Intensities')
        plt.ylabel('Regressed Intensities')
        plt.savefig('Figures/Regressed_vs_Experimental.png', format='png')
        plt.close()
        plt.figure(0, figsize=(7.2,5),dpi=400)
        plt.plot((np.min(MIX_STANDARDIZED),np.max(MIX_STANDARDIZED)),(np.min(MIX_STANDARDIZED),np.max(MIX_STANDARDIZED)),'k',zorder=0)
        plt.plot(MIX_STANDARDIZED.flatten(),pure_flattend,'o')
        plt.xlabel('Mixture Intensities')
        plt.ylabel('Summed Pure\n Component Intensities')
        plt.savefig('Figures/Summed_vs_Mixed.png', format='png')
        plt.close()
        
    def deconvolute_spectra(self):
        NUM_TARGETS = self.NUM_TARGETS
        NUM_MIXED = self.NUM_MIXED
        FREQUENCY_RANGE = self.FREQUENCY_RANGE
        CONCENTRATIONS_2_PURE_SPECTRA = self.CONCENTRATIONS_2_PURE_SPECTRA
        predictions = self.get_predictions()
        deconvoluted_spectra = []
        for i in range(NUM_TARGETS):
            concentration_coefficients = self.get_concentration_coefficients(predictions[:,i])
            deconvoluted_spectra.append(np.dot(concentration_coefficients,CONCENTRATIONS_2_PURE_SPECTRA[i]))
        reordered_spectra = []
        for i in range(NUM_MIXED):
            reordered_spectra_i = np.zeros((NUM_TARGETS, FREQUENCY_RANGE.size))
            for ii in range(NUM_TARGETS):
                reordered_spectra_i[ii] = deconvoluted_spectra[ii][i]
            reordered_spectra.append(reordered_spectra_i)
        return reordered_spectra
    
    def get_results(self):
        MIXTURE_FILES = self.MIXTURE_FILES
        FREQUENCY_RANGE = self.FREQUENCY_RANGE
        NUM_MIXED = self.NUM_MIXED
        NUM_TARGETS = self.NUM_TARGETS
        MIXTURE_NAMES = self.MIXTURE_NAMES
        MIXED_SPECTRA = np.array(self.MIXTURE_STANDARDIZED)
        deconvoluted_spectra = self.deconvolute_spectra()
        comparison_spectra = self.standardize_spectra(self.PURE_DATA_IN_MIXTURE)
        predictions = self.get_predictions()
        errors = self.get_95PI()
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
        
        print('R2 of mixed prediction: ' + str(self.get_r2()))
        print('RMSE of mixed prediction: ' + str(self.get_rmse()))
        print('Max Error mixed prediction: ' + str(self.get_max_error()))
        plt.tight_layout()
        plt.savefig('Figures/Model_Validation.png', format='png')
        plt.close()

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
            plt.savefig('Figures/'+MIXTURE_FILES[i][:-4]+'.png', format='png')
            plt.close()
            np.savetxt('Figures/'+MIXTURE_FILES[i][:-4]+'.csv',np.concatenate((self.FREQUENCY_RANGE.reshape((-1,1)),MIXED_SPECTRA[i].reshape((-1,1))\
                       ,deconvoluted_spectra[i].T,comparison_spectra[i].T),axis=1)\
                       ,delimiter=',',header='Frequency,Mixed_Spectra,'+'_Deconvoluted,'.join(MIXTURE_NAMES[0])\
                       +'_Deconvoluted,'+'_PURE_SPECTRA,'.join(MIXTURE_NAMES[0])+'_PURE_SPECTRA')
            
    def get_r2(self):
        y_pred = self.get_predictions().flatten()
        y_true = np.array(self.MIXTURE_CONCENTRATIONS).flatten()
        SStot = np.sum((y_true-y_true.mean())**2)
        SSres = np.sum((y_true-y_pred)**2)
        return 1 - SSres/SStot
    
    def get_rmse(self):
        y_pred = self.get_predictions().flatten()
        y_true = np.array(self.MIXTURE_CONCENTRATIONS).flatten()
        SSres = np.mean((y_true-y_pred)**2)
        return SSres**0.5
    def get_max_error(self):
        y_pred = self.get_predictions().flatten()
        y_true = np.array(self.MIXTURE_CONCENTRATIONS).flatten()
        return np.array(y_pred-y_true)[np.argmax(np.abs(y_pred-y_true))]
    
    def get_predictions(self, spectra = None):
        #if self.NN is not None:
           #NN = self.NN
            #y_pred = NN.predict(self.MIXTURE_STANDARDIZED)
        #else:
        if spectra is None:
            spectra = np.array(self.MIXTURE_STANDARDIZED)
        PCs_2_concentrations = self._get_PCs_2_concentrations()
        pca_components = self._get_PC_loadings()
        #CONCENTRATIONS_2_MIXED = self._regress_mixed_spectra()
        PCs = np.dot(spectra,pca_components.T)
        #predictions, res, rank, s = np.linalg.lstsq(CONCENTRATIONS_2_MIXED,X_fit.T,rcond=None)  
        predictions =  np.dot(PCs,PCs_2_concentrations)
        #y_pred = predictions.T[:,0:self.NUM_TARGETS]
        return predictions
    
    def get_95PI(self):
        pure_spectra, pure_concentrations  = self.get_pure_single_spectra()
        pca_components = self._get_PC_loadings()
        y_fit = self.get_predictions(spectra=pure_spectra)
        mixed_spectra = np.array(self.MIXTURE_STANDARDIZED)
        NUM_TARGETS = self.NUM_TARGETS
        #variance of nonzero fit
        #variance of nonzero values considering each set of estimators for fitting the species concentrations is regressed separately
        Xnew = np.dot(mixed_spectra,pca_components.T)
        Xfit = np.dot(pure_spectra,pca_components.T)
        var_yfit = np.zeros(NUM_TARGETS)
        var_ynew = np.zeros((mixed_spectra.shape[0],NUM_TARGETS))
        for i in range(NUM_TARGETS):
            var_yfit[i] = np.var(pure_concentrations[:,i]-y_fit[:,i],ddof=pca_components.shape[0])
            var_estimators = np.linalg.inv(np.dot(Xfit.T,Xfit))*var_yfit[i] 
            for ii in range(mixed_spectra.shape[0]):
                x1 = np.dot(Xnew[ii],var_estimators)
                x2 = np.dot(x1,Xnew[ii].reshape(-1,1))[0]
                var_ynew[ii][i] = var_yfit[i]+x2
        
        return stats.t.ppf(1-0.025,y_fit.shape[0]-pca_components.shape[0])*var_ynew**0.5