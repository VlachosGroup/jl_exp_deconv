import numpy as np
from scipy import stats

#----------------------------- PartialLeastSquares Class Definition --------------------------------

class PCR:

    def __init__(self, XMatrix, YMatrix, NUM_PCs, centered=False):
        if centered == True:
            self.Xmean = XMatrix.mean(axis=0,keepdims=True)
            self.Ymean = YMatrix.mean(axis=0, keepdims=True)
        else:
            self.Xmean = np.zeros((1,XMatrix.shape[1]))
            self.Ymean = np.zeros((1,YMatrix.shape[1]))
        self.NUM_TARGETS = YMatrix.shape[1]
        self.XMatrix = XMatrix.copy() # Each column of X stands for a predictor variable numpy.matrix([[]])
        self.YMatrix = YMatrix.copy() # Each column of Y stands for a predicted variable
        self.NUM_PCs = NUM_PCs   
        self.centered = centered
        self.pca_loadings, self.PCs_2_Y = self._get_PCs_and_regressors()


    def _get_PC_loadings(self):
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
        XMatrix = self.XMatrix - self.Xmean
        NUM_PCs = self.NUM_PCs
        U, S, V = np.linalg.svd(XMatrix, full_matrices=False)
        PC_loadings = V[:NUM_PCs]
        self.TOTAL_EXPLAINED_VARIANCE = np.sum(S[:NUM_PCs]**2)/np.sum(S**2)
        return PC_loadings
    
    def _get_PCs_and_regressors(self):
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
            
        PCs_2_Y : numpy.ndarray
            Regressed matrix to compute concentrations given the principal
            components of a mixed spectra.
        
        """
        
        XMatrix = self.XMatrix - self.Xmean
        YMatrix = self.YMatrix - self.Ymean
        _get_PC_loadings = self._get_PC_loadings
        pca_loadings = _get_PC_loadings()
        PCs = np.dot(XMatrix,pca_loadings.T)
        PCs_2_Y, res, rank, s = np.linalg.lstsq(PCs,YMatrix,rcond=None)
        self.res = res
        return pca_loadings, PCs_2_Y
    
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
        PCs_2_concentrations = self.PCs_2_Y
        pca_loadings = self.pca_loadings
        PCs = np.dot(spectra-self.Xmean,pca_loadings.T)  
        predictions =  np.dot(PCs,PCs_2_concentrations) + self.Ymean
        return predictions
    
    def get_PI(self, spectra,CI=0.95):
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
        if self.centered == True:
            used_mean = 1
        else:
            used_mean = 0
        alpha = (1-CI)/2
        spectra = np.copy(spectra)
        if len(spectra.shape)==1 and len(spectra[0].shape)==0:
            spectra = spectra.reshape(1,-1)
        elif len(spectra.shape)==1 or len(spectra.shape)==3:
            output_list = []
            for value in spectra:
                output_list.append(self.get_PI(value, CI))
            return output_list
        pure_spectra = self.XMatrix
        pure_concentrations = self.YMatrix
        pca_loadings = self.pca_loadings
        y_fit = self.get_predictions(spectra=pure_spectra)
        NUM_TARGETS = self.NUM_TARGETS
        Xnew = np.dot(spectra - self.Xmean,pca_loadings.T)
        Xfit = np.dot(pure_spectra-self.Xmean,pca_loadings.T)
        var_yfit = np.zeros(NUM_TARGETS)
        var_ynew = np.zeros((spectra.shape[0],NUM_TARGETS))
        for i in range(NUM_TARGETS):
            var_yfit[i] = np.var(pure_concentrations[:,i]-y_fit[:,i]
                                 - self.Ymean[0][i]
                                 ,ddof=pca_loadings.shape[0] + used_mean)
            var_estimators = np.linalg.inv(np.dot(Xfit.T,Xfit))*var_yfit[i] 
            for ii in range(spectra.shape[0]):
                x1 = np.dot(Xnew[ii],var_estimators)
                x2 = np.dot(x1,Xnew[ii].reshape(-1,1))[0]
                var_ynew[ii][i] = var_yfit[i]+x2
        prediction_interval = stats.t.ppf(1 - alpha
                                          ,y_fit.shape[0]
                                          - pca_loadings.shape[0] - used_mean
                                          )*var_ynew**0.5
        return prediction_interval
    
    

    

