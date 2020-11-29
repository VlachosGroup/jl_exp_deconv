__version__ = '1.0.4'
__author__  = "Avinash Kak (kak@purdue.edu)"
__date__    = '2020-October-3'
__url__     = 'https://engineering.purdue.edu/kak/distPLS/PartialLeastSquares-1.0.4.html'
__copyright__ = "(C) 2020 Avinash Kak. Python Software Foundation."



import numpy as np
from numpy.linalg import cholesky, det, lstsq
from scipy.optimize import minimize


#----------------------------- PartialLeastSquares Class Definition --------------------------------

class GPR(object):

    def __init__(self, X_train, Y_train, noise=10**-8):
        self.noise = noise
        self.NUM_TARGETS = Y_train.shape[1]
        self.X_train = X_train.copy()
        self.Y_train = Y_train.copy()
        l_opt, sigma_f_opt = self.optimize()
        self.l_opt = l_opt
        self.sigma_f_opt = sigma_f_opt


    def kernel(self, X1, X2, l=1.0, sigma_f=1.0):
        """
        Isotropic squared exponential kernel.
        
        Args:
            X1: Array of m points (m x d).
            X2: Array of n points (n x d).
    
        Returns:
            (m x n) matrix.
        """
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)
    
    def posterior(self, X_new, X_train, Y_train, l=1.0, sigma_f=1.0, sigma_y=1e-8):
        """
        Computes the suffifient statistics of the posterior distribution 
        from m training data X_train and Y_train and n new inputs X_s.
        
        Args:
            X_new: New input locations (n x d).
            X_train: Training locations (m x d).
            Y_train: Training targets (m x 1).
            l: Kernel length parameter.
            sigma_f: Kernel vertical variation parameter.
            sigma_y: Noise parameter.
        
        Returns:
            Posterior mean vector (n x d) and covariance matrix (n x n).
        """
        K = self.kernel(X_train, X_train, l, sigma_f) + sigma_y**2 * np.eye(len(X_train))
        K_s = self.kernel(X_train, X_new, l, sigma_f)
        K_ss = self.kernel(X_new, X_new, l, sigma_f) + 1e-8 * np.eye(len(X_new))
        K_inv = np.linalg.inv(K)
        
        # Equation (7)
        mu_s = K_s.T.dot(K_inv).dot(Y_train)
    
        # Equation (8)
        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
        
        return mu_s, cov_s
    
    def get_predictions(self, Xnew):
        predictions = []
        sigma_list = []
        for column in self.Y_train:
            mu_s, cov_s = self.posterior(Xnew, self.X_train
                                    , self.Y_train[:,column].reshape(-1,1)
                                    ,l=self.l_opt, sigma_f = self.sigma_f_opt
                                    , sigma_y = self.noise)
            predictions.append(mu_s)
            sigma_list.append(np.sqrt(np.diag(cov_s)))
         
        self.sigma_list = np.array(sigma_list).T
        return np.array(predictions).T
    
    def nll_fn(self, X_train, Y_train, noise, naive=True):
        """
        Returns a function that computes the negative log marginal
        likelihood for training data X_train and Y_train and given
        noise level.
    
        Args:
            X_train: training locations (m x d).
            Y_train: training targets (m x 1).
            noise: known noise level of Y_train.
            naive: if True use a naive implementation of Eq. (11), if
                   False use a numerically more stable implementation.
    
        Returns:
            Minimization objective.
        """
        
        Y_train = Y_train.ravel()
        
        def nll_naive(theta):
            # Naive implementation of Eq. (11). Works well for the examples 
            # in this article but is numerically less stable compared to 
            # the implementation in nll_stable below.
            K = self.kernel(X_train, X_train, l=theta[0], sigma_f=theta[1]) + \
                noise**2 * np.eye(len(X_train))
            return 0.5 * np.log(det(K)) + \
                   0.5 * Y_train.dot(np.linalg.inv(K).dot(Y_train)) + \
                   0.5 * len(X_train) * np.log(2*np.pi)
            
        def nll_stable(theta):
            # Numerically more stable implementation of Eq. (11) as described
            # in http://www.gaussianprocess.org/gpml/chapters/RW2.pdf, Section
            # 2.2, Algorithm 2.1.
            
            def ls(a, b):
                return lstsq(a, b, rcond=-1)[0]
            
            K = self.kernel(X_train, X_train, l=theta[0], sigma_f=theta[1]) + \
                noise**2 * np.eye(len(X_train))
            L = cholesky(K)
            return np.sum(np.log(np.diagonal(L))) + \
                   0.5 * Y_train.dot(ls(L.T, ls(L, Y_train))) + \
                   0.5 * len(X_train) * np.log(2*np.pi)
    
        if naive:
            return nll_naive
        else:
            return nll_stable
        
    def optimize(self):
        res = minimize(self.nll_fn(self.X_train, self.Y_train, self.noise, naive=False), [1, 1], 
               bounds=((1e-5, None), (1e-5, None)),
               method='L-BFGS-B')

        # Store the optimization results in global variables so that we can
        # compare it later with the results from other implementations.
        l_opt, sigma_f_opt = res.x
        return l_opt, sigma_f_opt

   
        
    

    

