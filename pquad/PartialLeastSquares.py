__version__ = '1.0.4'
__author__  = "Avinash Kak (kak@purdue.edu)"
__date__    = '2020-October-3'
__url__     = 'https://engineering.purdue.edu/kak/distPLS/PartialLeastSquares-1.0.4.html'
__copyright__ = "(C) 2020 Avinash Kak. Python Software Foundation."



import numpy as np
import numpy.linalg as np_linalg
from scipy import stats

#----------------------------- PartialLeastSquares Class Definition --------------------------------

class PartialLeastSquares(object):

    def __init__(self, XMatrix, YMatrix , NUM_PCs):
        self.NUM_TARGETS = YMatrix.shape[1]
        self.X = XMatrix                                # Each column of X stands for a predictor variable np.matrix([[]])
        self.Y = YMatrix                                # Each column of Y stands for a predicted variable
        self.mean0X = None                           # Store column-wise mean for X
        self.mean0Y = None                           #    and for Y
        self.Xtest = None                            # X matrix for evaluating PLS regression
        self.Ytest = None                            # Y matrix for evaluating PLS regression
        self.B = None                                # regression coefficients
        self.N = self.X.shape[0]
        self.num_predictor_vars = self.X.shape[1]
        self.debug=False
        self.NUM_PCs = NUM_PCs


    def PLS(self):
        """
        This implementation is based on the description of the algorithm by Herve
        Abdi in the article "Partial Least Squares Regression and Projection on
        Latent Structure Regression," Computational Statistics, 2010.  From my
        experiments with the different variants of PLS, this particular version
        generates the best regression results.  The Examples directory contains a
        script that carries out head-pose estimation using this version of PLS.
        """
        X,Y = self.X, self.Y
        self.mean0X = X.mean(0)
        if self.debug:
            print("\nColumn-wise mean for X:")
            print(self.mean0X)
        X = X - self.mean0X
        if self.debug:
            print("\nZero-mean version of X:")
            print(X)
        self.mean0Y = Y.mean(0)
        if self.debug:
            print("\nColumn-wise mean for Y is:")
            print(self.mean0Y)
        Y = Y - self.mean0Y
        if self.debug:
            print("\nZero-mean version of Y:")
            print(Y)
        T=U=W=C=P=Q=B=Bdiag=t=w=u=c=p=q=b=None        
        u = np.random.rand(1,self.N)
        u = np.asmatrix(u).T
        if self.debug:
            print("\nThe initial random guess for u: ")
            print(u)
        for i in range(self.NUM_PCs):                             
            j = 0
            while (True):
                w = X.T * u
                w = w / np_linalg.norm(w)
                t = X * w
                t = t / np_linalg.norm(t)      
                c = Y.T * t
                c = c / np_linalg.norm(c)        
                u_old = u
                u = Y * c
                error = np_linalg.norm(u - u_old)
                if error < 0.001: 
                    if self.debug:
                        print("Number of iterations for the %dth latent vector: %d" % (i,j+1))
                    break
                j += 1    
            b = t.T * u
            b = b[0,0]
            if T is None:
                T = t
            else:
                T = np.hstack((T,t))
            if U is None:
                U = u
            else:
                U = np.hstack((U,u))
            if W is None:
                W = w
            else:
                W = np.hstack((W,w))
            if C is None:
                C = c
            else:
                C = np.hstack((C,c))
            p = X.T * t / (np_linalg.norm(t) ** 2)
            q = Y.T * u / (np_linalg.norm(u) ** 2)
            if P is None:
                P = p
            else:
                P = np.hstack((P,p))
            if Q is None:
                Q = q
            else:
                Q = np.hstack((Q,q))
            if Bdiag is None:
                Bdiag = [b]
            else:
                Bdiag.append(b)
            X = X - t * p.T
            Y = Y - b * t * c.T
            i += 1
        if self.debug:
            print("\n\n\nThe T matrix:")
            print(T)
            print("\nThe U matrix:")
            print(U)
            print("\nThe W matrix:")
            print(W)
            print("\nThe C matrix:")
            print(C)
            print("\nThe P matrix:")
            print(P)
            print("\nThe b vector:")
            print(Bdiag)
            print("\nThe final deflated X matrix:")
            print(X)
            print("\nThe final deflated Y matrix:")
            print(Y)
        B = np.diag(Bdiag)
        B = np.asmatrix(B)  
        if self.debug:
            print("\nThe diagonal matrix B of b values:")
            print(B)
        self.B2 = B
        self.B = np_linalg.pinv(P.T) * B * C.T
        if self.debug:
            print("\nThe matrix B of regression coefficients:")
            print(self.B)
        # For testing, make a prediction based on the original X:
        if self.debug:
            Y_predicted = (self.X - self.mean0X) * self.B
            print("\nY_predicted from the original X:")
            print(Y_predicted)
            Y_predicted_with_mean = Y_predicted + self.mean0Y
            print("\nThe predicted Y with the original Y's column-wise mean added:")
            print(Y_predicted_with_mean)
            print("\nThe original Y for comparison:") 
            print(self.Y)
        self.Pinv = np_linalg.pinv(P.T)
        projections = np.dot((self.X - self.mean0X), self.Pinv)
        #return self.B
        Projections_2_Y, res, rank, s = np.linalg.lstsq(projections
                                                        ,self.Y.copy()
                                                        ,rcond=None)
        self.Projections_2_Y = Projections_2_Y
    
    def get_predictions(self,Xnew):
        Xnew = np.copy(Xnew)
        if len(Xnew.shape)==1 and len(Xnew[0].shape)==0:
            Xnew = Xnew.reshape(1,-1)
        elif len(Xnew.shape)==1 or len(Xnew.shape)==3:
            output_list = []
            for value in Xnew:
                output_list.append(self.get_predictions(value))
            return output_list
        #Y_predicted = (Xnew - self.mean0X) * self.B + self.mean0Y
        projections = np.dot((Xnew - self.mean0X), self.Pinv)
        Y_predicted =  np.dot(projections,self.Projections_2_Y) + self.mean0Y
        return np.array(Y_predicted)
    
    def get_PI(self, spectra,CI=0.95):
        alpha = (1-CI)/2
        spectra = np.copy(spectra)
        if len(spectra.shape)==1 and len(spectra[0].shape)==0:
            spectra = spectra.reshape(1,-1)
        elif len(spectra.shape)==1 or len(spectra.shape)==3:
            output_list = []
            for value in spectra:
                output_list.append(self.get_PI(value, CI))
            return output_list
        pure_spectra = self.X
        pure_concentrations = self.Y
        y_fit = self.get_predictions(pure_spectra)
        NUM_TARGETS = self.NUM_TARGETS
        Xnew = np.dot(spectra - self.mean0X, self.Pinv)
        Xfit = np.dot(pure_spectra - self.mean0X, self.Pinv)
        var_yfit = np.zeros(NUM_TARGETS)
        var_ynew = np.zeros((spectra.shape[0],NUM_TARGETS))
        for i in range(NUM_TARGETS):
            var_yfit[i] = np.var(pure_concentrations[:,i]-y_fit[:,i],ddof=self.Pinv.shape[1]+1)
            var_estimators = np_linalg.inv(np.dot(Xfit.T,Xfit))*var_yfit[i] 
            for ii in range(spectra.shape[0]):
                x1 = np.dot(Xnew[ii],var_estimators)
                x2 = np.dot(x1,Xnew[ii].reshape(-1,1))[0]
                var_ynew[ii][i] = var_yfit[i]+x2
        prediction_interval = stats.t.ppf(1 - alpha,y_fit.shape[0]-self.Pinv.shape[1]-1)*var_ynew**0.5
        return prediction_interval
        
    

    

