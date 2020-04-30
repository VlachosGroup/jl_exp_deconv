import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from pquad import plotting_tools
from itertools import combinations

params = {'lines.linewidth': 2,
            'lines.markersize': 5,
            'legend.fontsize': 10.7,
            'legend.borderpad': 0.2,
            'legend.labelspacing': 0.2,
            'legend.handletextpad' : 0.2,
            'legend.borderaxespad' : 0.2,
            'legend.scatterpoints' :1,
            'xtick.labelsize' : 10.7,
            'ytick.labelsize' : 10.7,
            'axes.titlesize' : 10.7,
            'axes.labelsize' : 10.7,
            'figure.autolayout': True,
            'font.family': 'Calibri',
            'font.size': 10.7}

plotting_tools.set_figure_settings('paper',**params)

def get_training_data(c,sigma,conc_error=False):
    
    intensities = np.array([[1, 0.25, 0, 0, 0, 0]\
                           ,[2, 0.50, 0, 0, 0, 0]\
                           ,[4, 1.00, 0, 0, 0, 0]\
                           ,[0, 0.50, 1, 0, 0, 0]\
                           ,[0, 1.00, 2, 0, 0, 0]\
                           ,[0, 2.00, 4, 0, 0, 0]\
                           ,[0, 0.00, 0, 1, 0, c]\
                           ,[0, 0.00, 0, 2, 0, c*2**4]\
                           ,[0, 0.00, 0, 4, 0, c*4**4]])
        
    intensities += sigma*np.random.randn(intensities.shape[0],intensities.shape[1])#*intensities
    
    
    concentrations = np.array([[1, 0, 0]\
                             ,[2, 0, 0]\
                             ,[4, 0, 0]\
                             ,[0, 1, 0]\
                             ,[0, 2, 0]\
                             ,[0, 4, 0]\
                             ,[0, 0, 1]\
                             ,[0, 0, 2]\
                             ,[0, 0, 4]],dtype=float)
        
    if conc_error == True:
        concentrations += sigma*np.random.randn(concentrations.shape[0]\
                                                ,concentrations.shape[1])#*concentrations
    
    return intensities, concentrations

def get_mixture(c, sigma, conc_error=False):
    intensities, concentrations = get_training_data(c, 0, conc_error=False)
    combos = [[0,1,2,3,4,5,6],[0,1,2,3,4,5,7],[0,1,2,3,4,5,8]]
    combos = [list(range(intensities.shape[0]))]
    mixture_indices = []
    for i in range(1,intensities.shape[0]+1):
        for combo in combos:
            for x in combinations(combo,i):
                mixture_indices.append(list(x))
                
    mixture_indices = np.array(mixture_indices)
    
    mixture_int = []
    mixture_conc = []
    for i in mixture_indices:
        mixture_int.append(intensities[i].sum(axis=0))
        mixture_conc.append(concentrations[i].sum(axis=0))
    mixture_int = np.array(mixture_int)
    
    mixture_conc = np.array(mixture_conc)
    if conc_error == True:
        mixture_conc += sigma*np.random.randn(mixture_conc.shape[0]\
                                                ,mixture_conc.shape[1])#*mixture_conc
    else:
        mixture_int += sigma*np.random.randn(mixture_int.shape[0],mixture_int.shape[1])#*mixture_int
       
    return mixture_int, mixture_conc

class PCA:
    def __init__(self,X,Y, center = False, scale = False, scale_type = 'var'\
                 ,add_constant = False):
        self.X = X
        self.mu = X.mean(axis=0).reshape(1,-1)
        self.ymean = Y.mean(axis=0).reshape(1,-1)
        self.Y = Y
        self.center = center
        self.scale = scale
        if scale_type == 'var':
            self.var = X.var(axis=0)
        else:
            self.var = X.std(axis=0)
        self.add_constant = add_constant
        #Need to add variable for constant to PCs if centered
    def get_PC_loadings(self, NUM_PCs, center):
        X = self.X.copy()
        if center in ['partial', True]:
            X -= self.mu
        if self.scale == True:
            X /= (self.var + 10**-9)
        U, S, V = np.linalg.svd(X, full_matrices=False)
        PC_loadings = V[:NUM_PCs]
        self.EXPLAINED_VARIANCE = S[:NUM_PCs]**2/np.sum(S**2)
        self.TOTAL_EXPLAINED_VARIANCE = np.sum(S[:NUM_PCs]**2)/np.sum(S**2)
        self.EIGEN_VALUES = S**2
        return PC_loadings
    
    def get_PCs_and_regressors(self,NUM_PCs):
        X = self.X.copy()
        Y = self.Y.copy()
        mu = self.mu.copy()
        if self.scale == True:
            mu /= (self.var + 10**-9)
            X /= (self.var + 10**-9)
        pca_loadings = self.get_PC_loadings(NUM_PCs,False)
        if self.add_constant == True:
            PCs = np.dot(X,pca_loadings.T)
            PCs_2_Y, res, rank, s = np.linalg.lstsq(np.concatenate((PCs\
                            ,np.ones((PCs.shape[0],1))),axis=1),Y,rcond=None)
        elif self.center == True:
            #pca_loadings = self.get_PC_loadings(X.shape[1],False)
            #PCs = np.dot(X,pca_loadings.T)
            #PCs_2_Y, res, rank, s = np.linalg.lstsq(PCs,Y,rcond=None)
            #inner = np.dot(pca_loadings,mu.T)
            #constants = np.dot(inner.T,PCs_2_Y)
            pca_loadings = self.get_PC_loadings(NUM_PCs,True)
            PCs = np.dot(X-mu,pca_loadings.T)
            PCs_2_Y, res, rank, s = np.linalg.lstsq(PCs,Y-self.ymean,rcond=None)
            self.constants = self.ymean
        else:    
            PCs = np.dot(X,pca_loadings.T)
            PCs_2_Y, res, rank, s = np.linalg.lstsq(PCs,Y,rcond=None)
        self.PCs = PCs
        self.PCs_2_Y = PCs_2_Y
        self.pca_loadings = pca_loadings
        self.res = res
    
    def predict(self,X):
        X = X.copy()
        if self.center in ['partial', True]:
            X -= self.mu
        if self.scale == True:
            X /= (self.var + 10**-9)
        PCs = np.dot(X,self.pca_loadings.T)
        if self.add_constant == True:
            prediction = np.dot(np.concatenate((PCs,np.ones((PCs.shape[0],1))),axis=1),self.PCs_2_Y)
        else:
            prediction = np.dot(PCs,self.PCs_2_Y)
        if self.center == True:
            prediction += self.constants
        return prediction
    
    @staticmethod
    def get_r2(y_true, y_pred):
        SStot = np.sum((y_true-y_true.mean())**2,axis=0)
        SSres = np.sum((y_true-y_pred)**2,axis=0)
        return 1 - SSres/SStot
    @staticmethod
    def get_rmse(y_true, y_pred):
        SSres = np.mean((y_true-y_pred)**2,axis=0)
        return SSres**0.5
 
def plot_processing_results(figure_name,num_PCs,training_error=0,train_conc=False\
                            ,mixture_error = 0, mix_conc=False):     
    r2 = []
    rmse = []
    r2_centered = []
    rmse_centered = []
    r2_cscaled = []
    rmse_cscaled = []
    r2_scaled = []
    rmse_scaled = []
    r2_scaled_std = []
    rmse_scaled_std = []
    r2_PLS = []
    rmse_PLS = []
    r2_PLS_scaled = []
    rmse_PLS_scaled = []
    
    num_PCs = num_PCs
    crange = 10**np.linspace(-3,1,num=25,endpoint=True)
    cratio = crange*4**4/4
    print(cratio)
    for c in crange:
        intensities, concentrations = get_training_data(c,training_error\
                                                        ,conc_error=train_conc)
        mixture_int, mixture_conc = get_mixture(c,mixture_error\
                                                , conc_error=mix_conc)
        
        pca = PCA(intensities,concentrations,center=False,scale=False,add_constant=True)
        pca.get_PCs_and_regressors(num_PCs)
        prediction = pca.predict(mixture_int)
        r2.append(PCA.get_r2(mixture_conc, prediction))
        rmse.append(PCA.get_rmse(mixture_conc,prediction))
        
        #centered
        pca_centered = PCA(intensities,concentrations,center=True,scale=False)
        pca_centered.get_PCs_and_regressors(num_PCs)
        prediction = pca_centered.predict(mixture_int)
        r2_centered.append(PCA.get_r2(mixture_conc, prediction))
        rmse_centered.append(PCA.get_rmse(mixture_conc,prediction))
        #centered and scaled sigma^2
        pca_cscaled = PCA(intensities,concentrations,center=True,scale=True)
        pca_cscaled.get_PCs_and_regressors(num_PCs)
        prediction = pca_cscaled.predict(mixture_int)
        r2_cscaled.append(PCA.get_r2(mixture_conc, prediction))
        rmse_cscaled.append(PCA.get_rmse(mixture_conc,prediction))
        
        #scaled sigma^2
        pca_scaled = PCA(intensities,concentrations,center=False,scale=True,add_constant=True)
        pca_scaled.get_PCs_and_regressors(num_PCs)
        prediction = pca_scaled.predict(mixture_int)
        r2_scaled.append(PCA.get_r2(mixture_conc, prediction))
        rmse_scaled.append(PCA.get_rmse(mixture_conc,prediction))
        
        #scaled sigma
        pca_scaled_std = PCA(intensities,concentrations,center=False,scale=True, scale_type='std',add_constant=True)
        pca_scaled_std.get_PCs_and_regressors(num_PCs)
        prediction = pca_scaled_std.predict(mixture_int)
        r2_scaled_std.append(PCA.get_r2(mixture_conc, prediction))
        rmse_scaled_std.append(PCA.get_rmse(mixture_conc,prediction))
        
        #PLS
        PLS = PLSRegression(n_components=num_PCs, scale=False, max_iter=500\
                            , tol=1e-06, copy=True, center=True)
        PLS.fit(intensities, concentrations)
        prediction = PLS.predict(mixture_int)
        r2_PLS.append(PCA.get_r2(mixture_conc, prediction))
        rmse_PLS.append(PCA.get_rmse(mixture_conc,prediction))
        
        #PLS scaled
        PLS_scaled = PLSRegression(n_components=num_PCs, scale=True\
                                   , max_iter=500, tol=1e-06, copy=True, center=True)
        PLS_scaled.fit(intensities, concentrations)
        prediction = PLS_scaled.predict(mixture_int)
        r2_PLS_scaled.append(PCA.get_r2(mixture_conc, prediction))
        rmse_PLS_scaled.append(PCA.get_rmse(mixture_conc,prediction))
    
    
    r2 = np.array(r2)
    r2_centered = np.array(r2_centered)
    r2_scaled = np.array(r2_scaled)
    r2_cscaled = np.array(r2_cscaled)
    r2_scaled_std = np.array(r2_scaled_std)
    r2_PLS = np.array(r2_PLS)    
    r2_PLS_scaled = np.array(r2_PLS_scaled)
    
    rmse = np.array(rmse)
    rmse_centered = np.array(rmse_centered)
    rmse_scaled = np.array(rmse_scaled)
    rmse_cscaled = np.array(rmse_cscaled)
    rmse_scaled_std = np.array(rmse_scaled_std)
    rmse_PLS = np.array(rmse_PLS)    
    rmse_PLS_scaled = np.array(rmse_PLS_scaled)
    
    """
    plt.figure(0)
    plt.plot(cratio,np.mean(r2,axis=1),'blue')
    plt.plot(cratio,np.mean(r2_centered,axis=1),'og')
    plt.plot(cratio,np.mean(r2_scaled,axis=1),':r')
    plt.plot(cratio,np.mean(r2_cscaled,axis=1),':g')
    plt.plot(cratio,np.mean(r2_scaled_std,axis=1),'^r')
    plt.plot(cratio,np.mean(r2_PLS,axis=1),'orange')
    plt.plot(cratio,np.mean(r2_PLS_scaled,axis=1),marker='s',color='orange',linewidth=0)
    plt.legend(['PCA unprocessed','PCA centered','PCA $\sigma^{2}$ scaled'
                ,'PCA centered and $\sigma^{2}$ scaled', 'PCA $\sigma$ scalled'
                ,'PLS centered', 'PLS centered and $\sigma$ scaled'])
    plt.xscale('log')
    plt.xlabel('max(squared elements)/max(linear elements)')
    plt.ylabel('Average R$^{2}$')
    plt.show()
    """
    
    plt.figure(1,figsize=(3.5,3.5))
    Figure_folder=os.path.join(os.path.expanduser("~"), 'Downloads')
    plt.plot(cratio,np.mean(rmse,axis=1),'blue')
    plt.plot(cratio,np.mean(rmse_centered,axis=1),'og')
    plt.plot(cratio,np.mean(rmse_scaled,axis=1), ':r')
    plt.plot(cratio,np.mean(rmse_cscaled,axis=1),':g')
    plt.plot(cratio,np.mean(rmse_scaled_std,axis=1),'^r')
    plt.plot(cratio,np.mean(rmse_PLS,axis=1),color='orange')
    plt.plot(cratio,np.mean(rmse_PLS_scaled,axis=1),marker='s',color='orange',linewidth=0)
    plt.legend(['PCA unprocessed','PCA centered','PCA $\sigma^{2}$ scaled'
                ,'PCA centered and $\sigma^{2}$ scaled', 'PCA $\sigma$ scalled'
                ,'PLS centered ', 'PLS centered and $\sigma$ scaled'],loc=4)
    plt.xlabel('Max(squared elements)/max(linear elements)')
    #plt.ylabel('Average RMSE')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim([6*10**-4,4])
    #plt.xticks([])
    #plt.yticks([])
    figure_file = os.path.join(Figure_folder,figure_name+'.jpg')
    plt.savefig(figure_file,format='jpg')
    plt.close()
    
    
    plt.figure(2)
    plt.plot(cratio,r2[:,2],'blue')
    plt.plot(cratio,r2_centered[:,2],'og')
    plt.plot(cratio,r2_scaled[:,2],':r')
    plt.plot(cratio,r2_cscaled[:,2],':g')
    plt.plot(cratio,r2_scaled_std[:,2],'^r')
    plt.plot(cratio,r2_PLS[:,2],'orange')
    plt.plot(cratio,r2_PLS_scaled[:,2],marker='s',color='orange',linewidth=0)
    plt.legend(['PCA unprocessed','PCA centered','PCA $\sigma^{2}$ scaled'
                ,'PCA centered and $\sigma^{2}$ scaled', 'PCA $\sigma$ scalled'
                ,'PLS centered', 'PLS centered and $\sigma$ scaled'])
    plt.xscale('log')
    plt.xlabel('Max(squared elements)/max(linear elements)')
    plt.ylabel('Average R$^{2}$')
    plt.show()
    
    plt.figure(3)
    plt.plot(cratio,rmse[:,2],'blue')
    plt.plot(cratio,rmse_centered[:,2],'og')
    plt.plot(cratio,rmse_scaled[:,2], ':r')
    plt.plot(cratio,rmse_cscaled[:,2],':g')
    plt.plot(cratio,rmse_scaled_std[:,2],'^r')
    plt.plot(cratio,rmse_PLS[:,2],color='orange')
    plt.plot(cratio,rmse_PLS_scaled[:,2],marker='s',color='orange',linewidth=0)
    plt.legend(['PCA unprocessed','PCA centered','PCA $\sigma^{2}$ scaled'
                ,'PCA centered and $\sigma^{2}$ scaled', 'PCA $\sigma$ scalled'
                ,'PLS centered ', 'PLS centered and $\sigma$ scaled'])
    plt.xlabel('Max(squared elements)/max(linear elements)')
    plt.ylabel('Average RMSE')
    plt.xscale('log')
    plt.show()
    

num_PCs = 3
plot_processing_results('no_error',num_PCs,0,False,0,False)

"""
plot_processing_results('train_0.001',num_PCs,training_error=0.001,train_conc=False\
                            ,mixture_error = 0, mix_conc=False)


plot_processing_results('train_0.01',num_PCs,training_error=0.01,train_conc=False\
                            ,mixture_error = 0, mix_conc=False)
    
plot_processing_results('train_0.1',num_PCs,training_error=0.1,train_conc=False\
                            ,mixture_error = 0, mix_conc=False)
   

plot_processing_results('train_0.001_conc',num_PCs,training_error=0.001,train_conc=True\
                            ,mixture_error = 0, mix_conc=False)
    
plot_processing_results('train_0.01_conc',num_PCs,training_error=0.01,train_conc=True\
                            ,mixture_error = 0, mix_conc=False)
    
plot_processing_results('train_0.1_conc',num_PCs,training_error=0.1,train_conc=True\
                            ,mixture_error = 0, mix_conc=False)


plot_processing_results('test_0.001',num_PCs,training_error=0,train_conc=False\
                            ,mixture_error = 0.001, mix_conc=False)
    
plot_processing_results('test_0.01',num_PCs,training_error=0,train_conc=False\
                            ,mixture_error = 0.01, mix_conc=False)
    
plot_processing_results('test_0.1',num_PCs,training_error=0,train_conc=False\
                            ,mixture_error = 0.1, mix_conc=False)
"""