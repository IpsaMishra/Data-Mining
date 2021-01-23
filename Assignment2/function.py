import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
from mpl_toolkits import mplot3d

  
def norm_data_generate(m, c, k = 1, bins = 10):      # Generates normally distributed data with given mean and variance/covariance
    if isinstance(m,float) == 1 or isinstance(m,int) == 1 or len(m) == 1:  # Convert into proper array format for use with numpy
        m = np.array([float(np.array([m]))])
        c = np.array([[float(np.array([c]))]])
        
    x = np.random.multivariate_normal(m,c,k)
    if bins == -1:          # Suppress the histogram plot
        return x
    elif len(m) == 1:
        plt.figure()
        plt.hist(x,bins)
    elif len(m) == 2:
        plt.figure()
        h = plt.hist2d(x[:,0], x[:,1])
        plt.colorbar(h[3])
    return x


def Gauss_mixt_data_generate(m0, c0, k0, m1, c1, k1, bins, scat):    # Generates data from a bi-modal Gaussian mixture
    x0 = norm_data_generate(m0, c0, k0, -1)
    x1 = norm_data_generate(m1, c1, k1, -1)
    x = np.concatenate((x0, x1))
    
    if isinstance(m0,float) == 1 or isinstance(m0,int) == 1 or len(m0) == 1:  # Convert into proper array format for use with numpy
        m = np.array([float(np.array([m0]))])
    else:
        m = m0
        
    if scat != 0:
        plt.figure()
        ax = plt.axes()
        ax.set_xlabel('x-axis')
        ax.set_ylabel('y-axis')
        ax.set_title('Scatter Plot')
        col = np.concatenate((2 * np.ones(k0), 3 * np.ones(k1)))
        plt.scatter(x[:,0], x[:,1], c = col, alpha = 0.5)
        
    if bins == -1:          # Suppress the histogram plot
        return x
    elif len(m) == 1:
        plt.figure()
        plt.hist(x,bins)
    elif len(m) == 2:
        plt.figure()
        h = plt.hist2d(x[:,0], x[:,1])
        plt.colorbar(h[3])
    return x

    
def label_gen(k0, k1):                                              # Generates binary classification labels for datasets
    l0 = np.zeros(k0)
    l1 = np.ones(k1)
    l = np.concatenate((l0, l1))
    return l


def mykde(x,h = 0.5):                   # Generates kernel density estimation values for a given bandwidth and plots it for 1D, 2D
    x = np.array(x)
    dims = np.shape(x)
    dmin = np.zeros((1,dims[1]))
    dmax = np.zeros((1,dims[1]))
    samp_rate = 8
    step = []
    for i in range(dims[1]):
        dmin[0,i] = np.min(x[:,i]) - 2 * h
        dmax[0,i] = np.max(x[:,i]) + 2 * h
        naxis = int((dmax[0,i] - dmin[0,i]) * samp_rate / h)   # Number of points along of axis where the value is computed
        step.append((dmax[0,i] - dmin[0,i]) / naxis)
        print('The domain of axis', i, '=', [dmin[0,i],dmax[0,i]])
    
    if dims[1] == 1:
        ax1 = dmin[0,i] + step[0] * np.arange(naxis + 1)
        vals = np.zeros(np.shape(ax1))
        for x_i in x:
            rv = ss.multivariate_normal(x_i, h*h)
            vals += rv.pdf(ax1) / len(x)
        
        plt.figure()
        ax = plt.axes()
        ax.set_xlabel('x')
        ax.set_ylabel('PDF')
        ax.set_title('Surface plot of a Gaussian KDE')
        plt.plot(ax1,vals)
    
    elif dims[1] == 2:
        ax1, ax2 = np.mgrid[dmin[0,0]:dmax[0,0]+0.01:step[0], dmin[0,1]:dmax[0,1]+0.01:step[1]]
        vals = np.zeros(np.shape(ax1))
        coord = np.empty(ax1.shape + (2,))
        for x_i in x:
            rv = ss.multivariate_normal(x_i, [[h, 0], [0, h]])
            coord[:, :, 0] = ax1 
            coord[:, :, 1] = ax2
            vals += rv.pdf(coord) / len(x)
        
        fig = plt.figure(figsize=(10, 5))
        ax = plt.axes(projection='3d')
        surf = ax.plot_surface(ax1, ax2, vals, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('PDF')
        ax.set_title('Surface plot of a 2D Gaussian KDE')
        fig.colorbar(surf, shrink=0.5, aspect=5) # add color bar indicating the PDF
        ax.view_init(60, 35)
        
        fig = plt.figure(figsize=(10, 5))
        ax = plt.axes(projection='3d')
        ax.plot_wireframe(ax1, ax2, vals)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('PDF')
        ax.set_title('Wireframe plot of a 2D Gaussian KDE');
   
    else:
        print('This function is defined for 1D, 2D datasets only')
        return
    

def myNB(X, Y, X_test, Y_test):                    # This is a binary classifier for 2D dataset
    x0 = []
    x1 = []
    for i in range(len(Y)):
        if int(Y[i]) == 0:
            x0.append(X[i])
        elif int(Y[i]) == 1:
            x1.append(X[i])
    x0 = np.array(x0)
    x1 = np.array(x1)
    
    c0 = len(x0) / len(Y)
    c1 = len(x1) / len(Y)
    
    m0 = np.mean(x0, axis = 0)
    m1 = np.mean(x1, axis = 0)
    m0_0, m0_1 = m0[0], m0[1]
    m1_0, m1_1 = m1[0], m1[1]
    
    # Calculating variance on each feature dimension separately as required for Naive Bayes
    v0_0, v0_1 = np.var(x0[:,0]) * len(x0) / (len(x0) - 1), np.var(x0[:,1]) * len(x0) / (len(x0) - 1)
    v1_0, v1_1 = np.var(x1[:,0]) * len(x1) / (len(x1) - 1), np.var(x1[:,1]) * len(x1) / (len(x1) - 1)
    
    rv0_0 = ss.multivariate_normal(m0_0, v0_0)
    rv0_1 = ss.multivariate_normal(m0_1, v0_1)
    rv1_0 = ss.multivariate_normal(m1_0, v1_0)
    rv1_1 = ss.multivariate_normal(m1_1, v1_1)
    
    pr0 = c0 * rv0_0.pdf(X_test[:,0]) * rv0_1.pdf(X_test[:,1])
    pr1 = c1 * rv1_0.pdf(X_test[:,0]) * rv1_1.pdf(X_test[:,1])
    tpr = pr0 + pr1
    
    pred = (pr0 < pr1) + 0
    posterior = np.transpose(np.array([pr0 / tpr, pr1 / tpr]))
    err = np.mean(pred != Y_test)
    return pred, posterior, err


def error_analysis(Y, posterior, roc_plot = 0):
    tp = np.sum((Y == 1) * (posterior[:,1] > posterior[:,0]))
    fp = np.sum((Y == 0) * (posterior[:,1] > posterior[:,0]))
    tn = np.sum((Y == 0) * (posterior[:,0] > posterior[:,1]))
    fn = np.sum((Y == 1) * (posterior[:,0] > posterior[:,1]))
    
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    conf_matr = np.array([[tp, fn],[fp, tn]])
    
    tpr = np.zeros(101)
    fpr = np.zeros(101)
    auc = 0
    for i in range(0,101):
        tp = np.sum((Y == 1) * (posterior[:,1] >= (i / 100.0)))
        fp = np.sum((Y == 0) * (posterior[:,1] >= (i / 100.0)))
        tn = np.sum((Y == 0) * (posterior[:,1] < (i / 100.0)))
        fn = np.sum((Y == 1) * (posterior[:,1] < (i / 100.0)))
        tpr[i] = tp / (tp + fn)
        fpr[i] = fp / (fp + tn)
        if i > 0:
            auc += (fpr[i-1] - fpr[i]) * (tpr[i] + tpr[i-1]) / 2
    
    if roc_plot != 0:
        plt.figure()
        ax = plt.axes()
        ax.set_xlabel('False positive rate')
        ax.set_ylabel('True positive rate')
        ax.set_title('ROC plot')
        plt.text(0.5, 0.5, 'AUC = ' + str(round(auc,5)))
        plt.plot(fpr, tpr)
        
    return prec, rec, conf_matr, auc