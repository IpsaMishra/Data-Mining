import numpy as np
import matplotlib.pyplot as plt

def generate_data(mu1, sigma1, mu2, sigma2, n_samples):            
    x1 = np.random.multivariate_normal(mu1, sigma1, n_samples)
    x2 = np.random.multivariate_normal(mu2, sigma2, n_samples)     
    x = np.concatenate((x1, x2), axis = 0)
    return x

def mykmeans(X, k, c):
    thres = 0.001
    n_iter = 10000
    n_X = np.shape(X)[0]
    tags = np.zeros(n_X)                        # Stores the association of data points to cluster index
    
    iter_centers = []
    for nc in range(len(c)):
        iter_centers.append([])
    
    for nc in range(len(c)):
        iter_centers[nc].append(list(c[nc]))
        
    for i in range(n_iter):                     # Run a loop for all iterations
        for j in range(n_X):                    # Run a loop over all data points
            dist = [np.linalg.norm(X[j] - c[n]) for n in range(len(c))]
            indx = np.where(dist == np.amin(dist))
            indx = list(indx[0])
            tags[j] = indx[0]
            
        check = 0                               # Reset the variable
        upd_c = c * 0.0
        for j in range(len(c)):                 # Update each of the centroids
            identify = (tags == j) + 0.0        # Marks 1 for the data which is part of cluster 'j'
            upd_c[j] = np.matmul(np.transpose(identify) , X) / np.sum(identify)
            if np.linalg.norm(upd_c[j]-c[j]) <= thres:
                check = 1
        
        c = upd_c
        for nc in range(len(c)):
            iter_centers[nc].append(list(c[nc]))
        
        if check == 1:                          # Checks if any one updated center is closer to its corresponding previous value by the given threshold or less
            for j in range(n_X):                    # Run a loop over all data points to recompute cluster assignments before exiting loop
                dist = [np.linalg.norm(X[j] - c[n]) for n in range(len(c))]
                indx = np.where(dist == np.amin(dist))
                indx = list(indx[0])
                tags[j] = indx[0]
            break
    
    iter_centers = np.array(iter_centers)
    
    for nc in range(len(c)):                    # Plots the values of center updates
        plt.scatter(iter_centers[nc,:,0],iter_centers[nc,:,1],marker = '.')
    
    print('# of iterations = ', len(iter_centers[0])-1)              # Prints the number of iterations
    return c, tags
    
##Generate Dataset
mu1 = np.array([1,0])     
mu2 = np.array([0,1.5])
sigma1 = np.array([[0.9,0.4],[0.4,0.9]])
sigma2 = np.array([[0.9,0.4],[0.4,0.9]])
n_samples = 500
X = generate_data(mu1, sigma1, mu2, sigma2, n_samples)
#
## Execute k-means to obtain the centers and tag each data point with a center
#k = 4
#c1 = (10,10)
#c2 = (-10, -10)
#c = np.array((c1, c2))
#c3 = (10, -10)
#c4 = (-10, 10)
#c = np.array((c1, c2, c3, c4))

