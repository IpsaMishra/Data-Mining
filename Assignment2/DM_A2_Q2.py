import function as f
import numpy as np
import matplotlib.pyplot as plt

bins = -1   # Supress histogram plot

### 1a & 2a
k = 500                             # Number of data points in each class in one experiment
conf_matr = np.zeros((2,2))
prec = 0
rec = 0
acc = 0
auc = 0
n_run = 10
for i in range(n_run):              # Running the experiment 'n_run' times to account for random datasets
    if i == 0:
        scat = 1
        roc_plot = 1
    else:
        scat = 0
        roc_plot = 0
    
    m0 = [1, 0]
    c0 = [[1, 0.75], [0.75, 1]]
    
    m1 = [0, 1]
    c1 = [[1, 0.75], [0.75, 1]]
    
    x = f.Gauss_mixt_data_generate(m0, c0, k, m1, c1, k, bins, scat)
    y = f.label_gen(k, k)
    
    xt = f.Gauss_mixt_data_generate(m0, c0, k, m1, c1, k, bins, 0)
    yt = f.label_gen(k, k)
    
    pred, posterior, err = f.myNB(x,y,xt,yt)
    prec0, rec0, conf_matr0, auc0 = f.error_analysis(yt, posterior, roc_plot)
    prec += prec0 / n_run
    rec += rec0 / n_run
    conf_matr += conf_matr0 / n_run
    auc += auc0 / n_run
    acc += (1 - err) / n_run
    
print('Average Accuracy:', acc )
print('Average AUC:', auc)
print('Average Precision:', prec)
print('Average Recall:', rec)
print('Average of Confusion Matrices:', conf_matr)

### 1b
klist = np.array([10, 20, 50, 100, 300, 500])
test_k = 500

acc = np.zeros(len(klist))
for k in range(len(klist)):
    n_run = 10
    for i in range(n_run):              # Running the experiment 'n_run' times to account for random datasets        
        m0 = [1, 0]
        c0 = [[1, 0.75], [0.75, 1]]
        
        m1 = [0, 1]
        c1 = [[1, 0.75], [0.75, 1]]
        
        x = f.Gauss_mixt_data_generate(m0, c0, klist[k], m1, c1, klist[k], bins, 0)
        y = f.label_gen(klist[k], klist[k])
        
        xt = f.Gauss_mixt_data_generate(m0, c0, test_k, m1, c1, test_k, bins, 0)
        yt = f.label_gen(test_k, test_k)
        
        pred, posterior, err = f.myNB(x,y,xt,yt)
        acc[k] += (1 - err) / n_run

plt.figure()
ax = plt.axes()
ax.set_xlabel('Training data set size')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy vs Number of training samples')
plt.plot(2 * klist,acc)

### 1c and 2c
k0 = 700                            # Number of data points in class '0' in each experiment
k1 = 300                            # Number of data points in class '1' in each experiment
k = 500                             # Number of data points in each class in the test data in each experiment
conf_matr = np.zeros((2,2))
prec = 0
rec = 0
acc = 0
auc = 0
n_run = 10
for i in range(n_run):              # Running the experiment 'n_run' times to account for random datasets
    if i == 0:
        roc_plot = 1
    else:
        roc_plot = 0
    
    m0 = [1, 0]
    c0 = [[1, 0.75], [0.75, 1]]
    
    m1 = [0, 1]
    c1 = [[1, 0.75], [0.75, 1]]
    
    x = f.Gauss_mixt_data_generate(m0, c0, k0, m1, c1, k1, bins, 0)
    y = f.label_gen(k0, k1)
    
    xt = f.Gauss_mixt_data_generate(m0, c0, k, m1, c1, k, bins, 0)
    yt = f.label_gen(k, k)
    
    pred, posterior, err = f.myNB(x,y,xt,yt)
    prec0, rec0, conf_matr0, auc0 = f.error_analysis(yt, posterior, roc_plot)
    prec += prec0 / n_run
    rec += rec0 / n_run
    conf_matr += conf_matr0 / n_run
    auc += auc0 / n_run
    acc += (1 - err) / n_run
    
print('Average Accuracy:', acc )
print('Average AUC:', auc)
print('Average Precision:', prec)
print('Average Recall:', rec)
print('Average of Confusion Matrices:', conf_matr)