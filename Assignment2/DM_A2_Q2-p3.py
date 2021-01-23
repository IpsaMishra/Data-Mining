import function as f
import numpy as np

fl = open('Amazon_Reviews.csv','r')          # Open a file in read mode
text = fl.read()                             # Read the file
text = text.replace('.','').replace('!','').replace('"','').replace(';','').replace(',','')                         # Removing puctuations & double quotes
text = text.replace('(','').replace(')','').replace('[','').replace(']','').replace('{','').replace('}','')         # Removing paranthesis
lines = text.split('\n')                    # Split up the entire text into lines
lines = lines[1:-1]                         # Remove final empty line and initial line which has header
fl.close()                                   # Close file pointer

review = []                                 # Stores the reviews text in a list
label = []                                  # Stores the corresponding labels

bagofWords = set()                          # Stores the set of words in all the reviews/docs
localbagofWords = []                        # Stores the bag of words and the word count for each review/doc
wordCount = []                              # Stores total word count for each review/doc

for l in lines:
    review_label = l.split('__label__')     # Split each line into review and label using '__label__' as delimiter
    review.append(review_label[0])
    label.append(review_label[1])
    words = review_label[0].split()         # Get all words in each review
    localbagofWords.append(dict())
    wordCount.append(len(words))
    for word in words:
        bagofWords.add(word)
        if word in localbagofWords[-1]:
            localbagofWords[-1][word] += 1
        else:
            localbagofWords[-1][word] = 1

bagofWords_pos = ['beautiful','good','awesome','great','amazing']                               ## Positive Words
bagofWords_neg = ['error','bad','beware','poor','incorrect']                                    ## Negative Words
bagofWords = bagofWords_pos + bagofWords_neg

idf = np.zeros(len(bagofWords))                     # Stores idf for words
tc = np.zeros((len(lines),len(bagofWords)))         # Stores term count
tf = np.zeros((len(lines),len(bagofWords)))         # Stores term frequency
tf_idf = np.zeros((len(lines),len(bagofWords)))
for i in range(len(bagofWords)):                    # Calculate idf for words
    word = bagofWords[i]
    n_t = 0
    for docu in review:
        words = docu.split()
        if word in words:
            n_t += 1
    idf[i] = np.log(len(lines) / n_t)
    
for i in range(len(review)):                        # Calculate tf-idf matrix
    for j in range(len(bagofWords)):
        word = bagofWords[j]
        if word in localbagofWords[i]:
            tc[i,j] = localbagofWords[i][word]
            tf[i,j] = localbagofWords[i][word] / wordCount[i]
            tf_idf[i,j] = idf[j] * localbagofWords[i][word] / wordCount[i]
        else:
            tc[i,j] = 0
            tf[i,j] = 0.0
            tf_idf[i,j] = 0.0

# Sum term frequency for positive and negative words
tf_pn = np.zeros((len(tf_idf),2))
tf_pn[:,0]=np.sum(tf[:,0:5],axis=1)
tf_pn[:,1]=np.sum(tf[:,5:10],axis=1)
lab = np.zeros(len(label))
for i in range(len(label)):
    if label[i] == '1 ':
        lab[i] = 0
    else:
        lab[i] = 1
        
n_data = len(label)
kfold = 5
acc = 0
prec = 0
rec = 0
for i in range(kfold):
    lb = int(i / kfold * n_data)
    ub = int((i+1) / kfold * n_data) 
    
    x = np.concatenate((tf_pn[0:lb],tf_pn[ub:n_data]))
    y = np.concatenate((lab[0:lb],lab[ub:n_data]))
    
    xt = tf_pn[lb:ub]
    yt = lab[lb:ub]

    pred, posterior, err = f.myNB(x,y,xt,yt)
    prec0, rec0, conf_matr, auc = f.error_analysis(yt, posterior, 0)
    prec += prec0 / kfold
    rec += rec0 / kfold
    acc += (1 - err) / kfold
    
print('Average Accuracy:', acc)
print('Average Precision:', prec)
print('Average Recall:', rec)