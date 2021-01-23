#import sys
import numpy as np
import matplotlib.pyplot as plt

f = open('Amazon_Reviews.csv','r')          # Open a file in read mode
text = f.read()                             # Read the file
text = text.replace('.','').replace('!','').replace('"','').replace(';','').replace(',','')                         # Removing puctuations & double quotes
text = text.replace('(','').replace(')','').replace('[','').replace(']','').replace('{','').replace('}','')         # Removing paranthesis
lines = text.split('\n')                    # Split up the entire text into lines
lines = lines[1:-1]                         # Remove final empty line and initial line which has header
f.close()                                   # Close file pointer

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

#bagofWords = list(bagofWords)
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

np.set_printoptions(precision=3)
#plt.figure(figsize=(40, 800), dpi = 80)             # For entire word set
plt.figure(figsize=(12,12), dpi = 80)               # For selected words
plt.matshow(tf_idf, cmap = plt.cm.Blues, fignum = 1)          # Print color map  
for i in range(len(tf_idf)):                        # Print term count
   print('Review ', str(i), ': ', tc[i])

#np.set_printoptions(threshold=sys.maxsize)
for i in range(len(tf_idf)):                        # Print tf-idf
   print('Review ', str(i), ': ', tf_idf[i])

# Sum term frequency for positive and negative words
tf_pn = np.zeros((len(tf_idf),2))
tf_pn[:,0]=np.sum(tf[:,0:5],axis=1)
tf_pn[:,1]=np.sum(tf[:,5:10],axis=1)
for i in range(len(tf_idf)):                        # Print term frequency
    print('Review ', str(i), ': ', tf_pn[i])