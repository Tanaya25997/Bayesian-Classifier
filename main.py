import scipy.io
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)


# Load the .mat file
mat_contents = scipy.io.loadmat('spamdata.mat')

# Access train and test variables in the .mat file
Training_X = mat_contents['X']
Training_y = mat_contents['y']
Test_X2 = mat_contents['X2']
Test_y2 = mat_contents['y2']


####### Implement Bayes Classifier - Training #######



Y_equals_1 = np.count_nonzero(Training_y == 1)
#print(Y_equals_1)
Y_equals_0 = len(Training_y) - Y_equals_1
Pr_y_1 = Y_equals_1/len(Training_y) 
Pr_y_0 = 1 - Pr_y_1


############### Train #################
k = 1
Word_1_spam = []
Word_1_notspam = []

    
for i in range(Training_X.shape[1]):

    Word_1_spam.append((np.sum((Training_X[:, i] == 1) & (Training_y.flatten() == 1))+k)/(Y_equals_1+2*k))
    Word_1_notspam.append((np.sum((Training_X[:, i] == 1) & (Training_y.flatten() == 0))+k)/(Y_equals_0+2*k))
    
    
################ Test #################

pred = []
for j in range(Test_X2.shape[0]):
    ##### Calcuate p(spam|words) ######
    spam = Pr_y_1
    for m in range(Test_X2.shape[1]):
        if Test_X2[j][m] == 1:
            spam = spam*(Word_1_spam[m])
        else:
            spam = spam*(1-Word_1_spam[m])
    
    ##### Calucltae p(notspam|words) #######
    not_spam = Pr_y_0
    for n in range(Test_X2.shape[1]):
        if Test_X2[j][n] == 1:
            not_spam = not_spam*(Word_1_notspam[n])
        else:
            not_spam = not_spam*(1-Word_1_notspam[n])
            
    if not_spam > spam:
        pred.append(0)
    else:
        pred.append(1)
        
        
######## Caluculate Accuracy #########
pred = np.array(pred)
Accuracy = np.sum(np.equal(pred, Test_y2.flatten())) / len(Test_y2)
print("Prediction Accuracy = ", Accuracy)



    






