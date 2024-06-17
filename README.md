# Bayesian Classifier to classify mails as spam or not spam

A simple bayesian classifier to classify mails as spam and not spam 

Lines 8 to 15 of the code load the .mat data and then accesses the train and test variables.
  
Lines 18 to 33 are the training phase. Here, we calculate the Pr(Xi = 1|Y = 1) and Pr(Xi = 1|Y =
0) for each word in the training dataset. The values are stored in Word 1 spam and Word 1 notspam
respectively.

Lines 38 to 59 are the testing phase. Testing is done on each row(email) of the test data. The
probabilities calculated in the training phase are being used here. And if the probability of the mail
being spam is more that that of it not being spam, then the mail is labelled as spam and vice versa.

Lines 62 to 64 calculate the prediction accuracy using the total number of correctly predicted data
divide by the total data.
y:
The Accuracy obtained is 0.941 or 94.1 %.
