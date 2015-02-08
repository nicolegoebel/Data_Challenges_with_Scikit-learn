# Challenge 2 
# Supervised Learning

# Data files: challengeTrainy.csv, challengeTrainx.csv, challengeTextX.csv
# Goals:
#    2. To provide accuracy estimates for the overall classification exercise 
import csv
import pandas as pd
#import pandas as DataFrame, read_csv
import pdb
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split

#--------------saving figures--------------------
def save(path, ext='png', close=True, verbose=True):
    """Save a figure from pyplot.

    Parameters
    ----------
    path : string
        The path (and filename, without the extension) to save the
        figure to.

    ext : string (default='png')
        The file extension. This must be supported by the active
        matplotlib backend (see matplotlib.backends module).  Most
        backends support 'png', 'pdf', 'ps', 'eps', and 'svg'.

    close : boolean (default=True)
        Whether to close the figure after saving.  If you want to save
        the figure multiple times (e.g., to multiple formats), you
        should NOT close it in between saves or you will have to
        re-plot it.

    verbose : boolean (default=True)
        Whether to print information about when and where the image
        has been saved.

        e.g. save("signal", ext="png", close=True, verbose=True)

    """
    
    # Extract the directory and filename from the given path
    directory = os.path.split(path)[0]
    filename = "%s.%s" % (os.path.split(path)[1], ext)
    if directory == '':
        directory = '.'
 
    # If the directory does not exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)
 
    # The final path to save to
    savepath = os.path.join(directory, filename)
 
    if verbose:
        print("Saving figure to '%s'..." % savepath),
 
    # Actually save the figure
    plt.savefig(savepath)
    
    # Close it
    if close:
        plt.close()
 
    if verbose:
        print("Done")
#--------------saving figures end--------------------
# 1. load data
dir=r"/Users/nicolegoebel/Google Drive/Challenge2/"
fileName = dir + "challengeTrainX.csv"
na_values = 'nan'
xTrain = pd.read_csv(fileName, na_values=na_values, header=None)# 3957x614
fileName = dir + "challengeTrainy.csv"
yTrain = pd.read_csv(fileName, na_values=na_values, header=None)# 3957 x 1
fileName = dir + "challengeTestX.csv"
xTest = pd.read_csv(fileName, na_values=na_values, header=None)# 2639 x 614 index_col="date", parse_dates=True

# 2. Clean Data (?)
#      Divide Data into Training and Validation Sets
# check to see if I need to nornmalize features - are they all in similar ranges? plot up or just use describe()`
# if need to scale data or standardize it to have a mean 0 and variance 1, be sure to also apply to tst vector. 
#      use StandardScaler to do this.
# NO NEED TO SCALE, I BELIEVE, AS ALL VARIABLES RANGE FROM ~-4 to  ~4
#scaler = StandardScaler()
#scaler.fit(X_train)  # Don't cheat - fit only on training data
#X_train = scaler.transform(X_train)
#X_test = scaler.transform(X_test)  # apply same transformation to test data
#split data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(xTrain, yTrain, test_size=0.33, random_state=42)
y_train = np.ravel(y_train)
y_val = np.ravel(y_val)
# 3. Train Model

# random forests
from sklearn.ensemble import RandomForestClassifier
n_estimators=50
oob_score=True
model = RandomForestClassifier(n_estimators=n_estimators, oob_score=oob_score)
model.fit(x_train, y_train)
# what is oob_score?
print "out of bag score for {0} n_estimators is {1}".format(n_estimators, model.oob_score_) #.90
# What is your model's mean accuracy score on the validation and test sets?
test_scores = cross_val_score(model, x_val, y_val)
print "For {0} trees the mean of the test scores = {1}".format(n_estimators, test_scores.mean())
# make prediction
predicted = model.predict(x_val)
# What is your model's precision, recall, and F1 score on the test set?
# summarize the fit of the model
print "classifcation report for {} trees:".format(n_estimators)
# plot confusion matrix
print(metrics.confusion_matrix(y_val, predicted)

# CART 
# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(x_train, y_train)
# make predictions
predicted = model.predict(x_val)
# summarize the fit of the model
print(metrics.classification_report(y_val, predicted))
print(metrics.confusion_matrix(y_val, predicted))
cm = metrics.confusion_matrix(y_val, predicted)
plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('confusion_matrix_CART.png')

#SVM
from sklearn.linear_model import SGDClassifier
clf = SGDClassifier(loss="hinge", penalty="l2", n_iter = 10) #SVM
clf.fit(x_train, y_train)
# concrte loss function is set via loss parameter where loss="hing" ==linear SVM - a lazy loss function as only updates model params if an example violates the margin constraint. - makes for effcieint training but may reault in spaser models even when L2 penalty is used.
#SGDClassifier(alpha=0.0001, class_weight=None, epsilon=0.1, eta0=0.0,
#       fit_intercept=True, l1_ratio=0.15, learning_rate='optimal',
#       loss='hinge', n_iter=5, n_jobs=1, penalty='l2', power_t=0.5,
#       random_state=None, shuffle=False, verbose=0, warm_start=False)

# 4. Test Model  #predict new values and test for accuracy
preds = clf.predict(x_val)
score = clf.score(x_val, y_val) #mean accuracy
# summarize the fit of the model
print(metrics.classification_report(y_val, preds))
print(metrics.confusion_matrix(y_val, preds))
#test_scores = cross_val_score(clf, x_test, y_test)
cm = metrics.confusion_matrix(y_val, preds)
plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('confusion_matrix_linearSVC.png')
#from sklearn.metrics import accuracy_score
#print accuracy_score(y_test, preds)#my own calculation of accuracy.
correct = 0
for x in range(len(y_val)):
    if y_val[x]== preds[x]:
        correct += 1
acc= correct/float(len(y_val)) * 100.0

def naive_accuracy(true, pred):
    number_correct = 0
    i = 0
    for i, y in enumerate(true):
        if pred[i] == y:
            number_correct += 1.0
    return 1 - number_correct / len(true)

print classification_report(y_val, preds)
report = classification_report(y_val, preds)
dfreport = pd.DataFrame(report)
#dfout.columns = ['row_number', 'class']
with open("class_reportn10.txt", "w") as f:
    for line in report:
        f.write(line)

report.to_csv('report_precision_recall.csv', index=False)

from sklearn import metrics
print metrics.confusion_matrix(y_val, preds)
pdb.set_trace()
#clf.coef_   #model Parameters
#clf.intercept_  #intercept = offset or bias
clf.decision_function( ) #gives signed distance to hyperplane

predTest = clf.predict(xTest)
#rownum = range(1,len(xTest)+1)
#xrow_class = zip(rownum, predTest)
#dfclass=pd.DataFrame(xrow_class)
#dfclass.columns = ['row_number', 'prediction']
#dfclass.to_csv('challenge2_rows_labels', index=False)
#for x in range(len(c)):
#    if c[x]== k_means_labels[x]:
#        correct += 1
#acc= correct/float(len(c)) * 100.0


#  lets try another algo
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(x_train, y_train) 
#KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
#               metric_params=None, n_neighbors=5, p=2, weights='uniform')
knnpreds= knn.predict(x_val)
knnscore = knn.score(x_val, y_val)
#knnreport = classification_report(y_val, knnpreds)

# SVM algo
from sklearn import svm
#  linear
linsvc = svm.SVC(kernel='linear')
linsvc.fit(x_train, y_train)   
#SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0,
#  kernel='linear', max_iter=-1, probability=False, random_state=None,
#  shrinking=True, tol=0.001, verbose=False)

linsvcPreds = linsvc.predict(x_val)
linsvcScore = linsvc.score(x_val, y_val)
#linsvcReport = classifcation_report(y_val, linsvcPreds)
cm = metrics.confusion_matrix(y_val, linsvcPreds)
plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('confusion_matrix_linSVC.png')
#polynomia
poly3svc = svm.SVC(kernel='poly', degree=3)
poly3svc.fit(x_train, y_train)   
#SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0,
#  kernel='poly', max_iter=-1, probability=False, random_state=None,
#  shrinking=True, tol=0.001, verbose=False)

poly3svcPreds = poly3svc.predict(x_val)
poly3svcScore = poly3svc.score(x_val, y_val)
#poly3svcReport = classifcation_report(y_val, polysvcPreds)

# radial basis function
rbfsvc = svm.SVC(kernel='rbf')
rbfsvc.fit(x_train, y_train)   
rbfsvcPreds = rbfsvc.predict(x_val)
rbfsvcScore = rbfsvc.score(x_val, y_val)
#rbfsvcReport = classifcation_report(y_val, rbfsvcPreds)

from random import randint
colors = []
for i in range(23):
    #colors.append('%06X' % randint(0, 0xFFFFFF))
    tmp='%06X' % randint(0, 0xFFFFFF)
    label = "#{0}".format(tmp)
    #print label
    colors.append(label)
color_col=[]
for x in preds:
    color_col.append(colors[x])
#colors = [
#    "red" if x == 0 else "blue" if x == 1 else "green" for x in kmpp.clusters]
x1=x_train[:,1:2]
x2=x_train[:,2:3]
plt.scatter(x1, x2, s=22, c=color_col)
plt.savefig('svm1vs2.png')
