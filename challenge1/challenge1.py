# Challenge 1 
# Unsupervised Learning

# Data file:  allData.csv (1 GB)
#                  13 parameters measured over a period of 6 months
#                   the first column contains row numbres, 
#                    comma separated columns, 
#                    some columns correspond to different components of a given parameter
#                    3, 368, 016 rows by 27 columns (one is row number, one is date)
#                    all columns have nans other than number, date, V18, V19, V20
# Goals:
#  1. find relationships between parameters
#  2. classify data into at least 6 groups (all data points need to be classified)
#  3. chactrize each of these groups
#  provdie a file that contains row numbers and corresponding identifiers as two columsns
#%matplotlib inline
import pandas as pd
import csv
import pdb
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
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
# 1. Load, Clean and Organize Data
dir='/Users/nicolegoebel/Google Drive/Challenge1/'
fileName = dir + "allData.csv"
na_values = 'nan'
df = pd.read_csv(fileName, na_values=na_values, header=None)#index_col="date", parse_dates=True
#give column header (temporary?) names (guess)
df.columns = ['number', 'date', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25']
#convert data column to dates and set index to date
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')
# investigate messiness
#df.describe()

# 1a.  raw plots
#df1_5 = df.iloc[:, 1:6]
#df6_10 = df.iloc[:, 6:11]
#df11_15 = df.iloc[:, 11:16]
#df16_20 = df.iloc[:, 16:21]
#df21_25 = df.iloc[:, 21:26]
#df1_5.plot(subplots=True, sharex=True);plt.savefig('V1_5_raw2.png')
#df6_10.plot(subplots=True, sharex=True);plt.savefig('V6_10_raw2.png')
#df11_15.plot(subplots=True, sharex=True);plt.savefig('V11_15_raw2.png')
#df16_20.plot(subplots=True, sharex=True);plt.savefig('V16_20_raw2.png')
#df21_25.plot(subplots=True, sharex=True);plt.savefig('V21_25_raw2.png')

# 1b. Clean Data column by column (!!)
# In future, remove data +/- > 2-3 Stdevs
# also keep code DRY (use lambda function)
dfnums = df.ix[:, 'number']
df01=df.ix[:, ['V1']]
df02=df.ix[:, ['V2']]
df03=df.ix[:, ['V3']]
df04=df.ix[:, ['V4']]
df05=df.ix[:, ['V5']]
df06=df.ix[:, ['V6']]
df07=df.ix[:, ['V7']]
df08=df.ix[:, ['V8']]
df09=df.ix[:, ['V9']]
df10=df.ix[:, ['V10']]
df11=df.ix[:, ['V11']]
df12=df.ix[:, ['V12']]
df13=df.ix[:, ['V13']]
df14=df.ix[:, ['V14']]
df15=df.ix[:, ['V15']]
df16=df.ix[:, ['V16']]
df17=df.ix[:, ['V17']]
df18=df.ix[:, ['V18']]
df19=df.ix[:, ['V19']]
df20=df.ix[:, ['V20']]
df21=df.ix[:, ['V21']]
df22=df.ix[:, ['V22']]
df23=df.ix[:, ['V23']]
df24=df.ix[:, ['V24']]
df25=df.ix[:, ['V25']]

# remove values from each variable, and create a vector which will be plotted below
df02=df02.where(df02<2000, np.nan);df02=df02.where(df02>-1000);#df02.plot();plt.savefig('V02tmp.png')
df03=df03.where(df03<1000, np.nan);df03=df03.where(df03>-1000);#df03.plot();#plt.savefig('V03tmp.png')
df04=df04.where(df04<700, np.nan);df04=df04.where(df04>-1000);#df04.plot();#plt.savefig('V04tmp.png')
df05=df05.where(df05<1500, np.nan);df05=df05.where(df05>-2000);#df05.plot();#plt.savefig('V05tmp.png')
df06=df06.where(df06<2000, np.nan);df06=df06.where(df06>-2000);#df06.plot();#plt.savefig('V06tmp.png')
df07=df07.where(df07<700, np.nan);df07=df07.where(df07>-700);#df07.plot();#plt.savefig('V07tmp.png')
df08=df08.where(df08<500, np.nan);df08=df08.where(df08>-1e+5);#df08.plot();#plt.savefig('V08tmp.png')
df09=df09.where(df09<2000, np.nan);df09=df09.where(df09>-1e+5);#df09.plot();#plt.savefig('V09tmp.png')
df10=df10.where(df10<200, np.nan);df10=df10.where(df10>-1e+5);#df10.plot();#plt.savefig('V10tmp.png')
df11=df11.where(df11<20, np.nan);df11=df11.where(df11>-1e+5);#df11.plot();#plt.savefig('V11tmp.png')
df12=df12.where(df12<15, np.nan);df12=df12.where(df12>-15);#df12.plot();#plt.savefig('V12tmp.png')
df13=df13.where(df13<2, np.nan);df13=df13.where(df13>-1e+5);#df13.plot();#plt.savefig('V13tmp.png')
df14=df14.where(df14<20, np.nan);df14=df14.where(df14>-2);#df14.plot();#plt.savefig('V14tmp.png')
df15=df15.where(df15<2, np.nan);df15=df15.where(df15>-1e+5);#df15.plot();#plt.savefig('V15tmp.png')
df16=df16.where(df16<20, np.nan);df16=df16.where(df16>0);#df16.plot();#plt.savefig('V16tmp.png')
df17=df17.where(df17<1.5, np.nan);df17=df17.where(df17>-1e+5);#df17.plot();#plt.savefig('V17tmp.png')
df18=df18.where(df18<20, np.nan);df18=df18.where(df18>-2);#df18.plot();#plt.savefig('V18tmp.png')
df19=df19.where(df19<20, np.nan);df19=df19.where(df19>-1e+5);#df19.plot();#plt.savefig('V19tmp.png')
df20=df20.where(df20<20, np.nan);df20=df20.where(df20>-1e+5);#df20.plot();#plt.savefig('V20tmp.png')
df21=df21.where(df21<600, np.nan);df21=df21.where(df21>-600);#df21.plot();#plt.savefig('V21tmp.png')
df22=df22.where(df22<600, np.nan);df22=df22.where(df22>-600);#df22.plot();#plt.savefig('V22tmp.png')
df23=df23.where(df23<600, np.nan);df23=df23.where(df23>-600);#df23.plot();#plt.savefig('V23tmp.png')
df24=df24.where(df24<600, np.nan);df24=df24.where(df24>-600);#df24.plot();#plt.savefig('V24tmp.png')
df25=df25.where(df25<2000, np.nan);df25=df25.where(df25>-1e+5);#df25.plot();#plt.savefig('V25tmp.png')

# create a data frame with cleaned data
dfClean= df01
dfClean['V2'] = df02
dfClean['V3'] = df03
dfClean['V4'] = df04
dfClean['V5'] = df05
dfClean['V6'] = df06
dfClean['V7'] = df07
dfClean['V8'] = df08
dfClean['V9'] = df09
dfClean['V10'] = df10
dfClean['V11'] = df11
dfClean['V12'] = df12
dfClean['V13'] = df13
dfClean['V14'] = df14
dfClean['V15'] = df15
dfClean['V16'] = df16
dfClean['V17'] = df17
dfClean['V18'] = df18
dfClean['V19'] = df19
dfClean['V20'] = df20
dfClean['V21'] = df21
dfClean['V22'] = df22
dfClean['V23'] = df23
dfClean['V24'] = df24
dfClean['V25'] = df25
dfClean.dropna(inplace=True)   #drop rows with na  13% of rows have a nan
#pdb.set_trace()

#test features selection VarianceThreshold which reomoves all features whos variance doesnt meeet a grheshold.
#  removes low variance features - only considers X (not y)
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold = 0.95) # for 2 std, 99.7 for 3 stds ,  68% for 1std
# selects 1-10 and 21-25, wheras my method selected 21-25,  1-4, 8, 13, 14, 17
sel.fit_transform(dfClean)
## plot up cleaned data
#df1_5 = df01
#df1_5['V2'] = df02
#df1_5['V3'] = df03
#df1_5['V4'] = df04
#df1_5['V5'] = df05
#pdb.set_trace()
#fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(8, 5))
#df1_5['V1'].plot(ax=axes[0], ylim=(0,200)); axes[0].set_title('V1'); axes[0].set_xticklabels(' '); axes[0].set_xlabel(' ')#label.set_visible(False)
#df1_5['V2'].plot(ax=axes[1], ylim=(-1000,1000)); axes[1].set_title('V2');axes[1].set_xticklabels(' '); axes[1].set_xlabel(' ');
#df1_5['V3'].plot(ax=axes[2], ylim=(-500,500)); axes[2].set_title('V3');axes[2].set_xticklabels(' '); axes[2].set_xlabel(' ')
#df1_5['V4'].plot(ax=axes[3], ylim=(-400,400)); axes[3].set_title('V4');axes[3].set_xticklabels(' '); axes[3].set_xlabel(' ');
#df1_5['V5'].plot(ax=axes[4], ylim=(-1000,1000)); axes[4].set_title('V5');axes[4].set_xticklabels(' '); axes[4].set_xlabel(' ');
#fig.savefig('V1_5_new.png')
#plt.close("all")

#df6_10 = df06
#df6_10['V7'] = df07
#df6_10['V8'] = df08
#df6_10['V9'] = df09
#df6_10['V10'] = df10
#fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(8, 5))
#df6_10['V6'].plot(ax=axes[0], ylim=(-500,500)); axes[0].set_title('V6');axes[0].set_xticklabels(' ');axes[0].set_xlabel(' ')#label.set_visible(False)
#df6_10['V7'].plot(ax=axes[1], ylim=(-500,500)); axes[1].set_title('V7');axes[1].set_xticklabels(' ');axes[1].set_xlabel(' ');
#df6_10['V8'].plot(ax=axes[2], ylim=(0,120)); axes[2].set_title('V8');axes[2].set_xticklabels(' ');axes[2].set_xlabel(' ')
#df6_10['V9'].plot(ax=axes[3], ylim=(0,130)); axes[3].set_title('V9');axes[3].set_xticklabels(' ');axes[3].set_xlabel(' ');
#df6_10['V10'].plot(ax=axes[4], ylim=(0,120)); axes[4].set_title('V10');axes[4].set_xticklabels(' ');axes[4].set_xlabel(' ');
#fig.savefig('V6_10_new.png')
#plt.close("all")

#df11_15 = df11
#df11_15['V12'] = df12
#df11_15['V13'] = df13
#df11_15['V14'] = df14
#df11_15['V15'] = df15

#plt.close("all")
#fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(8, 5))
#df11_15['V11'].plot(ax=axes[0], ylim=(0, 7)); axes[0].set_title('V11', fontsize=8);axes[0].set_xticklabels(' ');axes[0].set_xlabel(' ');axes[0].#tick_params(axis='both', which='major', labelsize=10)
#df11_15['V12'].plot(ax=axes[1], ylim=(0, 7)); axes[1].set_title('V12', fontsize=8);axes[1].set_xticklabels(' ');axes[1].set_xlabel(' ');axes[1].tick_params(axis='both', which='major', labelsize=10)
#df11_15['V13'].plot(ax=axes[2], ylim=(-.5,1)); axes[2].set_title('V13', fontsize=8);axes[2].set_xticklabels(' ');axes[2].set_xlabel(' ');axes[2].tick_params(axis='both', which='major', labelsize=10)
#df11_15['V14'].plot(ax=axes[3], ylim=(-.5, .5)); axes[3].set_title('V14', fontsize=8);axes[3].set_xticklabels(' ');axes[3].set_xlabel(' ');axes[3].tick_params(axis='both', which='major', labelsize=10)
#df11_15['V15'].plot(ax=axes[4], ylim=(0,2)); axes[4].set_title('V15', fontsize=8);axes[4].tick_params(axis='both', which='major', labelsize=10)
#fig.savefig('V11_15_new3.png')
#fig.tight_layout()

#df16_20 = df16
#df16_20['V17'] = df17
#df16_20['V18'] = df18
#df16_20['V19'] = df19
#df16_20['V20'] = df20
#fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(8, 5))
###df16_20['V16'].plot(ax=axes[0], style = 'r', label='Series', ylim=(0,8)); axes[0].set_title('V16', fontsize=12)
#df16_20['V16'].plot(ax=axes[0], ylim=(0,8)); axes[0].set_title('V16', fontsize=12);axes[0].set_xticklabels(' ');axes[0].set_xlabel(' ')
#df16_20['V17'].plot(ax=axes[1], ylim=(-1.0, 1.0)); axes[1].set_title('V17', fontsize=12);axes[1].set_xticklabels(' ');axes[1].set_xlabel(' ');
#df16_20['V18'].plot(ax=axes[2], ylim=(-0.5, 1.0)); axes[2].set_title('V18', fontsize=12);axes[2].set_xticklabels(' ');axes[2].set_xlabel(' ')
#df16_20['V19'].plot(ax=axes[3], ylim=(-0.5, 1.0)); axes[3].set_title('V19', fontsize=12);axes[3].set_xticklabels(' ');axes[3].set_xlabel(' ');
#df16_20['V20'].plot(ax=axes[4], ylim=(0,7)); axes[4].set_title('V20', fontsize=12);axes[4].set_xticklabels(' ');axes[4].set_xlabel(' ');
#fig.savefig('V16_20_new.png')
#plt.close("all")

#df21_25 = df21
#df21_25['V22'] = df22
#df21_25['V23'] = df23
#df21_25['V24'] = df24
#df21_25['V25'] = df25
#pdb.set_trace()
#fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(8, 5))
#df21_25['V21'].plot(ax=axes[0], ylim=(-500, 600)); axes[0].set_title('V1'); axes[0].set_xticklabels(' ');axes[0].set_xlabel(' ')#label.set_visible(False)
#df21_25['V22'].plot(ax=axes[1], ylim=(-600, 600)); axes[1].set_title('V2');axes[1].set_xticklabels(' ');axes[1].set_xlabel(' ');
#df21_25['V23'].plot(ax=axes[2], ylim=(-400, 400)); axes[2].set_title('V3');axes[2].set_xticklabels(' ');axes[2].set_xlabel(' ')
#df21_25['V24'].plot(ax=axes[3], ylim=(0,600)); axes[3].set_title('V4');axes[3].set_xticklabels(' ');axes[3].set_xlabel(' ');
#df21_25['V25'].plot(ax=axes[4], ylim=(0,1500)); axes[4].set_title('V5');axes[4].set_xticklabels(' ');axes[4].set_xlabel(' ');
#fig.savefig('V21_25_new.png')
#plt.close("all")
##----------------------descriptive statistics--------------------------------
print(df.describe())
print(df.corr())
dfCorr = df.corr()
dfCorr.to_csv('rawData_corr.csv')
print(df.cov())
dfCov = df.cov()
dfCov.to_csv('rawData_cov.csv')
print(dfClean.describe())
print(dfClean.corr())
dfCorr = dfClean.corr()
#----------------------- Plot Correlation Matrix-------------------------------------
#import matplotlib
from pylab import *
##import numpy as np
dfCorrVals=abs(dfCorr.values)
##Create test data with zero valued diagonal:
rows, cols = np.indices((25,25))
dfCorrVals[np.diag(rows, k=0), np.diag(cols, k=0)] = 0
#Create new colormap, with white for zero 
# create mask
mask = np.diag(np.ones(25))
#Apply mask to data:
masked_data = np.ma.masked_array(dfCorrVals, mask)
#Set mask color to white:
cm.jet.set_bad(color='white', alpha=None)
#for this to work we use pcolormesh instead of pcolor:
pcolormesh(masked_data, cmap=cm.jet)
xlabel ('Variable 1 through 25')
ylabel ('Variable 1 through 25')
colorbar()
savefig('CorrmapAbs.png')

#dfCorr.to_csv('cleanData_corr.csv')
#print(dfClean.cov())
#dfCov = dfiCean.cov()
#dfCov.to_csv('cleanData_cov.csv')

#-----------------------------method for selecting low corr vars--------------------------------------
#take first component(first column of the plot), it is correlated with 11,12,16, and 20. so you only need to pick one of them.
#keep 1, remove 11,12,16,20
#keep={1}; reject={11,12,,16,20}
#second column is correlated with 5, so keep 2, remove 5. 
#keep={1,2}; reject={11,12,,16,20, 5}
#third column is correlated with 6
#keep={1, 2, 3};reject={11,12,,16,20, 5,6}
#fourth: is corelated with 7
#keep={1, 2, 3,4};reject={11,12,,16,20, 5,6,7}
#skip column 5-7 because they are in the reject list. '
#next 8,9,10 are correlated, keep only one of them
#keep={1, 2, 3,4,8};reject={11,12,,16,20, 5,6,7,9,10}
#11 and 12 are in the reject list
#column 13 is corelated with 15:
#keep={1, 2, 3,4,8,13};reject={11,12,,16,20, 5,6,7,9,10,15}
#following this procedure:
#keep={1, 2, 3,4,8,13,14};reject={11,12,,16,20, 5,6,7,9,10,15,18}
#keep={1, 2, 3,4,8,13,14};reject={11,12,,16,20, 5,6,7,9,10,15,18}
#keep={1, 2, 3,4,8,13,14,17};reject={11,12,,16,20, 5,6,7,9,10,15,18,19}
#keep={1, 2, 3,4,8,13,14,17};reject={11,12,,16,20, 5,6,7,9,10,15,18,19}
#21-25 are not strongly correlated with other vars
#keep={1, 2, 3,4,8,13,14,17,21,22,23,24,25};reject={11,12,,16,20, 5,6,7,9,10,15,18,19}
#length(keep)=13

#put selected features into a data frame
dfFinal= df01
dfFinal['V1'] = df01
dfFinal['V2'] = df02
dfFinal['V3'] = df03
dfFinal['V4'] = df04
dfFinal['V8'] = df08
dfFinal['V13'] = df13
dfFinal['V14'] = df14
dfFinal['V17'] = df17
dfFinal['V21'] = df21
dfFinal['V22'] = df22
dfFinal['V23'] = df23
dfFinal['V24'] = df24
dfFinal['V25'] = df25
dfFinal['row_number'] = dfnums

del dfClean, df, df01, df02, df03, df04, df05, df06, df07, df08, df09, df10, df11, df12, df13, df14, df15, df16, df17, df18, df19, df20, df21, df22, df23, df24, df25 #,df1_5, df6_10, df11_15, df16_20, df21_25
#remove rows with NaN
npArray=dfFinal.dropna(how='any') #if any column has an nan, drop that row.  # 2932362 rows, 14 columns, was 3368018, 14 - lost 435,000 rows
# get numpy array (omit index)
dfrow_number = npArray.ix[:, 'row_number']
dfrow_number = dfrow_number.values #1 x 2932362
npArray = npArray.values
npArray = npArray[:,0:13] #get rid of row numbers column  (save above so that I can attach labels to them)

# -------------1. kmeans test of clusters: determine number of clusters based on within cluster SOSs---------------------
## kmeans
#from scipy.spatial.distance import cdist
#from scipy.cluster import vq##### cluster data into K=3..10 clusters #####
##K, KM, centroids,D_k,cIdx,dist,avgWithinSS = kmeans.run_kmeans(npArray,10)
#K = range(1,15)
## scipy.cluster.vq.kmeans
#KM = [vq.kmeans(npArray,k) for k in K] # apply kmeans 1 to 10
#centroids = [cent for (cent,var) in KM]   # cluster centroids
#D_k = [cdist(npArray, cent, 'euclidean') for cent in centroids]
#cIdx = [np.argmin(D,axis=1) for D in D_k]
#dist = [np.min(D,axis=1) for D in D_k]
#avgWithinSS = [sum(d)/npArray.shape[0] for d in dist]  
##plot elbow curve
#kIdx = 5
#kIdx2 = 3
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.plot(K, avgWithinSS, 'b*-')
#ax.plot(K[kIdx], avgWithinSS[kIdx], marker='o', markersize=12, markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
#ax.plot(K[kIdx2], avgWithinSS[kIdx2], marker='o', markersize=12, markeredgewidth=2, markeredgecolor='g', markerfacecolor='None')
#plt.grid(True)
#plt.xlabel('Number of clusters')
#plt.ylabel('Average within-cluster sum of squares')
#pdb.set_trace()
#plt.savefig('elbowPlot1to15.png')
#save("elbowPlot1", ext='png', close=True, verbose=True)
#plt.title('Elbow Plot for K-Means clustering')  
#plt.savefig('elbowPlot1to15_title.png')

#----------------2. With known cluster number use Kmeans in Scipy to calculate clusters/groups/classes
#cluster the data as per our requirements and it returns the centroids of the clusters
#centers,dist = vq.kmeans(npArray,6)
#label the data with vq.vq() which takes test data and centroids as inputs 
# and outputs the labelled data,called 'labels' and distance between each data and corresponding centroids.
#labels, distance = vq.vq(npArray,centers)
#rows_labels = zip(dfrow_number, labels)
#dfout=pd.DataFrame(rows_labels)
#dfout.columns = ['row_number', 'class']
#dfout.to_csv('challenge1_rows_labels.csv')
# --------------- 2. With known cluster number, use KMeans model to fit clusters to data, then plot clusters
from sklearn.cluster import KMeans
#initialize and carry out clustering
n_clusters=6
km = KMeans(n_clusters = n_clusters, init='k-means++') # initialize
km.fit(npArray)
k_means_labels = km.labels_
k_means_cluster_centers = km.cluster_centers_
k_means_labels_unique = np.unique(k_means_labels)
k_means_cluster_centers[k_means_cluster_centers<0] = 0 #the minimization function may find very small negative numbers, we threshold them to 0
k_means_cluster_centers = k_means_cluster_centers.round(2)

#c = km.predict(npArray) # classify into six clusters

#plot
colors = ['#a020f0', '#ff4500','#ffff00', '#32cd32', '#00ffff', '#ff1493']
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(1, 1, 1)
for k, col in zip(range(n_clusters), colors):
    my_members = k_means_labels == k
    cluster_center = k_means_cluster_centers[k]
    ax.plot(npArray[my_members, 0], npArray[my_members, 1], 'w',markerfacecolor=col, marker='.')
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,markeredgecolor='k', markersize=6, label = "{0}".format(k))
#ax.set_title('KMeans {0} Clusters'.format(n_clusters))
ax.set_xticks(())
ax.set_yticks(())
plt.legend(frameon=False)

plt.savefig("Kmeans_{0}clusters_0vs1a.png".format(n_clusters))



colors = ['#a020f0', '#ff4500','#ffff00', '#32cd32', '#00ffff', '#ff1493']
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(1, 1, 1)
for cc in range(n_clusters):
    plt.close("all")
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    for k, col in zip(range(n_clusters), colors):
        my_members = k_means_labels == k
        cluster_center = k_means_cluster_centers[k]
        ax.plot(npArray[my_members, 2], npArray[my_members, cc], 'w',markerfacecolor=col, marker='.')
        ax.plot(cluster_center[0], cluster_center[2], 'o', markerfacecolor=col,markeredgecolor='k', markersize=6, label = "{0}".format(k))
#ax.set_title('KMeans {0} Clusters'.format(n_clusters))
    ax.set_xticks(())
    ax.set_yticks(())
    plt.legend(frameon=False)
    plt.savefig("Kmeans_{0}clusters_2vs{1}.png".format(n_clusters,cc))
pdb.set_trace()
#---------------2b. minibatch Kmeans ---------------------------------
batch_size = 45
n_clusters=6

mbk = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, batch_size=batch_size,n_init=10, max_no_improvement=10, verbose=0)
t0 = time.time()
mbk.fit(npArray)
t_mini_batch = time.time() - t0
mbk_means_labels = mbk.labels_
mbk_means_cluster_centers = mbk.cluster_centers_
mbk_means_labels_unique = np.unique(mbk_means_labels)
#write to file
# get numpy array (omit index)
row_labels = np.vstack((dfrow_number, mbk_means_labels)).T
dfout=pd.DataFrame(row_labels)
dfout.columns = ['row_number', 'class']
dfout.to_csv('challenge1_rows_labels_mbkmeans.csv', index=False)

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(1, 1, 1)
for k, col in zip(range(n_clusters), colors):
    my_members = mbk_means_labels == order[k]
    cluster_center = mbk_means_cluster_centers[order[k]]
    ax.plot(npArray[my_members, 0], npArray[my_members, 1], 'w',
            markerfacecolor=col, marker='.')
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
            markeredgecolor='k', markersize=6, label = "{0}".format(k))
#ax.set_title("MiniBatch KMeans {0} Clusters".format(n_clusters))
ax.set_xticks(())
ax.set_yticks(())
plt.legend(frameon=False)
plt.savefig("minibatchKmeans_{0}clusters.png".format(n_clusters))
