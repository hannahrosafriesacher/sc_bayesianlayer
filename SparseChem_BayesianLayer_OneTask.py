import sparsechem as sc
import numpy as np
import scipy
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from sklearn.utils.estimator_checks import check_estimator
from sklearn.metrics import mean_squared_error as mse
from skbayes.linear_models import VBLogisticRegression, EBLinearRegression
import matplotlib.pyplot as plt
import math
import scipy.stats as sci


shape=1
rate=0.01

values=[0.0, 0.1, 0.2 ,0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#---------Some Useful Functions---------------
#split array according to condition
def split(arr, cond):
    return arr[cond]

#Calculate positive ratio (=accuracy)
#if there are no measurements (=no predictions) in a split: add 0 to acc list
#Note: if 0 is added to the list, the difference between acc and  conf is the conf of this split
def posRatio(arr, dimension):
    if np.unique(arr, axis=dimension).shape[dimension]>1:
        return (arr==1).sum()/arr.shape[dimension]
    else:
        return np.array(0)

#Calculate Mean of Probablities in Column (=confidence)
#if there are no measurements(=no predictions) in a split: the confidence is calculated from the values list
def ProbMean(arr, dimension, ind):
    if arr.shape[dimension]!=0:
        mean=np.mean(arr)
        return(mean)
    else:
        return values[ind]+0.5

#count positives/negatives in each class
def selectPos(arr):
    pos=np.count_nonzero(arr==1)
    #print('pos', pos)
    return pos
def selectNeg(arr):
    neg=np.count_nonzero(arr==-1)
    #print('neg', neg)
    return neg
 
#AUC_ROC
def auc_roc(y_hat, y_class):
    return roc_auc_score(y_class, y_hat)
#AUC_PR
def auc_pr(y_hat,y_class):
    precision, recall, thresholds = precision_recall_curve(y_class, y_hat)
    return auc(recall, precision)
    return 
    
    
#function ECE
def calculate_ECE(y_hat, y_class):
    prob=[]
    acc=[]
    conf=[]
    clas=[]
    NumPos=[]
    NumNeg=[]   
    j=0
    k=0
    m=0
    #split values according to values-list (0.0, 0.1, 0.2...) 
    for j in range(10):
        clas.append(split(y_class,np.logical_and(y_hat>=values[j], y_hat<values[j+1])))
        prob.append(split(y_hat,np.logical_and(y_hat>=values[j], y_hat<values[j+1])))
        j+=1
        #print('clas:', clas)

    #Obtain positive ratio (=acc calculated from true values) and 
    # probablity mean (=conf calculated from predictions) for each split
    for k in range(10):
        acc.append(posRatio(clas[k], 0))
        conf.append(ProbMean(prob[k], 0, k))
        NumPos.append(selectPos(clas[k]))
        NumNeg.append(selectNeg(clas[k]))
        k+=1

    #obtain ECE for this specific target:
    ece=0
    for m in range(10):
        ece+=(np.abs(np.array(acc[m])-np.array(conf[m]))*clas[m].shape[0])
        #      |               acc(b)-         conf(n)| * nb

    ece_final=ece/(y_class.shape[0])
    return ece_final, NumPos, NumNeg

def calculate_ACE(y_hat, y_class):
    m=0
    y_hat_selected_ace=y_hat.flatten()
    y_class_selected_ace=y_class.flatten()

    #sort class and hat file by ascending probablity values in hat file
    index_sort_y_hat=np.argsort(y_hat_selected_ace)
    y_hat_sorted=y_hat_selected_ace[index_sort_y_hat]
    y_class_sorted=y_class_selected_ace[index_sort_y_hat]

    #divide in 10 classes with equal numbers of predictions
    y_hat_split=np.array_split(y_hat_sorted, 10)
    y_class_split=np.array_split(y_class_sorted, 10)

    acc_ace=[]
    conf_acc=[]

    #Obtain positive ratio (=acc calculated from true values) and 
    #probablity mean (=conf calculated from predictions) for each split
    for m in range(10):
        acc_ace.append(posRatio(y_class_split[m], 0))
        conf_acc.append(ProbMean(y_hat_split[m], 0, m))

    acc_ace=np.array(acc_ace)
    conf_acc=np.array(conf_acc)

    #obtain ACE for this specific target:
    ace=np.sum(np.abs(acc_ace-conf_acc))/10
    #     SumOverAllR(|acc(b)-conf(b)|)/R
    
    return ace

#-------------------calculate ACE/ACE for NN with Bayesian Layer------------------------------------

#Load files with more than 5 actives/inactives in each fold
hidden_layer_vafold=np.load('/home/rosa/git/SparseChem_new2/src/sparsechem/examples/chembl/sc_run_h1000_ldo_r_wd0.0001_lr0.001_lrsteps10_ep20_fva1_TargetsWithMoreThan4Actives-hidden.npy')
hidden_layer_tefold=np.load('/home/rosa/git/SparseChem_new2/src/sparsechem/examples/chembl/sc_run_h1000_ldo_r_wd0.0001_lr0.001_lrsteps10_ep20_fte0_TargetsWithMoreThan4Actives-hidden.npy')
x=sc.load_sparse('/home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/chembl_29_X.npy')
y_class=sc.load_sparse('/home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/SelectedTargetsWithMoreThanNMeasurements/Chembl29_y_class_TargetsWithMoreThan4ActivesInactives.npy')
folding=np.load('/home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/folding.npy')

TargetID=617

keep_te    = np.isin(folding, 1)
keep_va    =np.isin(folding,0)

y_class_te = sc.keep_row_data(y_class, keep_te)
y_class_va = sc.keep_row_data(y_class, keep_va)
y_class_te=y_class_te.toarray()
y_class_va=y_class_va.toarray()


#select current target
y_class_te_now=y_class_te[:,TargetID]
y_class_va_now=y_class_va[:,TargetID]
#obtain nonzero-arrays ()
y_class_nonzero_te=y_class_te_now[np.nonzero(y_class_te_now)]
hidden_nonzero_te=hidden_layer_tefold[np.nonzero(y_class_te_now)]
y_class_nonzero_va=y_class_va_now[np.nonzero(y_class_va_now)]
hidden_nonzero_va=hidden_layer_vafold[np.nonzero(y_class_va_now)]

#Exchange -1 by 0
#y_class_nonzero_te[y_class_nonzero_te==-1]=0
#y_class_nonzero_va[y_class_nonzero_va==-1]=0
    
#Train model on Validation fold
vblr=VBLogisticRegression(a=rate, b=shape)
vblr.fit(hidden_nonzero_va, y_class_nonzero_va)
#Predict Test fold
y_hat_te=vblr.predict_proba(hidden_nonzero_te)[:, 1]

    
#ECE/ACE
#AUC_ROC
auc_roc_target=auc_roc(y_hat_te, y_class_nonzero_te)
#AUC_PR
auc_pr_target=auc_pr(y_hat_te, y_class_nonzero_te)
#ECE:
result=calculate_ECE(y_hat_te, y_class_nonzero_te)
ECE_target=result[0]
NumPos=result[1]
NumNeg=result[2]
#ACE
ACE_target=calculate_ACE(y_hat_te, y_class_nonzero_te)


print('AUC_ROC: ', auc_roc_target)
print('AUC_Pr: ', auc_pr_target)
print('ECE: ', ECE_target)
print('ACE: ', ACE_target)
print('Parameters of Gamma Prior on precision parameter of coefficients')
print('shape: ', shape)
print('rate: ', rate)

#Plot Calibration plot
#Obtain positive ratio and positive/negative counts for compounds with predicted values betwen 0.0-0.1, 0.1-0.2, ...

Q1=[]
Q3=[]
Me=[]
Med=[]
i=0
WhisLo=[]
WhisHi=[]
stats_box=[]
for i in range(10):
    Stats={}
    q1=0
    q3=0
    whislo=0
    whishi=0
    me=0
    med=0
    q1,q3=sci.beta.interval(0.5, NumPos[i], NumNeg[i])
    whislo, whishi=sci.beta.interval(0.95, NumPos[i], NumNeg[i])
    me=sci.beta.mean(NumPos[i], NumNeg[i])
    med=sci.beta.median(NumPos[i], NumNeg[i])

    Stats['med']=med
    Stats['q1']=q1
    Stats['q3']=q3
    Stats['whislo']=whislo
    Stats['whishi']=whishi

    stats_box.append(Stats)
    i+=1

#Prepare for Plotting
X_axis_bar= ['0.0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5','0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0-9', '0.9-1.0'] 
 
fig, axs=plt.subplots(2, 1, figsize=(9,9))
#plot a box plot for each 'probability class'
axs[0].bxp(stats_box, showfliers=False, meanline=True)
axs[0].set_xticklabels(X_axis_bar)
axs[0].set_title('Positive Ratio', fontsize='x-large')
axs[0].set_xlabel('predicted activity')
axs[0].set_ylabel('positive ratio')
axs[0].axline([1,0.05], [10,0.95], color='r', linestyle='--')
#PLot number of compounds for each 'probability class'
num_measurements=np.add(np.array(NumNeg), np.array(NumPos))
heights, bins= np.histogram(num_measurements/np.sum(num_measurements))
axs[1].bar(bins[:-1],num_measurements/np.sum(num_measurements), align='center', tick_label=X_axis_bar, width=bins[1]-bins[0])
axs[1].set_title('Counts', fontsize='x-large')
axs[1].set_xlabel('predicted activity')
axs[1].set_ylabel('relative frequency')

#fig.suptitle('Target-ID:'+ str(TargetID) + '\n Total Number of Bioactivities:' + str(y_hat_selected.shape[1]),fontsize='xx-large' )
fig.tight_layout()

#save figure
plt.savefig('/home/rosa/git/SparseChem_new2/src/sparsechem/examples/chembl/SparseChem_BayesianLayer_CalibrationPlots/Logistic_CalibrationPlot_TargetID'+str(TargetID)+'_shape'+str(shape)+'_rate'+str(rate)+'.png')







