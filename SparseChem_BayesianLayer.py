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


shape=10
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
    j=0
    k=0
    m=0
    #split values according to values-list (0.0, 0.1, 0.2...) 
    for j in range(10):
        clas.append(split(y_class,np.logical_and(y_hat>=values[j], y_hat<values[j+1])))
        prob.append(split(y_hat,np.logical_and(y_hat>=values[j], y_hat<values[j+1])))
        j+=1

    #Obtain positive ratio (=acc calculated from true values) and 
    # probablity mean (=conf calculated from predictions) for each split
    for k in range(10):
        
        acc.append(posRatio(clas[k], 0))
        conf.append(ProbMean(prob[k], 0, k))
        k+=1

    #obtain ACE for this specific target:
    ece=0
    for m in range(10):
        ece+=(np.abs(np.array(acc[m])-np.array(conf[m]))*clas[m].shape[0])
        #      |               acc(b)-         conf(n)| * nb

    ece_final=ece/(y_class.shape[0])
    return ece_final

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
hidden_layer_vafold=np.load('/home/rosa/git/SparseChem_new2/src/sparsechem/examples/chembl/sc_run_h1000_ldo_r_wd0.0001_lr0.001_lrsteps10_ep20_fva1_ftrain34_TargetsWithMoreThan4Actives-hidden.npy')
hidden_layer_tefold=np.load('/home/rosa/git/SparseChem_new2/src/sparsechem/examples/chembl/sc_run_h1000_ldo_r_wd0.0001_lr0.001_lrsteps10_ep20_fte0_ftrain34_TargetsWithMoreThan4Actives-hidden.npy')
hidden_layer_refold=np.load('/home/rosa/git/SparseChem_new2/src/sparsechem/examples/chembl/sc_run_h1000_ldo_r_wd0.0001_lr0.001_lrsteps10_ep20_fre2_ftrain34_TargetsWithMoreThan4Actives-hidden.npy')
x=sc.load_sparse('/home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/chembl_29_X.npy')
y_class=sc.load_sparse('/home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/SelectedTargetsWithMoreThanNMeasurements/Chembl29_y_class_TargetsWithMoreThan4ActivesInactives.npy')
folding=np.load('/home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/folding.npy')



keep_te    = np.isin(folding, 0)
keep_va    =np.isin(folding,1)
keep_re    =np.isin(folding,2)

y_class_te = sc.keep_row_data(y_class, keep_te)
y_class_va = sc.keep_row_data(y_class, keep_va)
y_class_re = sc.keep_row_data(y_class, keep_re)
y_class_te=y_class_te.toarray()
y_class_va=y_class_va.toarray()
y_class_re=y_class_re.toarray()

ECE_sum=0
ACE_sum=0
AUC_ROC_sum=0
AUC_PR_sum=0

for i in range(y_class.shape[1]):
    #select current target
    print(i)
    y_class_te_now=y_class_te[:,i]
    y_class_va_now=y_class_va[:,i]
    y_class_re_now=y_class_re[:,i]
    #obtain nonzero-arrays ()
    y_class_nonzero_te=y_class_te_now[np.nonzero(y_class_te_now)]
    hidden_nonzero_te=hidden_layer_tefold[np.nonzero(y_class_te_now)]
    y_class_nonzero_va=y_class_va_now[np.nonzero(y_class_va_now)]
    hidden_nonzero_va=hidden_layer_vafold[np.nonzero(y_class_va_now)]
    y_class_nonzero_re=y_class_re_now[np.nonzero(y_class_re_now)]
    hidden_nonzero_re=hidden_layer_refold[np.nonzero(y_class_re_now)]

    
    #Train model on Validation fold
    vblr=VBLogisticRegression(a=rate, b=shape)
    vblr.fit(hidden_nonzero_re, y_class_nonzero_re)
    
    #################################################################
    #Predict Validation fold
    y_hat_va=vblr.predict(hidden_nonzero_va)
    
    #ECE/ACE
    #AUC_ROC
    auc_roc_target=auc_roc(y_hat_va, y_class_nonzero_va)
    AUC_ROC_sum+=auc_roc_target
    #AUC_PR
    auc_pr_target=auc_pr(y_hat_va, y_class_nonzero_va)
    AUC_PR_sum+=auc_pr_target
    #ECE:
    ECE_target=calculate_ECE(y_hat_va, y_class_nonzero_va)
    ECE_sum+=ECE_target
    #ACE
    ACE_target=calculate_ACE(y_hat_va, y_class_nonzero_va)
    ACE_sum+=ACE_target
    
    #################################################################
    '''#Predict Test fold
    y_hat_te=vblr.predict(hidden_nonzero_te)
    
    #ECE/ACE
    #AUC_ROC
    auc_roc_target=auc_roc(y_hat_te, y_class_nonzero_te)
    AUC_ROC_sum+=auc_roc_target
    #AUC_PR
    auc_pr_target=auc_pr(y_hat_te, y_class_nonzero_te)
    AUC_PR_sum+=auc_pr_target
    #ECE:
    ECE_target=calculate_ECE(y_hat_te, y_class_nonzero_te)
    ECE_sum+=ECE_target
    #ACE
    ACE_target=calculate_ACE(y_hat_te, y_class_nonzero_te)
    ACE_sum+=ACE_target'''
    

AUC_ROC=AUC_ROC_sum/i
AUC_PR=AUC_PR_sum/i

ECE=ECE_sum/i
ACE=ACE_sum/i

print('AUC_ROC: ', AUC_ROC)
print('AUC_Pr: ', AUC_PR)
print('ECE: ', ECE)
print('ACE: ', ACE)
print('Parameters of Gamma Prior on precision parameter of coefficients')
print('shape: ', shape)
print('rate: ', rate)





