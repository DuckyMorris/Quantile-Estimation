import numpy as np
import xgboost as xgb
import pandas as pd
import scipy
import random
import math
#Source used for quantile regression:
#https://gist.github.com/Nikolay-Lysenko/06769d701c1d9c9acb9a66f2f9d7a6c7
a = 0.05

#Quantile regression objective function:
def objective(preds, dtrain, alpha = a, eps = 0.1):
    labels = dtrain.get_label()
    x= preds - labels

    left_mask = x < 0
    right_mask = x > 0

    grad = -alpha * left_mask + (1 - alpha) * right_mask
    hess = np.ones_like(preds)

    return grad, hess


def evalfn(preds, dtrain, alpha =a, eps = 0.1):
    labels = dtrain.get_label()
    s = np.nanmean((preds >= labels) * (1 - alpha) * (preds - labels) +(preds < labels) * alpha * (labels - preds))
    
    return 'error',s

#-----------------------------------------------------------------------------Previous Custom objective:

def objective1 (preds, dtrain, alpha = a, eps = 0.1):
   

    x = dtrain.get_label()
    #print(preds)
    x = np.array(x)
    n = len(x)

    grad = (1+eps) * (1-alpha)/n * (preds -x)**eps
    grad[x>preds] = -((1+eps) *alpha/n * (x-preds)**eps)[x>preds]
    
    hess = (1+eps)*eps * (1-alpha)/n * (preds-x)**(eps-1)
    hess[x>preds] = (eps* (1+eps) * (alpha)/n * (x-preds)**(eps-1))[x>preds]
    #hess = [1]*n

    grad = np.array(grad)
    hess = np.array(hess)
    return grad, hess

def evalfn1 (preds, dtrain, alpha =a, eps = 0.1):
    x = dtrain.get_label()
    n = len(x)
    x = np.array(x)

    o = (1-alpha)/n * (preds-x)**(1+eps)
    o[x>preds] = ( (alpha)/n * (x-preds)**(1+eps) )[x>preds]
           
    s = float(sum(o))
    return 'error', s


        
    
#---------------------------------------------------------------------------

        

x1 = pd.read_csv ('x1.csv')
x1 = x1.to_numpy()[:,1]
x2 = pd.read_csv ('x2.csv')
x2 = x2.to_numpy()[:,1]
x3 = pd.read_csv ('x3.csv')
x3 = x3.to_numpy()[:,1]
y = pd.read_csv ('y.csv')
y = y.to_numpy()[:,1]

data = np.zeros(shape = (2,10000))
data[0] = x1
data[1] = x2

data1 = pd.read_csv ('data.csv')

dtrain = xgb.DMatrix(data1[0:8000], label = y[0:8000])
dtest = xgb.DMatrix(data1[8000:10000], label = y[8000:10000])

min_error = 10000
trees_best = 0
best_depth = 0


deepest_depth = 5
maxtrees = 1000


for x in range (deepest_depth):
    tree_depth = x+1
    random.seed(0)
    params = {
    'max_depth':tree_depth,
    'min_child_weight': 0,
    'eta': 0.1,
    'nthread': 3}
    
    cv = xgb.cv(params, dtrain =dtrain, num_boost_round = maxtrees, nfold = 5, verbose_eval = False, obj = objective1, feval= evalfn1)
    
    cv_error = np.array(cv['test-rmse-mean'])
    cv_error_std = np.array(cv['test-rmse-std'])

    index = np.where(cv_error == np.amin(cv_error))
    one_sd_limit = cv_error[index] + cv_error_std[index]
    trees_used = np.where(cv_error<=one_sd_limit)[0]
    #print(trees_used)
    if (cv_error[trees_used[0]] < min_error):
        min_error = cv_error[trees_used[0]]
        best_depth = tree_depth
        best_trees = trees_used[0]

params = {
'max_depth':best_depth,
'min_child_weight': 0,
'eta': 0.1,
'nthread': 3}


xgfit= xgb.train(params, dtrain = dtrain, num_boost_round = best_trees, verbose_eval = False, obj=objective1, feval = evalfn1)

preds = xgfit.predict(dtest)
preds = np.array(preds)

print("Min: ", np.min(preds))
print("Q1: ", np.quantile(preds, 0.25))
print("Median: ", np.quantile(preds, 0.5))
print("Mean: ",np.mean(preds))
print("Q3: ", np.quantile(preds, 0.75))
print("Max: ", np.max(preds))
true = 10*x1[8000:10000]*x2[8000:10000] + scipy.stats.norm.ppf(a)*x2[8000:10000]

L1 = sum(abs(preds-true)/sum(abs(true)))
L2 = math.sqrt(sum((preds-true)**2)/sum(true**2))
print()
print(L1, " ", L2)

