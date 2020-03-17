import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso



def getFileNamesFromArguments():
    """ Returns the file names of TrainingDataInputs, TrainingDataOutputs 
        from from command-line arguments passed to the script at locations 1 and 2 """
    return str(sys.argv[1]), str(sys.argv[2])

def select_best_features(x,y,p_value_threshold):
    """ Returns a list of the best features that acgieve a p_value <= p_value_threshold 
        using Backward Elimination of features """

    features = list(x.columns)
    pmax = 1

    while (len(features)>0):
        p = []
        Nset = x[features]
        Nset = sm.add_constant(Nset)

        model = sm.OLS(y,Nset).fit()

        p = pd.Series(model.pvalues.values[1:],index = features)      
        pmax = max(p)
        feature_with_p_max = p.idxmax()

        if ( pmax > p_value_threshold ):
            features.remove(feature_with_p_max)
        else:
            break

    return features

############################################
###   Start of Model Selection Script    ###
############################################

train_set = pd.read_csv('A3_training_dataset.tsv',sep='\t')
x = train_set.drop('group',axis=1)
y = train_set['group']

p_threshold = 0.001 
selected_features = select_best_features(x,y,p_threshold)
print("Selected Features: ",selected_features,len(selected_features))

# Extraxt X and Y from dataset
x = x[selected_features]

# Split data into train and test sets (should be changed to cross validation) 
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

print(y_test.value_counts())

# scale
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#X_train[:2]


# SVC
# ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ 
classifier_svc = svm.SVC()
params_grid_svc = [
  {'C': [.1, 1, 10, 100], 'kernel': ['linear']},
  {'C': [.1, 1, 10], 'degree': [2, 3, 4], 'kernel': ['poly']},
  {'C': [.1, 1, 10, 100], 'gamma': [0.001, 0.0001], 'kernel': ['rbf', 'sigmoid']},
 ]


# Grid Search Cross Validation
grid_svc = GridSearchCV(estimator=classifier_svc, param_grid=params_grid_svc, scoring='average_precision', n_jobs=-1)
grid_svc.fit(X_train, y_train)

# Summarize the results of the grid search
#print(grid)
#print('------------------------------------------------------------------------------------')
results = pd.DataFrame.from_dict(grid_svc.cv_results_).sort_values(by=['rank_test_score'])
print('cv_results_',results[["params","mean_test_score","std_test_score","rank_test_score"]])
print('------------------------------------------------------------------------------------')
print('best_estimator:',grid_svc.best_estimator_)
print('------------------------------------------------------------------------------------')
print('best_score:',grid_svc.best_score_)
print('------------------------------------------------------------------------------------')
print('best_params:',grid_svc.best_params_)
print('------------------------------------------------------------------------------------')
print('refit_time:',grid_svc.refit_time_)
print('------------------------------------------------------------------------------------')

# Test Performance
y_score_svc = grid_svc.decision_function(X_test)
y_predict_svc = grid_svc.predict(X_test)
print(np.vstack((y_score_svc[1:10], y_predict_svc[1:10])).T)

# Model Performance
print(classification_report(y_test,y_predict_svc))
print(confusion_matrix(y_test,y_predict_svc))
tn, fp, fn, tp = confusion_matrix(y_test,y_predict_svc).ravel()
print('TP={}\tFP={}'.format(tp,fp))
print('FN={}\tTN={}'.format(fn,tn))

average_precision = average_precision_score(y_test, y_score_svc)
disp = plot_precision_recall_curve(grid_svc, X_test, y_test)
disp.ax_.set_title('2-class Precision-Recall curve: AP={0:0.8f}'.format(average_precision))

# Random Forests
classifier_rf = RandomForestClassifier()
params_grid_rf = { 
    'n_estimators': [200, 500, 1000],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}

# Grid Search Cross Validation
grid_rf = GridSearchCV(estimator=classifier_rf, param_grid=params_grid_rf, scoring='average_precision', n_jobs=-1)
grid_rf.fit(X_train, y_train)

# Summarize the results of the grid search
#print(grid)
#print('------------------------------------------------------------------------------------')
results = pd.DataFrame.from_dict(grid_rf.cv_results_).sort_values(by=['rank_test_score'])
print('cv_results_',results[["params","mean_test_score","std_test_score","rank_test_score"]])
print('------------------------------------------------------------------------------------')
print('best_estimator:',grid_rf.best_estimator_)
print('------------------------------------------------------------------------------------')
print('best_score:',grid_rf.best_score_)
print('------------------------------------------------------------------------------------')
print('best_params:',grid_rf.best_params_)
print('------------------------------------------------------------------------------------')
print('refit_time:',grid_rf.refit_time_)
print('------------------------------------------------------------------------------------')

# Test Performance
y_score_rf = grid_rf.predict_proba(X_test)[:,1]
y_predict_rf = grid_rf.predict(X_test)
print(np.vstack((y_score_rf[1:10], y_predict_rf[1:10])).T)

# Model Performance
print(classification_report(y_test,y_predict_rf))
print(confusion_matrix(y_test,y_predict_rf))
tn, fp, fn, tp = confusion_matrix(y_test,y_predict_rf).ravel()
print('TP={}\tFP={}'.format(tp,fp))
print('FN={}\tTN={}'.format(fn,tn))

average_precision = average_precision_score(y_test, y_score_rf)
disp = plot_precision_recall_curve(grid_rf, X_test, y_test)
disp.ax_.set_title('2-class Precision-Recall curve: AP={0:0.8f}'.format(average_precision))

def get_hidden_layer_sizes_param():
    hidden_layers_count = [2,3,10,50]
    neuron_counts = [2,8,16,32]
    hidden_layer_sizes = []
    for layer_count in hidden_layers_count:
        for neuron_count in neuron_counts:
            hidden_layer_sizes.append(tuple(layer_count*[neuron_count]))
    return hidden_layer_sizes
print(get_hidden_layer_sizes_param())

# Random Forests
classifier_nn = MLPClassifier(max_iter=5000)
params_grid_nn = { 
    'hidden_layer_sizes': get_hidden_layer_sizes_param(),
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'solver': ['lbfgs', 'sgd', 'adam'],
    'alpha' : [0.1, 0.01, 0.001, 0.0001, 0.0001],
    'learning_rate': ['constant', 'invscaling', 'adaptive']
}

# Randomized Grid Search Cross Validation
grid_nn = RandomizedSearchCV(estimator=classifier_nn, param_distributions=params_grid_nn, scoring='average_precision', n_jobs=-1, n_iter=100)
grid_nn.fit(X_train, y_train)

# Summarize the results of the grid search
#print(grid)
#print('------------------------------------------------------------------------------------')
results = pd.DataFrame.from_dict(grid_nn.cv_results_).sort_values(by=['rank_test_score'])
print('cv_results_',results[["params","mean_test_score","std_test_score","rank_test_score"]])
print('------------------------------------------------------------------------------------')
print('best_estimator:',grid_nn.best_estimator_)
print('------------------------------------------------------------------------------------')
print('best_score:',grid_nn.best_score_)
print('------------------------------------------------------------------------------------')
print('best_params:',grid_nn.best_params_)
print('------------------------------------------------------------------------------------')
print('refit_time:',grid_nn.refit_time_)
print('------------------------------------------------------------------------------------')

# Test Performance
y_score_nn = grid_nn.predict_proba(X_test)[:,1]
y_predict_nn = grid_nn.predict(X_test)
print(np.vstack((y_score_nn[1:10], y_predict_nn[1:10])).T)

# Model Performance
print(classification_report(y_test,y_predict_nn))
print(confusion_matrix(y_test,y_predict_nn))
tn, fp, fn, tp = confusion_matrix(y_test,y_predict_nn).ravel()
print('TP={}\tFP={}'.format(tp,fp))
print('FN={}\tTN={}'.format(fn,tn))

average_precision = average_precision_score(y_test, y_score_nn)
disp = plot_precision_recall_curve(grid_nn, X_test, y_test)
disp.ax_.set_title('2-class Precision-Recall curve: AP={0:0.8f}'.format(average_precision))

# Best Predictions
def best_model_predict_proba(x):
    best_model = grid_svc
    predict_proba = []
    
    if(grid_rf.best_score_ > best_model.best_score_):
        best_model = grid_rf
    if(grid_nn.best_score_ > best_model.best_score_):
        best_model = grid_nn
    
    if(type(best_model.estimator)==type(grid_svc.estimator)):
        predict_proba = best_model.decision_function(x)
    else:
        predict_proba = grid_nn.predict_proba(x)[:,1]
        
    return best_model, predict_proba


instances_to_predict = pd.read_csv('A3_test_dataset.tsv',sep='\t')
x_to_predict = instances_to_predict[selected_features]
x_to_predict = sc.transform(x_to_predict)

best_model, y_predictions = best_model_predict_proba(x_to_predict)

print(y_predictions)

pd.DataFrame(y_predictions).to_csv("g01_predictions.txt", header=None, index=None)
