import sys
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


###########################
###   Configurations    ###
###########################

# Set to True to print progress at each step to console
print_progress = True

# Set to True to plot performace of best models
plot_output = True

#############################
###   Helper Functions    ###
#############################

def printP(*args):
    if(print_progress):
        print(*args)

def getFileNamesFromArguments():
    """ Returns the file names of 2 params from from command-line arguments passed to 
        the script at locations 1 and 2 """
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

def get_hidden_layer_sizes_param():
    hidden_layers_count = [2,3,10,50]
    neuron_counts = [2,8,16,32]
    hidden_layer_sizes = []

    for layer_count in hidden_layers_count:
        for neuron_count in neuron_counts:
            hidden_layer_sizes.append(tuple(layer_count*[neuron_count]))

    return hidden_layer_sizes

def print_search_summary(grid):
    printP('------------------------------------------------------------------------------------')
    results = pd.DataFrame.from_dict(grid.cv_results_).sort_values(by=['rank_test_score'])
    printP('cv_results_',results[["params","mean_test_score","std_test_score","rank_test_score"]])
    printP('------------------------------------------------------------------------------------')
    printP('best_estimator:',grid.best_estimator_)
    printP('------------------------------------------------------------------------------------')
    printP('best_score:',grid.best_score_)
    printP('------------------------------------------------------------------------------------')
    printP('best_params:',grid.best_params_)
    printP('------------------------------------------------------------------------------------')
    printP('refit_time:',grid.refit_time_)
    printP('------------------------------------------------------------------------------------')

def get_grid_preditions(grid,x):
    """ Returns the predictions for class '1' """
    predict_proba = []
    svc_classifier = svm.SVC()

    if(type(grid.estimator)==type(svc_classifier)):
        predict_proba = grid.decision_function(x)
    else:
        predict_proba = grid.predict_proba(x)[:,1]

    return predict_proba


# Best Predictions
def get_best_classifier(grids):
    best_model = grids[0]

    for c in grids:
        if(c.best_score_ > best_model.best_score_):
            best_model = c

    return best_model

def generate_perf_table(classifiers):
    names = []
    means = []
    stds = []

    performance_table = []
    performance_table.append(['============================='])
    performance_table.append(['Best Model Per ML Method:'])
    performance_table.append(['============================='])
    performance_table.append(['Method','Mean','STD','Prameters'])

    for c in classifiers:
        #name = str(type(g.estimator)).split('.')[1]
        g = c['grid']
        name = c['name']
        mean = g.best_score_
        std = g.cv_results_['std_test_score'][g.best_index_]
        params = str(g.best_params_)

        performance_table.append([name,mean,std,params])

        names.append(name)
        means.append(mean)
        stds.append(std)

    pd.DataFrame(performance_table).to_csv("performance_table.txt", header=None, index=None,sep='\t')

    # Create x-axis values
    x_pos = np.arange(len(names))

    # Build the plot
    fig, ax = plt.subplots()
    ax.bar(x_pos, means, yerr=stds, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('Average Precision Score')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names)
    ax.set_title('Best Average Precision Score per ML Method')
    ax.set_ylim(bottom=.8, top=.9)
    ax.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    plt.savefig('bar_plot_with_error_bars.png')


############################################
###   Start of Model Selection Script    ###
############################################

# Get file names of inputs and outputs files
training_set_filename, to_predict_set_filename = getFileNamesFromArguments()


# Import training and to_preditct set 
train_set = pd.read_csv(training_set_filename,sep='\t')
instances_to_predict = pd.read_csv(to_predict_set_filename,sep='\t')

# Extract training set features and outputs for all instances
x = train_set.drop('group',axis=1)
y = train_set['group']

# Select best features using Backward Elimination to keep only features with p_value <= p_threshold
p_threshold = 0.001 
selected_features = select_best_features(x,y,p_threshold)
printP("Selected Features: ",selected_features,len(selected_features))

# Keep only best features
x = x[selected_features]
x_to_predict = instances_to_predict[selected_features]

# Split data into train and test sets 
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
printP("Split Data, training data counts:")
printP(y_test.value_counts())

# Scale features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
x_to_predict = sc.transform(x_to_predict)

################################################
###   Setup Methods and their parameters     ###
################################################

# SVC
# ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ 
classifier_svc = svm.SVC()
params_grid_svc = [
  {'C': [.1, 1, 10, 100], 'kernel': ['linear']},
  {'C': [.1, 1, 10], 'degree': [2, 3, 4], 'kernel': ['poly']},
  {'C': [.1, 1, 10, 100], 'gamma': [0.001, 0.0001], 'kernel': ['rbf', 'sigmoid']},
 ]

# Random Forests
classifier_rf = RandomForestClassifier()
params_grid_rf = { 
    'n_estimators': [200, 500, 1000],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}

# Neural Network
classifier_nn = MLPClassifier(max_iter=5000)
params_grid_nn = { 
    'hidden_layer_sizes': get_hidden_layer_sizes_param(),
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'solver': ['lbfgs', 'sgd', 'adam'],
    'alpha' : [0.1, 0.01, 0.001, 0.0001, 0.0001],
    'learning_rate': ['constant', 'invscaling', 'adaptive']
}

classifiers = [
  {'classifier': classifier_svc, 'params': params_grid_svc, 'type': 'grid', 'name': "SVM"},
  {'classifier': classifier_rf, 'params': params_grid_rf, 'type': 'grid', 'name': "Random Forests"},
  {'classifier': classifier_nn, 'params': params_grid_nn, 'type': 'random', 'name': "Neural Network"}
]

grids = []
names = []

for c in classifiers:
    classifier = c['classifier']
    params = c['params']
    search_type = c['type']
    grid = None

    if(search_type == 'grid'):
        # Grid Search Cross Validation
        grid = GridSearchCV(estimator=classifier, param_grid=params, scoring='average_precision', n_jobs=-1)
    else:
        grid = RandomizedSearchCV(estimator=classifier, param_distributions=params, scoring='average_precision', n_jobs=-1, n_iter=100)

    # Do search
    grid.fit(X_train, y_train)

    # Summarize the results of the grid search
    print_search_summary(grid)

    # Test Performance
    y_score = get_grid_preditions(grid,X_test)
    y_predict = grid.predict(X_test)
    printP(np.vstack((y_score[1:10], y_predict[1:10])).T)

    # Model Performance
    printP(classification_report(y_test,y_predict))
    printP(confusion_matrix(y_test,y_predict))
    tn, fp, fn, tp = confusion_matrix(y_test,y_predict).ravel()
    printP('TP={}\tFP={}'.format(tp,fp))
    printP('FN={}\tTN={}'.format(fn,tn))

    if(plot_output):
        average_precision = average_precision_score(y_test, y_score)
        disp = plot_precision_recall_curve(grid, X_test, y_test)
        disp.ax_.set_title('2-class Precision-Recall curve: AP={0:0.8f}'.format(average_precision))
        plt.savefig("{}-PRC.png".format(c['name']))

    grids.append(grid)
    c['grid']=grid

# Generate best models performance measurements table
if(plot_output):
    generate_perf_table(classifiers)

# Get best model and use it to perform prediction 
best_model = get_best_classifier(grids)
predict_proba = get_grid_preditions(best_model,x_to_predict)
printP(predict_proba)

pd.DataFrame(predict_proba).to_csv("g01_predictions.txt", header=None, index=None)
