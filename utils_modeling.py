import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
import statsmodels.api as sm
import statsmodels.formula.api as smf
import cPickle


def ensemble_predict(fitted_models_dict, X_test):
    '''
    prediction using the ensemble method
    O: [0,1]
    '''    
    outcome = [fitted_models_dict[key].predict(X_test) for key in fitted_models_dict.keys()] 
    master_array = np.array(outcome).T
    ensemble_output = np.array([1 if master_array[i].sum() > master_array.shape[1] / 2. else 0 \
                                for i in xrange(master_array.shape[0])])
    return ensemble_output

def ensemble_predict_proba_true(fitted_models_dict, X_test, output_median=False, output_mean=False):
    '''
    np.c_[ensemble_predict(fitted_models_dict, test_set), \
    ensemble_predict_proba_true(fitted_models_dict, test_set,output_median=True)]
    '''
    outcome = [fitted_models_dict[key].predict_proba(X_test)[:,1] for key in fitted_models_dict.keys()]
    master_array = np.array(outcome).T
    
    if output_median:
        print 'median'
        np_median_obj = lambda x: np.median(x)
        median_vector = map(np_median_obj ,master_array[:])
        ensemble_output = median_vector
    elif output_mean:
        print 'mean'
        np_median_obj = lambda x: np.mean(x)
        median_vector = map(np_median_obj ,master_array[:])
        ensemble_output = median_vector
    else:
        #Need to come back to this
        #Going off of majority vote basically and then taking the probability
        print 'majority'
        ensemble_output = np.array([master_array[i].sum() / master_array.shape[1]  \
                                    for i in xrange(master_array.shape[0])])

    return np.array(ensemble_output)


def pickle_tuned_models(models_dict, file_name='tuned_models_dict.pkl'):
    with open(file_name, 'wb') as filename:
        cPickle.dump(models_dict, filename)

def get_rsquared(y, x):
    '''
    Get RMSE given response variable (y) and predictor set (X)
    USE CASE: helper function for VIF
    '''
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
 
    return model.rsquared

def VIF(feature_matrix_X):
    ''' 
    Solution Multicollinearirty
    '''
    all_rsquared_dict = {}
    predictors = feature_matrix_X.columns.tolist()

    for predictor in predictors:
        X = feature_matrix_X[:]
        y = X.pop(predictor)
        rsquared = get_rsquared(y, X)
        all_rsquared_dict[predictor] = (rsquared, 1./(1 - rsquared))

    #sort
    df_rsqaured = pd.DataFrame(
                    [(key, val[0], val[1]) for key, val in all_rsquared_dict.items()], 
                    columns=['Dependent', 'R-Squared', 'VIF'], 
                    index=None).sort(ascending=False, columns='VIF'
                    )
    return df_rsqaured

def multicollinearity_check_pairwise(feature_M, threshold=0.8):
    '''
    Multicollinearity PAIRWISE
    '''
    pairwise_correlation = feature_M.corr()[feature_M.corr() > threshold]
    pairwise_correlation = pairwise_correlation.sum()[pairwise_correlation.sum() > 1]
    
    return pairwise_correlation

def model_cv_score(model, feat_matrix, labels, folds=10, scoring='roc_auc'):
    return np.mean(cross_validation.cross_val_score(model, feat_matrix, labels, cv=folds, scoring='roc_auc'))


def get_scores(classifier, X_train, X_test, y_train, y_test, **kwargs):
    '''
    INPUT: model, Feature Matrix train, Feature Matrix test, label train, label test        
    OUTPUT: accuracy (float), precision (float), recall (float) 
    '''

    model = classifier(**kwargs)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    return model.score(X_test, y_test), \
           precision_score(y_test, y_predict), \
           recall_score(y_test, y_predict)
           # f1_score(y_test, y_predict)

def get_scores_model(model_obj, X_test, y_test):
    '''
    INPUT: model_obj, Feature Matrix train, Feature Matrix test, label train, label test        
    OUTPUT: accuracy (float), precision (float), recall (float), f1 (float), roc_auc (float)
    '''

    y_predict = model_obj.predict(X_test)
    
    #change this to 'micro'
    return model_obj.score(X_test, y_test), \
           precision_score(y_test, y_predict), \
           recall_score(y_test, y_predict), \
           f1_score(y_test, y_predict), \
           roc_auc_score(y_test, y_predict)

def get_scores_cross_val(model_obj, feature_M, labels):
    scoring_types = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    score_stor = [model_cv_score(model_obj, feature_M, labels, scoring=score_type) for score_type in scoring_types]
    

    return tuple(score_stor)

def get_model_eval_df(models_dict, X_test, y_test, cv=False):
    '''
    INPUT: model dict, X_test, y_test
    OUTPUT: model_scores df['accuracy, precision, recall, f1, roc_auc']
    '''
    
    if cv:
        print 'K-fold Cross-validated'
        model_scores_dict = {model_str : get_scores_cross_val(tuned_clf, X_test, y_test) for model_str, tuned_clf in models_dict.iteritems()}
    else:
        model_scores_dict = {model_str : get_scores_model(tuned_clf, X_test, y_test) for model_str, tuned_clf in models_dict.iteritems()}
    df_score = pd.DataFrame(model_scores_dict).T
    df_score.columns = ['Accuracy', 'Precision', 'Recall', 'F1', 'Roc_Auc']

    #formatting
    for k in df_score.keys():
        df_score[k] = df_score[k].apply(lambda x: float("{0:.2f}".format(x)))
    
    return df_score
