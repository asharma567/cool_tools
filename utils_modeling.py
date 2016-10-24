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
from sklearn.preprocessing import StandardScaler


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
        print 'majority'
        ensemble_output = np.array([master_array[i].sum() / master_array.shape[1]  \
                                    for i in xrange(master_array.shape[0])])

    return np.array(ensemble_output)

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
    Find out which predictors have the highest multicollinearirty 
    w.r.t the rest of them in the context of composition as oppose 
    to pairwise_correlation
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

def model_cv_score(model, feat_matrix, labels, folds=10, scoring='roc_auc'):
    '''
    I: persisted model object, feature matrix (numpy or pandas datafram), labels, k-folds, scoring metric
    O: mean of scores over each k-fold (float)
    '''
    return np.mean(cross_validation.cross_val_score(model, feat_matrix, labels, cv=folds, scoring=scoring))
    
def get_scores_cross_val(model_obj, feature_M, labels):
    '''
    I: persisted model object, feature matrix (numpy or pandas datafram), labels
    O: 'accuracy', 'precision', 'recall', 'f1', 'roc_auc'
    '''
    scoring_types = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    score_stor = [model_cv_score(model_obj, feature_M, labels, scoring=score_type) for score_type in scoring_types]
    model_name = str(model_obj).split('(')[0]
    df_output = pd.DataFrame(score_stor, index=scoring_types).T
    df_output.index = [model_name]
    return df_output

def df_get_model_eval(models_dict, X_test, y_test, cv=False):
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

def scale_feature_matrix(feature_M, linear=False):
    binary_fields = []
    for feature_name in feature_M.columns:
        
        unique_values_for_features = feature_M[feature_name].value_counts().index
        #edge case where it won't work is when there's only one value for the dummy
        #since it's an example with only one value do we even need to worry about it?
        if len(unique_values_for_features) == 2 and np.all(unique_values_for_features == np.array([0,1])):
            binary_fields.append(feature_name)
            print feature_name

    #Scale 0 mean & unit variance
    scaler_obj = StandardScaler()
    
    X_scaled = scaler_obj.fit_transform(feature_M.drop(binary_fields, axis=1))
    X_scaled_w_cats = np.c_[X_scaled, feature_M[binary_fields].as_matrix()]
    
    return X_scaled_w_cats, scaler_obj

def time_feature_maker(df_with_time_series, name_of_date_time_col):
    import pandas as pd    
    '''
    I:dataframe, name of the time-series column (str)
    O:None, does transformation inplace

    parses the time-series column of it's different time attributes e.g. hour, day of week, year..
    forms them into individual series and puts them back into the same DataFrame.
    '''
    
    df_with_time_series[name_of_date_time_col] =  pd.to_datetime(df_with_time_series[name_of_date_time_col])
    
    df_with_time_series[name_of_date_time_col + '_hour'] = pd.DatetimeIndex(df_with_time_series[name_of_date_time_col]).hour
    df_with_time_series[name_of_date_time_col + '_second'] = pd.DatetimeIndex(df_with_time_series[name_of_date_time_col]).second
    df_with_time_series[name_of_date_time_col + '_minute'] = pd.DatetimeIndex(df_with_time_series[name_of_date_time_col]).minute

    df_with_time_series[name_of_date_time_col + '_week'] = pd.DatetimeIndex(df_with_time_series[name_of_date_time_col]).week
    df_with_time_series[name_of_date_time_col + '_day_of_week'] = pd.DatetimeIndex(df_with_time_series[name_of_date_time_col]).dayofweek
    df_with_time_series[name_of_date_time_col + '_day'] = pd.DatetimeIndex(df_with_time_series[name_of_date_time_col]).day
    df_with_time_series[name_of_date_time_col + '_month'] = pd.DatetimeIndex(df_with_time_series[name_of_date_time_col]).month
    df_with_time_series[name_of_date_time_col + '_year'] = pd.DatetimeIndex(df_with_time_series[name_of_date_time_col]).year

    return None

def model_cv_score(model, feat_matrix, labels, folds=10, scoring='r2'):
    from sklearn import cross_validation
    '''
    I: persisted model object, feature matrix (numpy or pandas datafram), labels, k-folds, scoring metric
    O: mean of scores over each k-fold (float)
    '''
    return cross_validation.cross_val_score(model, feat_matrix, labels, cv=folds, scoring=scoring)

def get_multi_label_roc_score(input_labels, X, instatiated_clf):
    '''
    I: labels (pd.Series), Feature matrix (DataFrame), classifier 
    O: weighted average roc auc score of all labels

    Each class is first binarized and scored stepe-wise. 
    Then it calculates the weighted average roc auc for each class.
    '''
    from sklearn.cross_validation import StratifiedKFold, cross_val_score    
    output_list = []
    all_class_labels = dict(input_labels.value_counts(1))

    for label_tag in all_class_labels.keys():

        # binarize labels
        temp_label = input_labels.copy()

        # this sequence matters
        temp_label[temp_label != label_tag] = 0
        temp_label[temp_label == label_tag] = 1
        Y_train = np.asarray(temp_label, dtype=int)

        # score
        skf = StratifiedKFold(Y_train, n_folds=10, shuffle=False)
        scores = cross_val_score(instatiated_clf, X, Y_train, cv=skf, scoring='roc_auc')
        output_list.append((all_class_labels[label_tag], scores.mean()))
    
    return sum([weight*score for weight, score in output_list])

def score_classifier(clf, feature_M, labels, class_imbalance=True):
    
    #scoring mechanism
    if class_imbalance:
        skf = StratifiedKFold(labels, n_folds=5, shuffle=True)
    
    #put the else here for non-strat
    scores_for_each_fold = cross_val_score(
        clf, 
        feature_M, 
        labels, 
        cv=skf, 
        scoring='roc_auc'
    )
    
    median = np.median(scores_for_each_fold)
    std = np.std(scores_for_each_fold)
    
    return median, std, scores_for_each_fold

def feature_importance_wrapper(clf, feature_M, labels, class_imbalance=True):
    
    feat_imp_dict = {}
    
    #compute the benchmark with all features
    benchmark_median_score = score(clf, feature_M, labels)[0]
    feat_imp_dict['bench_mark'] = benchmark_median_score

    for feature in feature_M.columns:
        
        feature_M_dropped = feature_M.drop(feature, axis=1)
        current_median_score, std, all_scores = score(clf, feature_M_dropped, labels)
        feat_imp_dict[feature] = benchmark_median_score - current_median_score
    
    return feat_imp_dict

def feature_importance_linear(feature_names, coefficients):
    df_beta_weights = pd.DataFrame(sorted(zip(feature_names, coefficients),key=lambda x:x[1], reverse=True))
    df_beta_weights.index = df_beta_weights[0]

    df_beta_weights[::-1].plot(kind='barh', legend=False)
    return None


def get_beta_weights(subset_feats, total_feats, feature_M, y_labels, significance=0.05):
    wts = get_pearson_correlation(subset_feats, total_feats, feature_M, y_labels)
    return [item for item in sorted(wts, key=lambda x: x[1]) if item[-1] < significance]

def get_pearson_correlation(target_feature_names, feature_names_from_X, feature_M, Y_labels):
    from scipy.stats import pearsonr
    
    output = []
    for feat in target_feature_names[::-1]:
        index, str_ = [(i,item) for i,item in enumerate(feature_names_from_X) if feat == item ][0]
        coef, p_val = pearsonr(feature_M[:,index].toarray().T[0], Y_labels)
        output.append((str_, coef, p_val))
    
    return output

