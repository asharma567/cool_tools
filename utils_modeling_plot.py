from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from utils_modeling import get_model_eval_df
from StringIO import StringIO
import prettytable 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def roc_plot(X, y, X_hold_out, y_hold_out, clf_class, **kwargs):
    '''
    I: Feature Matrix, labels, model, hyper-parameter settings
    O: None, it plots a ROC of the Cross-Validation set
    '''
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    kf = KFold(len(y), n_folds=5, shuffle=True)
    y_prob = np.zeros((len(y),2))
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []
    
    plt.figure(figsize=(10,10))
    
    for i, (train_index, test_index) in enumerate(kf):
        
        # Splitting the folds in the data for Cross-Val set
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        
        # Model fitting
        clf = clf_class(**kwargs)
        clf.fit(X_train, y_train)
        
        # Predict probabilities, not classes
        y_prob[test_index] = clf.predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y[test_index], y_prob[test_index, 1])
        
        # Used for mean roc plot
        mean_tpr += np.interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        
        # Drawing the curve
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
    
    mean_tpr /= len(kf)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--', label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
    
    #plot the holdout
    y_prob_hold_out = clf.predict_proba(X_hold_out)
    fpr, tpr, thresholds = roc_curve(y_hold_out, y_prob_hold_out[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, 'r-.', lw=2, label='ROC hold out (area = %0.2f)' % roc_auc)
     
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    
    plt.show()
def feature_importance(feature_names, model, top=10):
    '''
    Plotting top x most important features
    
    I: Feature Matrix (DataFrame), model, labels (y), top x
    O: Bar plot of most normalized by the most important feature
    '''
    
    #Grabbing names of the features
    feature_importance_series = pd.Series(model.feature_importances_, index=feature_names)
    feature_importance_series /= feature_importance_series.max()
    feature_importance_series.sort()
    
    #parsing the model date
    model_name = str(model).split('(')[0]

    # create a figure of given size
    # arg to play with the size: figsize=(25,25)
    fig = plt.figure(figsize=(9, 9))

    # add a  subplot
    ax = fig.add_subplot(111)
    
    plt.title(model_name)
    feature_importance_series[::-1][:top][::-1].plot(kind='barh')
    
    #remove weird dotted line on axis
    ax.lines[0].set_visible(False)
    pass
    
def plot_cm(y_true, y_pred, labels=['True', 'False'], model_name=None):
    '''
    INPUT: , labels (list)
    OUPUT: none, plots a graph

    eg
    y_true = y_test.ravel()
    y_pred = RF.predict(X_scaled_test)

    plot_cm(y_true, y_pred, ['sale','not_sale'], 'LogisticRegression')
    '''
    
    if model_name: print model_name + '\n' + len(model_name) * '-'
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Compute percentanges    
    percent = (cm * 100.0) / np.array(np.matrix(cm.sum(axis=1)).T) 
    print 'Confusion Matrix Stats'
    for i, label_i in enumerate(labels):
        for j, label_j in enumerate(labels):
            print "predicted %s/actual %s: %.2f%% (%d/%d)" % (label_i, label_j, (percent[i][j]), cm[i][j], cm[i].sum())

    # Show confusion matrix
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid(b=False)
    cax = ax.matshow(percent, cmap='coolwarm',vmin=0,vmax=100)

    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

def plot_model_evaluation(models_dict, X_feature_matrix, y_labels):
    '''
    INPUT: dictionary of FITTED models, X_feature_matrix, y_labels
    OUPUT: plots bar charts and dataframe of scores
    '''

    df_score = get_model_eval_df(models_dict, X_feature_matrix, y_labels)

    for series_name in df_score:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.suptitle(series_name)
        df_score[series_name].ix[df_score[series_name].argsort()].plot(kind='barh')
        ax.lines[0].set_visible(False)
        plt.show()

    #pretty printing the DataFrame
    pprint_df(df_score.sort(['Precision', 'Roc_Auc'], ascending=[0,0]))
    pass

def pprint_df(data_frame):
    '''
    INPUT: Pandas Data Frame
    OUTPUT: A Pretty Pandas Data Frame printed to std out; returns None
    '''
    
    output = StringIO()
    data_frame.to_csv(output)
    output.seek(0)
    print prettytable.from_csv(output)
    pass

def scree_plot(pca_obj, top_k_components=None):
    
    plt.plot(pca_obj.explained_variance_ratio_)
    if top_k_components: plt.axvline(x=top_k_components, color='r', linestyle='--', alpha=0.5)
    plt.show()
    
    plt.plot(np.cumsum(pca_obj.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    if top_k_components: plt.axvline(x=top_k_components, color='r', linestyle='--', alpha=0.5)
    plt.show();
    
    pass

def plot_optimal_number_of_components_variance(scale_feature_collinear_feature_M, train_target, variance_threshold=.9):
    '''
    typical rule of thumb says keep the number of compenonents that get ~90% of 
    variance (50) by looking at the cumsum graph. but if we look at the scree plot 
    and abide by kairsers rule which argues to keep as many explanatory components 
    i.e. when the slope doesn't change -- elbow method
    '''
    
    n_col = scale_feature_collinear_feature_M.shape[1]
    pca = PCA(n_components=n_col)
    train_components = pca.fit_transform(scale_feature_collinear_feature_M)

    pca_range = np.arange(n_col) + 1
    xbar_names = ['PCA_%s' % xtick for xtick in pca_range]
    cumsum_components_eig = np.cumsum(pca.explained_variance_ratio_)
    
    target_component = np.where(cumsum_components_eig > variance_threshold)[0][0] + 1
    
    print 'number of components that explain target amt of variance explained: ' \
            + str(target_component) + ' @ ' + str(cumsum_components_eig[target_component - 1])

    kaiser_rule = len(pca.explained_variance_ratio_[np.mean(pca.explained_variance_ratio_) > pca.explained_variance_ratio_])
    label1_str = str(100 * cumsum_components_eig[target_component - 1])[:3] + '%'

    #cumsum plot                                                 
    plt.axvline(target_component, color='r', ls='--', alpha=.3, label= str(100 * cumsum_components_eig[target_component - 1])[:4] + '%')
    plt.axvline(kaiser_rule, ls='--', alpha=.3, label= str(100 * cumsum_components_eig[kaiser_rule])[:4] + '%')
    plt.plot(cumsum_components_eig)
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.legend(loc='best')
    plt.show()
    
    #Scree
    plt.axvline(target_component, color='r', ls='--', alpha=.3, label= str(100 * cumsum_components_eig[target_component - 1])[:4] + '%')
    plt.axvline(kaiser_rule, ls='--', alpha=.3, label= str(100 * cumsum_components_eig[kaiser_rule])[:4] + '%')
    plt.plot(pca.explained_variance_ratio_)
    plt.xlabel('number of components')
    plt.ylabel('variance of component (eigenvalues)')
    plt.legend(loc='best')
    plt.show()
    
    return target_component