#HEADER
from IPython.display import HTML

HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
<form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>''')

plt.style.use('ggplot')
%pylab inline

import warnings
warnings.filterwarnings('ignore')​

######










import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def value_counts_to_barplot(value_counts_series, pal=sns.color_palette("Greens_d", len(groupedvalues))):
    import seaborn as sns

    g = sns.barplot(
        x = value_counts_series,
        y = value_counts_series.index, 
        palette = np.array(pal[::-1]),
        
    )

    for i, val in enumerate(value_counts_series):
        g.text(
                x = value_counts_series[i],
                y = i,
                s = round(val, 2), 
                color = 'blue', 
                ha = "center",
                fontsize=15
            )
    plt.show()

def plot_distribution(data_points_list):
    '''
    kurtosis:
    Is about the fatness of the tails which is also indicative out of outliers.

    skew: 
    when a distribution "leans" to the right or left, it's called skewed the right/left. 
    Think of a skewer. This it's a indication of outliers that live on that side of the distribution.

    *these are both aggregate stats and very subjectable to the size of the target sample

    '''
   
    print(pd.Series(data_points_list).describe())
    
    skew, kurtosis = _skew_and_kurtosis(data_points_list)
    print ('skew -- ', skew)
    print ('kurtosis --', kurtosis)
    
    plot_transformation(data_points_list, 'no_transformation');
    plt.violinplot(
       data_points_list,
       showmeans=False,
       showmedians=True
    );

def plot_transformation(data, name_of_transformation):

    #setting up canvas
    figure = plt.figure(figsize=(10,5))
    
    plt.suptitle(name_of_transformation)
    
    figure.add_subplot(121)
    
    plt.hist(data, alpha=0.75, bins=100) 
    
    figure.add_subplot(122)
    plt.boxplot(data)
    
    plt.show()

def _skew_and_kurtosis(data_points_list): 
    from scipy.stats import skew, kurtosis
    return (skew(data_points_list), kurtosis(data_points_list))

def log_transform_distribution(data_points_list)
    constant = (1 + - 1 * data_points_list.min())
    log_transformed_data = preprocessing.scale(np.log(data_points_list + constant))
    return log_transformed_data


def plot_no_transformation(data_points_scaled):
   
    print(pd.Series(data_points_scaled).describe())
    print (_skew_and_kurtosis(data_points_scaled))
    plot_transformation(data_points_scaled, 'no_transformation');
    plt.violinplot(
       data_points_scaled,
       showmeans=False,
       showmedians=True
    );


def roc_plot(X, y, X_hold_out, y_hold_out, clf_class, **kwargs):
    '''
    I: Feature Matrix, labels, model, hyper-parameter settings
    O: None, it plots a ROC of the Cross-Validation set
    '''
    from sklearn.cross_validation import KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_curve, auc

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
    import pandas as pd
    import matplotlib.pyplot as plt
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
    return feature_importance_series[::-1][:top][::-1].index

def feature_importance_plot(feature_names, model, top_x=None):
    '''
    Ploting top x most important features
    
    I: feature name (list), model (fitted tree-based model), top x (int)
    Plots: bar plot of features importance with standard error among trees.
    
    *importances weights (normalized by sum) paired with names
    '''

    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    
    if top_x:
        indices = np.argsort(importances)[::-1][:top_x]
    else:
        indices = np.argsort(importances)[::-1]
    
    _plot_barchart_with_error_bars(importances[indices], std[indices], feature_names[indices])
        
    return None


def _plot_barchart_with_error_bars(importance_weights, errors, feature_names):
    '''
    I: importance_weights (np array), errors (np array), feature_names
    O: None

    does the actual plotting of feature weights
    goes with feature_importance_plot
    '''
    
    #if we don't reverse everything it'll plot it upside-down
    importance_weights_reversed = importance_weights[::-1] 
    errors_reversed = errors[::-1]
    feature_names_reversed = feature_names[::-1] 

    number_of_features = range(len(importance_weights_reversed))

    plt.figure(figsize=(12,10))
    plt.title("Feature importances")
    plt.barh(
        number_of_features, 
        importance_weights_reversed,
        color="r", 
        xerr=errors_reversed, 
        align="center"
    )
    plt.yticks(number_of_features, feature_names_reversed)
    plt.show()


def plot_feature_imp_random_stumps(feature_M, labels, feature_names):
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd
    
    rf = RandomForestClassifier(
        n_estimators=1000, 
        max_depth=2, 
        n_jobs=-1, 
        class_weight='balanced_subsample', 
        random_state=40
    )
    
    rf.fit(feature_M, labels)

    std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
    most_important_feat = pd.Series([tree.feature_importances_.argmax() for tree in rf.estimators_]).value_counts()
    most_important_feat.index = [feature_names[idx] for idx in most_important_feat.index]
    most_important_feat.sort_values().plot(kind='barh')

    return most_important_feat.index

def plot_cm(y_true, y_pred, labels=['True', 'False'], model_name=None):
    '''
    INPUT: , labels (list)
    OUPUT: none, plots a graph

    eg
    y_true = y_test.ravel()
    y_pred = RF.predict(X_scaled_test)

    plot_cm(y_true, y_pred, ['sale','not_sale'], 'LogisticRegression')
    '''
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    if model_name: print model_name + '\n' + len(model_name) * '-'
    
    
    
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
    from utils_modeling import get_model_eval_df

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
    from StringIO import StringIO
    import prettytable 

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

def plot_optimal_number_of_components_variance(scale_feature_collinear_feature_M, variance_threshold=0.99):
    '''
    typical rule of thumb says keep the number of compenonents that get ~90% of 
    variance (50) by looking at the cumsum graph. but if we look at the scree plot 
    and abide by kairsers rule which argues to keep as many explanatory components 
    i.e. when the slope doesn't change -- elbow method
    '''
    from sklearn.decomposition import PCA
    n_col = scale_feature_collinear_feature_M.shape[1]
    pca = PCA(n_components=n_col)
    train_components = pca.fit_transform(scale_feature_collinear_feature_M)

    pca_range = np.arange(n_col) + 1
    xbar_names = ['PCA_%s' % xtick for xtick in pca_range]
    cumsum_components_eig = np.cumsum(pca.explained_variance_ratio_)

    target_component = np.where(cumsum_components_eig > variance_threshold)[0][0] + 1

    print ('number of components that explain target amt of variance explained: ' \
            + str(target_component) + ' @ ' + str(cumsum_components_eig[target_component - 1]))

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

def plot_transformation(data, name_of_transformation):
    '''
    I: some feature in a training set(pd.Series), name for a label
    O: None, just plots the value_counts and includes the labels Skew, Mean, Std

    refer to this for actual use: 
    https://github.com/asharma567/multiclass_with_major_class_imbalance/blob/master/analysis.ipynb
    '''
    #setting up canvas
    figure = plt.figure(figsize=(10,5))
    plt.suptitle(name_of_transformation)
    figure.add_subplot(121)
    plt.hist(data, alpha=0.75) 
    distribution_stats_text_label(-2, 1500, data)

    figure.add_subplot(122)
    plt.boxplot(data)

    plt.show()

def distribution_stats_text_label(position_x, position_y, data):
    '''
    you'll have to play around with the label placement 
    but I try to make some effort to automate this
    '''
    label_position_decrement = 0.08 * position_y
    plt.text(position_x, position_y, "Skewness: {0:.2f}".format(skew(data))) 
    plt.text(position_x, position_y - label_position_decrement, "Mean: {0:.2f}".format(data.mean())) 
    plt.text(position_x, position_y - 2 * label_position_decrement, "Std: {0:.2f}".format(data.std())) 
    return None

def magnify():
    return [dict(selector="th",
                 props=[("font-size", "7pt")]),
            dict(selector="td",
                 props=[('padding', "0em 0em")]),
            dict(selector="th:hover",
                 props=[("font-size", "12pt")]),
            dict(selector="tr:hover td:hover",
                 props=[('max-width', '200px'),
                        ('font-size', '12pt')])
]

def plot_stacked_bar(df_to_plot, label, color_map = 'YlOrBr'):
    '''
    INPUT: DF, label(string) for the x-axis to be displayed at the top
    OUTPUT: Stacked Bar Chart
    '''
    
    # create a figure of given size
    fig = plt.figure(figsize=(15,15))

    # add a subplot
    ax = fig.add_subplot(111)

    # set color transparency (0: transparent; 1: solid)
    a = 0.8

    # set x axis label on top of plot, set label text
    xlab = label
    ax.set_xlabel(xlab, fontsize=20, alpha=a, ha='left')
    ax.xaxis.set_label_coords(0, 1.04)

    # position x tick labels on top
    ax.xaxis.tick_top()

    # remove tick lines in x and y axes
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')

    # remove grid lines (dotted lines inside plot)
    ax.grid(False)

    # Remove plot frame
    ax.set_frame_on(False)

    # using the actual data to plot
    df_to_plot[::-1].plot(
        ax=ax, 
        kind='barh', 
        alpha=a, 
        edgecolor='w',
        fontsize=12, 
        grid=True, 
        width=.8, 
        stacked=True,
        cmap=get_cmap(color_map)
    )


    # multiply xticks by format into pct
    fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
    xticks = matplotlib.ticker.FormatStrFormatter(fmt)
    ax.xaxis.set_major_formatter(xticks)
    ax.xaxis.set_ticks(ax.xaxis.get_majorticklocs()[:-1])

    plt.legend(prop={'size':20}, frameon=False, fancybox=None, loc="upper left", bbox_to_anchor=(1,1))
    plt.tight_layout()
    plt.show()
    ;


COLOR_PALETTE = [    
   "#348ABD",
   "#A60628",
   "#7A68A6",
   "#467821",
   "#CF4457",
   "#188487",
   "#E24A33"
  ]

def plot_time_series_in_context_sequence(list_of_ordered_datapoints, title_str):
    '''
    normally timeseries have a time domain functionality with it however this 
    function assume thats the point are just in order and each time interval 
    between points is normalized to 1
    '''
    plt.figure(figsize=(15, 10))
    plt.subplot(2,1,1);

    for i, pt in enumerate(list_of_ordered_datapoints):
        if pt < 0.0:
            plt.scatter(i, pt, c='grey')
        else:
            plt.scatter(i, pt, c=COLOR_PALETTE[-1])

    plt.axhline(y=0.0, color='blue', linewidth=1)
    plt.title(title_str)


def plot_pricevsyear(X, y, model=None, title=None):
    '''
    Calculate the residuals versus the model and sorts by largest negatives first
    INPUT  numpy array predicted (modeled) points and observed points
    OUTPUT numpy array of sorted indices
    '''
    fig, ax = plt.subplots(figsize=(10,8))
    ax.scatter(X, y, alpha=0.5, color='grey', label='flight options')
    fig.suptitle(title)

    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    ax.set_xlabel(r'$Duration$', fontsize=16)
    ax.set_ylabel(r'$Price$', fontsize=16)
    fig.tight_layout(pad=2)
    ax.grid(False)
    
    if model:
        #make a line
        min_duration_for_line, max_duration_for_line = X.min(), X.max()
        line = np.arange(min_duration_for_line, max_duration_for_line)
        years = line.reshape(len(line), 1)
        prediction = np.array(model.predict(years))
        print (np.mean(prediction), np.median(prediction))
        plt.plot(
            years, 
            prediction, 
            '-', 
            alpha=0.5, 
            label='baseline', 
            color='grey',
            linewidth=2
        )
    
    plt.legend()

