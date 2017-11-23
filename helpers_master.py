
def missing_values_finder(df):
    '''
    finds missing values in a data frame returns to you the value counts
    '''
    import pandas as pd
    missing_vals_dict= {col : df[col].dropna().shape[0] / float(df[col].shape[0]) for col in df.columns}
    output_df = pd.DataFrame().from_dict(missing_vals_dict, orient='index').sort_index()
    return output_df

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

    
def feature_importance_wrapper(clf, feature_M, labels, class_imbalance=True):
    
    feat_imp_dict = {}
    
    #compute the benchmark with all features
    benchmark_mean_score = score_classifier(clf, feature_M, labels, class_imbalance=True)[0]
    feat_imp_dict['bench_mark'] = benchmark_mean_score

    for feature in feature_M.columns:
        
        feature_M_dropped = feature_M.drop(feature, axis=1)
        current_mean_score, median, std, all_scores = score_classifier(clf, feature_M_dropped, labels)
        feat_imp_dict[feature] = benchmark_mean_score - current_mean_score
    
    return feat_imp_dict

def score_classifier(clf, feature_M, labels, class_imbalance=True):
    from sklearn.cross_validation import StratifiedKFold, cross_val_score    
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
    mean = np.mean(scores_for_each_fold)
    std = np.std(scores_for_each_fold)
    
    return mean, median, std, scores_for_each_fold

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



def plot_coverage_precision(df_matches_scored_and_validated_from_bea, df_budget_itemization_pairs_which_dont_exist_on_bea):
    '''
    df_matches_scored_and_validated_from_bea are budget-itemization pairs scored to be a match under a given threshold same 
    for df_budget_itemization_pairs_which_dont_exist_on_bea. However, both are validated by checking membership on the bea table, 
    not always the corresponding budget is returned.
    '''

    perf_metrics_per_theshold = []
    for threshold in [i/10.0 for i in range(1,10,1)]:

        df_matches_scored_and_validated_from_bea_trimmed, df_budget_itemization_pairs_which_dont_exist_on_bea_trimmed = \
            _filter_everything_by_threshold(
                df_matches_scored_and_validated_from_bea, 
                df_budget_itemization_pairs_which_dont_exist_on_bea, 
                threshold
            )
        
        *_ , precision, coverage, _, _ = _compute_performance_metrics(
                    df_matches_scored_and_validated_from_bea_trimmed, 
                    df_budget_itemization_pairs_which_dont_exist_on_bea_trimmed,
                    df_matches_scored_and_validated_from_bea
                )
        perf_metrics_per_theshold.append((threshold , precision, coverage))

    df = pd.DataFrame(perf_metrics_per_theshold)
    df.columns = ['threshold' , 'precision', 'coverage']
    df.index = df.threshold
    
    df['precision'].plot(label='precision')
    df['coverage'].plot(label='coverage')
    return df

def print_scores(df_matches_scored_and_validated_from_bea, df_budget_itemization_pairs_which_dont_exist_on_bea, threshold=0.5):
    '''
    df_matches_scored_and_validated_from_bea are budget-itemization pairs scored to be a match under a given threshold same 
    for df_budget_itemization_pairs_which_dont_exist_on_bea. However, both are validated by checking membership on the bea table, 
    not always the corresponding budget is returned.
    '''

    df_matches_scored_and_validated_from_bea_trimmed, df_budget_itemization_pairs_which_dont_exist_on_bea_trimmed = \
        _filter_everything_by_threshold(
            df_matches_scored_and_validated_from_bea, 
            df_budget_itemization_pairs_which_dont_exist_on_bea, 
            threshold
        )
    
    stats = _compute_performance_metrics(
                df_matches_scored_and_validated_from_bea_trimmed, 
                df_budget_itemization_pairs_which_dont_exist_on_bea_trimmed,
                df_matches_scored_and_validated_from_bea
            )
    
    _print_performance_metrics(*stats)
    
    return None                                                                                    

def _filter_everything_by_threshold(df_matches_scored_and_validated_from_bea, df_budget_itemization_pairs_which_dont_exist_on_bea, threshold=0.5):
    
    print('applying threshold-- ', threshold)

    df_matches_scored_and_validated_from_bea = df_matches_scored_and_validated_from_bea[df_matches_scored_and_validated_from_bea['scores'] >= threshold]
    
    df_budget_itemization_pairs_which_dont_exist_on_bea = df_budget_itemization_pairs_which_dont_exist_on_bea[df_budget_itemization_pairs_which_dont_exist_on_bea['scores'] >= threshold]
    return df_matches_scored_and_validated_from_bea, df_budget_itemization_pairs_which_dont_exist_on_bea

def _print_performance_metrics(
            length_of_truths, 
            length_of_falses_not_on_bea, 
            total_TPs, 
            total_FPs, 
            total_TPs_prethreshold, 
            precision, 
            coverage, 
            total_TPs_prethreshold_budget_level, 
            coverage_at_budget_level
        ):

    print ('total budget-expense pairs scored as a match (predicted true): ', 
            length_of_truths + length_of_falses_not_on_bea)
    print ('total scored as a match (with associated budgets): ', length_of_truths)
    print ('total scored as a match (with no associated budgets): ', length_of_falses_not_on_bea)
    print ('precision: ', precision)
    print ('total_number_of_matches (truths): ', total_TPs)
    print ('total_number_of_matches (total_FPs): ', total_FPs)
    print ('FNs:',  total_TPs_prethreshold - total_TPs)
    print ('total_TPs/total_TPs_prethreshold (coverage): ', coverage)
    print ('total_TPs_prethreshold_budget_level: ', total_TPs_prethreshold_budget_level)
    print ('total_TPs/total_TPs_prethreshold_budget_level (coverage): ', coverage_at_budget_level)

    
    return None



#have no idea if this has been tested
def memoize(fn):
    import joblib
    '''
    memoization for any function i.e. checks a hash-map to 
    see if the same work's already been done avoid unnecessary 
    computation
    '''
    try: 
        file_name_str = 'memoized_stored_results_' + str(fn).split()[1] + '.pkl' 
        stored_results = joblib.load(file_name_str)
    except EOFError:
        stored_results = {}
        
    def memoized(*args):
        if args in stored_results:
            result = stored_results[args]
        else:
            result = stored_results[args] = fn(*args)
            file_name_str = 'memoized_stored_results_' + str(fn).split()[1] + '.pkl'
            joblib.dump(stored_results, file_name_str)
        return result
    return memoized



def find_field_in_db(potential_field_name, connection_to_db):
    from sqlalchemy import create_engine
    
    qry = '''
    show tables
    '''

    df = pd.read_sql(qry, connection_to_db)


    all_tables_in_db = list(df.ix[:,0])

    stor =[]
    for name in all_tables_in_db:
        try:
            qry = '''DESCRIBE ''' + name

            df = pd.read_sql(qry, connection_to_db)
            if potential_field_name in list(df['Field']): stor.append(name)
        except:
            print ('error',name)
    return stor

def find_table(sub_str_in_table_name, connection_to_db):
    table_names = pd.read_sql("show tables;", connection_to_db)

    return [name for name in table_names.Tables_in_rocketrip_production if sub_str_in_table_name in name]

def connect_to_prod():
    engine_prod = create_engine('mysql+pymysql://analytician:crouchingtigerhiddenmouse@production-vpc-enc-readonly.cvhe9o57xgm1.us-east-1.rds.amazonaws.com/rocketrip_production', echo=False)
    return engine_prod.connect()

def connect_to_expenses():
    engine_exp = create_engine('mysql+pymysql://expense_ro:iwillexpensethelaborcost@expenses-vpc-enc.cvhe9o57xgm1.us-east-1.rds.amazonaws.com/rocketrip_expenses', echo=False)
    return engine_exp.connect()


def connect_to_prod():
    engine_prod = create_engine('mysql+pymysql://analytician:crouchingtigerhiddenmouse@production-vpc-enc-readonly.cvhe9o57xgm1.us-east-1.rds.amazonaws.com/rocketrip_production', echo=False)
    return engine_prod.connect()

def connect_to_expenses():
    engine_exp = create_engine('mysql+pymysql://expense_ro:iwillexpensethelaborcost@expenses-vpc-enc.cvhe9o57xgm1.us-east-1.rds.amazonaws.com/rocketrip_expenses', echo=False)
    return engine_exp.connect()


def numpy_array_to_str(arr): 
    '''
    numpy array
    - replaces brackets with () and 
    - replaces spaces with commas
    - removes newline char
    
    used to be inserted into a sql query
    '''
    #convert numpy array to list
    arr_list = list(arr)
    return '(' + str(arr_list).strip('[]').replace('\n','') + ')'




STOPWORDS_SET = set(stopwords.words('english'))
SNOWBALL = SnowballStemmer('english')
WORDNET = WordNetLemmatizer()


def find_stop_words(corpus):
    '''
    takes in a normalized corpus and returns stop words in pandas Series
    '''
    unpacked_list = [word for document in corpus for word in document.split()]
    
    return pd.Series(unpacked_list).value_counts()


ABREVIATIONS_DICT = {
    "'m":' am',
    "'ve":' have',
    "'ll":" will",
    "'d":" would",
    "'s":" is",
    "'re":" are",
    "  ":" ",
    "' s": " is",
    
    #debatable between and/or
    "/":" and "
}

def _multiple_replace(text, adict=ABREVIATIONS_DICT):
    import re
    '''
    Does a multiple find/replace
    '''
    rx = re.compile('|'.join(map(re.escape, adict)))
    def one_xlat(match):
        return adict[match.group(0)]
    return rx.sub(one_xlat, text.lower())

def _special_char_translation(doc):    
    return ' '.join([unidecode(word) for word in doc.split()])
    
def _remove_stop_words(doc):
    return ' '.join([word for word in doc.split() if word.lower() not in STOPWORDS_SET])


def normalize(document, post_normalization_stop_words={}):
    WHITE_SPACE = ' '

    decoded_doc = _special_char_translation(document)
    abbreviations_removed_doc = _multiple_replace(decoded_doc)
    stops_removed_doc = _remove_stop_words (abbreviations_removed_doc)
    punc_removed = ''.join([char for char in stops_removed_doc if char not in set(punctuation)])    
    
    stripped_lemmatized = map(WORDNET.lemmatize, punc_removed.split())
    stripped_lemmatized_stemmed = map(SNOWBALL.stem, stripped_lemmatized)
    
    return WHITE_SPACE.join([word for word in stripped_lemmatized_stemmed if word not in post_normalization_stop_words])

import numba as nb
#this was never tested
@nb.jit(nopython=True)
def generate_negative_labels(
        df_merged, 
        subject_fields=RELEVANT_BUDGET_FIELDS
        ): 
    '''
    This iterates through the entire dataframe per user. Once it's at a user level and it'll iterate for each and 
    every budget (hotel budget's in this case). At the end, it should produce a much larger dataset.

    For example, if there's 2 hotel budgets for a particular user It'll produce negative labels by 
    iterating the all the line-items not matched with that budget.

    todos
    -----
    - make a field that literally labels in DONE
    - refactor s.t. it's more modularized and the time complexity is better. It takes for ever, consider multithreadhing
    - there's bound to be a major class imbalance issue with this method. Perhaps ensembling might be a viable solution.
    - Find a small user number to double check stuff
    - Do the analysis!

    #this is also deprecated
    df_merged = pd.merge(
        

        #postive label subset of budget and line item data
        df_master_table_deduped_hotel_tax, \
        
        #below was produced by constraining all line-items to the set of positive labels
        df_line_items_positive_label_subset, \ 
        
        #joining on both keys
        left_on=['rocketrip_user_id', 'itemization_id'],\
        right_on=['rocketrip_user_id_expense', 'itemization_id_itemization'],\
        
        #the motivation is to capture all expense items for a user
        #which would be ideal for a training set
        how='right'
        )

    '''
    df_empty = pd.DataFrame() 
 
    #iterate through the entire dataframe per user
    for user_id in df_merged.rocketrip_user_id_expense.unique():
        df_user_level = df_merged[df_merged.rocketrip_user_id_expense == user_id]


        #iterate through each budget ID

        for budget_id in df_user_level.budget_id.dropna().unique():
            df_subject_budget = df_user_level[df_user_level.budget_id == budget_id]
            
            #label field
            # df_subject_budget['label'] = 'positive'
            df_subject_budget['label'] = df_subject_budget.budget_type_budget
            
            #append positive
            df_empty = df_empty.append(df_subject_budget)

            #grab budget-side data
            df_budget_data = df_subject_budget[subject_fields]
            
            #grab every expense not matched to subject budget ID i.e. negative labels
            df_all_unmatched_lineitems_to_be_filled = \
                df_user_level[df_user_level.budget_id != budget_id]

            #iterates through columns for budget data
            #filling in the budget data for the negative label line items
            for column_to_be_downfilled in df_budget_data.columns:

                #this unpacks a uniquefied value
                subject_budget_value = df_budget_data[column_to_be_downfilled].unique()[0]
                #passes the subject value to all the fields
                df_all_unmatched_lineitems_to_be_filled[column_to_be_downfilled] = subject_budget_value

            #negative label literal
            df_all_unmatched_lineitems_to_be_filled['label'] = 'not_a_match'

            #append negatives
            df_empty = df_empty.append(df_all_unmatched_lineitems_to_be_filled)
        
    return df_empty


def str_to_countidf_array(input_str, corpus):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_extraction.text import CountVectorizer
    import numpy as np
    '''
    This is a modification to tfidf. It maintains the scarcity weighting of IDF: log(N/df) but it yields 
    a wordcount not relative to the document size. So if there's a search: 'some word' the size of the 
    actual search term document length has no diminishing affects for searching the underlying terms.

    It just takes away the normalization by document length in tf. That's the only diff versus tfidf.

    todos:
    -----
    needs to be modularized
    '''

    # compute the idfs from corpus
    tfv = TfidfVectorizer(
                    norm=None,
                    analyzer='word',
                    sublinear_tf=True,
                    ngram_range=(1,1),
                    smooth_idf=True,
                )

    cnt_vect = CountVectorizer(
            analyzer='word',
            ngram_range=(1,1)
        )

    cnt_vect.fit(corpus)
    tfv.fit(corpus)
    
    #get these for analysis sake
    vocab_w_idf = dict(zip(tfv.get_feature_names(), tfv.idf_))
    
    #check if columns are aligned
    non_matches = [item for item in zip(cnt_vect.get_feature_names(), tfv.get_feature_names()) if  item[0] != item[1]]
    
    if non_matches:
        print non_matches
        raise Exception("vocabs don't match")    

    #grab indices for the word counts
    #motivation being that it's a sparse matrix
    input_arr = cnt_vect.transform([input_str]).todense()
    input_arr[input_arr != 0]
    _, indices_for_counts = input_arr.nonzero()

    #indices for counts
    counts = input_arr[0, indices_for_counts]
    idf_weights = tfv.idf_[indices_for_counts]

    #multiply the word counts and idf weights
    ctidf_arr = [ct * idf for ct, idf in zip(counts.A.tolist()[0], idf_weights.tolist())]

    #make an array
    zeroes_arr = np.zeros(input_arr.shape)
    zeroes_arr[:, indices_for_counts] = ctidf_arr

    return zeroes_arr


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

def plot_no_transformation(data_points_scaled):
   
    print(pd.Series(data_points_scaled).describe())
    print (_skew_and_kurtosis(data_points_scaled))
    plot_transformation(data_points_scaled, 'no_transformation');
    plt.violinplot(
       data_points_scaled,
       showmeans=False,
       showmedians=True
    );

    
def log_transform_distribution(data_points_list)
    constant = (1 + - 1 * data_points_list.min())
    log_transformed_data = preprocessing.scale(np.log(data_points_list + constant))
    return log_transformed_data


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

def score_classifier(clf, feature_M, labels, class_imbalance=True):
    from sklearn.cross_validation import StratifiedKFold, cross_val_score    
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
    mean = np.mean(scores_for_each_fold)
    std = np.std(scores_for_each_fold)
    
    return mean, median, std, scores_for_each_fold

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

#time_series
COLOR_PALETTE = [    
   "#348ABD",
   "#A60628",
   "#7A68A6",
   "#467821",
   "#CF4457",
   "#188487",
   "#E24A33"
  ]

def get_median_filtered(signal, threshold = 3):
    """
    signal: is numpy array-like
    returns: signal, numpy array 
    """
    difference = np.abs(signal - np.median(signal))
    median_difference = np.median(difference)
    s = 0 if median_difference == 0 else difference / float(median_difference)
    mask = s > threshold
    signal[mask] = np.median(signal)
    return signal

def get_mean_filtered(signal, threshold = 3):
    """
    signal: is numpy array-like
    returns: signal, numpy array 
    """
    difference = np.abs(signal - np.mean(signal))
    median_difference = np.mean(difference)
    s = 0 if median_difference == 0 else difference / float(median_difference)
    mask = s > threshold
    signal[mask] = np.mean(signal)
    return signal

def transform_to_signal_list(series_of_points, func, window_size = 20):

    outlier_s = series_of_points
    median_filtered_signal = []

    for ii in range(0, len(series_of_points), window_size):
        median_filtered_signal += func(np.asanyarray(outlier_s[ii: ii+20])).tolist() 
    return median_filtered_signal


def _detect_sign_change(x,y):    
    return (x * y) < 0.0

def _count_number_of_inflections(time_series):
    ct_of_inflections = 0 
    for idx in range(1, time_series.shape[0]):
        trailing_idx = idx - 1 
        
        if _detect_sign_change(time_series[idx], time_series[trailing_idx]):
            ct_of_inflections += 1

    return ct_of_inflections

def find_and_plot_converters(cohort, super_set):
    
    for emp_id in cohort:
        all_of_subject_employees_travel = super_set[super_set.employee_id.isin([emp_id])]
        time_series = all_of_subject_employees_travel.pct_savings
        time_series.index = pd.to_datetime(all_of_subject_employees_travel.approved_at)    

        if _count_number_of_inflections(time_series) > 2.0: 
            continue
        
        time_series.plot(figsize=(15,15), label=str(emp_id))
        
        plt.legend()

#make time-frame
def make_window(data, rolling_window_size):
    l_mask = data.approved_at < data.approved_at.min() + relativedelta(months=rolling_window_size)
    u_mask = data.approved_at > data.approved_at.max() - relativedelta(months=rolling_window_size)
    print (len(get_distance_of_months(data)), data.approved_at.min(), data.approved_at.max())
    return l_mask, u_mask

def get_metrics(Series):
    pct_below_budget = Series[Series > 0.0].shape[0]/Series.shape[0]
    return Series.describe()[['25%', '50%', '75%','count']], pct_below_budget

def get_distance_of_months(df):
    delta =  df.approved_at.max() - df.approved_at.min()
    distance_in_terms_of_months = int(delta.days/30)
    return [(df.approved_at.min().month + i)  % 12 for i in range(0, distance_in_terms_of_months)]from sqlalchemy import create_engine


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


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def find_and_plot_converters(cohort, super_set):
    
    for emp_id in cohort:
        all_of_subject_employees_travel = super_set[super_set.employee_id.isin([emp_id])]
        time_series = all_of_subject_employees_travel.pct_savings
        time_series.index = pd.to_datetime(all_of_subject_employees_travel.approved_at)    

        if count_number_of_sign_changes(time_series) > 2.0: 
            continue
        
        time_series.plot(figsize=(15,15), label=str(emp_id))
        
        plt.legend()

def transform_to_signal_list(series_of_points, func, window_size = 20):

    outlier_s = series_of_points
    median_filtered_signal = []

    for ii in range(0, len(series_of_points), window_size):
        median_filtered_signal += func(np.asanyarray(outlier_s[ii: ii+20])).tolist() 
    return median_filtered_signal

def plot_time_series_in_context_sequence_lollipop(median_filtered_signal, title_str='Median Filtered Signal'):
    plt.figure(figsize=(15, 10))
    plt.subplot(2,1,1)
    values = np.array(median_filtered_signal)
    (markers, stemlines, baseline) = plt.stem(values, basefmt='r--', base)
    useless_output = plt.setp(stemlines, linestyle="-", color="olive", linewidth=0.5, alpha=0.7)
    plt.title(title_str)

def plot_rolling_avg(subject_cohort, data, func=get_median_filtered, label_str='median'):
    
    mean_filtered_signal = transform_to_signal_list(subject_cohort.pct_savings.tolist(), func)
    
    ts = pd.Series(mean_filtered_signal)
    ts.index = pd.to_datetime(subject_cohort.approved_at)
    ts.plot(figsize=(15,15), label=label_str, linewidth=5)

def get_median_filtered(signal, threshold = 3):
    """
    signal: is numpy array-like
    returns: signal, numpy array 
    """
    difference = np.abs(signal - np.median(signal))
    median_difference = np.median(difference)
    s = 0 if median_difference == 0 else difference / float(median_difference)
    mask = s > threshold
    signal[mask] = np.median(signal)
    return signal

def get_mean_filtered(signal, threshold = 3):
    """
    signal: is numpy array-like
    returns: signal, numpy array 
    """
    difference = np.abs(signal - np.mean(signal))
    median_difference = np.mean(difference)
    s = 0 if median_difference == 0 else difference / float(median_difference)
    mask = s > threshold
    signal[mask] = np.mean(signal)
    return signal

def detect_sign_change(x,y):    
    return (x * y) < 0.0

def count_number_of_sign_changes(time_series):
    ct_of_inflections = 0 
    for idx in range(1, time_series.shape[0]):
        trailing_idx = idx - 1 
        
        if detect_sign_change(time_series[idx], time_series[trailing_idx]):
            ct_of_inflections += 1

    return ct_of_inflections


def plot_each_employee_as_timeseries(subject_cohort):
    
    for emp_id in subject_cohort:
        employee_level_cohort = data[data.employee_id.isin([emp_id])]
        time_series = employee_level_cohort.pct_savings
        time_series.index = pd.to_datetime(employee_level_cohort.approved_at)
        time_series.plot(figsize=(15,15),label=str(emp_id), alpha=0.3)

    pass


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

