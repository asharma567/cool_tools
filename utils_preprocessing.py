import pandas as pd
import numpy as np
from utils_modeling_plot import pprint_df


def isfloat(x):
    try:
        a = float(x)
    except ValueError:
        return False
    else:
        return True

def isint(x):
    try:
        a = float(x)
        b = int(a)
    except ValueError:
        return False
    else:
        return a == b

def clean_up_null_data(df):
    '''
    when we started pulling queries from SQLPRO it showed 
    '<null>' instead of leaving it blank so here's a 
    subsitution
    '''
    df = df.applymap(lambda x: np.nan if x == '<null>' else x)
    df = df.applymap(lambda x: np.nan if x == 'NULL' else x)
    
    #convert the columns back to their proper data type
    df = df.applymap(lambda x: float(x) if isfloat(x) else x)
    df = df.applymap(lambda x: int(x) if isint(x) else x)
    
    return df

def detect_categorical_variables(input_df):
    '''
    I: Pandas DataFrame with categorical features
    O: categorical features list
    '''
    categorical_variables = []
    for col in input_df.columns:
        if not (input_df[col].dtype == 'float64' or input_df[col].dtype == 'int64'):
            categorical_variables.append(col)
                
    # print out all the categoricals which are being dummytized
    if categorical_variables: 
        print 'Categorical Variables Detected: '
        print '=' * 10
        print '\n'.join(categorical_variables)
    
    return categorical_variables


def join_to_feature(main_dataframe, additional_dataframe, rename_cnt_to_str, key_str):
    '''
    pass into two dataframes and a join field with a field to rename from 'cnt'
    '''
    main_dataframe = pd.merge(
                        main_dataframe, 
                        additional_dataframe, 
                        how='left', 
                        on=key_str
                    )
    main_dataframe = main_dataframe.rename(columns={'cnt':rename_cnt_to_str})
    return main_dataframe

def dummy_it(input_df, linear_model=False):
    '''
    I: Pandas DataFrame with categorical features
    O: Same df with binarized categorical features

    *check the dummy variable trap thing
    '''
    
    # base_case empty DF to append to
    base_case_df = pd.DataFrame()
    categorical_variables = []
    dropped_variables = []
    
    # every column that's not a categorical column, we dummytize
    for col in input_df.columns:
        if str(input_df[col].dtype) != 'object':
            base_case_df = pd.concat([base_case_df, input_df[col]], axis=1)
        else:
            if linear_model:
                dropped_variables.append(pd.get_dummies(input_df[col]).ix[:, -1].name)
                #leaves the last one out
                base_case_df = pd.concat([base_case_df, pd.get_dummies(input_df[col]).ix[:, :-1]], axis=1)
            else:
                base_case_df = pd.concat([base_case_df, pd.get_dummies(input_df[col])], axis=1)
            categorical_variables.append(col)
    
    # print out all the categoricals which are being dummytized
    if categorical_variables:
        print 'Variables Being Dummytized: '
        print '=' * 10
        print '\n'.join(categorical_variables)

    if dropped_variables:
        print 'Dropped to avoid dummy variable trap: '
        print '=' * 10
        print '\n'.join(dropped_variables)

    return base_case_df

def throw_away_series(df_feature_matrix, threshold=1000):
    '''
    I: Feature Matrix (Pandas DataFrame), threshold of observations
    O: df and pretty print df to std out
    throw away features with limited observations

    This function takes a feature matrix as input loops 
    through each feature and deleting each one that's less 
    than the threshold.

    eg.
    IN:df_truncated = throw_away_series(df, 1000)
    OUT:
    +---+--------------------------+---------------------+
    |   |  deleted feature names   | cnt of observations |
    +---+--------------------------+---------------------+
    | 0 |        GeoRegion         |        986.0        |
    | 1 | MostRecentAppraisedValue |        145.0        |
    | 2 |         NbrUnits         |        343.0        |
    | 3 |           Noi            |        329.0        |
    | 4 |   OriginalAppraisedVal   |        127.0        |
    | 5 |    PreviouslyValuedTo    |         18.0        |
    | 6 |    RentableSqrFootage    |         14.0        |
    | 7 |    Value_of_Property     |         25.0        |
    +---+--------------------------+---------------------+
    
    +----+-------------------------+---------------------+
    |    |    kept feature names   | cnt of observations |
    +----+-------------------------+---------------------+
    | 0  |    global_property_id   |        1000.0       |
    | 1  |       Appreciation      |        1000.0       |
    | 2  |       Depreciation      |        1000.0       |
    | 3  |    StateAppreciation    |        1000.0       |
    ...

    '''
    deleted_features = {}
    
    for key, val in df_feature_matrix.describe().ix['count'].iteritems():
        if val < threshold:
            deleted_features[key] = df_feature_matrix[key].describe().ix['count']
            del df_feature_matrix[key]
    
    df_deleted_cols = pd.DataFrame(deleted_features, index=[0]).T
    df_deleted_cols.reset_index(inplace=True)
    df_deleted_cols.rename(
        columns={
            'index' : 'deleted feature names', 
            0 : 'cnt of observations'
            }, 
        inplace=True
        )    

    pprint_df(df_deleted_cols)
    series_of_observation_cnts = df_feature_matrix.describe().ix['count']
    df_count_of_remaining_feature_M = pd.DataFrame(series_of_observation_cnts).reset_index().rename(
        columns={
            'index' : 'kept feature names', 
            'count' : 'cnt of observations'
            }
        )
    pprint_df(df_count_of_remaining_feature_M)
    
    return df_feature_matrix

def scale_feature_matrix(feature_M, linear=False, outliers=False):
    from sklearn.preprocessing import StandardScaler, RobustScaler
    import numpy as np
    
    binary_fields = [col for col in feature_M.columns if len(set(feature_M[col])) == 2]
            
    if outliers:
        #Scaling 0 median & unit variance
        scaler_obj = RobustScaler()
        print 'centering around median'

    else:
        #Scale 0 mean & unit variance
        scaler_obj = StandardScaler()
        print 'centering around mean'
    
    print 'found these binaries'
    print '-' * 10
    print '\n'.join(binary_fields)

        
    X_scaled = scaler_obj.fit_transform(feature_M.drop(binary_fields, axis=1))
    X_scaled_w_cats = np.c_[X_scaled, feature_M[binary_fields].as_matrix()]
    
    return X_scaled_w_cats, scaler_obj

def missing_values_finder(df):
    '''
    finds missing values in a data frame returns to you the value counts
    '''
    import pandas as pd
    missing_vals_dict= {col : df[col].dropna().shape[0] / float(df[col].shape[0]) for col in df.columns}
    output_df = pd.DataFrame().from_dict(missing_vals_dict, orient='index').sort_index()
    return output_df

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


def get_lat_lon(str_, stop_words=None):
    geocode_result = []
    gmaps = googlemaps.Client(key='AIzaSyCol8kK-GVXAIukXhICNXuaBIgqzENNp7I')
    
    try:

        if stop_words:
            str_ = ' '.join([word for word in str_.split() if word not in set(stop_words)])

        geocode_result = gmaps.geocode(str_)

        return geocode_result[0]['geometry']['location']
    except:
        print (geocode_result)
        return None
