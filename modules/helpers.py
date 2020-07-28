import pandas as pd
import numpy as np

#read in an csv file and display the columns and head
def read_in_dataset(fname, verbose = False):
    data = pd.read_csv(fname)
    if verbose:
        print('\n{0:*^80}'.format('Reading in the {0} dataset'.format(fname)))
        print('\nit has {0} rows and {1} columns'.format(*data.shape))
        print('\n{0:*^80}'.format('it has the following columns\n'))
        print(data.columns)
        print('\n{0:*^80}\n'.format('the first 5 rows looks like this'))
        print(data.head())
    return data

#get the title from the name and aggregate based on survival pattern
def get_name_title_adv(df):
    df['name_title'] = df.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())
    df['name_title_adv'] = df.name_title.apply(lambda x: 'title_survive' if x in 
                                                 ['Lady','Mlle','Mme', 'the Countess','Ms','Sir'] else 
                                                 'title_nonsurvive' if x in 
                                                 ['Capt','Don','Jonkheer'] else x)
    df = df.drop(columns = 'name_title')
    return df

#use the first letter of the string to simplify categorical variables with many different values.
def get_first_letter(df, colname):
    df[colname + '_firstletter'] = df[colname].apply(lambda x: 'None' if str(x)[0].isdigit() else str(x)[0])
    return df

#if a value of a categorical variable in the testing set does not exist in the training set, set the value to be the mode of the training set of that categorical variable
def fill_extra_categories_with_train_mode(df, colname, train):
    extra_values = set(df[colname]) - set(train[colname])
    replacement = train[colname].mode()[0]
    
    for value in extra_values:
        bool1 = df[colname] == value
        print('\nreplaced {0} {1} with {2}'.format(sum(bool1), value, replacement))
        df.loc[bool1,colname] = replacement
    return df

#calculate the total number of words in a value (eg. cabins) associated with the record
#In this case the total word count is the number of cabins
def get_word_count(df, colname):
    df[colname +'_total'] = df[colname].apply(lambda x:0 if pd.isna(x) else len(x.split(' ')))
    return df

#remove columns that are not going to the models
def remove_extra_columns(df, colnames):
    df = df.drop(columns = colnames)
    return df

#fill in the empty cells for column1 base on the aggregated value from column2, pivot table is obtained from the training set
#For example, it can be used to replace the empty cells in Age, using the mean Age of each title (Ms, Mr, Miss, etc)
def fill_colname1_by_train_colname2(df,train, colname1, colname2):
    bool1 = pd.isnull(df[colname1])
    if sum(bool1) > 0:
        print('\nimputing {0} using aggregated info baesd on name_title_adv from the training set.......\n'.format(colname1))
        colname1_by_colname2_table_train = pd.pivot_table(train, index = colname2, values = colname1)

        nan_colname1_by_colname2_table = pd.pivot_table(df[bool1], index = colname2, 
                                    values = 'Ticket_firstletter', aggfunc = 'count')
        for index in nan_colname1_by_colname2_table.index:
            value_from_pivot_table = colname1_by_colname2_table_train.loc[index][colname1]
            bool1 = pd.isnull(df[colname1]) 
            bool2 = df[colname2] == index
            to_replace = np.logical_and(bool1, bool2)
            df.loc[to_replace,colname1] = value_from_pivot_table
            print('filled {0} title {1} with age value {2}'.format(sum(to_replace),index, value_from_pivot_table))
    else:
        print('\nNo NaN in {0}\n'.format(colname1))
    return df

#fill in the empty cells of Embarked using the training set mode value
def fill_with_train_mode(df,colname, train):
    bool1 = pd.isnull(df[colname])
    if sum(bool1) > 0:
        print('\nimputing {0} using mode from the training set......\n'.format(colname))
        df[colname] = df[colname].fillna(train[colname].mode()[0])
        print('{0} {1} imputed with {2}\n'.format(sum(bool1), colname, train[colname].mode()[0]))
    else:
        print('\nNo NaN in {0}\n'.format(colname))
    return df

#fill in the empty cells of Fare using the training set median value
def fill_with_train_median(df, colname, train):
    bool1 = pd.isnull(df[colname])
    if sum(bool1) > 0:
        print('\nimputing {0} using median from the training set......\n'.format(colname))
        df[colname] = df[colname].fillna(train[colname].median())
        print('{0} {1} imputed\n'.format(sum(bool1), colname))
    else:
        print('No NaN in {0}\n'.format(colname))
    return df

#convert a numeric variable to categorical
def convert_col_to_str(df, colname):
    df[colname] = df[colname].astype(str)
    return df

#normalize a variable
def normalize(df, colname):
    df['norm_' + colname] = np.log(df[colname] + 1)
    return df

#chain the whole preprocess
def preprocess(df,train):
    
    #cleaning:
    
    #convert Pclass to categorical
    df = convert_col_to_str(df, 'Pclass')
    
    #fill in the empty cells of Embarked
    df = fill_with_train_mode(df,'Embarked',train)

    #normalize then fill in the empty cells for Fare
    df = normalize(df, 'Fare')
    df = fill_with_train_median(df,'norm_Fare',train)
    
    #Feature Engineering:
    
    #create name_title_adv
    df = get_name_title_adv(df)
    #Use the first letter of Ticket and Cabin, and total Cabin count to simplify the values
    df = get_first_letter(df,'Ticket')
    df = get_first_letter(df,'Cabin')
    df = get_word_count(df,'Cabin')
    
    #Replace the values in 'name_title_adv' in the test set that is absent in the training with training mode
    df = fill_extra_categories_with_train_mode(df, 'name_title_adv', train)

    #Fill the empty cells of Age by aggregrated values from name_title_adv
    df = fill_colname1_by_train_colname2(df,train,'Age','name_title_adv')

    #remove extra columns
    df = remove_extra_columns(df, ['PassengerId','Name','Ticket','Cabin', 'Fare'])

    return df

#scale the numeric columns
from sklearn.preprocessing import StandardScaler
def apply_scaler(train, test):
    scale = StandardScaler()
    train_scaled = train.copy()
    test_scaled = test.copy()
    colnames = list(train_scaled.select_dtypes(include=['float64','int64']).columns)
    print('\nThe following columns are scaled:\n')
    print(colnames)
    train_scaled[colnames] =  scale.fit_transform(train_scaled[colnames])
    test_scaled[colnames] =  scale.transform(test_scaled[colnames])
    return [train_scaled, test_scaled]

#feature importance
def get_model_feature_importances(model, feature_df):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        importances = [0] * len(feature_df.columns)
    
    feature_importances = pd.DataFrame({'feature': feature_df.columns, 'importance': importances})
    feature_importances.sort_values(by = 'importance', ascending = False, inplace = True)
    ''' set the index to 'feature' '''
    feature_importances.set_index('feature', inplace = True, drop = True)
    return feature_importances

def clf_performance(classifier, model_name):
    print(model_name)
    print('Best Score: ' + str(classifier.best_score_))
    print('Best Parameters: ' + str(classifier.best_params_))