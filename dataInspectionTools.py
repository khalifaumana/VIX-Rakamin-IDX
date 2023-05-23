import pandas as pd

def missing_cols(x):
        '''
        Returns a table in dataframe format.

                Parameters:
                        x (pandas DataFrame): DataFrame input to be inspected for the missing data
                Returns:
                        missing_value_df (pandas DataFrame): DataFrame output showing the missing data information for each columns in x (how many and its percentage)
        '''

        #acquiring column name, missing count and its percentage to the size of dataframe
        column_name =  x.columns
        missing_count = x.isnull().sum()
        percent_missing = missing_count * 100 / len(x)

        #creating a dataframe to store the missing row count
        missing_value_df = pd.DataFrame({
                                    'column_name': column_name,
                                    'missing_count' : missing_count,
                                    'percent_missing': percent_missing
                                    }).reset_index(drop=True)

        return missing_value_df

def missing_info(x):
        '''
        Returns 2 table in dataframe format.

                Parameters:
                        x (pandas DataFrame): DataFrame input to be inspected for the missing data
                Returns:
                        missing_value_df (pandas DataFrame): DataFrame output showing the missing data information for each columns in x (how many and its percentage)
                        missing_info (pandas DataFrame): DataFrame output showing a category of missing data (No Null, Semi Null, and fully null),
                                                                                how many columns for each category,
                                                                                and list of columns for each category
        '''
        missing_value_df = missing_cols(x)
        
        missing_category = ['No Null', 'Semi Null', 'Fully Null']
        
        NoNull_count = len(missing_value_df[missing_value_df['missing_count'] == 0])
        SemiNull_count = len(missing_value_df[(
                                        (missing_value_df['missing_count'] != 0) & 
                                        (missing_value_df['percent_missing'] != 100)
                                        )])
        FullyNull_count = len(missing_value_df[missing_value_df['percent_missing'] == 100])

        CountList = [NoNull_count, SemiNull_count, FullyNull_count]
        
        NoNull_tables = missing_value_df[missing_value_df['missing_count'] == 0]['column_name'].tolist()
        SemiNull_tables = missing_value_df[(
                                        (missing_value_df['missing_count'] != 0) & 
                                        (missing_value_df['percent_missing'] != 100)
                                        )]['column_name'].tolist()
        FullyNull_tables = missing_value_df[missing_value_df['percent_missing'] == 100]['column_name'].tolist()

        Tables = [NoNull_tables, SemiNull_tables, FullyNull_tables]

        missing_info = pd.DataFrame({
                                    'missing_category': missing_category,
                                    'table_number' : CountList,
                                    'table_list': Tables
                                    }).reset_index(drop=True)

        return missing_value_df, missing_info

def getNumericalCategorical(x):
    '''
        Returns 2 list of columns for numerical data and categorical data.

                Parameters:
                        x (pandas DataFrame): DataFrame input to be inspected for the missing data
                Returns:
                        numCols (list): list of numerical columns from DataFrame x
                        catCols (list): list of cateforical columns from DataFrame x
    '''


    # getting columns list of numerical data from DataFrame x using _get_numeric_data() function
    numCols = x._get_numeric_data().columns.tolist()

    # determining cateforical columns by substracting all columns list by numCols
    catCols = list(set(x.columns) - set(numCols))

    return numCols, catCols