
from sklearn.model_selection import train_test_split as tts
#
def find_const_columns(df):

    const_col = []
    for col in df.columns:
        unique_val = df[col].unique()
        if len(unique_val)==1:
            unique_val.append(col)

    return const_col

#
def const_col(df,cols_to_delete):

    new_df = df.drop(cols_to_delete, axis=1)
    return new_df

#
def find_col_with_few_values(df, threshold):
    col_with_few_values = []

    for col in df.columns:
        if df[col].nunique()< threshold:
            col_with_few_values.append(col)

    return col_with_few_values


#
def find_duplicate_rows(df):
    new_df = df[df.duplicated()]
    return new_df

#
def delete_duplicate_rows(df):
    new_df = df.drop_duplicate(keep='first')
    return new_df

#
def drop_fill(df):

    col_to_drop =  df.columns[df.isnull().mean()>0.5]

    new_df = df.drop(col_to_drop,axis=1)
    new_df = new_df.fillna(df.mean())

    return new_df


#
def split_data(df, target_col):

    x = df.drop(target_col, axis=1)
    y= df[target_col]

    x_train, x_test, y_train, y_test = tts(x, y, test_size=0.2, random_state=0)

    return x_train, x_test, y_train, y_test

