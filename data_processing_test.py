

from data_ingest import ingest_data
from data_processing import (
    drop_fill,
    find_col_with_few_values,
    const_col,
    find_const_columns
)
from feature_engineering import bin_to_num, cat_to_col, one_hot_encoding

Ingest_Data = ingest_data()
df = Ingest_Data.get_data(r'C:\Users\vishw\PycharmProjects\MLR US cancer mortality\data\cancer_reg.csv')

Constant_columns = find_const_columns(df)
print('Columns that contain same value in all rows = ',Constant_columns)

Cols_with_few_values = find_col_with_few_values(df, 10)


df = bin_to_num(df)

df = cat_to_col(df)
df = one_hot_encoding(df)
df = drop_fill(df)
print(df.shape)

df.to_csv(r'C:\Users\vishw\PycharmProjects\MLR US cancer mortality\data/cancer_reg_processed.csv', index=False)

