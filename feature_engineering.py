import pandas as pd
from sklearn.preprocessing import OneHotEncoder

#
def bin_to_num(data):
    binnedinc= []

    for i in data['binnedinc']:
        i = i.strip('()[]')
        i = i.split(',')
        i = tuple(i)
        i = tuple(map(float,i))
        i = list(i)

        binnedinc.append(i)

    data['binnedinc'] = binnedinc

    data['upper_bound'] = [i[1] for i in data['binnedinc'] ]
    data['lower_bound'] = [i[0] for i in data['binnedinc'] ]

    data['median']  =  (data['upper_bound'] + data['lower_bound'])/2

    data.drop('binnedinc', axis=1, inplace=True)
    return data

#
def cat_to_col(data):
    data['county'] = [i.split(',')[0] for i in data['geography']]
    data['state']   = [i.split(',')[1] for i in data['geography']]

    data.drop('geography', axis=1, inplace=True)
    return data


#
def one_hot_encoding(data):
    categorical_col = data.select_dtype(include=['object']).columns

    one_hot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    one_hot_encoded = one_hot_encoder.fit_transform(data['categorical_col'])

    one_hot_encoded = pd.DataFrame(one_hot_encoded, columns = one_hot_encoder.get_feature_names_out(categorical_col))

    data.drop(categorical_col, axis=1, inplace=True)
    data = pd.concate([data, one_hot_encoded], axis=1)

    return data
