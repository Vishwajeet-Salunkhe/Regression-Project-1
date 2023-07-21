import warnings
warnings.filterwarnings("ignore")

import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd

from data_processing import split_data


#
def correlation_among_numeric_features(df, col):

    corr_features = set()
    corr = df.corr()

    for i in range(len(corr.columns)):
        for j in range(i):
            if corr.iloc[i,j]>0.8:
                colname = corr.columns[i]
                corr_features.add(colname)

    return corr_features

#
def lr_model(xTrain, yTrain):
    #create a fitted model
    xTrain_with_intercept = sm.add_constant(xTrain)
    lr = sm.OLS(yTrain, xTrain_with_intercept).fit()

    return lr





#
def identify_significant_vars(lr, p_value_threshold=0.05):
    # print the p-values
    print(lr.pvalues)
    # print the r-squared value for the model
    print(lr.rsquared)
    # print the adjusted r-squared value for the model
    print(lr.rsquared_adj)
    # identify the significant variables
    significant_vars = [var for var in lr.pvalues.keys() if lr.pvalues[var] < p_value_threshold]
    return significant_vars



if __name__ == "__main__":
    capped_data = pd.read_csv(r'C:\Users\vishw\PycharmProjects\MLR US cancer mortality\data\capped_data.csv')
    print(capped_data)

    corr_features = correlation_among_numeric_features(capped_data, capped_data.columns)
    print('correlated_features = ', corr_features)
    print(len(corr_features))

    cols = [col for col in capped_data.columns if col not in corr_features]

    xtrain, xtest, ytrain, ytest = split_data(capped_data[cols], 'target_deathrate')

    lr = lr_model(xtrain, ytrain)
    summary = lr.summary()
    print(summary)
