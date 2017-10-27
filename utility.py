

import numpy as np
import pandas as pd #not of your use
import logging
import json

FILE_NAME_TRAIN = 'train.csv' #replace this file name with the train file
FILE_NAME_TEST = 'test.csv' #replace
ALPHA = 1e-3
EPOCHS = 100000
MODEL_FILE = 'models/model2'


logging.basicConfig(filename='output.log',level=logging.DEBUG)


#utility functions
def loadData(file_name):
    df = pd.read_csv(file_name)
    logging.info("Number of data points in the data set "+str(len(df)))
    y_df = df['output']
    keys = ['overall_rating', 'bought_at', 'months_used', 'issues_rating', 'resale_value']
    X_df = df.get(keys)
    return X_df, y_df


def normalizeData(X_df, y_df, model):
    #save the scaling factors so that after prediction the value can be again rescaled
    model['input_scaling_factors'] = [list(X_df.mean()),list(X_df.std())]
    model['output_scaling_factors'] = [y_df.mean(), y_df.std()]
    X = np.array((X_df-X_df.mean())/X_df.std())
    #y = np.array((y_df - y_df.mean())/y_df.std())
    return X, y_df, model

def normalizeTestData(X_df, y_df, model):
    meanX = model['input_scaling_factors'][0]
    stdX = model['input_scaling_factors'][1]
    meany = model['output_scaling_factors'][0]
    stdy = model['output_scaling_factors'][1]

    X = 1.0*(X_df - meanX)/stdX
    #y = 1.0*(y_df - meany)/stdy

    return X, y_df


def accuracy(X, y, model):
    #print "Y is"
    #print y
    total_data_points= y.shape[0]
    #print total_data_points
    theta = np.array(model["theta"])
    y_pred = predict(X, theta)
    #print y_pred
    y_pred = np.subtract(y_pred, y)

    total_datapoint_classified_correct_by_your_model= total_data_points- np.count_nonzero(y_pred)
    #print total_datapoint_classified_correct_by_your_model

    acc = (total_datapoint_classified_correct_by_your_model*1.0 /total_data_points)*100
    print "Accuracy is:",
    print acc

def predict(X,theta):
    #print X.shape
    theta = np.transpose(theta)
    #print theta.shape
    arrr = sigmoid(np.dot(X, theta))
    ar1 = []
    for x in arrr:
        if x>= 0.5:
            ar1.append(1)
        else:
            ar1.append(0)
    ar1 = np.array(ar1)
    return ar1


def sigmoid( param ):
    return (1/(1+ np.exp(-param)))
