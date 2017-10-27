
import numpy as np
import logging
import json
from utility import * #custom methods for data cleaning

FILE_NAME_TRAIN = 'train.csv' #replace this file name with the train file
FILE_NAME_TEST = 'test.csv' #replace
ALPHA = 0.002
EPOCHS = 10000#keep this greater than or equl to 5000 strictly otherwise you will get an error
MODEL_FILE = 'models/model2'
train_flag = False

logging.basicConfig(filename='output.log',level=logging.DEBUG)

np.set_printoptions(suppress=True)
#################################################################################################
#####################################write the functions here####################################
#################################################################################################
#this function appends 1 to the start of the input X and returns the new array
def appendIntercept(X):
    #steps
    #make a column vector of ones
    #stack this column vector infront of the main X vector using hstack
    #return the new matrix
    (x,y) = X.shape

    arr = np.ones((x,1))
    X = np.hstack((arr, X))
    return X
    #remove this line once you finish writing




 #intitial guess of parameters (intialize all to zero)
 #this func takes the number of parameters that is to be fitted and returns a vector of zeros
def initialGuess(n_thetas):
     vec = np.zeros((1, n_thetas))
     #print vec
     return vec



def train(theta, X, y, model):

     EPOCHS = 5000
     J = [] #this array should contain the cost for every iteration so that you can visualize it later when you plot it vs the ith iteration
     #train for the number of epochs you have defined
     m = len(y)
     #your  gradient descent code goes here
     #steps
     #run you gd loop for EPOCHS that you have defined
        #calculate the predicted y using your current value of theta
        # calculate cost with that current theta using the costFunc function
        #append the above cost in J
        #calculate your gradients values using calcGradients function
        # update the theta using makeGradientUpdate function (don't make a new variable assign it back to theta that you received)

     while EPOCHS>0 :

         y_predicted = predict(X, theta)
         y_predicted = y_predicted.flatten()
         cost = costFunc(m, y, y_predicted)
         J.append(cost)
         grad = calcGradients(X, y, y_predicted, m)
         theta = makeGradientUpdate(theta, grad )

         EPOCHS = EPOCHS -1


     model['J'] = J
     model['theta'] = list(theta.flatten())
     print "training....."
     return model


#this function will calculate the total cost and will return it
def costFunc(m,y,y_predicted):
    #takes three parameter as the input m(#training examples), (labeled y), (predicted y)
    #steps
    #apply the formula learnt
    return ((np.sum((np.multiply(-y,np.log(y_predicted)))+(np.multiply(y-1, np.log(1-y_predicted)))))/m)
    #return (( np.sum(np.square(np.subtract(y_predicted ,y))))/(2*m))


def calcGradients(X,y,y_predicted,m):
    #apply the formula , this function will return cost with respect to the gradients
    # basically an numpy array containing n_params
    #y.reshape((y.shape[0],1))
    #arr = y_predicted - y
    #print y_predicted.shape
    #print y.shape
    arr = np.subtract(y_predicted,y)

    #print arr.shape
    arr = arr.reshape((m,1))
    X = X * arr
    return (np.sum( X, axis=0)/m)
       #pass

#this function will update the theta and return it
def makeGradientUpdate(theta, grads):
    return (np.subtract(theta, np.multiply(ALPHA, grads)))


def sigmoid( param ):
    return (1/(1+ np.exp(-param)))


#this function will take two paramets as the input

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




########################main function###########################################
def main():
    if(train_flag):
        model = {}
        X_df,y_df = loadData(FILE_NAME_TRAIN)
        X,y, model = normalizeData(X_df, y_df, model)
        X = appendIntercept(X)
        theta = initialGuess(X.shape[1])
        model = train(theta, X, y, model)



        with open(MODEL_FILE,'w') as f:
            f.write(json.dumps(model))


    else:
        model = {}
        with open(MODEL_FILE,'r') as f:
            model = json.loads(f.read())
            X_df, y_df = loadData(FILE_NAME_TEST)
            X,y = normalizeTestData(X_df, y_df, model)
            X = appendIntercept(X)
            accuracy(X,y,model)



if __name__ == '__main__':
    main()
