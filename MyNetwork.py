
def computeClosestCenters(input, cent):
    k = cent.shape[0]
    m = input.shape[0]
    member = np.zeros((m, 1))
    dist = np.zeros((m, k))
    for i in range(k):
        diff = input-cent[i,:]
        SquaredDiffs = np.square(diff)
        dist[:, i] = SquaredDiffs.sum(axis=1)
    member = dist.argmin(axis = 1)
    return member 
def computeCenters(input, prev_cent, member, k):
    m,n = input.shape
    cent = np.zeros((k, n))
    for i in range(k):
        if (~np.any(member == i)):
            cent[i, :] = prev_cent[i, :]
        else:
            p = input[member == i, :]
            cent[i, :] = np.mean(p,axis=0) 
    return cent
        
def getActivations(cent, beta, input):
    diff = cent-input
    squaredDists= np.square(diff).sum(axis=1)
    squaredDists = squaredDists.reshape((squaredDists.shape[0],1))
    z = np.exp(np.multiply(np.negative(beta),squaredDists))
    z = z.reshape(z.shape[0])

    return z
def computeBetas(input, center, member):
    numofNeurons = center.shape[0]
    sigmas = np.zeros((numofNeurons, 1))
    for i in range(numofNeurons):
        cent = center[i, :]
        members = input[member == i, :]
        diff = members-cent
        squaredDiffs = np.square(diff)
        squredDists= squaredDiffs.sum(axis=1)
        dist = np.sqrt(squredDists)
        sigmas[i, :] = np.mean(dist)
    betas = 1/(2*np.square(sigmas))
    return betas

def kMeans(input, init_cent, max_itereration):
    k = init_cent.shape[0];
    cent = init_cent;
    prevCent = cent;
    for i in range(max_itereration):
        member = computeClosestCenters(input, cent);
        cent = computeCenters(input, cent, member, k);
        if ((prevCent==cent).all()):
            break
        prevCent = cent
    return (cent, member)
def get_activations(X_train, y_train, numOfCentersPerCategory):
    numOfCats = 10;
    m = X_train.shape[0]
    Cent = np.array([])
    betas = np.array([])
    for c in range(numOfCats):
        X = X_train[y_train == c, :]
        init_Cent = X[0:numOfCentersPerCategory, :];
        Cent_c, member_c = kMeans(X, init_Cent, 100)
        
        betas_c = computeBetas(X, Cent_c, member_c);
        Cent = np.vstack([Cent, Cent_c]) if Cent.size else Cent_c;
        betas = np.vstack([betas, betas_c]) if betas.size else betas_c;
    
    numOfRBFNeurons = Cent.shape[0];
    X_active = np.zeros((m, numOfRBFNeurons));
    for i in range(m):
        input = X_train[i, :];
        z = getActivations(Cent, betas, input);

        X_active[i, :] = z.T

        


    return X_active, Cent, betas

def evaluate(Cent, betas, W, b,input):
    diffs = getActivations(Cent, betas, input)
    z = W.T.dot(diffs)+b
    return z





def one_hot(input, no_of_classes):
    out = np.array(input).reshape(-1)
    return np.eye(no_of_classes)[out]

def safe_ln(x, minimumVal=0.0000000001):
    return np.log(x.clip(min=minimumVal))

def softmax(input):  
    return np.exp(input) / np.sum(np.exp(input))

def stable_softmax(input):
    out = np.exp(input - np.max(input))
    return out / np.sum(out)

def loss(out, target):
    return - (target * safe_ln(out)).sum()

def sigmoid(input):
    return 1. / (1. + np.exp(-input))
  
def output(input, W, b):
    return stable_softmax((input @ W) + b)

                   
def error_of_output(out, target):

    return out - target

def weight_out(input, error):
    return  input.T @ error

def bias_out(input):
    return  np.sum(input, axis=0, keepdims=True)

def backprop(input, target,W, b):

    out = output(input, W, b)
    error = error_of_output(out, target)
    delta_W = weight_out(input, error)
    delta_b = bias_out(error)
    return [delta_W, delta_b]


def momentum_update(input, target, list_of_parameters, M, momentum, learning_rate):

    J = backprop(input, target, *list_of_parameters)
    return [momentum * i - learning_rate * j for i,j in zip(M, J)]

def parameters_update(list_of_parameters, M):
  
    return [i + j for i,j in zip(list_of_parameters, M)]
                   
                   

def train(train_txt,lr,k,num_epochs):
    print('Loading images')
    X_train = []
    y_train = []
    num_cats=10
    cd = os.getcwd()
    file = open(train_txt, "r") 
    for line in file:
        im = imageio.imread(cd+line[:-3])
        X_train.append(im.reshape(784))
        y_train.append(int(line[-2]))
    X_train = np.array(X_train)
    X_train = (255-X_train)/255
    y_train = np.array(y_train)
    y_train_one_hot = one_hot(y_train,num_cats)

    print('Computing rbf activations')
    X_activ, Centers, betas = get_activations(X_train, y_train, k);

    print('Learning Weights')
    X = X_activ
    T = y_train_one_hot
    Wo = np.random.normal(-1,1,(k*num_cats,num_cats))
    bo = np.random.normal(-1,1,(1,num_cats))
    learning_rate = lr
    momentum_term = 0.9
    Ms = [np.zeros_like(M) for M in [Wo, bo]]
    step_size = 1

    
    for i in range(num_epochs):
        for j in range(0,X.shape[0],step_size):
            Ms = momentum_update(X[j:j+step_size,:], T[j:j+step_size], [Wo, bo], Ms, momentum_term, learning_rate)
            Wo, bo = parameters_update([Wo, bo], Ms)

    print('Saving Weights to the file')
    import pickle

    with open('netWeights.txt', 'wb') as f:
        pickle.dump([Wo, bo], f)

def test(text_txt,W,k):
    print('Loading images')
    X_test = []
    y_test = []
    num_cats = 10
    cd = os.getcwd()
    file = open(text_txt, "r") 
    for line in file:
        im = imageio.imread(cd+line[:-3])
        X_test.append(im.reshape(784))
        y_test.append(int(line[-2]))
    X_test = np.array(X_test)
    X_test = (255-X_test)/255
    y_test = np.array(y_test)
    y_test_one_hot = one_hot(y_test,num_cats)
    numRight = 0;
    m = X_test.shape[0]

    import pickle

    with open(W,'rb') as f:  
        Wo, bo = pickle.load(f)

    X_activ, Centers, betas = get_activations(X_test, y_test, k);
    print('Computing accuracy')
    for i in range(m):
        scores =  evaluate(Centers, betas, Wo,bo,X_test[i,:])
        category = np.argmax(scores)
        if (category == y_test[i]):
            numRight = numRight + 1


    accuracy = (numRight / m )* 100;
    print('test accuracy =', accuracy )




import numpy as np
import imageio
import os
import sys
func_to_call = sys.argv[1]
if (func_to_call=='train'):
    train_txt = sys.argv[2]
    lr = sys.argv[3]
    lr = float(lr)
    k = sys.argv[4]
    num_epochs = sys.argv[4]
    train(train_txt,lr,k,num_epochs)
elif(func_to_call=='test'):
    text_txt = sys.argv[2]
    W = sys.argv[3]
    k = sys.argv[4]
    test(text_txt,W,k)
    
else:
    print('Error:', func_to_call,' is not defined.')







