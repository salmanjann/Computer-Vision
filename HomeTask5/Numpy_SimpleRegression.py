import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#data loading
df = pd.read_csv('NumpyRegCSV_Data.csv')
df = df.dropna(subset=['Calories', 'Duration'])
y_data = df['Calories'].to_numpy()
x_data = df['Duration'].to_numpy()

# Normalize inputs and outputs (standardization)
x_data = (x_data - x_data.mean()) / x_data.std()
y_data = (y_data - y_data.mean()) / y_data.std()

# print("NaNs in Duration:", df['Duration'].isnull().sum())
# print("NaNs in Calories:", df['Calories'].isnull().sum())
# print("Infs in Duration:", np.isinf(x_data).sum())
# print("Infs in Calories:", np.isinf(y_data).sum())
#
# print("x mean:", x_data.mean(), "x std:", x_data.std())
# print("y mean:", y_data.mean(), "y std:", y_data.std())


#Splitting Data into train and validation
N= x_data.size
idx=np.arange(N)
np.random.shuffle(idx)
idx_train=idx[:int(0.8*N)]
idx_test=idx[int(0.8*N):]
x_train, y_train = x_data[idx_train],y_data[idx_train]
x_val, y_val = x_data[idx_test],y_data[idx_test]

#training loop
#initializing parameters
trainLosses=[]
valLosses=[]
lr=0.01
w=np.random.randn(1) * 0.01
b=np.random.randn(1) * 0.01
for i in range(200):
    #forward pass
    yhat=w*x_train+b #note vectorized operation
    #MSE loss
    error=yhat-y_train
    loss= (error**2).mean()
    trainLosses.append(loss)
    #computing gradients
    db=2*error.mean()
    dw=2*(x_train*error).mean()
    #weight update
    b=b-lr*db
    w=w-lr*dw

    #val MSE loss
    yhatVal=w*x_val+b
    errorVal=yhatVal-y_val
    valLoss= (errorVal**2).mean()
    valLosses.append(valLoss)

    #stopping condition
    if(valLoss<0.001 or i==199):
        print(f'train loss={loss}, val loss={valLoss}, w={w}, b={b}')

        # training data plot
        plt.figure('1')
        plt.cla()
        plt.scatter(x_train,y_train)
        plt.scatter(x_train,yhat)
        plt.title(f'epoch={i}, loss={loss}, w={w}, b={b}')
        plt.show(block=False)
        plt.pause(1)

        #validation data plot
        plt.figure('2')
        plt.cla()
        plt.scatter(x_val,y_val)
        plt.scatter(x_val,yhatVal)
        plt.title(f'epoch={i}, ValLoss={valLoss}, w={w}, b={b}')
        plt.show(block=False)
        plt.pause(1)

        #trainLoss vs Epoch
        plt.figure('3')
        plt.cla()
        plt.plot(trainLosses)
        plt.xlabel('Epoch')
        plt.ylabel('trainLoss')
        plt.title(f'Training Loss Vs Epoch')
        plt.show(block=False)
        plt.pause(1)

        #validationLoss vs Epoch
        plt.figure('4')
        plt.cla()
        plt.plot(valLosses,color='m')
        plt.xlabel('Epoch')
        plt.ylabel('valLoss')
        plt.title(f'Validation Loss Vs Epoch')
        plt.show(block=False)
        plt.pause(1)

        break

# true_w=2
# true_b=1
# N=100
#
# np.random.seed(100)
# #get N uniformly distributed values
# x=np.random.rand(N,1)
# #get N noise values from standard normal distribution
# epsilon=0.1*np.random.randn(N,1)
# y=true_w*x+true_b+epsilon
#

#
#
# #plotting tain and val data
# plt.figure('1')
# plt.scatter(x_train,y_train)
# plt.xlabel('x_train')
# plt.ylabel('y_train')
# plt.title(f'true_w={true_w}, true_b={true_b}')
# plt.figure('2')
# plt.scatter(x_val,y_val,color = 'm')
# plt.xlabel('x_val')
# plt.ylabel('y_val')
# plt.title(f'true_w={true_w}, true_b={true_b}')
# plt.show(block=True)
#
#


    



    

















xx=0