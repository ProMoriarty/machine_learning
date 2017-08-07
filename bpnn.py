#from tensorflow_test.bp import *
import numpy as np
def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1.0 - np.tanh(x)*np.tanh(x)

def logistic(x):
    return 1/(1 + np.exp(-x))

def logistic_derivative(x):
    return logistic(x)*(1-logistic(x))

class NeuralNetwork:
    def __init__(self,layers,activation="tanh"):
        if activation =="tanh":
            self.activation = tanh
            self.activation_deriv = tanh_deriv
        if activation == "logistic":
            self.activation=logistic
            self.activation_deriv = logistic_derivative
        self.weights = []
        for i in range(1,len(layers)-1): #对于权重参数w，只有一个中间层
            self.weights.append((2*np.random.random((layers[i - 1] + 1, layers[i] + 1))-1)*0.25)  #根据开始设定的节点数来生成权重矩阵
            self.weights.append((2*np.random.random((layers[i] + 1, layers[i + 1]))-1)*0.25)      #根据开始设定的节点数来生成权重矩阵
        print(self.weights)
    def fit(self,X,y,learning_rate = 0.2 , epoch=10000):#学习率0.2,迭代10000次
        X = np.atleast_2d(X)
        temp = np.zeros((X.shape[0],X.shape[1]+1))#多出来的一列是biases，wx+b
        temp[:,0:-1] = X
        X = temp #numpy的复制
        y = np.array(y)

        for k in range(epoch):
            i = np.random.randint(X.shape[0])
            a = [X[i]]
            for l in range(len(self.weights)):
                a.append(self.activation(np.dot(a[l],self.weights[l]))) #正向求值
            error = y[i]-a[-1]                                          #计算误差
            deltas = [error*self.activation_deriv(a[-1])]               #

            for l in range(len(a) - 2, 0, -1):
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_deriv(a[l]))
            deltas.reverse()
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

    def predict(self,x):
        x = np.atleast_2d(x)
        temp = np.zeros((x.shape[0],x.shape[1]+1))
        temp[:,0:-1] = x
        a = temp
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a

if __name__ == '__main__':
    nn = NeuralNetwork([2,2,1],"tanh")
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([0,1,1,0])
    nn.fit(X,y)
    for i in [[0,0],[0,1],[1,0],[1,1]]:
        print(i,nn.predict(i))