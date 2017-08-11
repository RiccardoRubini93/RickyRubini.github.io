from scipy import optimize
import pylab as pl
import numpy as np

class Neural_Net(object):
	def __init__(self,Lambda=0):
	#define parameters
		self.inputLayerSize = 1
		self.outputLayerSize = 1
		self.hiddenLayerSize = 20

	#weight matrices
		self.W1 = np.random.rand(self.inputLayerSize,self.hiddenLayerSize)
		self.W2 = np.random.rand(self.hiddenLayerSize,self.outputLayerSize)
	
	#regularization parameters
		
		self.Lambda = Lambda
	def forward(self,X):
	#propagate input
		self.z2 = np.dot(X,self.W1)
		self.a2 = self.act_func(self.z2)
		self.z3 = np.dot(self.a2,self.W2)
		y_out = self.act_func(self.z3)
		return y_out

	def act_func(self,z):
		
		return np.tanh(z)

	def act_funcPrime(self,z):
		
		return 1 - np.tanh(z)*np.tanh(z)

	def costFunction(self,X,Y):
		self.y_out = self.forward(X)
		J = 0.5*sum((Y-self.y_out)**2)/X.shape[0] + (self.Lambda/2)*(np.sum(self.W1**2)+np.sum(self.W2**2))
		return J
	
	def costFunctionPrime(self, X, Y):

		self.y_out = self.forward(X)

		delta3 = np.multiply(-(Y-self.y_out), self.act_funcPrime(self.z3))
		dJdW2 = np.dot(self.a2.T, delta3)

		delta2 = np.dot(delta3, self.W2.T)*self.act_funcPrime(self.z2)
		dJdW1 = np.dot(X.T, delta2)  

		return dJdW1, dJdW2

	#Helper Functions for interacting with other classes:
	def getParams(self):
	#Get W1 and W2 unrolled into vector:
		params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
		return params
    
	def setParams(self, params):
	#Set W1 and W2 using single paramater vector.
		W1_start = 0
		W1_end = self.hiddenLayerSize * self.inputLayerSize
		self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize ,self.hiddenLayerSize))
		W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
		self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))

	def computeGradients(self, X, Y):
		dJdW1, dJdW2 = self.costFunctionPrime(X, Y)
		return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))

	def computeNumericalGradient(N, X, Y):
		paramsInitial = N.getParams()
		numgrad = np.zeros(paramsInitial.shape)
		perturb = np.zeros(paramsInitial.shape)
		e = 1e-4 # defalt -4

		for p in range(len(paramsInitial)):
    		#Set perturbation vector
			perturb[p] = e
			N.setParams(paramsInitial + perturb)
			loss2 = N.costFunction(X, Y)    
			N.setParams(paramsInitial - perturb)
			loss1 = N.costFunction(X, Y)
			
		#Compute Numerical Gradient
			numgrad[p] = (loss2 - loss1) / (2*e)

		#Return the value we changed to zero:
			perturb[p] = 0
    
		#Return Params to original value:
		N.setParams(paramsInitial)

		return numgrad 

class Trainer(object):
    
	def __init__(self, N):
	#Make Local reference to network:
		self.N = N

	def callbackF(self, params):
		self.N.setParams(params)
		self.J.append(self.N.costFunction(self.X, self.Y))   
    
	def costFunctionWrapper(self,params,X,Y):
		self.N.setParams(params)
		cost = self.N.costFunction(X,Y)
		grad = self.N.computeGradients(X,Y)
		return cost, grad

    
	def train(self,X,Y):
	#Make an internal variable for the callback function:
		self.X = X
		self.Y = Y

	#Make empty list to store costs:
		self.J = []

		params0 = self.N.getParams()
		max_iter = 5000
		options = {'maxiter': max_iter, 'disp' : True}
		_res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS', args=(X, Y), options=options, callback=self.callbackF)

		self.N.setParams(_res.x)
		self.optimizationResults = _res

def input(x_min,x_max,N):
	
	x  = np.linspace(0,x_max,N)
	X = x/np.amax(x,axis = 0)
	X = X.reshape((N,1))

	y =  np.sin(x) + 0*np.random.rand(len(x))
	Y = y/np.amax(y,axis = 0)
	Y = y.reshape((N,1))

	return X,Y,x,y

def main():

	N = 100
	x_min = 0
	x_max = 10

	X,Y,x,y = input(x_min,x_max,N)
	NN = Neural_Net(Lambda=0)
	T = Trainer(NN)
	T.train(X,Y)

	fig1 = pl.figure(1)

	pl.plot(T.J)
	pl.grid(1)
	pl.xlabel('Iteration')
	pl.ylabel('Cost')
	pl.show()

	x_coord = np.linspace(x_min,x_max,100)
	x_coordNorm = x_coord/x_max

	allInputs = np.zeros((x_coord.size,1))
	allInputs[:,0] = x_coordNorm

	allOutputs = NN.forward(allInputs)

	#plot
	fig2 = pl.figure(2)

	pl.plot(x_coordNorm*np.amax(x,axis = 0),allOutputs*np.amax(y,axis = 0),'b--^',label='NN')
	pl.plot(X*np.amax(x,axis = 0),Y*np.amax(y,axis = 0),'r',lw=2,label='sin(x)')
	pl.legend()
	pl.title('Sin function Interpolation')
	pl.xlabel('X')
	pl.ylabel('Y')
	pl.show()

if __name__=="__main__":
	main()



