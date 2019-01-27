import numpy as np
import pickle
import copy


config = {}
config['layer_specs'] = [784, 100, 100, 10]  # The length of list denotes number of hidden layers; each element denotes number of neurons in that layer; first element is the size of input layer, last element is the size of output layer.
config['activation'] = 'tanh' #Takes values 'sigmoid', 'tanh' or 'ReLU'; denotes activation function for hidden layers
config['batch_size'] = 1000  # Number of training samples per batch to be passed to network
config['epochs'] = 50  # Number of epochs to train the model
config['early_stop'] = True  # Implement early stopping or not
config['early_stop_epoch'] = 5  # Number of epochs for which validation loss increases to be counted as overfitting
config['L2_penalty'] = 0  # Regularization constant
config['momentum'] = False  # Denotes if momentum is to be applied or not
config['momentum_gamma'] = 0.9  # Denotes the constant 'gamma' in momentum expression
config['learning_rate'] = 0.0001 # Learning rate of gradient descent algorithm

def softmax(x):
  """
  Write the code for softmax activation function that takes in a numpy array and returns a numpy array.
  """
  if x.ndim != 1:
        output = (np.exp(x)/ np.array([np.sum(np.exp(x),axis=1)]).T)
  else:
        output= np.exp(x)/np.sum(np.exp(x))
  return output


def onehotencoding(labels):
    '''
    Does one-hot encoding from the labels
    Args
        labels : List containing the labels
    Returns
        onehotCoded : Matrix containing the one-hot encoded values
    '''
    onehotCoded = list()
    for value in labels:
        letter = [0 for i in range(10)]
        letter[value] = 1
        onehotCoded.append(letter)
    return np.array(onehotCoded)



def load_data(fname):
  """
  Write code to read the data and return it as 2 numpy arrays.
  Make sure to convert labels to one hot encoded format.
  """

  fobject = open(fname,"rb")
  fArray = pickle.load(fobject)
  images = fArray[:,:-1]
  labels = onehotencoding(fArray[:,-1])
  return images, labels


class Activation:
  def __init__(self, activation_type = "sigmoid"):
    self.activation_type = activation_type
    self.x = None # Save the input 'x' for sigmoid or tanh or ReLU to this variable since it will be used later for computing gradients.

  def forward_pass(self, a):
    if self.activation_type == "sigmoid":
      return self.sigmoid(a)

    elif self.activation_type == "tanh":
      return self.tanh(a)

    elif self.activation_type == "ReLU":
      return self.ReLU(a)

  def backward_pass(self, delta):
    if self.activation_type == "sigmoid":
      grad = self.grad_sigmoid()

    elif self.activation_type == "tanh":
      grad = self.grad_tanh()

    elif self.activation_type == "ReLU":
      grad = self.grad_ReLU()
    if type(delta) is not float:
        return (grad * delta.T).T
    else:
        return grad * delta

  def sigmoid(self, x):
    """
    Write the code for sigmoid activation function that takes in a numpy array and returns a numpy array.
    """
    self.x = x
    return 1 / (1 + np.exp(-x))

  def tanh(self, x):
    """
    Write the code for tanh activation function that takes in a numpy array and returns a numpy array.
    """
    self.x = x
    return np.tanh(x)

  def ReLU(self, x):
    """
    Write the code for ReLU activation function that takes in a numpy array and returns a numpy array.
    """
    self.x = x
    return x * (x > 0)


  def grad_sigmoid(self):
    """
    Write the code for gradient through sigmoid activation function that takes in a numpy array and returns a numpy array.
    """
    grad = (1 / (1 + np.exp(-self.x)))*(1-(1 / (1 + np.exp(-self.x))))
    return grad

  def grad_tanh(self):
    """
    Write the code for gradient through tanh activation function that takes in a numpy array and returns a numpy array.
    """
    grad = 1.0 - np.tanh(self.x)**2
    return grad

  def grad_ReLU(self):
    """
    Write the code for gradient through ReLU activation function that takes in a numpy array and returns a numpy array.
    """
    return 1.0 * (self.x > 0)


class Layer():
  def __init__(self, in_units, out_units):
    np.random.seed(42)
    self.w = np.random.randn(in_units, out_units)  # Weight matrix
    self.b = np.zeros((1, out_units)).astype(np.float32)  # Bias
    self.x = None  # Save the input to forward_pass in this
    self.a = None  # Save the output of forward pass in this (without activation)
    self.d_x = None  # Save the gradient w.r.t x in this
    self.d_w = None  # Save the gradient w.r.t w in this
    self.d_b = None  # Save the gradient w.r.t b in this

  def forward_pass(self, x):
    """
    Write the code for forward pass through a layer. Do not apply activation function here.
    """
    self.x = x
    W = np.concatenate((self.b, self.w), axis=0)
    X = np.concatenate((np.ones((1, x.shape[0])).astype(np.float32), self.x), axis=1)
    self.a = np.dot(X, W)
    return self.a

  def backward_pass(self, delta):
    """
    Write the code for backward pass. This takes in gradient from its next layer as input,
    computes gradient for its weights and the delta to pass to its previous layers.
    """
    self.d_w = self.x.T.dot(delta.T)
    self.d_b = delta.T
    self.d_x = self.w.dot(delta) #Bias not involved because bias has no input!
    return self.d_x


class Neuralnetwork():
  def __init__(self, config):
    self.layers = []
    self.x = None  # Save the input to forward_pass in this
    self.y = None  # Save the output vector of model in this
    self.targets = None  # Save the targets in forward_pass in this variable
    for i in range(len(config['layer_specs']) - 1):
      self.layers.append( Layer(config['layer_specs'][i], config['layer_specs'][i+1]) )
      if i < len(config['layer_specs']) - 2:
        self.layers.append(Activation(config['activation']))

  def forward_pass(self, x, targets=None):
    """
    Write the code for forward pass through all layers of the model and return loss and predictions.
    If targets == None, loss should be None. If not, then return the loss computed.
    """
    self.x = x
    self.targets = targets
    output = self.x
    if targets.all():
        loss = None
    else:
        for i in range(len(self.layers)):
            output = self.layers[i].forward_pass(output)
        self.y = softmax(output)
        loss = self.loss_func(self.y,targets)
    return loss, self.y

  def loss_func(self, logits, targets):
    '''
    find cross entropy loss between logits and targets
    '''
    output = -np.average(targets * np.log(logits))
    return output

  def backward_pass(self):
    '''
    implement the backward pass for the whole network.
    hint - use previously built functions.
    '''
    delta = (self.targets - self.y).T

    for i in reversed(range(len(self.layers))):
        delta = self.layers[i].backward_pass(delta)

    #changing weights
    for layer in self.layers:
        alpha = config['learning_rate']
        if isinstance(layer,Layer):
            layer.w = layer.w - alpha * layer.d_w
            layer.b = layer.b - alpha * layer.d_b



def trainer(model, X_train, y_train, X_valid, y_valid, config):
        """
  Write the code to train the network. Use values from config to set parameters
  such as L2 penalty, number of epochs, momentum, etc.
        """
        y = onehotencoding(y_train)
        yho = onehotencoding(y_valid)
        cost_array = []#np.zeros(config['epochs']);
        #cost_array = []
        hocost_array = []#np.zeros(config['epochs']);
        curr_ho_cost = np.inf
        for epoch in range(0,config['epochs']):
            cost = 0
            hocost = 0
            #print(random_order)
            batch_size = config['batch_size']
            for i in range(len(x_train)/batch_size):
                X_ib = X_train[i*batch_size:(i+1)*batch_size,:]
                y_ib = y[i*batch_size:(i+1)*batch_size,:]
                cost_ib,logits = model.forward_pass(X_ib,y_ib)
                cost += cost_ib
                model.backward_pass()
            cost_array.append(cost)
            hocost,logits = model.forward_pass(X_valid,yho)
            hocost_array.append(hocost)
            #cost_array[i] /= len(X)
            if hocost < curr_ho_cost:
                best_model = copy.deepcopy(model);
                curr_ho_cost = hocost;
            if sorted(hocost[-5:])  == hocost[-5:]:
                break;
        model = copy.deepcopy(best_model)



def test(model, X_test, y_test, config):
  """
  Write code to run the model on the data passed as input and return accuracy.
  """
  ytest = onehotencoding(y_test)
  cost_test,logits = model.forward_pass(X_test,ytest)
  predicted = np.argmax(logits,axis = 1)
  accuracy = np.sum(predicted == y_test)/len(y_test)
  return accuracy


if __name__ == "__main__":
  train_data_fname = 'MNIST_train.pkl'
  valid_data_fname = 'MNIST_valid.pkl'
  test_data_fname = 'MNIST_test.pkl'

  ### Train the network ###
  model = Neuralnetwork(config)
  X_train, y_train = load_data(train_data_fname)
  X_valid, y_valid = load_data(valid_data_fname)
  X_test, y_test = load_data(test_data_fname)
  trainer(model, X_train, y_train, X_valid, y_valid, config)
  test_acc = test(model, X_test, y_test, config)

