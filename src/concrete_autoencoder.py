#This file is based upon the original implementation of the CAE, which can be found here: https://github.com/mfbalin/Concrete-Autoencoders/blob/master/concrete_autoencoder/concrete_autoencoder
import math
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Softmax, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import Constant, glorot_normal
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

class ConcreteSelect(Layer):
    
    def __init__(self, output_dim, start_temp = 10.0, min_temp = 0.1, alpha = 0.99999, **kwargs):
        self.output_dim = output_dim
        self.start_temp = start_temp
        self.min_temp = K.constant(min_temp)
        self.alpha = K.constant(alpha)
        super(ConcreteSelect, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.temp = self.add_weight(name = 'temp', shape = [], initializer = Constant(self.start_temp), trainable = False)
        self.logits = self.add_weight(name = 'logits', shape = [self.output_dim, input_shape[1]], initializer = glorot_normal(), trainable = True)
        super(ConcreteSelect, self).build(input_shape)
        
    def call(self, X, training=None):
        uniform = tf.random.uniform(tf.shape(self.logits), minval=K.epsilon(), maxval=1.0)
        gumbel = -tf.math.log(-tf.math.log(uniform))

        new_temp = tf.maximum(self.min_temp, self.temp * self.alpha)
        self.temp.assign(new_temp)

        noisy_logits = (self.logits + gumbel) / self.temp
        samples = tf.nn.softmax(noisy_logits, axis=-1)

        discrete_logits = tf.one_hot(
            tf.argmax(self.logits, axis=-1),
            depth=tf.shape(self.logits)[1]
        )

        if training:
            self.selections = samples
        else:
            self.selections = discrete_logits

        Y = tf.matmul(X, tf.transpose(self.selections))
        return Y
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
    
class StopperCallback(EarlyStopping):
    
    def __init__(self, mean_max_target = 0.998):
        self.mean_max_target = mean_max_target
        super(StopperCallback, self).__init__(monitor = '', patience = float('inf'), verbose = 1, mode = 'max', baseline = self.mean_max_target)
    
    def on_epoch_begin(self, epoch, logs = None):
        if epoch % 50 == 0:
                print('Epoch', epoch, '- mean max of probabilities:', self.get_monitor_value(logs), '- temperature', K.get_value(self.model.get_layer('concrete_select').temp))
        #print('mean max of probabilities:', self.get_monitor_value(logs), '- temperature', K.get_value(self.model.get_layer('concrete_select').temp))
        #print( K.get_value(K.max(K.softmax(self.model.get_layer('concrete_select').logits), axis = -1)))
        #print(K.get_value(K.max(self.model.get_layer('concrete_select').selections, axis = -1)))
    
    def get_monitor_value(self, logs):
        monitor_value = K.get_value(K.mean(K.max(K.softmax(self.model.get_layer('concrete_select').logits), axis = -1)))
        return monitor_value


class ConcreteAutoencoderFeatureSelector():
    
    def __init__(
            self,
            K,
            output_function,
            loss='mean_squared_error',
            num_epochs=300,
            batch_size=None,
            learning_rate=0.001,
            start_temp=10.0,
            min_temp=0.1,
            tryout_limit=1
        ):
        self.K = K
        self.output_function = output_function
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.start_temp = start_temp
        self.min_temp = min_temp
        self.tryout_limit = tryout_limit
        self.loss = loss
        
    def fit(self, X, Y = None, val_X = None, val_Y = None):
        if Y is None:
            Y = X
        assert len(X) == len(Y)
        validation_data = None
        if val_X is not None and val_Y is not None:
            assert len(val_X) == len(val_Y)
            validation_data = (val_X, val_Y)
        
        if self.batch_size is None:
            self.batch_size = max(len(X) // 256, 16)
        
        num_epochs = self.num_epochs
        steps_per_epoch = (len(X) + self.batch_size - 1) // self.batch_size
                               
        inputs = Input(shape = X.shape[1:])

        alpha = math.exp(math.log(self.min_temp / self.start_temp) / (num_epochs * steps_per_epoch))
        
        self.concrete_select = ConcreteSelect(self.K, self.start_temp, self.min_temp, alpha, name = 'concrete_select')

        selected_features = self.concrete_select(inputs)

        outputs = self.output_function(selected_features)

        self.model = Model(inputs, outputs)

        self.model.compile(Adam(self.learning_rate), loss = self.loss) 
        
        stopper_callback = StopperCallback()
        
        hist = self.model.fit(X, Y, self.batch_size, num_epochs, verbose = 0, callbacks = [stopper_callback], validation_data = validation_data)#, validation_freq = 10)
            
    
        self.probabilities = K.get_value(K.softmax(self.model.get_layer('concrete_select').logits))
        self.indices = K.get_value(K.argmax(self.model.get_layer('concrete_select').logits))
            
        return self
    
    def get_indices(self):
        return K.get_value(K.argmax(self.model.get_layer('concrete_select').logits))
    
    def get_mask(self):
        return K.get_value(K.sum(K.one_hot(K.argmax(self.model.get_layer('concrete_select').logits), self.model.get_layer('concrete_select').logits.shape[1]), axis = 0))
    
    # Check if columns and rows are correct 
    def transform(self, X):
        return X[:, self.get_indices()]
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
    
    def get_support(self, indices = False):
        return self.get_indices() if indices else self.get_mask()
    
    def get_params(self):
        return self.model