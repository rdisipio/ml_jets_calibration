from keras.models import load_model
from keras.models import Model, Sequential
from keras.layers import Merge, Dense, Activation, Input, LSTM, Permute, Reshape
from keras.layers import Lambda
from keras.layers import Dropout
from keras.layers import concatenate, maximum, dot, average
from keras.layers.pooling import MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
from keras.layers.merge import *
from keras.layers.convolutional import Conv1D
from keras.optimizers import *
from keras.regularizers import l2

from keras.wrappers.scikit_learn import KerasRegressor

from features import *

#############################

def custom_loss(y_true, y_pred):
#   y_t = K.clip( y_true, K.epsilon(), 1. )
#   y_p = K.clip( y_pred, K.epsilon(), 1. )
#   return K.mean( K.abs( y_p/y_t - 1 ) )
#    return K.mean(K.square((y_pred/y_true) - 1))
#   return K.mean( (y_pred/y_true) - 1. )
   return K.mean( (y_pred - y_true)/K.sqrt(y_true) )

#############################

from keras.layers.merge import _Merge
class Subtract(_Merge):
   def _merge_function(self, inputs):
      output = inputs[0]
      for i in range(1, len(inputs)):
         output -= inputs[i]
      return output

def subtract(inputs, **kwargs):
   return Subtract(**kwargs)(inputs)


############################

def create_model_calib_SingleNet():

   kreg = None
   #kreg = kernel_regularizer=l2(0.01)

   input_pT  = Input( shape=(n_input_pT,) )
   input_eta = Input( shape=(n_input_eta,) )
   input_E   = Input( shape=(n_input_E,) )
   input_M   = Input( shape=(n_input_M,) )

   x_pT   = Dense(n_input_pT)(input_pT)
   x_eta  = Dense(n_input_eta)(input_eta)
   x_E    = Dense(n_input_E)(input_E)
   x_M    = Dense(n_input_M)(input_M)

   input_calib = concatenate( [ x_pT, x_eta, x_E, x_M ] )

# , kernel_regularizer=l2(0.1)
   dnn_calib   = Dense( 500, activation='relu' )(input_calib)
   dnn_calib   = Dense( 300, activation='relu', kernel_regularizer=kreg )(dnn_calib)
   dnn_calib   = Dense( 200, activation='relu', kernel_regularizer=kreg )(dnn_calib)
   dnn_calib   = Dense( 100, activation='relu', kernel_regularizer=kreg )(dnn_calib)
   dnn_calib   = Dense(  50, activation='relu', kernel_regularizer=kreg )(dnn_calib)
   dnn_calib   = Dense(   4 )(dnn_calib)
   dnn_model   = Model( inputs=[input_pT,input_eta,input_E,input_M], outputs=dnn_calib )

   dnn_model.compile( optimizer='adam', loss='mean_squared_error' )
#   dnn_model.compile( optimizer='adam', loss='mean_absolute_error' )
#   dnn_model.compile( optimizer='adam', loss=custom_loss )
   print "INFO: DNN calibration model SingleNet compiled"
   return dnn_model

############################

def create_model_calib_4p():
   input_pT  = Input( shape=(n_input_pT,) )
   input_eta = Input( shape=(n_input_eta,) )
   input_E   = Input( shape=(n_input_E,) )
   input_M   = Input( shape=(n_input_M,) )

   tower_pT = Dense(50, kernel_initializer='glorot_normal' )(input_pT)
   for n_nodes in [ 30, 20 ]:
      tower_pT = Dense(n_nodes)(tower_pT)
      tower_pT = Activation('relu')(tower_pT)
   tower_pT = Dense(1)(tower_pT)

   tower_eta = Dense(10, kernel_initializer='glorot_normal' )(input_eta)
   for n_nodes in [ 5 ]:
      tower_eta = Dense(n_nodes)(tower_eta)
      tower_eta = Activation('relu')(tower_eta)
   tower_eta = Dense(1)(tower_eta)

   tower_E = Dense(50, kernel_initializer='glorot_normal' )(input_E)
   for n_nodes in [ 30, 20 ]:
      tower_E = Dense(n_nodes)(tower_E)
      tower_E = Activation('relu')(tower_E)
   tower_E = Dense(1)(tower_E)

   tower_M = Dense(100, kernel_initializer='glorot_normal' )(input_M)
   for n_nodes in [ 50, 30 ]:
      tower_M = Dense(n_nodes)(tower_M)
      tower_M = Activation('relu')(tower_M)
   tower_M = Dense(1)(tower_M)

   tower_pT_eta_E_M = concatenate( [ tower_pT, tower_eta, tower_E, tower_M ] )
   tower_pT_eta_E_M = Dense(4)(tower_pT_eta_E_M)
   tower_pT_eta_E_M = Dense(4)(tower_pT_eta_E_M)

   calibrated = Dense( 4, activation='linear', name='calibrated' )(tower_pT_eta_E_M)
   input_calib = [ input_pT, input_eta, input_E, input_M ]
   dnn_model  = Model( inputs=input_calib, outputs=calibrated )

   dnn_model.compile( optimizer='adam', loss='mean_squared_error' )

   print "INFO: DNN calibration model 4p compiled"
   return dnn_model


############################


def create_model_calib_4p_resnet():
   def _block_resnet(_input):
      n = K.int_shape(_input)[1]
      _output = Dense(n)(_input)
      #_output = BatchNormalization()(_output)
      _output = Activation('relu')(_output)
      _output = Dense(n)(_output)
      _output = add( [ _output, _input ] )
      _output = Activation('relu')(_output)
      return _output

   input_pT  = Input( shape=(n_input_pT,) )
   input_eta = Input( shape=(n_input_eta,) )
   input_E   = Input( shape=(n_input_E,) )
   input_M   = Input( shape=(n_input_M,) )

   tower_pT = Dense(50, kernel_initializer='glorot_normal' )(input_pT)
   for n_nodes in [ 30, 20 ]:
      tower_pT = Dense(n_nodes)(tower_pT)
      tower_pT = _block_resnet(tower_pT)
   tower_pT = Dense(1)(tower_pT)
   
   tower_eta = Dense(10, kernel_initializer='glorot_normal' )(input_eta)
   for n_nodes in [ 5 ]:
      tower_eta = Dense(n_nodes)(tower_eta)
      tower_eta = _block_resnet(tower_eta)
   tower_eta = Dense(1)(tower_eta)

   tower_E = Dense(50, kernel_initializer='glorot_normal' )(input_E)
   for n_nodes in [ 30, 20 ]:
      tower_E = Dense(n_nodes)(tower_E)
      tower_E = _block_resnet(tower_E)
   tower_E = Dense(1)(tower_E)

   tower_M = Dense(100, kernel_initializer='glorot_normal' )(input_M)
   for n_nodes in [ 50, 30 ]:
      tower_M = Dense(n_nodes)(tower_M)
      tower_M = _block_resnet(tower_M)
   tower_M = Dense(1)(tower_M)

   tower_pT_eta_E_M = concatenate( [ tower_pT, tower_eta, tower_E, tower_M ] )  
   tower_pT_eta_E_M = Dense(4)(tower_pT_eta_E_M)
   tower_pT_eta_E_M = Dense(4)(tower_pT_eta_E_M)

   calibrated = Dense( 4, activation='linear', name='calibrated' )(tower_pT_eta_E_M)
   input_calib = [ input_pT, input_eta, input_E, input_M ] 
   dnn_model  = Model( inputs=input_calib, outputs=calibrated )

   dnn_model.compile( optimizer='adam', loss='mean_squared_error' )

   print "INFO: DNN calibration model 4p ResNet compiled"
   return dnn_model

############

def create_model_calib_433():

   input_pT  = Input( shape=(n_input_pT,) )
   input_eta = Input( shape=(n_input_eta,) )
   input_E   = Input( shape=(n_input_E,) )
   input_M   = Input( shape=(n_input_M,) )

   tower_pT = Dense(500, kernel_initializer='glorot_normal' )(input_pT)
   for n_nodes in [ 300, 100, 50 ]:
      tower_pT = Dense(n_nodes)(tower_pT)
      tower_pT = Activation('relu')(tower_pT)
   tower_pT = Dense(1)(tower_pT)

   tower_eta = Dense(500, kernel_initializer='glorot_normal' )(input_eta)
   for n_nodes in [ 300, 100, 50 ]:
      tower_eta = Dense(n_nodes)(tower_eta)
      tower_eta = Activation('relu')(tower_eta)
   tower_eta = Dense(1)(tower_eta)

   tower_E = Dense(500, kernel_initializer='glorot_normal' )(input_E)
   for n_nodes in [ 300, 100, 50 ]:
      tower_E = Dense(n_nodes)(tower_E)
      tower_E = Activation('relu')(tower_E)
   tower_E = Dense(1)(tower_E)

   tower_M = Dense(500, kernel_initializer='glorot_normal' )(input_M)
   for n_nodes in [ 300, 100, 50 ]:
      tower_M = Dense(n_nodes)(tower_M)
      tower_M = Activation('relu')(tower_M)
   tower_M = Dense(1)(tower_M)

   dnn_calib_pT_eta_E = concatenate( [ tower_pT, tower_eta, tower_E ] ) 
   dnn_calib_pT_eta_M = concatenate( [ tower_pT, tower_eta, tower_M ] )
   dnn_calib_pT_eta_E_M = concatenate( [ dnn_calib_pT_eta_E, dnn_calib_pT_eta_M ] )

   calibrated = Dense( 4, activation='linear', name='calibrated' )(dnn_calib_pT_eta_E_M)

   input_calib = [ input_pT, input_eta, input_E, input_M ]
   dnn_model  = Model( inputs=input_calib, outputs=calibrated )

   dnn_model.compile( optimizer='adam', loss='mean_squared_error' )

   print "INFO: DNN calibration model 433 compiled"
   return dnn_model

######################


