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

from keras.wrappers.scikit_learn import KerasRegressor

from features import *

#############################


def eta_range( y ):
   ymax = 2.5
   if abs(ymax) <  abs(y): return 0.
   return y

############################

#   model_pT.add(Dropout(0.2))

def create_model_pT( do_compile=True ):
   inputs = Input( shape=(n_input_pT,))

   x = Dense( 300, activation='relu' )(inputs)
   x = Dense( 200, activation='relu' )(x)
   x = Dense( 100, activation='relu' )(x)
   x = Dense(   1, activation='relu' )(x)

   model_pT = Model( inputs=inputs, outputs=x )

   if do_compile == True:
      print "DEBUG: compiling model pT"
      model_pT.compile( loss='mean_squared_error', optimizer='adam' )

   return model_pT

def create_model_eta( do_compile=True ):
  
   inputs = Input( shape=(n_input_eta,))
   x = Dense( 300 )(inputs)
   x = Dense( 1 )(x)

   model_eta = Model( inputs=inputs, outputs=x )

   if do_compile == True:
      print "DEBUG: compiling model eta"
      model_eta.compile( loss='mean_squared_error', optimizer='adam' )

   return model_eta

def create_model_E( do_compile=True ):
   inputs = Input( shape=(n_input_E,))

   x = Dense( 300, activation='relu' )(inputs)
   x = Dense( 200, activation='relu' )(x)
   x = Dense( 100, activation='relu' )(x)
   x = Dense(   1, activation='relu' )(x)

   model_E = Model( inputs=inputs, outputs=x )

   if do_compile == True:
      print "DEBUG: compiling model E"
      model_E.compile( loss='mean_squared_error', optimizer='adam' )

   return model_E

def create_model_M( do_compile=True ):
   inputs = Input( shape=(n_input_M,))

   x = Dense( 300, activation='relu' )(inputs)
   x = Dense( 200, activation='relu' )(x)
   x = Dense( 100, activation='relu' )(x)
   x = Dense(   1, activation='relu' )(x)

   model_M = Model( inputs=inputs, outputs=x )

   if do_compile == True:
      print "DEBUG: compiling model M"
      model_M.compile( loss='mean_squared_error', optimizer='adam' )

   return model_M

def create_model_merged( do_compile=True ):
   model_pt  = create_model_pT( do_compile=False )
   model_eta = create_model_eta( do_compile=False )
   model_E   = create_model_E( do_compile=False )
   model_M   = create_model_M( do_compile=False )

   model_pt.load_weights( "weights.pt.h5" )
   model_eta.load_weights( "weights.eta.h5" )
   model_E.load_weights( "weights.E.h5" )
   model_M.load_weights( "weights.M.h5" )

   model_pt.trainable  = False
   model_eta.trainable = False
   model_E.trainable   = False
   model_M.trainable   = False

   merged = Merge( [ model_pt, model_eta, model_E, model_M ], mode='concat' )
#   merged = concatenate( [ model_pt, model_eta, model_E, model_M ], axis=-1 )

   model = Sequential()
   model.add(merged)

   model.add( Dense( 4 ) )

#   model.trainable   = False

   if do_compile == True:
      model.compile( loss='mean_squared_error', optimizer='adam' )
#   model.compile( loss='mean_squared_logarithmic_error', optimizer='adam' )

   return model

#################

def create_model_all():
  '''Trains all variables at once'''

  model_all = Sequential()

  model_all.add( Dense( 500, input_dim=n_input_all ) )
  model_all.add( ELU() )

  model_all.add( Dense(100) )
  model_all.add( ELU() )

  model_all.add( Dense(4) )
#  model_all.add( ELU() )

  model_all.compile( loss='mean_squared_error', optimizer='adam' )

  return model_all

#################

def create_model_pT_E_M_parallel():
   model_pT  = create_model_pT()
   model_E   = create_model_E()
   model_M   = create_model_M()

   merged = Merge( [ model_pT, model_E, model_M ], mode='concat' )

   model = Sequential()
   model.add(merged)
   model.add( Dense( 3 ) )

   model.compile( loss='mean_squared_error', optimizer='adam' )
#   model.compile( loss='mean_squared_logarithmic_error', optimizer='adam' )

   return model

#################

def create_model_pT_E_single():
   model = Sequential()

   model.add( Dense( 400, input_dim=n_input_pT ) )
   model.add( ELU() )

   model.add( Dense(2) )
   model.add( ELU() )

   model.compile( loss='mean_squared_error', optimizer='adam' )

   return model

#################

def create_model_pT_eta_single():
   model = Sequential()

   model.add( Dense( 400, input_dim=n_input_pT ) )
#   model.add( ELU() )

   model.add( Dense(2) )
#   model.add( ELU() )

   model.compile( loss='mean_squared_error', optimizer='adam' )

   return model

#################
  

def create_model_pT_E_M_single( do_compile = True ):
   model = Sequential()

   model.add( Dense( 500, input_dim=n_input_pT_E_M) )
   model.add( ELU() )

   model.add( Dense( 300 ) )
   model.add( ELU() )

   model.add( Dense( 200) )
   model.add( ELU() )

   model.add( Dense( 100) )
   model.add( ELU() )

   model.add( Dense(1) )
   model.add( ELU() )

   if do_compile == True:
      model.compile( loss='mean_squared_error', optimizer='adam' )
#   model.compile( loss='mean_squared_logarithmic_error', optimizer='adam' )

   return model

#################


def create_model_funnel_pretrained():

   model_pt  = create_model_pT( do_compile=False )
   model_eta = create_model_eta( do_compile=False )
   model_E   = create_model_E( do_compile=False )
   model_M   = create_model_M( do_compile=False )

   model_pt.load_weights( "weights.pt.h5" )
   model_eta.load_weights( "weights.eta.h5" )
   model_E.load_weights( "weights.E.h5" )
   model_M.load_weights( "weights.M.h5" )

   merged_pT_E_M = keras.layers.concatenate( [model_pt, model_E, model_M ] )
   merged_pT_E_M = Dense( 3, activation='relu')(merged_pT_E_M)
   merged_pT_E_M = Dense( 3, activation='relu')(merged_pT_E_M)
   merged_pT_E_M = Dense( 3, activation='relu')(merged_pT_E_M)

   merged_final = keras.layers.concatenate( [ merged_pT_E_M, model_eta ] )

#   merged_pT_E_M = Merge( [ model_pt, model_E, model_M ], mode='concat' )
#   model_pT_E_M = Sequential()
#   model_pT_E_M.add(merged_pT_E_M)

#   merged_final = Merge( [ merged_pT_E_M, model_eta ], mode='concat' )
#   model_final = Sequential()
#   model_final.add(merged_final)

#   model.add( Permute( (1,4,2,3) )

   model = Model(inputs=[inputs_pT, inputs_eta, inputs_E, inputs_M], outputs=[merged_final] )
   model.compile( loss='mean_squared_error', optimizer='adam' )
#   model.compile( loss='mean_squared_logarithmic_error', optimizer='adam' )

   return model

#################

def create_model_funnel():
   inputs_pT  = Input( shape=(n_input_pT,),  name='inputs_pT' )
   inputs_eta = Input( shape=(n_input_eta,), name='inputs_eta' )
   inputs_E   = Input( shape=(n_input_E,),   name='inputs_E')
   inputs_M   = Input( shape=(n_input_M,),   name='inputs_M')

   x_pT = Dense( 300 )(inputs_pT)
#   x_pT = BatchNormalization()(x_pT)
   x_pT = Dense( 200, activation='linear' )(x_pT)
   x_pT = Dense( 100, activation='linear' )(x_pT)
   x_pT = Dense(   1, activation='linear' )(x_pT)

   x_eta = Dense( 100 )(inputs_eta)
#   x_eta = BatchNormalization()(x_eta)
   x_eta = Dense(   1 )(x_eta)

   x_E = Dense( 300 )(inputs_E)
#   x_E = BatchNormalization()(x_E)
   x_E = Dense( 200, activation='linear' )(x_E)
   x_E = Dense( 100, activation='linear' )(x_E)
   x_E = Dense(   1, activation='linear' )(x_E)

   x_M = Dense( 300 )(inputs_M)
#   x_M = BatchNormalization()(x_M)
   x_M = Dense( 200, activation='linear' )(x_M)
   x_M = Dense( 100, activation='linear' )(x_M)
   x_M = Dense(   1, activation='linear' )(x_M)

#   x_pT_E = concatenate( [ x_pT, x_E ] )
#   x_pT_E = Dense(   2, activation='relu' )(x_pT_E)
#   x_pT_E = Dense( 100, activation='relu' )(x_pT_E)
#   x_pT_E = Dense( 100, activation='relu' )(x_pT_E)
#   x_pT_E = Dense( 100, activation='relu' )(x_pT_E)
#   x_pT_E = Dense(   2, activation='relu' )(x_pT_E)

#   x_pT_E_M = concatenate( [ x_pT_E, x_M ] )
#   x_pT_E_M = concatenate( [x_pT, x_E, x_M] )
#   x_pT_E_M = Dense(   3, kernel_initializer='normal' )( x_pT_E_M )
#   x_pT_E_M = Dense( 100, activation='relu' )(x_pT_E_M)
#   x_pT_E_M = Dense( 100, activation='relu' )(x_pT_E_M)
#   x_pT_E_M = Dense( 100, activation='relu' )(x_pT_E_M)
#   x_pT_E_M = Dense(   3, activation='relu' )(x_pT_E_M)
#   x_pT_eta_E = concatenate( [ x_pT, x_eta, x_E ] )
#   x_pT_eta_E = Dense( 3, kernel_initializer='normal' )(x_pT_eta_E)

#   x_pT_eta_E_M = concatenate( [ x_pT_E, x_eta, x_M ] )
   x_pT_eta_E_M = concatenate( [ x_pT, x_eta, x_E, x_M ] )
#   x_pT_eta_E_M = concatenate( [ x_pT_E_M, x_eta ] )
#   x_pT_eta_E_M = concatenate( [ x_pT_eta_E, x_M ] )
   x_pT_eta_E_M = Dense( 4 )(x_pT_eta_E_M)
#   x_pT_eta_E_M = BatchNormalization()(x_pT_eta_E_M)
#   x_pT_eta_E_M = Dense( 300 )(x_pT_eta_E_M)
#   x_pT_eta_E_M = Dense( 200 )(x_pT_eta_E_M)
#   x_pT_eta_E_M = Dense( 100 )(x_pT_eta_E_M)
#   x_pT_eta_E_M = Dense( 4 )(x_pT_eta_E_M)

   calibrated = Dense( 4, activation='linear', name='calibrated' )(x_pT_eta_E_M)   

   model = Model( inputs = [ inputs_pT, inputs_eta, inputs_E, inputs_M ], outputs=calibrated )
 
   model.compile( optimizer='adam', loss={'calibrated' : 'mean_squared_error'}, loss_weights={'calibrated':1.0} )
#   model.compile( optimizer='adam', loss={'calibrated' : 'mean_absolute_error'}, loss_weights={'calibrated':1.0} )
#   model.compile( loss='mean_squared_error', optimizer='adam' )

   return model

##################
