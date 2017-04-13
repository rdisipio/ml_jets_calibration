from keras.models import Sequential

from keras.layers import Merge
from keras.layers import Dense, Activation
from keras.layers import Lambda
from keras.layers import Dropout
from keras.layers.advanced_activations import ELU

from keras.wrappers.scikit_learn import KerasRegressor

from features import *

#############################


def eta_range( y ):
   ymax = 2.5
   if abs(ymax) <  abs(y): return 0.
   return y

############################

#   model_pT.add(Dropout(0.2))

def create_model_pT():
   model_pT = Sequential()

   model_pT.add( Dense( 300, input_dim=n_input_pT) )
   model_pT.add( ELU() )

   model_pT.add( Dense( 200, input_dim=n_input_pT) )
   model_pT.add( ELU() )

   model_pT.add( Dense( 100, input_dim=n_input_pT) )
   model_pT.add( ELU() )

   model_pT.add( Dense(1))
   model_pT.add( ELU() )

   return model_pT

def create_model_eta():
   model_eta = Sequential()

   model_eta.add( Dense( 50, input_dim=n_input_eta) )

   model_eta.add( Dense( 10, activation='tanh' ) )

   model_eta.add( Dense(1) )

   return model_eta

def create_model_E():
   model_E = Sequential()

   model_E.add( Dense( 300, input_dim=n_input_E) )
   model_E.add( ELU() )

   model_E.add( Dense( 200) )
   model_E.add( ELU() )

   model_E.add( Dense( 100) )
   model_E.add( ELU() )

   model_E.add( Dense(1) )
   model_E.add( ELU() )

   return model_E

def create_model_M():
   model_M = Sequential()

   model_M.add( Dense( 300, input_dim=n_input_M) )
   model_M.add( ELU() )

   model_M.add( Dense( 200 ) )
   model_M.add( ELU() )

   model_M.add( Dense( 100 ) )
   model_M.add( ELU() )

   model_M.add( Dense(1) )
   model_M.add( ELU() )

   return model_M

def create_model_merged():
   model_pT  = create_model_pT()
   model_eta = create_model_eta()
   model_E   = create_model_E()
   model_M   = create_model_M()

   merged = Merge( [ model_pT, model_eta, model_E, model_M ], mode='concat' )

   model = Sequential()
   model.add(merged)
   model.add( Dense( 4 ) )

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

  

def create_model_pT_E_parallel():
   model_pT  = create_model_pT()
   model_E   = create_model_E()
   merged = Merge( [ model_pT, model_E ], mode='concat' )

   model = Sequential()
   model.add(merged)
   model.add( Dense( 2 ) )

   model.compile( loss='mean_squared_error', optimizer='adam' )
#   model.compile( loss='mean_squared_logarithmic_error', optimizer='adam' )

   return model

#################

def create_model_4( wgt_filename = "weights.model_pT_E_parallel.h5" ):
   model_pT_E = create_model_pT_E_merged()
   model_pT_E.load_weights( wgt_filename )
   model_pT_E.trainable = False

   model_M   = create_M()

   merged = Merge( [ model_pT_E, model_M ], mode='concat' )

   model = Sequential()
   model.add(merged)
   model.add( Dense( 3 ) )

   model.compile( loss='mean_squared_error', optimizer='adam' )
#   model.compile( loss='mean_squared_logarithmic_error', optimizer='adam' )

   return model


#################


def create_model_5():
   model_pT_E_M  = create_model_pT_E_M()
   model_pT_E_M.load_weights( "model_pT_E_M.wgt" )
   model_pT_E_M.trainable = False

   model_eta = create_eta()

   merged = Merge( [ model_pT_E_M, model_eta ], mode='concat' )

   model = Sequential()
   model.add(merged)
   model.add( Dense( pT_E_M ) )

   model.compile( loss='mean_squared_error', optimizer='adam' )
#   model.compile( loss='mean_squared_logarithmic_error', optimizer='adam' )

   return model

#################

