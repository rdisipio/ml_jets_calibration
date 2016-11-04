#!/usr/bin/env python

import os, sys

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import SGD

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score

import cPickle as pickle
import numpy as np
import pandas as pd

from ROOT import *

np.set_printoptions( precision=2, suppress=True )


# change this to increase the number of eta slices
etaregions = [ [0.0,0.1], [0.1,0.2], [0.2,0.3], [0.3,0.4], [0.4,0.5], [0.5,0.6], [0.6,0.7], [0.7,0.8], [0.8,0.9], [0.9,1.0], [1.0,2.5] ]


#################


def FindEtaRegion( eta ):
   eta = abs(eta) 

   ieta = 0
   for region in etaregions:
     etamin = region[0]
     etamax = region[1]

     if eta >= etamin and eta <= etamax:
       return ieta
     ieta += 1
   print "WARNING: eta =", eta

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

with open( "scaler.pkl", "rb" ) as file_scaler:
  scaler = pickle.load( file_scaler )
#scaler = StandardScaler() 
#scaler = RobustScaler()

testing_filename  = sys.argv[1]

model_filename = "dnn.h5"
if len(sys.argv) > 2:
   model_filename = sys.argv[2]

dnn = load_model( model_filename )

#testing_dataset = np.loadtxt( testing_filename, delimiter=",")
testing_dataset = pd.read_csv( testing_filename, delimiter="," ).values

event_test   = testing_dataset[:,:2]
calib_test   = testing_dataset[:,2:5]
nocalib_test = testing_dataset[:,5:8]
truth_test   = testing_dataset[:,8:]

#calib_test   = scaler.transform( calib_test )
nocalib_test = scaler.transform( nocalib_test )
#truth_test   = scaler.transform( truth_test )
#E_truth_test = truth_test[:,2]
#truth_test = truth_test[:,1:]

predict_dnn = dnn.predict( nocalib_test )

#score_test = dnn.score( nocalib_test, E_truth_test )
#print
#print "INFO: Score (dnn, testing) =", score_test

# Create ROOT output file
outfilename = testing_filename.split("/")[-1].replace("output.csv","") +  model_filename.replace(".h5",".histograms.root")
print "INFO: output file:", outfilename
outfile = TFile.Open( outfilename, "RECREATE" )

h_pT_nocalib  = TH2F( "pT_nocalib_vs_truth", "non-calib vs truth-level;truth-level jet p_{T} [GeV];Reco-level non-calib jet p_{T} [GeV]", 100, 0., 500., 100, 0., 500. )
h_pT_calib    = TH2F( "pT_calib_vs_truth", "calib vs truth-level;truth-level jet p_{T} [GeV];Reco-level calib jet p_{T} [GeV]", 100, 0., 500., 100, 0., 500. )
h_pT_dnncalib  = TH2F( "pT_dnncalib_vs_truth", "DNN vs truth-level;truth-level jet p_{T} [GeV];Reco-level DNN jet p_{T} [GeV]", 100, 0., 500., 100, 0., 500. )

h_eta_nocalib  = TH2F( "eta_nocalib_vs_truth", "non-calib vs truth-level;truth-level jet #eta;Reco-level non-calib jet #eta", 50, 0., 2.5, 50, 0., 2.5 )
h_eta_calib    = TH2F( "eta_calib_vs_truth", "calib vs truth-level;truth-level jet #eta;Reco-level calib jet #eta", 50, 0., 2.5, 50, 0., 2.5 )
h_eta_dnncalib = TH2F( "eta_dnncalib_vs_truth", "DNN vs truth-level;truth-level jet #eta;Reco-level DNN jet #eta", 50, 0., 2.5, 50, 0., 2.5 )

h_E_nocalib  = TH2F( "E_nocalib_vs_truth", "non-calib vs truth-level;truth-level jet E [GeV];Reco-level non-calib jet E [GeV]", 100, 0., 500., 100, 0., 500. )
h_E_calib    = TH2F( "E_calib_vs_truth", "calib vs truth-level;truth-level jet E [GeV];Reco-level calib jet E [GeV]", 100, 0., 500., 100, 0., 500. )
h_E_dnncalib = TH2F( "E_dnncalib_vs_truth", "DNN vs truth-level;truth-level jet E [GeV];Reco-level DNN jet E [GeV]", 100, 0., 500., 100, 0., 500. )

histograms = {}
for ieta in range(len(etaregions)):
  etamin = etaregions[ieta][0]
  etamax = etaregions[ieta][1]

  histograms['pT_response_nocalib_%i'%ieta]  = TH2F( "pT_response_nocalib_%i"%ieta, \
             "p_{T} response non-calib jets (%2.1f < |#eta| < %2.1f);p_{T}^{reco};p_{T}^{reco}/p_{T}^{truth}" % (etamin,etamax), 25, 0., 500., 25, 0., 2. )
  histograms['pT_response_calib_%i'%ieta]    = TH2F( "pT_response_calib_%i"%ieta, \
             "p_{T} response calib jets (%2.1f < |#eta| < %2.1f);p_{T}^{reco};p_{T}^{reco}/p_{T}^{truth}" % (etamin,etamax), 25, 0., 500., 25, 0., 2. )
  histograms['pT_response_dnncalib_%i'%ieta] = TH2F( "pT_response_dnncalib_%i"%ieta, \
              "p_{T} response dnn-calib jets (%2.1f < |#eta| < %2.1f);p_{T}^{reco};p_{T}^{reco}/p_{T}^{truth}" % (etamin,etamax), 25, 0., 500., 25, 0., 2. )


  histograms['E_response_nocalib_%i'%ieta]  = TH2F( "E_response_nocalib_%i"%ieta, \
             "Energy response non-calib jets (%2.1f < |#eta| < %2.1f);E^{reco};E^{reco}/E^{truth}" % (etamin,etamax), 25, 0., 500., 25, 0., 2. )
  histograms['E_response_calib_%i'%ieta]    = TH2F( "E_response_calib_%i"%ieta, \
             "Energy response calib jets (%2.1f < |#eta| < %2.1f);E^{reco};E^{reco}/E^{truth}" % (etamin,etamax), 25, 0., 500., 25, 0., 2. )
  histograms['E_response_dnncalib_%i'%ieta] = TH2F( "E_response_dnncalib_%i"%ieta, \
              "Energy response dnn-calib jets (%2.1f < |#eta| < %2.1f);E^{reco};E^{reco}/E^{truth}" % (etamin,etamax), 25, 0., 500., 25, 0., 2. )



# transform back to usual representation
#calib_test   = scaler.inverse_transform( calib_test )
nocalib_test = scaler.inverse_transform( nocalib_test )
#truth_test   = scaler.inverse_transform( truth_test )
#predict_dnn  = scaler.inverse_transform( predict_dnn )

# Print out example
for i in range(10):
  print "  ", nocalib_test[i], "----> E(dnn) =", predict_dnn[i], ":: Truth =", truth_test[i], " :: Event w =", event_test[i][1]


for i in range(len(truth_test) ):
  w = event_test[i][1]

  pT_truth   = truth_test[i][0]
  pT_calib   = calib_test[i][0]
  pT_nocalib = nocalib_test[i][0]
  pT_dnncalib  = predict_dnn[i][0]

  eta_truth   = abs( truth_test[i][1] )
  eta_calib   = abs( calib_test[i][1] )
  eta_nocalib = abs( nocalib_test[i][1] )
#  eta_dnncalib  = abs( predict_dnn[i][0] )

  E_truth   = truth_test[i][2]
  E_calib   = calib_test[i][2]
  E_nocalib = nocalib_test[i][2]
  E_dnncalib     = predict_dnn[i][1]
#  E_dnncalib     = predict_dnn[i]
 
  h_pT_calib.Fill( pT_truth, pT_calib )
  h_pT_nocalib.Fill( pT_truth, pT_nocalib )
  h_pT_dnncalib.Fill( pT_truth, pT_dnncalib ) 

  h_eta_calib.Fill( eta_truth, eta_calib )
  h_eta_nocalib.Fill( eta_truth, eta_nocalib )
#  h_eta_dnncalib.Fill( eta_truth, eta_dnncalib )

  h_E_calib.Fill( E_truth, E_calib )
  h_E_nocalib.Fill( E_truth, E_nocalib )
  h_E_dnncalib.Fill( E_truth, E_dnncalib )

  pT_response_nocalib  = pT_nocalib  / pT_truth if pT_truth > 0. else -1.
  pT_response_calib    = pT_calib    / pT_truth if pT_truth > 0. else -1.
  pT_response_dnncalib = pT_dnncalib / pT_truth if pT_truth > 0. else -1.

  E_response_nocalib  = E_nocalib  / E_truth if E_truth > 0. else -1.
  E_response_calib    = E_calib    / E_truth if E_truth > 0. else -1.
  E_response_dnncalib = E_dnncalib / E_truth if E_truth > 0. else -1.

  ieta = FindEtaRegion( abs(eta_calib) )

  histograms['pT_response_nocalib_%i'%ieta].Fill( pT_nocalib, pT_response_nocalib, w )
  histograms['pT_response_calib_%i'%ieta].Fill( pT_calib, pT_response_calib, w )
  histograms['pT_response_dnncalib_%i'%ieta].Fill( pT_dnncalib, pT_response_dnncalib, w )

  histograms['E_response_nocalib_%i'%ieta].Fill( E_nocalib, E_response_nocalib, w )
  histograms['E_response_calib_%i'%ieta].Fill( E_calib, E_response_calib, w )
  histograms['E_response_dnncalib_%i'%ieta].Fill( E_dnncalib, E_response_dnncalib, w )

outfile.Write()
outfile.Close()
