#!/usr/bin/env python

import os, sys

from math import atan, exp, log, pow

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
etabins = [ [0.0,0.1], [0.1,0.2], [0.2,0.3], [0.3,0.4], [0.4,0.5], [0.5,0.6], [0.6,0.7], [0.7,0.8], [0.8,0.9], [0.9,1.0], [1.0,2.5] ]
ptbins  = [ [0., 20.], [ 20., 30.], [30., 40], [40., 60.], [60., 80.], [80., 100.], [100., 400.], [400., 1200.], [1200.,3000.] ] 

#################

def FindEtaBin( eta ):
   eta = abs(eta) 

   ieta = 0
   for bin in etabins:
     etamin = bin[0]
     etamax = bin[1]

     if eta >= etamin and eta <= etamax:
       return ieta
     ieta += 1
   print "WARNING: eta =", eta


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def FindPtBin( pt ):

   ipt = 0
   for bin in ptbins:
     ptmin = bin[0]
     ptmax = bin[1]

     if pt >= ptmin and pt <= ptmax:
       return ipt
     ipt += 1
   print "WARNING: pt =", pt

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

testing_filename  = sys.argv[1]

calibration = "pT_eta_E"
#calibration = "pT_E"
#calibration = "eta"

with open( "scaler.smallR.%s.pkl" % calibration, "rb" ) as file_scaler:
  scaler = pickle.load( file_scaler )
  poly   = pickle.load( file_scaler )
#scaler = StandardScaler() 
#scaler = RobustScaler()

model_filename = "dnn.smallR.%s.h5" % calibration
if len(sys.argv) > 3:
  model_filename = sys.argv[3]

dnn = load_model( model_filename )

print "INFO: calibration scheme:", calibration
print "INFO: model loaded from file", model_filename
print dnn.model.summary()

#testing_dataset = np.loadtxt( testing_filename, delimiter=",")
testing_dataset = pd.read_csv( testing_filename, delimiter="," ).values

# load four-vectors in (pT,eta,phi,E) representation
event_test   = testing_dataset[:,:3]
calib_test   = testing_dataset[:,3:8]
nocalib_test = testing_dataset[:,8:13]
truth_test   = testing_dataset[:,13:]

X_test = nocalib_test

print "INFO: testing calib:"
print calib_test
print "INFO: testing nocalib:"
print nocalib_test
print "INFO: testing truth:"
print truth_test
print "INFO: test X:" 
print X_test

#nocalib_test = poly.transform( nocalib_test )
#nocalib_test = scaler.transform( nocalib_test )

predict_dnn = dnn.predict( X_test )

#score_test = dnn.model.score( nocalib_test, truth_test )
#print
#print "INFO: Score (dnn, testing) =", score_test

# Create ROOT output file
outfilename = testing_filename.split("/")[-1].replace("csv","") +  model_filename.replace(".h5",".histograms.root")
print "INFO: output file:", outfilename
outfile = TFile.Open( outfilename, "RECREATE" )

h_pT_nocalib  = TH2F( "pT_nocalib_vs_truth", "non-calib vs truth-level;truth-level jet p_{T} [GeV];Reco-level non-calib jet p_{T} [GeV]", 100, 0., 1000., 100, 0., 1000. )
h_pT_calib    = TH2F( "pT_calib_vs_truth", "calib vs truth-level;truth-level jet p_{T} [GeV];Reco-level calib jet p_{T} [GeV]", 100, 0., 1000., 100, 0., 1000. )
h_pT_dnncalib  = TH2F( "pT_dnncalib_vs_truth", "DNN vs truth-level;truth-level jet p_{T} [GeV];Reco-level DNN jet p_{T} [GeV]", 100, 0., 1000., 100, 0., 1000. )

h_eta_nocalib  = TH2F( "eta_nocalib_vs_truth", "non-calib vs truth-level;truth-level jet #eta;Reco-level non-calib jet #eta", 50, 0., 2.5, 50, 0., 2.5 )
h_eta_calib    = TH2F( "eta_calib_vs_truth", "calib vs truth-level;truth-level jet #eta;Reco-level calib jet #eta", 50, 0., 2.5, 50, 0., 2.5 )
h_eta_dnncalib = TH2F( "eta_dnncalib_vs_truth", "DNN vs truth-level;truth-level jet #eta;Reco-level DNN jet #eta", 50, 0., 2.5, 50, 0., 2.5 )

h_E_nocalib  = TH2F( "E_nocalib_vs_truth", "non-calib vs truth-level;truth-level jet E [GeV];Reco-level non-calib jet E [GeV]", 100, 0., 1000., 100, 0., 1000. )
h_E_calib    = TH2F( "E_calib_vs_truth", "calib vs truth-level;truth-level jet E [GeV];Reco-level calib jet E [GeV]", 100, 0., 1000., 100, 0., 1000. )
h_E_dnncalib = TH2F( "E_dnncalib_vs_truth", "DNN vs truth-level;truth-level jet E [GeV];Reco-level DNN jet E [GeV]", 100, 0., 1000., 100, 0., 1000. )


h_pT_resolution_nocalib  = TH2F( "pT_resolution_nocalib",  "p_{T} resolution (nocalib);truth-level jet p_{T} [GeV];(p_{T}^{reco}-p_{T}^{truth}) / p_{T}^{truth}",  100, 0., 1000., 20, 0., 2. )
h_pT_resolution_calib    = TH2F( "pT_resolution_calib",    "p_{T} resolution (calib);truth-level jet p_{T} [GeV];(p_{T}^{reco}-p_{T}^{truth}) / p_{T}^{truth}",    100, 0., 1000., 20, 0., 2. )
h_pT_resolution_dnncalib = TH2F( "pT_resolution_dnncalib", "p_{T} resolution (dnncalib);truth-level jet p_{T} [GeV];(p_{T}^{reco}-p_{T}^{truth}) / p_{T}^{truth}", 100, 0., 1000., 20, 0., 2. )
 
h_pT_response_nocalib  = TH2F( "pT_response_nocalib",  "p_{T} response non-calib jets;truth-level jet p_{T} [GeV];p_{T}^{reco}/p_{T}^{truth}",20, 0., 1000., 20, 0., 2. )
h_pT_response_calib    = TH2F( "pT_response_calib",    "p_{T} response calib jets;truth-level jet p_{T} [GeV];p_{T}^{reco}/p_{T}^{truth}",    20, 0., 1000., 20, 0., 2. )
h_pT_response_dnncalib = TH2F( "pT_response_dnncalib", "p_{T} response dnn-calib jets;truth-level jet p_{T} [GeV];p_{T}^{reco}/p_{T}^{truth}",20, 0., 1000., 20, 0., 2. )


histograms = {}
for ieta in range(len(etabins)):
  etamin = etabins[ieta][0]
  etamax = etabins[ieta][1]

  histograms['pT_response_nocalib_etabin_%i'%ieta]  = TH2F( "pT_response_nocalib_etabin_%i"%ieta, \
             "p_{T} response non-calib jets (%2.1f < |#eta| < %2.1f);p_{T}^{truth};p_{T}^{reco}/p_{T}^{truth}" % (etamin,etamax),20, 0., 1000., 25, 0., 2. )
  histograms['pT_response_calib_etabin_%i'%ieta]    = TH2F( "pT_response_calib_etabin_%i"%ieta, \
             "p_{T} response calib jets (%2.1f < |#eta| < %2.1f);p_{T}^{truth};p_{T}^{reco}/p_{T}^{truth}" % (etamin,etamax),20, 0., 1000., 25, 0., 2. )
  histograms['pT_response_dnncalib_etabin_%i'%ieta] = TH2F( "pT_response_dnncalib_etabin_%i"%ieta, \
              "p_{T} response dnn-calib jets (%2.1f < |#eta| < %2.1f);p_{T}^{truth};p_{T}^{reco}/p_{T}^{truth}" % (etamin,etamax),20, 0., 1000., 25, 0., 2. )


  histograms['E_response_nocalib_etabin_%i'%ieta]  = TH2F( "E_response_nocalib_etabin_%i"%ieta, \
             "Energy response non-calib jets (%2.1f < |#eta| < %2.1f);p_{T}^{truth};E^{reco}/E^{truth}" % (etamin,etamax),20, 0., 1000., 25, 0., 2. )
  histograms['E_response_calib_etabin_%i'%ieta]    = TH2F( "E_response_calib_etabin_%i"%ieta, \
             "Energy response calib jets (%2.1f < |#eta| < %2.1f);p_{T}^{truth};E^{reco}/E^{truth}" % (etamin,etamax),20, 0., 1000., 25, 0., 2. )
  histograms['E_response_dnncalib_etabin_%i'%ieta] = TH2F( "E_response_dnncalib_etabin_%i"%ieta, \
              "Energy response dnn-calib jets (%2.1f < |#eta| < %2.1f);p_{T}^{truth};E^{reco}/E^{truth}" % (etamin,etamax),20, 0., 1000., 25, 0., 2. )


for ipt in range(len(ptbins)):
  ptmin = ptbins[ipt][0]
  ptmax = ptbins[ipt][1]

  histograms['pT_response_nocalib_ptbin_%i'%ipt] = TH2F( "pT_response_nocalib_ptbin_%i"%ipt, \
             "p_{T} response non-calib jets (%4.0f < p_{T} < %4.0f);|#eta_{det}|;p_{T}^{reco}/p_{T}^{truth}" % (ptmin,ptmax), 15, 0., 3.0, 20, 0., 2. )
  histograms['pT_response_calib_ptbin_%i'%ipt] = TH2F( "pT_response_calib_ptbin_%i"%ipt, \
             "p_{T} response calib jets (%4.0f < p_{T} < %4.0f);|#eta_{det}|;p_{T}^{reco}/p_{T}^{truth}" % (ptmin,ptmax), 15, 0., 3.0, 20, 0., 2. )
  histograms['pT_response_dnncalib_ptbin_%i'%ipt] = TH2F( "pT_response_dnncalib_ptbin_%i"%ipt, \
             "p_{T} response dnn-calib jets (%4.0f < p_{T} < %4.0f);|#eta_{det}|;p_{T}^{reco}/p_{T}^{truth}" % (ptmin,ptmax), 15, 0., 3.0, 20, 0., 2. )

  histograms['E_response_nocalib_ptbin_%i'%ipt] = TH2F( "E_response_nocalib_ptbin_%i"%ipt, \
             "E response non-calib jets (%4.0f < p_{T} < %4.0f);|#eta_{det}|;E^{reco}/E^{truth}" % (ptmin,ptmax), 15, 0., 3.0, 20, 0., 2. )
  histograms['E_response_calib_ptbin_%i'%ipt] = TH2F( "E_response_calib_ptbin_%i"%ipt, \
             "E response calib jets (%4.0f < p_{T} < %4.0f);|#eta_{det}|;E^{reco}/E^{truth}" % (ptmin,ptmax), 15, 0., 3.0, 20, 0., 2. )
  histograms['E_response_dnncalib_ptbin_%i'%ipt] = TH2F( "E_response_dnncalib_ptbin_%i"%ipt, \
             "E response dnn-calib jets (%4.0f < p_{T} < %4.0f);|#eta_{det}|;E^{reco}/E^{truth}" % (ptmin,ptmax), 15, 0., 3.0, 20, 0., 2. )



# transform back to usual representation
#nocalib_test = scaler.inverse_transform( nocalib_test )

# Print out example
for i in range(10):
  print "  ", X_test[i], "----> DNN =", predict_dnn[i], ":: Truth =", truth_test[i] 
#, " :: Event info =", event_test[i]

n_entries = len( truth_test )
print "INFO: looping over %i entries" % n_entries
for i in range( n_entries ):
  w = event_test[i][1]

  pT_truth   = truth_test[i][0]
  eta_truth  = truth_test[i][1]
  E_truth    = truth_test[i][2]
  v_truth = TLorentzVector()
  v_truth.SetPtEtaPhiE( pT_truth, eta_truth, 0., E_truth )

  pT_calib   = calib_test[i][0]
  eta_calib  = calib_test[i][1]
  E_calib    = calib_test[i][2]
  foo = calib_test[i][2]
  v_calib = TLorentzVector()
  v_calib.SetPtEtaPhiE( pT_calib, eta_calib, 0., foo )

  pT_nocalib  = nocalib_test[i][0]
  eta_nocalib = nocalib_test[i][1]
  E_nocalib   = nocalib_test[i][2]
  #print pT_nocalib, eta_nocalib, E_nocalib
  v_nocalib = TLorentzVector()
  v_nocalib.SetPtEtaPhiE( pT_nocalib, eta_nocalib, 0., E_nocalib )  

  pT_dnncalib  = -1
  eta_dnncalib = -100000
  E_dnncalib   = -1

  if calibration   == "eta":
     eta_dnncalib  = predict_dnn[i]

     theta = 2. * atan( exp( -eta_dnncalib ) )
     v_nocalib.SetTheta( theta )

     pT_dnncalib  = v_nocalib.Pt()
     eta_dnncalib = v_nocalib.Eta()
     E_dnncalib   = v_nocalib.E()

  elif calibration == "pT_eta_E":
     pT_dnncalib   = predict_dnn[i][0]
     eta_dnncalib  = predict_dnn[i][1]
     E_dnncalib    = predict_dnn[i][2]
  elif calibration == "pT_E":
     pT_dnncalib   = predict_dnn[i][0]
     eta_dnncalib  = eta_nocalib
     E_dnncalib    = predict_dnn[i][1]
  elif calibration == "pT_eta":
     pT_dnncalib   = predict_dnn[i][0]
     eta_dnncalib  = predict_dnn[i][1]
     E_dnncalib    = E_nocalib
  elif calibration == "eta_E":
     pT_dnncalib   = pT_nocalib
     eta_dnncalib  = predict_dnn[i][0]
     E_dnncalib    = predict_dnn[i][1]
  else:
     print "Unknown calibration scheme", calibration 

  h_pT_calib.Fill(    pT_truth, pT_calib, w )
  h_pT_nocalib.Fill(  pT_truth, pT_nocalib, w )
  h_pT_dnncalib.Fill( pT_truth, pT_dnncalib, w ) 

  h_eta_calib.Fill(    abs(eta_truth), abs(eta_calib), w )
  h_eta_nocalib.Fill(  abs(eta_truth), abs(eta_nocalib), w )
  h_eta_dnncalib.Fill( abs(eta_truth), abs(eta_dnncalib), w )

  h_E_calib.Fill(    E_truth, E_calib, w )
  h_E_nocalib.Fill(  E_truth, E_nocalib, w )
  h_E_dnncalib.Fill( E_truth, E_dnncalib, w )

  pT_response_nocalib  = pT_nocalib  / pT_truth if pT_truth > 0. else -1.
  pT_response_calib    = pT_calib    / pT_truth if pT_truth > 0. else -1.
  pT_response_dnncalib = pT_dnncalib / pT_truth if pT_truth > 0. else -1.

  E_response_nocalib  = E_nocalib  / E_truth if E_truth > 0. else -1.
  E_response_calib    = E_calib    / E_truth if E_truth > 0. else -1.
  E_response_dnncalib = E_dnncalib / E_truth if E_truth > 0. else -1.

  h_pT_response_nocalib.Fill(  pT_nocalib,  pT_response_nocalib, w )
  h_pT_response_calib.Fill(    pT_calib,    pT_response_calib, w )
  h_pT_response_dnncalib.Fill( pT_dnncalib, pT_response_dnncalib, w )

  # fill the same, but divided into eta bins
  ieta = FindEtaBin( abs(eta_nocalib) )

  histograms['pT_response_nocalib_etabin_%i'%ieta].Fill(  pT_truth, pT_response_nocalib, w )
  histograms['pT_response_calib_etabin_%i'%ieta].Fill(    pT_truth, pT_response_calib, w )
  histograms['pT_response_dnncalib_etabin_%i'%ieta].Fill( pT_truth, pT_response_dnncalib, w )

  histograms['E_response_nocalib_etabin_%i'%ieta].Fill(  pT_truth, E_response_nocalib, w )
  histograms['E_response_calib_etabin_%i'%ieta].Fill(    pT_truth, E_response_calib, w )
  histograms['E_response_dnncalib_etabin_%i'%ieta].Fill( pT_truth, E_response_dnncalib, w )

  # fill the same, but divided into pT bins
  ipt = FindPtBin( pT_truth )
  histograms['pT_response_nocalib_ptbin_%i'%ipt].Fill(  abs(eta_truth), pT_response_nocalib, w )
  histograms['pT_response_calib_ptbin_%i'%ipt].Fill(    abs(eta_truth), pT_response_calib, w )
  histograms['pT_response_dnncalib_ptbin_%i'%ipt].Fill( abs(eta_truth), pT_response_dnncalib, w )

  histograms['E_response_nocalib_ptbin_%i'%ipt].Fill(  abs(eta_truth), E_response_nocalib, w )
  histograms['E_response_calib_ptbin_%i'%ipt].Fill(    abs(eta_truth), E_response_calib, w )
  histograms['E_response_dnncalib_ptbin_%i'%ipt].Fill( abs(eta_truth), E_response_dnncalib, w )

  # Resolution
  pT_resolution_nocalib  = ( pT_nocalib  - pT_truth ) / pT_truth if pT_truth > 0. else -1.
  pT_resolution_calib    = ( pT_calib    - pT_truth ) / pT_truth if pT_truth > 0. else -1.
  pT_resolution_dnncalib = ( pT_dnncalib - pT_truth ) / pT_truth if pT_truth > 0. else -1.

  h_pT_resolution_nocalib.Fill(  pT_truth,  pT_resolution_nocalib, w )
  h_pT_resolution_calib.Fill(    pT_truth,  pT_resolution_calib, w )
  h_pT_resolution_dnncalib.Fill( pT_truth,  pT_resolution_dnncalib, w )

print "INFO: saved output file", outfilename
outfile.Write()
outfile.Close()
