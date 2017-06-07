#!/usr/bin/env python

import os, sys

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasRegressor

from keras.optimizers import SGD

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import cross_val_score

import cPickle as pickle
import numpy as np
import pandas as pd

from ROOT import *

from models import *

np.set_printoptions( precision=2, suppress=True )


# change this to increase the number of eta slices
etabins  = [ [0.0, 0.1], [0.1,0.2], [0.2,0.3], [0.3,0.4], [0.4,0.5], [0.5,0.6], [0.6,0.7], [0.7,0.8], [0.8,0.9], [0.9,1.0], [1.0,1.5],[1.5,2.0] ]
ptbins   = [ [250., 400.], [400., 500.], [500.,600.], [600.,800.], [800.,1200.] ]
Ebins    = [ [0., 200.], [200.,400.], [400.,600.], [600.,800.], [800.,1000.],[1000.,1500.] ]
massbins = [ [30., 60.], [ 60., 110.], [110.,140], [140,200], [200,300] ] 
#                LOW          W/Z            H          t       QCD          

#################

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
#   print "WARNING: eta =", eta
   return -1


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def FindPtBin( pt ):

   i_pT = 0
   for bin in ptbins:
     ptmin = bin[0]
     ptmax = bin[1]

     if pt >= ptmin and pt <= ptmax:
       return i_pT
     i_pT += 1
#   print "WARNING: pt =", pt
   return - 1

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def FindMassBin( m ):

   i_M = 0
   for bin in massbins:
     Mmin = bin[0]
     Mmax = bin[1]

     if m >= Mmin and m <= Mmax:
       return i_M
     i_M += 1
#   print "WARNING: pt =", pt
   return -1

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

calibration = "pT_eta_E_M"

filename_scaler = "scaler.largeR_substructure.pkl"
with open( filename_scaler, "rb" ) as file_scaler:
  X_scaler = pickle.load( file_scaler )
  y_scaler = pickle.load( file_scaler )

testing_filename  = sys.argv[1]

model_filename = "model.calib4.h5"
dnn = load_model( model_filename )

from features import *

df_testing = pd.read_csv( testing_filename, delimiter=',', names=header )
#print df_testing

event_info = df_testing[ [ "mc_Weight" ] ].values

X_test_all = df_testing[features_all].values
X_test_all = X_scaler.transform( X_test_all )

# Create autoencoder
n_input_all = len( features_all )
encoder = load_model( "encoder.h5" )
encoding_dim = encoder.layers[-1].output_shape[1]
print "INFO: loaded encoder %i -> %i" % ( n_input_all, encoding_dim )

# these are the compressed data
X_test_all_encoded = encoder.predict(X_test_all)
print "INFO: example of compressed data:"
print X_test_all_encoded

# these should be standard
y_nocalib = df_testing[y_features_nocalib].values
y_truth   = df_testing[y_features_truth].values
y_calib   = df_testing[y_features_calib].values

# run the calibration
y_dnncalib = dnn.predict( X_test_all_encoded )
y_dnncalib = y_scaler.inverse_transform( y_dnncalib )

# Create ROOT output file
outfilename = testing_filename.split("/")[-1].replace("csv","") +  model_filename.replace(".h5",".histograms.root")
print "INFO: output file:", outfilename
outfile = TFile.Open( outfilename, "RECREATE" )

#tree = TNtuple( "nominal", "nominal", "

E_min  = Ebins[0][0]
E_max  = Ebins[-1][1]
pT_min = ptbins[0][0]
pT_max = ptbins[-1][1]
M_min  = massbins[0][0]
M_max  = massbins[-1][1]
eta_min = etabins[0][0]
eta_max = etabins[-1][1]

h_pT_nocalib  = TH2F( "pT_nocalib_vs_truth", "non-calib vs truth-level;truth-level jet p_{T} [GeV];Reco-level non-calib jet p_{T} [GeV]", 100, pT_min, pT_max, 100, pT_min, pT_max )
h_pT_calib    = TH2F( "pT_calib_vs_truth", "calib vs truth-level;truth-level jet p_{T} [GeV];Reco-level calib jet p_{T} [GeV]", 100, pT_min, pT_max, 100, pT_min, pT_max )
h_pT_dnncalib  = TH2F( "pT_dnncalib_vs_truth", "DNN vs truth-level;truth-level jet p_{T} [GeV];Reco-level DNN jet p_{T} [GeV]", 100, pT_min, pT_max, 100, pT_min, pT_max )

h_eta_nocalib  = TH2F( "eta_nocalib_vs_truth", "non-calib vs truth-level;truth-level jet #eta;Reco-level non-calib jet #eta", 40, eta_min, eta_max, 40, 0., eta_max )
h_eta_calib    = TH2F( "eta_calib_vs_truth", "calib vs truth-level;truth-level jet #eta;Reco-level calib jet #eta",           40, eta_min, eta_max, 40, 0., eta_max )
h_eta_dnncalib = TH2F( "eta_dnncalib_vs_truth", "DNN vs truth-level;truth-level jet #eta;Reco-level DNN jet #eta",            40, eta_min, eta_max, 40, 0., eta_max )

h_E_nocalib  = TH2F( "E_nocalib_vs_truth", "non-calib vs truth-level;truth-level jet E [GeV];Reco-level non-calib jet E [GeV]", 150, E_min, E_max, 150, E_min, E_max )
h_E_calib    = TH2F( "E_calib_vs_truth", "calib vs truth-level;truth-level jet E [GeV];Reco-level calib jet E [GeV]",           150, E_min, E_max, 150, E_min, E_max )
h_E_dnncalib = TH2F( "E_dnncalib_vs_truth", "DNN vs truth-level;truth-level jet E [GeV];Reco-level DNN jet E [GeV]",            150, E_min, E_max, 150, E_min, E_max )

h_M_nocalib  = TH2F( "M_nocalib_vs_truth", "non-calib vs truth-level;truth-level jet M [GeV];Reco-level non-calib jet M [GeV]", 100, M_min, M_max, 100, M_min, M_max )
h_M_calib    = TH2F( "M_calib_vs_truth", "calib vs truth-level;truth-level jet M [GeV];Reco-level calib jet M [GeV]",           100, M_min, M_max, 100, M_min, M_max )
h_M_dnncalib = TH2F( "M_dnncalib_vs_truth", "DNN vs truth-level;truth-level jet M [GeV];Reco-level DNN jet M [GeV]",            100, M_min, M_max, 100, M_min, M_max )


# Response = reco / truth
h_pT_response_nocalib  = TH2F( "pT_response_nocalib",  "p_{T} response non-calib jets;truth-level jet p_{T} [GeV];p_{T}^{reco}/p_{T}^{truth}",20, pT_min, pT_max, 20, 0., 2. )
h_pT_response_calib    = TH2F( "pT_response_calib",    "p_{T} response calib jets;truth-level jet p_{T} [GeV];p_{T}^{reco}/p_{T}^{truth}",    20, pT_min, pT_max, 20, 0., 2. )
h_pT_response_dnncalib = TH2F( "pT_response_dnncalib", "p_{T} response dnn-calib jets;truth-level jet p_{T} [GeV];p_{T}^{reco}/p_{T}^{truth}",20, pT_min, pT_max, 20, 0., 2. )

h_E_response_nocalib  = TH2F( "E_response_nocalib",  "E response non-calib jets;truth-level jet E [GeV];E^{reco}/E^{truth}",30, E_min, E_max, 20, 0., 2. )
h_E_response_calib    = TH2F( "E_response_calib",    "E response calib jets;truth-level jet E [GeV];E^{reco}/E^{truth}",    30, E_min, E_max, 20, 0., 2. )
h_E_response_dnncalib = TH2F( "E_response_dnncalib", "E response dnn-calib jets;truth-level jet E [GeV];E^{reco}/E^{truth}",30, E_min, E_max, 20, 0., 2. )

h_M_response_calib    = TH2F( "M_response_calib",    "M response calib jets;truth-level jet M [GeV];M^{reco}/M^{truth}",    30, M_min, M_max, 20, 0., 2. )
h_M_response_nocalib  = TH2F( "M_response_nocalib",  "M response non-calib jets;truth-level jet M [GeV];M^{reco}/M^{truth}",30, M_min, M_max, 20, 0., 2. )
h_M_response_dnncalib = TH2F( "M_response_dnncalib", "M response dnn-calib jets;truth-level jet M [GeV];M^{reco}/M^{truth}",30, M_min, M_max, 20, 0., 2. )

h_eta_response_calib    = TH2F( "eta_response_calib",    "#eta response calib jets;truth-level jet #eta;#eta^{reco}/#eta^{truth}",     20, -eta_max, eta_max, 20, 0., 2. )
h_eta_response_nocalib  = TH2F( "eta_response_nocalib",  "#eta response non-calib jets;truth-level jet #eta;#eta^{reco}/#eta^{truth}", 20, -eta_max, eta_max, 20, 0., 2. )
h_eta_response_dnncalib = TH2F( "eta_response_dnncalib", "#eta response dnn-calib jets;truth-level jet #eta;#eta^{reco}/#eta^{truth}", 20, -eta_max, eta_max, 20, 0., 2. )


histograms = {}

print "INFO: response in %i eta bins" % len(etabins)
for ieta in range(len(etabins)):
  etamin = etabins[ieta][0]
  etamax = etabins[ieta][1]

  histograms['pT_response_nocalib_etabin_%i'%ieta]  = TH2F( "pT_response_nocalib_etabin_%i"%ieta, \
             "p_{T} response non-calib jets (%2.1f < |#eta| < %2.1f);p_{T}^{truth};p_{T}^{reco}/p_{T}^{truth}" % (etamin,etamax),20, pT_min, pT_max, 25, 0., 2. )
  histograms['pT_response_calib_etabin_%i'%ieta]    = TH2F( "pT_response_calib_etabin_%i"%ieta, \
             "p_{T} response calib jets (%2.1f < |#eta| < %2.1f);p_{T}^{truth};p_{T}^{reco}/p_{T}^{truth}" % (etamin,etamax),20, pT_min, pT_max, 25, 0., 2. )
  histograms['pT_response_dnncalib_etabin_%i'%ieta] = TH2F( "pT_response_dnncalib_etabin_%i"%ieta, \
              "p_{T} response dnn-calib jets (%2.1f < |#eta| < %2.1f);p_{T}^{truth};p_{T}^{reco}/p_{T}^{truth}" % (etamin,etamax),20, pT_min, pT_max, 25, 0., 2. )


  histograms['E_response_nocalib_etabin_%i'%ieta]  = TH2F( "E_response_nocalib_etabin_%i"%ieta, \
             "Energy response non-calib jets (%2.1f < |#eta| < %2.1f);p_{T}^{truth};E^{reco}/E^{truth}" % (etamin,etamax),20, pT_min, pT_max, 25, 0., 2. )
  histograms['E_response_calib_etabin_%i'%ieta]    = TH2F( "E_response_calib_etabin_%i"%ieta, \
             "Energy response calib jets (%2.1f < |#eta| < %2.1f);p_{T}^{truth};E^{reco}/E^{truth}" % (etamin,etamax),20, pT_min, pT_max, 25, 0., 2. )
  histograms['E_response_dnncalib_etabin_%i'%ieta] = TH2F( "E_response_dnncalib_etabin_%i"%ieta, \
              "Energy response dnn-calib jets (%2.1f < |#eta| < %2.1f);p_{T}^{truth};E^{reco}/E^{truth}" % (etamin,etamax),20, pT_min, pT_max, 25, 0., 2. )


  histograms['M_response_nocalib_etabin_%i'%ieta]  = TH2F( "M_response_nocalib_etabin_%i"%ieta, \
             "Mass response non-calib jets (%2.1f < |#eta| < %2.1f);p_{T}^{truth};M^{reco}/M^{truth}" % (etamin,etamax), 20, pT_min, pT_max, 25, 0., 2. )
  histograms['M_response_calib_etabin_%i'%ieta]    = TH2F( "M_response_calib_etabin_%i"%ieta, \
             "Mass response calib jets (%2.1f < |#eta| < %2.1f);p_{T}^{truth};M^{reco}/M^{truth}" % (etamin,etamax), 20, pT_min, pT_max, 25, 0., 2 )
  histograms['M_response_dnncalib_etabin_%i'%ieta] = TH2F( "M_response_dnncalib_etabin_%i"%ieta, \
              "Mass response dnn-calib jets (%2.1f < |#eta| < %2.1f);p_{T}^{truth};M^{reco}/M^{truth}" % (etamin,etamax), 20, pT_min, pT_max, 25, 0., 2 )

print "INFO: response in %i pT bins" % len(ptbins)
for i_pT in range(len(ptbins)):
  ptmin = ptbins[i_pT][0]
  ptmax = ptbins[i_pT][1]

  histograms['pT_response_nocalib_ptbin_%i'%i_pT] = TH2F( "pT_response_nocalib_ptbin_%i"%i_pT, \
             "p_{T} response non-calib jets (%4.0f < p_{T} < %4.0f);|#eta_{det}|;p_{T}^{reco}/p_{T}^{truth}" % (ptmin,ptmax), 20, eta_min, eta_max, 20, 0., 2. )
  histograms['pT_response_calib_ptbin_%i'%i_pT] = TH2F( "pT_response_calib_ptbin_%i"%i_pT, \
             "p_{T} response calib jets (%4.0f < p_{T} < %4.0f);|#eta_{det}|;p_{T}^{reco}/p_{T}^{truth}" % (ptmin,ptmax), 20, eta_min, eta_max, 20, 0., 2. )
  histograms['pT_response_dnncalib_ptbin_%i'%i_pT] = TH2F( "pT_response_dnncalib_ptbin_%i"%i_pT, \
             "p_{T} response dnn-calib jets (%4.0f < p_{T} < %4.0f);|#eta_{det}|;p_{T}^{reco}/p_{T}^{truth}" % (ptmin,ptmax), 20, eta_min, eta_max, 20, 0., 2. )

  histograms['E_response_nocalib_ptbin_%i'%i_pT] = TH2F( "E_response_nocalib_ptbin_%i"%i_pT, \
             "E response non-calib jets (%4.0f < p_{T} < %4.0f);|#eta_{det}|;E^{reco}/E^{truth}" % (ptmin,ptmax), 20, eta_min, eta_max, 20, 0., 2. )
  histograms['E_response_calib_ptbin_%i'%i_pT] = TH2F( "E_response_calib_ptbin_%i"%i_pT, \
             "E response calib jets (%4.0f < p_{T} < %4.0f);|#eta_{det}|;E^{reco}/E^{truth}" % (ptmin,ptmax), 20, eta_min, eta_max, 20, 0., 2. )
  histograms['E_response_dnncalib_ptbin_%i'%i_pT] = TH2F( "E_response_dnncalib_ptbin_%i"%i_pT, \
             "E response dnn-calib jets (%4.0f < p_{T} < %4.0f);|#eta_{det}|;E^{reco}/E^{truth}" % (ptmin,ptmax), 20, eta_min, eta_max, 20, 0., 2. )


  histograms['M_response_nocalib_ptbin_%i'%i_pT] = TH2F( "M_response_nocalib_ptbin_%i"%i_pT, \
             "M response non-calib jets (%4.0f < p_{T} < %4.0f);|#eta_{det}|;M^{reco}/M^{truth}" % (ptmin,ptmax), 10, 0., 2., 25, 0., 2 )
  histograms['M_response_calib_ptbin_%i'%i_pT] = TH2F( "M_response_calib_ptbin_%i"%i_pT, \
             "M response calib jets (%4.0f < p_{T} < %4.0f);|#eta_{det}|;M^{reco}/M^{truth}" % (ptmin,ptmax), 10, 0., 2., 25, 0., 2)
  histograms['M_response_dnncalib_ptbin_%i'%i_pT] = TH2F( "M_response_dnncalib_ptbin_%i"%i_pT, \
             "M response dnn-calib jets (%4.0f < p_{T} < %4.0f);|#eta_{det}|;M^{reco}/M^{truth}" % (ptmin,ptmax), 10, 0., 2., 25, 0., 2)

print "INFO: response in %i mass bins" % len(massbins)
for i_M in range(len(massbins)):
  Mmin = massbins[i_M][0]
  Mmax = massbins[i_M][1]

  histograms['pT_response_nocalib_massbin_%i'%i_M]  = TH2F( "pT_response_nocalib_massbin_%i"%i_M, \
             "p_{T} response non-calib jets (%4.0f < M < %4.0f);p_{T}^{truth};p_{T}^{reco}/p_{T}^{truth}"  % (Mmin,Mmax),20, pT_min, pT_max, 25, 0., 2. )
  histograms['pT_response_calib_massbin_%i'%i_M]    = TH2F( "pT_response_calib_massbin_%i"%i_M, \
             "p_{T} response calib jets (%4.0f < M < %4.0f);p_{T}^{truth};p_{T}^{reco}/p_{T}^{truth}"      % (Mmin,Mmax),20, pT_min, pT_max, 25, 0., 2. )
  histograms['pT_response_dnncalib_massbin_%i'%i_M] = TH2F( "pT_response_dnncalib_massbin_%i"%i_M, \
              "p_{T} response dnn-calib jets (%4.0f < M < %4.0f);p_{T}^{truth};p_{T}^{reco}/p_{T}^{truth}" % (Mmin,Mmax),20, pT_min, pT_max, 25, 0., 2. )


  histograms['E_response_nocalib_massbin_%i'%i_M]  = TH2F( "E_response_nocalib_massbin_%i"%i_M, \
             "Energy response non-calib jets (%4.0f < M < %4.0f);p_{T}^{truth};E^{reco}/E^{truth}"  % (Mmin,Mmax),20, pT_min, pT_max, 25, 0., 2. )
  histograms['E_response_calib_massbin_%i'%i_M]    = TH2F( "E_response_calib_massbin_%i"%i_M, \
             "Energy response calib jets (%4.0f < M < %4.0f);p_{T}^{truth};E^{reco}/E^{truth}"      % (Mmin,Mmax),20, pT_min, pT_max, 25, 0., 2. )
  histograms['E_response_dnncalib_massbin_%i'%i_M] = TH2F( "E_response_dnncalib_massbin_%i"%i_M, \
              "Energy response dnn-calib jets (%4.0f < M < %4.0f);p_{T}^{truth};E^{reco}/E^{truth}" % (Mmin,Mmax),20, pT_min, pT_max, 25, 0., 2. )


  histograms['M_response_nocalib_massbin_%i'%i_M]  = TH2F( "M_response_nocalib_massbin_%i"%i_M, \
             "Mass response non-calib jets (%4.0f < M < %4.0f);p_{T}^{truth};M^{reco}/M^{truth}"    % (Mmin,Mmax), 20, pT_min, pT_max, 25, 0., 2. )
  histograms['M_response_calib_massbin_%i'%i_M]    = TH2F( "M_response_calib_massbin_%i"%i_M, \
             "Mass response calib jets (%4.0f < M < %4.0f);p_{T}^{truth};M^{reco}/M^{truth}"        % (Mmin,Mmax), 20, pT_min, pT_max, 25, 0., 2 )
  histograms['M_response_dnncalib_massbin_%i'%i_M] = TH2F( "M_response_dnncalib_massbin_%i"%i_M, \
              "Mass response dnn-calib jets (%4.0f < M < %4.0f);p_{T}^{truth};M^{reco}/M^{truth}"   % (Mmin,Mmax), 20, pT_min, pT_max, 25, 0., 2 )


# transform back to usual representation
#X_test = scaler.inverse_transform( X_test )

# Print out example
for i in range(10):
  print "  ", y_nocalib[i], "----> DNN =", y_dnncalib[i], ":: Truth =", y_truth[i]

k_pt = 0
k_eta = 1
k_E  = 2
k_M  = 3

#, " :: Event info =", event_test[i]
n_entries = len( y_truth )
print "INFO: looping over %i entries" % n_entries
for i in range( n_entries ):
  w = event_info[i][0]

  if ( n_entries < 10 ) or ( (i+1) % int(float(n_entries)/10.)  == 0 ):
    perc = 100. * i / float(n_entries)
    print "INFO: Event %-9i  (%3.0f %%)" % ( i, perc )

  pT_truth   = y_truth[i][k_pt]
  eta_truth  = y_truth[i][k_eta]
  E_truth    = y_truth[i][k_E]
  M_truth    = y_truth[i][k_M]
#  v_truth = TLorentzVector()
#  v_truth.SetPtEtaPhiM( pT_truth, eta_truth, 0., M_truth )
#  E_truth = v_truth.E()

  pT_calib   = y_calib[i][k_pt]
  eta_calib  = y_calib[i][k_eta]
  E_calib    = y_calib[i][k_E]
  M_calib    = y_calib[i][k_M]
#  v_calib = TLorentzVector()
#  v_calib.SetPtEtaPhiM( pT_calib, eta_calib, 0., M_calib )
#  E_calib = v_calib.E()

  pT_nocalib  = y_nocalib[i][k_pt]
  eta_nocalib = y_nocalib[i][k_eta]
  E_nocalib   = y_nocalib[i][k_E]
  M_nocalib   = y_nocalib[i][k_M]
#  v_nocalib = TLorentzVector()
#  v_nocalib.SetPtEtaPhiM( pT_nocalib, eta_nocalib, 0., M_nocalib )  
#  E_nocalib = v_nocalib.E()

  pT_dnncalib  = y_dnncalib[i][k_pt]
  eta_dnncalib = y_dnncalib[i][k_eta]
  E_dnncalib   = y_dnncalib[i][k_E]
  M_dnncalib   = y_dnncalib[i][k_M]
#  v_dnncalib = TLorentzVector() 
#  v_dnncalib.SetPtEtaPhiM( pT_dnncalib, eta_dnncalib, 0., M_dnncalib )
#  E_dnncalib = v_dnncalib.E()

  # Now fill histograms

  h_pT_calib.Fill(    pT_truth, pT_calib, w )
  h_pT_nocalib.Fill(  pT_truth, pT_nocalib, w )
  h_pT_dnncalib.Fill( pT_truth, pT_dnncalib, w ) 

  h_eta_calib.Fill(    abs(eta_truth), abs(eta_calib), w )
  h_eta_nocalib.Fill(  abs(eta_truth), abs(eta_nocalib), w )
  h_eta_dnncalib.Fill( abs(eta_truth), abs(eta_dnncalib), w )

  h_E_calib.Fill(    E_truth, E_calib, w )
  h_E_nocalib.Fill(  E_truth, E_nocalib, w )
  h_E_dnncalib.Fill( E_truth, E_dnncalib, w )

  h_M_calib.Fill(    M_truth, M_calib, w )
  h_M_nocalib.Fill(  M_truth, M_nocalib, w )
  h_M_dnncalib.Fill( M_truth, M_dnncalib, w )

  pT_response_nocalib  = pT_nocalib  / pT_truth if pT_truth > 0. else -1.
  pT_response_calib    = pT_calib    / pT_truth if pT_truth > 0. else -1.
  pT_response_dnncalib = pT_dnncalib / pT_truth if pT_truth > 0. else -1.

  E_response_nocalib  = E_nocalib  / E_truth if E_truth > 0. else -1.
  E_response_calib    = E_calib    / E_truth if E_truth > 0. else -1.
  E_response_dnncalib = E_dnncalib / E_truth if E_truth > 0. else -1.

  eta_response_nocalib  = eta_nocalib/ eta_truth if not eta_truth == 0. else -1000.
  eta_response_calib    = eta_calib/ eta_truth if not eta_truth == 0. else -1000.
  eta_response_dnncalib = eta_dnncalib/ eta_truth if not eta_truth == 0. else -1000.

  M_response_nocalib  = M_nocalib  / M_truth if M_truth > 0. else -1.
  M_response_calib    = M_calib    / M_truth if M_truth > 0. else -1.
  M_response_dnncalib = M_dnncalib / M_truth if M_truth > 0. else -1.

  h_pT_response_nocalib.Fill(  pT_nocalib,  pT_response_nocalib, w )
  h_pT_response_calib.Fill(    pT_calib,    pT_response_calib, w )
  h_pT_response_dnncalib.Fill( pT_dnncalib, pT_response_dnncalib, w )

  h_E_response_nocalib.Fill(  E_nocalib,  E_response_nocalib, w )
  h_E_response_calib.Fill(    E_calib,    E_response_calib, w )
  h_E_response_dnncalib.Fill( E_dnncalib, E_response_dnncalib, w )

  h_eta_response_nocalib.Fill(  eta_nocalib,  eta_response_nocalib, w )
  h_eta_response_calib.Fill(    eta_calib,    eta_response_calib, w )
  h_eta_response_dnncalib.Fill( eta_dnncalib, eta_response_dnncalib, w )

  h_M_response_nocalib.Fill(  M_nocalib,  M_response_nocalib, w )
  h_M_response_calib.Fill(    M_calib,    M_response_calib, w )
  h_M_response_dnncalib.Fill( M_dnncalib, M_response_dnncalib, w )


  # fill the same, but divided into eta bins
  ieta = FindEtaBin( abs(eta_nocalib) )
  if ieta >= 0:
     histograms['pT_response_nocalib_etabin_%i'%ieta].Fill(  pT_truth, pT_response_nocalib, w )
     histograms['pT_response_calib_etabin_%i'%ieta].Fill(    pT_truth, pT_response_calib, w )
     histograms['pT_response_dnncalib_etabin_%i'%ieta].Fill( pT_truth, pT_response_dnncalib, w )

     histograms['E_response_nocalib_etabin_%i'%ieta].Fill(  pT_truth, E_response_nocalib, w )
     histograms['E_response_calib_etabin_%i'%ieta].Fill(    pT_truth, E_response_calib, w )
     histograms['E_response_dnncalib_etabin_%i'%ieta].Fill( pT_truth, E_response_dnncalib, w )

     histograms['M_response_nocalib_etabin_%i'%ieta].Fill(  pT_truth, M_response_nocalib, w )
     histograms['M_response_calib_etabin_%i'%ieta].Fill(    pT_truth, M_response_calib, w )
     histograms['M_response_dnncalib_etabin_%i'%ieta].Fill( pT_truth, M_response_dnncalib, w )

  # fill the same, but divided into pT bins
  i_pT = FindPtBin( pT_truth )
  if i_pT >= 0:
     histograms['pT_response_nocalib_ptbin_%i'%i_pT].Fill(  abs(eta_truth), pT_response_nocalib, w )
     histograms['pT_response_calib_ptbin_%i'%i_pT].Fill(    abs(eta_truth), pT_response_calib, w )
     histograms['pT_response_dnncalib_ptbin_%i'%i_pT].Fill( abs(eta_truth), pT_response_dnncalib, w )

     histograms['E_response_nocalib_ptbin_%i'%i_pT].Fill(  abs(eta_truth), E_response_nocalib, w )
     histograms['E_response_calib_ptbin_%i'%i_pT].Fill(    abs(eta_truth), E_response_calib, w )
     histograms['E_response_dnncalib_ptbin_%i'%i_pT].Fill( abs(eta_truth), E_response_dnncalib, w )

     histograms['M_response_nocalib_ptbin_%i'%i_pT].Fill(  abs(eta_truth), M_response_nocalib, w )
     histograms['M_response_calib_ptbin_%i'%i_pT].Fill(    abs(eta_truth), M_response_calib, w )
     histograms['M_response_dnncalib_ptbin_%i'%i_pT].Fill( abs(eta_truth), M_response_dnncalib, w )

  # fill the same, but divided into M bins
  i_M = FindMassBin( M_truth )
  if i_M >= 0:
     histograms['pT_response_nocalib_massbin_%i'%i_M].Fill(  pT_truth, pT_response_nocalib, w )
     histograms['pT_response_calib_massbin_%i'%i_M].Fill(    pT_truth, pT_response_calib, w )
     histograms['pT_response_dnncalib_massbin_%i'%i_M].Fill( pT_truth, pT_response_dnncalib, w )

     histograms['E_response_nocalib_massbin_%i'%i_M].Fill(  pT_truth, E_response_nocalib, w )
     histograms['E_response_calib_massbin_%i'%i_M].Fill(    pT_truth, E_response_calib, w )
     histograms['E_response_dnncalib_massbin_%i'%i_M].Fill( pT_truth, E_response_dnncalib, w )

     histograms['M_response_nocalib_massbin_%i'%i_M].Fill(  pT_truth, M_response_nocalib, w )
     histograms['M_response_calib_massbin_%i'%i_M].Fill(    pT_truth, M_response_calib, w )
     histograms['M_response_dnncalib_massbin_%i'%i_M].Fill( pT_truth, M_response_dnncalib, w )

  # Resolution
#  pT_resolution_nocalib  = ( pT_nocalib  - pT_truth ) / pT_truth if pT_truth > 0. else -1.
#  pT_resolution_calib    = ( pT_calib    - pT_truth ) / pT_truth if pT_truth > 0. else -1.
#  pT_resolution_dnncalib = ( pT_dnncalib - pT_truth ) / pT_truth if pT_truth > 0. else -1.

#  h_pT_resolution_nocalib.Fill(  pT_truth,  pT_resolution_nocalib, w )
#  h_pT_resolution_calib.Fill(    pT_truth,  pT_resolution_calib, w )
#  h_pT_resolution_dnncalib.Fill( pT_truth,  pT_resolution_dnncalib, w )

print "INFO: saved output file", outfilename
outfile.Write()
outfile.Close()

