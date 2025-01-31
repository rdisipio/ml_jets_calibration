#!/usr/bin/env python

import os, sys
import csv
from ROOT import *

GeV = 1000.
R = 40

infilename = sys.argv[1]
infile = TFile.Open( infilename )


dts    = infilename.split("/")[-2]
usr    = dts.split('.')[1]
dsid   = dts.split('.')[2]
sname  = dts.split('.')[3]
deriv  = dts.split('.')[4]

outfilename = "csv/" + infilename.split("/")[-1].replace( "output.root", "%s.%s.%s.csv" % (dsid,sname,deriv)  )
outfile = open( outfilename, "wt" )
csvwriter = csv.writer( outfile )

tree = infile.Get( "nominal" )

nentries = tree.GetEntries()
#print "INFO: found %i entries" % nentries

for ientry in range(nentries):
  tree.GetEntry( ientry )

  eventNumber = tree.eventNumber
  weight   = tree.weight_mc * tree.weight_pileup #* tree.weight_jvt
  mu       = tree.mu

  jets_reco_calib   = []
  jets_reco_nocalib = []
  jets_truth        = []

  jets_all_n = len( tree.htt_caSmallR_reco_pt )

  for i in range(jets_all_n):
     thisR = tree.htt_caSmallR_R[i]
     if not R == thisR: continue

     dR = tree.htt_caSmallR_dR[i]
     if dR > 0.1: continue

     # Calibrated
     pt  = tree.htt_caSmallR_reco_pt[i] 
     eta = tree.htt_caSmallR_reco_eta[i]
     phi = 0.# tree.htt_caSmallR_reco_phi[i]
     E   = tree.htt_caSmallR_reco_e[i] 

     if pt < 20*GeV: continue
     if abs(eta) > 2.5: continue

     jets_reco_calib += [ TLorentzVector() ]
     jet = jets_reco_calib[-1]

     jet.SetPtEtaPhiE( pt, eta, phi, E )

     # Uncalibrated
     pt  = tree.htt_caSmallR_reco_nocalib_pt[i]
     eta = tree.htt_caSmallR_reco_nocalib_eta[i]
     phi = 0. #tree.htt_caSmallR_reco_nocalib_phi[i]
     E   = tree.htt_caSmallR_reco_nocalib_e[i]

     jets_reco_nocalib += [ TLorentzVector() ]
     jet = jets_reco_nocalib[-1]

     jet.SetPtEtaPhiE( pt, eta, phi, E )

     # Truth
     pt  = tree.htt_caSmallR_truth_pt[i]
     eta = tree.htt_caSmallR_truth_eta[i]
     phi = 0. #tree.htt_caSmallR_truth_phi[i]
     E   = tree.htt_caSmallR_truth_e[i]

     jets_truth += [ TLorentzVector() ]
     jet = jets_truth[-1]

     jet.SetPtEtaPhiE( pt, eta, phi, E )

  # Dump only good jets
  jets_n = len( jets_reco_calib )
  for i in range( jets_n ):
    j_calib   = jets_reco_calib[i]

    j_calib_pt  = j_calib.Pt()/GeV
    j_calib_eta = j_calib.Eta()
    j_calib_E   = j_calib.E()/GeV
    j_calib_M   = j_calib.M()/GeV
    j_calib_P   = j_calib.P()/GeV

    j_calib_P2  = j_calib_P*j_calib_P
    j_calib_E2  = j_calib_E*j_calib_E
    j_calib_M2  = j_calib_M*j_calib_M

    j_nocalib = jets_reco_nocalib[i]

    j_nocalib_pt  = j_nocalib.Pt()/GeV
    j_nocalib_eta = j_nocalib.Eta()
    j_nocalib_E   = j_nocalib.E()/GeV
    j_nocalib_M   = j_nocalib.M()/GeV
    j_nocalib_P   = j_nocalib.P()/GeV

    j_nocalib_P2  = j_nocalib_P*j_nocalib_P
    j_nocalib_E2  = j_nocalib_E*j_nocalib_E
    j_nocalib_M2  = j_nocalib_M*j_nocalib_M

    j_truth        = jets_truth[i]

    j_truth_pt  = j_truth.Pt()/GeV
    j_truth_eta = j_truth.Eta()
    j_truth_E   = j_truth.E()/GeV
    j_truth_M   = j_truth.M()/GeV
    j_truth_P   = j_truth.P()/GeV

    j_truth_P2  = j_truth_P*j_truth_P
    j_truth_E2  = j_truth_E*j_truth_E
    j_truth_M2  = j_truth_M*j_truth_M

    
## Write (pT, eta, E, M )
    csvwriter.writerow( (
           "%i"    % eventNumber, \
           "%4.2f" % weight, \
           "%4.2f" % mu, \

           "%4.1f" % j_calib_pt, \
           "%3.2f" % j_calib_eta, \
           "%4.1f" % j_calib_E, \
           "%4.1f" % j_calib_P, \
           "%4.1f" % j_calib_M, \

           "%4.1f" % j_nocalib_pt, \
           "%3.2f" % j_nocalib_eta, \
           "%4.1f" % j_nocalib_E, \
           "%4.1f" % j_nocalib_P, \
           "%4.1f" % j_nocalib_M, \

           "%4.1f" % j_truth_pt, \
           "%3.2f" % j_truth_eta, \
           "%4.1f" % j_truth_E, \
           "%4.1f" % j_truth_P, \
           "%4.1f" % j_truth_M, \
           ) )

outfile.close()
