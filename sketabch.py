#!/usr/bin/env python

import os, sys
import csv
from ROOT import *

basedir = os.environ['PWD'] + "/"
GeV = 1000.

def DumpFourVector( p ):
  print "(pT,eta,phi,E;M) = ( %4.1f, %4.3f, %4.3f, %4.1f ; %4.1f )" % ( p.Pt(), p.Eta(), p.Phi(), p.E(), p.M() )
 

infilename = sys.argv[1]

outfilename = "csv/" + infilename.split("/")[-1].replace( ".root", ".csv" )
outfile = open( outfilename, "wt" )
csvwriter = csv.writer( outfile )
print "INFO: output file:", outfilename

tree = TChain( "Tree", "tree" )
txt = open( infilename, 'r' )

for rootf in txt.readlines():
  rootf = rootf.strip()
  if rootf=="": continue
  tree.Add( rootf )
  
nentries = tree.GetEntries()
print "INFO: found %i entries" % nentries

i = 0
for ientry in range(nentries):
  tree.GetEntry(ientry)

#don't have any of these
  # eventNumber = tree.eventNumber
  # weight      = tree.mcEventWeight 
  # mu          = tree.averageInteractionsPerCrossing
  # prw         = tree.prw

  #calibrated
  jet_calib = TLorentzVector()
  jet_calib.SetPtEtaPhiM( tree.calib_jet_pt, tree.calib_jet_eta, tree.calib_jet_phi, tree.calib_jet_m )
  
  jet_calib.Pt = jet_calib.Pt()/GeV
  jet_calib.E =jet_calib.E()/GeV
  jet_calib.P = jet_calib.P()/GeV
  jet_calib.M = jet_calib.M()/GeV

  # nocalibrated
  jet_nocalib = TLorentzVector()
  jet_nocalib.SetPtEtaPhiM( tree.pt, tree.eta, tree.phi, tree.m )
  
  jet_nocalib.Pt = jet_nocalib.Pt()/GeV
  jet_nocalib.E =jet_nocalib.E()/GeV
  jet_nocalib.P = jet_nocalib.P()/GeV
  jet_nocalib.M = jet_nocalib.M()/GeV
  # truth
  jet_truth = TLorentzVector()
  jet_truth.SetPtEtaPhiM( tree.truth_jet_pt, tree.truth_jet_eta, tree.truth_jet_phi, tree.truth_jet_m )

  jet_truth.Pt = jet_truth.Pt()/GeV
  jet_truth.E =jet_truth.E()/GeV
  jet_truth.P = jet_truth.P()/GeV
  jet_truth.M = jet_truth.M()/GeV

  jet_Nconstit = tree.n_constituents
  #fjet1_untrimNtrk500 = tree.fjet1_untrimNtrk500
  #fjet1_mTA        = tree.fjet1_mTA
  #fjet1_mTAS       = tree.fjet1_mTAS
  jet_D2         = tree.D2
  jet_C2         = tree.C2

  jet_Tau1_wta  = tree.Tau1_wta
  jet_Tau2_wta  = tree.Tau2_wta
  jet_Tau3_wta  = tree.Tau3_wta

  jet_Tau21_wta  = tree.Tau21_wta
  jet_Tau32_wta  = tree.Tau32_wta

  jet_Angularity = tree.angularity 
  jet_Aplanarity = tree.aplanarity
  jet_PlanarFlow = tree.planarflow
  jet_Sphericity = tree.sphericity
  # fjet1_ThrustMaj = tree.fjet1_ThrustMaj
  # fjet1_ThrustMin = tree.fjet1_ThrustMin

  jet_ECF1    = tree.ECF1
  jet_ECF2    = tree.ECF2
  jet_ECF3    = tree.ECF3

  jet_Width   = tree.width
  jet_Weight  = tree.weight

  skip = 0

  # if abs( fjet1_calib.Eta() )   > 2.0: skip = 1
  # if abs( fjet1_nocalib.Eta() ) > 2.0: skip = 1
  # if abs( fjet1_truth.Eta() )   > 2.0: skip = 1

  if jet_calib.Pt < 250.: skip = 1
  # if fjet1_calib.Pt() > 2000.: skip = 1

  if jet_nocalib.Pt < 250.: skip = 1
  # if fjet1_nocalib.Pt() > 2000.: skip = 1

  if jet_truth.Pt < 250.: skip = 1
  # if fjet1_truth.Pt() > 2000.: skip = 1

  if jet_calib.M < 30.: skip = 2
  # if fjet1_calib.M() > 1000.: skip = 2

  if jet_nocalib.M < 30.: skip = 3
  # if fjet1_nocalib.M() > 1000.: skip = 3

  if jet_truth.M < 30.: skip = 4
  # if fjet1_truth.M() > 1000.: skip = 4

  if skip == 0:
    i+=1
  # calibration constants 
    # alpha = [ jet_truth.Pt() / jet_nocalib.Pt(), 
    #         jet_truth.Eta()  / jet_nocalib.Eta(),
    #         jet_truth.E()    / jet_nocalib.E(),
    #         jet_truth.M()    / jet_nocalib.M() 
    #       ]

    # Write (pT, eta, E, P, M )
    csvwriter.writerow( (
           "%4.1f" % jet_Weight, #no eventNumber, mu or prw\
           "%4.1f" % jet_truth.Pt,   "%4.3f" % jet_truth.Eta(),   "%4.1f" % jet_truth.E,   "%4.1f" % jet_truth.P,   "%4.1f" % jet_truth.M, \
           "%4.1f" % jet_nocalib.Pt, "%4.3f" % jet_nocalib.Eta(), "%4.1f" % jet_nocalib.E, "%4.1f" % jet_nocalib.P, "%4.1f" % jet_nocalib.M, \
           "%i"    % jet_Nconstit,\
           "%f" % jet_D2, jet_C2, \
           "%4.3f" % jet_Tau1_wta, \
           "%4.3f" % jet_Tau2_wta, \
           "%4.3f" % jet_Tau3_wta, \
           "%4.3f" % jet_Tau21_wta, jet_Tau32_wta, \
           "%4.3f" % jet_Angularity, jet_Aplanarity, jet_PlanarFlow, jet_Sphericity, \
           "%.3f" % jet_Width, \
           # "%f" % jet_ECF1, \
           # "%f" % jet_ECF2, \
           # "%f" % jet_ECF3, \
           "%4.1f" % jet_calib.Pt,   "%4.3f" % jet_calib.Eta(),   "%4.1f" % jet_calib.E,   "%4.1f" % jet_calib.P,   "%4.1f" % jet_calib.M, \
           # "%4.3f" % alpha[0], "%4.3f" % alpha[1], "%4.3f" % alpha[2], "%4.3f" % alpha[3], \
           ) )

outfile.close()
print i

print "INFO: file %s done" % outfilename

