#!/usr/bin/env python

import os, sys
import csv
from ROOT import *

basedir = os.environ['PWD'] + "/"
GeV = 1000.

def DumpFourVector( p ):
  print "(pT,eta,phi,E;M) = ( %4.1f, %4.3f, %4.3f, %4.1f ; %4.1f )" % ( p.Pt(), p.Eta(), p.Phi(), p.E(), p.M() )
 

infilename = sys.argv[1]
#infile = TFile.Open( infilename )

outfilename = "csv/" + infilename.split("/")[-1].replace( ".dat", ".csv" )
outfile = open( outfilename, "wt" )
csvwriter = csv.writer( outfile )
print "INFO: output file:", outfilename

#tree = infile.Get( "tree" )
tree = TChain( "tree", "tree" )
txt = open( infilename, 'r' )
for rootf in txt.readlines():
  rootf = rootf.strip()
  tree.Add( rootf )

nentries = tree.GetEntries()
print "INFO: found %i entries" % nentries

for ientry in range(nentries):
  tree.GetEntry( ientry )

  eventNumber = tree.eventNumber
  weight      = tree.mcEventWeight 
  mu          = tree.averageInteractionsPerCrossing
  prw         = tree.prw

  fjet1_calib = TLorentzVector()
  fjet1_calib.SetPtEtaPhiM( tree.fjet1_pt, tree.fjet1_eta, tree.fjet1_phi, tree.fjet1_m )

  # nocalibrated
  fjet1_nocalib = TLorentzVector()
  fjet1_nocalib.SetPtEtaPhiM( tree.fjet1_upt, tree.fjet1_ueta, tree.fjet1_uphi, tree.fjet1_um )

  # truth
  fjet1_truth = TLorentzVector()
  fjet1_truth.SetPtEtaPhiM( tree.ftruthjet1_pt, tree.ftruthjet1_eta, tree.ftruthjet1_phi, tree.ftruthjet1_m )

#  fjet1_moments = {}
  fjet1_Nconstit = tree.fjet1_Nconstit
  fjet1_untrimNtrk500 = tree.fjet1_untrimNtrk500
  fjet1_mTA        = tree.fjet1_mTA
  fjet1_mTAS       = tree.fjet1_mTAS
  fjet1_D2         = tree.fjet1_D2
  fjet1_C2         = tree.fjet1_C2

  fjet1_Tau1      = tree.fjet1_Tau1
  fjet1_Tau1_wta  = tree.fjet1_Tau1_wta
  fjet1_Tau2      = tree.fjet1_Tau2 
  fjet1_Tau2_wta  = tree.fjet1_Tau2_wta
  fjet1_Tau3      = tree.fjet1_Tau3 
  fjet1_Tau3_wta  = tree.fjet1_Tau3_wta

  fjet1_Angularity = tree.fjet1_Angularity 
  fjet1_Aplanarity = tree.fjet1_Aplanarity
  fjet1_PlanarFlow = tree.fjet1_PlanarFlow
  fjet1_Sphericity = tree.fjet1_Sphericity
  fjet1_ThrustMaj = tree.fjet1_ThrustMaj
  fjet1_ThrustMin = tree.fjet1_ThrustMin

  fjet1_Dip12     = tree.fjet1_Dip12
  fjet1_Dip13     = tree.fjet1_Dip13
  fjet1_Dip23     = tree.fjet1_Dip23 
  fjet1_DipExcl12 = tree.fjet1_DipExcl12

  # C2 = ecf3 * ecf1 / pow(ecf2, 2.0)
  # D2 = ecf3 * pow(ecf1, 3.0) / pow(ecf2, 3.0)
  fjet1_ECF1    = tree.fjet1_ECF1
  fjet1_ECF2    = tree.fjet1_ECF2
  fjet1_ECF3    = tree.fjet1_ECF3

  fjet1_KtDR    = tree.fjet1_KtDR
  fjet1_Mu12    = tree.fjet1_Mu12

  fjet1_Split12 = tree.fjet1_Split12
  fjet1_Split23 = tree.fjet1_Split23
  fjet1_Split34 = tree.fjet1_Split34

  fjet1_Width   = tree.fjet1_Width
  fjet1_Qw      = tree.fjet1_Qw 

  skip = 0

  if abs( fjet1_calib.Eta() )   > 2.0: skip = 1
  if abs( fjet1_nocalib.Eta() ) > 2.0: skip = 1
  if abs( fjet1_truth.Eta() )   > 2.0: skip = 1

  if fjet1_calib.Pt() < 250.: skip = 1
  if fjet1_calib.Pt() > 2000.: skip = 1

  if fjet1_nocalib.Pt() < 250.: skip = 1
  if fjet1_nocalib.Pt() > 2000.: skip = 1

  if fjet1_truth.Pt() < 250.: skip = 1
  if fjet1_truth.Pt() > 2000.: skip = 1

  if fjet1_calib.M() < 30.: skip = 2
  if fjet1_calib.M() > 1000.: skip = 2

  if fjet1_nocalib.M() < 30.: skip = 3
  if fjet1_nocalib.M() > 1000.: skip = 3

  if fjet1_truth.M() < 30.: skip = 4
  if fjet1_truth.M() > 1000.: skip = 4


  if skip == 0:
  # calibration constants 
    alpha = [ fjet1_truth.Pt() / fjet1_nocalib.Pt(), 
            fjet1_truth.Eta()  / fjet1_nocalib.Eta(),
            fjet1_truth.E()    / fjet1_nocalib.E(),
            fjet1_truth.M()    / fjet1_nocalib.M() 
          ]

    # Write (pT, eta, E, P, M )
    csvwriter.writerow( (
           "%i"    % eventNumber, \
           "%4.1f" % weight, \
           "%3.1f" % mu, \
           "%f"    % prw, \
           "%4.1f" % fjet1_truth.Pt(),   "%4.3f" % fjet1_truth.Eta(),   "%4.1f" % fjet1_truth.E(),   "%4.1f" % fjet1_truth.P(),   "%4.1f" % fjet1_truth.M(), \
           "%4.1f" % fjet1_nocalib.Pt(), "%4.3f" % fjet1_nocalib.Eta(), "%4.1f" % fjet1_nocalib.E(), "%4.1f" % fjet1_nocalib.P(), "%4.1f" % fjet1_nocalib.M(), \
           "%i"    % fjet1_Nconstit, \
           "%i"    % fjet1_untrimNtrk500, \
#           "%f" % fjet1_mTA, \
#           "%f" % fjet1_mTAS, \
           "%4.3f" % fjet1_D2, "%4.3f" % fjet1_C2, \
           "%4.3f" % fjet1_Tau1, "%4.3f" % fjet1_Tau1_wta, \
           "%4.3f" % fjet1_Tau2, "%4.3f" % fjet1_Tau2_wta, \
           "%4.3f" % fjet1_Tau3, "%4.3f" % fjet1_Tau3_wta, \
           "%4.3f" % fjet1_Angularity, \
           "%4.3f" % fjet1_Aplanarity, \
           "%4.3f" % fjet1_PlanarFlow, \
           "%4.3f" % fjet1_Sphericity, \
           "%4.3f" % fjet1_ThrustMaj, \
           "%4.3f" % fjet1_ThrustMin, \
#           "%f" % fjet1_Dip12, \
#           "%f" % fjet1_Dip13, \
#           "%f" % fjet1_Dip23, \
#           "%f" % fjet1_DipExcl12, \
#           "%f" % fjet1_ECF1, \
#           "%f" % fjet1_ECF2, \
#           "%f" % fjet1_ECF3, \
           "%.3f" % fjet1_KtDR, \
           "%.3f" % fjet1_Mu12, \
           "%.3f" % fjet1_Width, \
           "%.1f" % fjet1_Qw, \
           "%.1f" % fjet1_Split12, \
           "%.1f" % fjet1_Split23, \
           "%.1f" % fjet1_Split34, \
           "%4.1f" % fjet1_calib.Pt(),   "%4.3f" % fjet1_calib.Eta(),   "%4.1f" % fjet1_calib.E(),   "%4.1f" % fjet1_calib.P(),   "%4.1f" % fjet1_calib.M(), \
           "%4.3f" % alpha[0], "%4.3f" % alpha[1], "%4.3f" % alpha[2], "%4.3f" % alpha[3], \
           ) )

outfile.close()

print "INFO: file %s done" % outfilename
