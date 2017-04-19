#!/usr/bin/env python

import os, sys
import csv
from ROOT import *

basedir = os.environ['PWD'] + "/"
GeV = 1000.

xsec = {
 '361020' : 76634376600.0,
 '361021' : 51525861.0,
 '361022' : 806353.671,
 '361023' : 8513.42628,
 '361024' : 134.9768167,
 '361025' : 4.204018875,
 '361026' : 0.242119405,
 '361027' : 0.006369576,
 '361028' : 0.006351453,
 '361029' : 0.000236729,
 '361030' : 7.05e-06,
 '361031' : 1.14e-07,
 '361032' : 0.0004*1.0367e-06,
}
#sumw = {
#}

def DumpFourVector( p ):
  print "(pT,eta,phi,E;M) = ( %4.1f, %4.3f, %4.3f, %4.1f ; %4.1f )" % ( p.Pt(), p.Eta(), p.Phi(), p.E(), p.M() )
 

infilename = sys.argv[1]

outfilename = infilename.replace( ".dat", ".csv" )
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

  mc_weight  = tree.weight

  #don't have any of these
  # eventNumber = tree.eventNumber
  # mu          = tree.averageInteractionsPerCrossing
  # prw         = tree.prw

  # track jet
  jet_track = TLorentzVector()
  jet_track.SetPtEtaPhiM( tree.track_jet_pt, tree.track_jet_eta, tree.track_jet_phi, tree.track_jet_m )

  # ATLAS-calibrated jet
  jet_calib = TLorentzVector()
  jet_calib.SetPtEtaPhiM( tree.calib_jet_pt, tree.calib_jet_eta, tree.calib_jet_phi, tree.calib_jet_m )

  # Truth jet
  jet_truth = TLorentzVector()
  jet_truth.SetPtEtaPhiM( tree.truth_jet_pt, tree.truth_jet_eta, tree.truth_jet_phi, tree.truth_jet_m )

  # Non-calibrated jet
  jet_nocalib = TLorentzVector()
  jet_nocalib.SetPtEtaPhiM( tree.pt, tree.eta, tree.phi, tree.m )

  #######################
  # Apply event selection

  skip = 0

  if jet_calib.Pt() < 200.: skip = 1
  if jet_calib.Pt() > 3000.: skip = 1

  if jet_nocalib.Pt() < 200.: skip = 1
  if jet_nocalib.Pt() > 3000.: skip = 1

  if jet_truth.Pt() < 200.: skip = 1
  if jet_truth.Pt() > 3000.: skip = 1

  if jet_calib.M() < 30.: skip = 1
  if jet_calib.M() > 300.: skip = 1

  if jet_nocalib.M() < 30.: skip = 1
  if jet_nocalib.M() > 300.: skip = 1

  if jet_truth.M() < 30.: skip = 1
  if jet_truth.M() > 300.: skip = 1

  if not skip == 0: continue

  # Create other variables
  jet_truth.m_over_pt = jet_truth.M() / jet_truth.Pt()
  
  jet_track.m_over_pt = jet_track.M() / jet_track.Pt()

  jet_track.tau21_wta = tree.track_jet_Tau21_wta
  jet_track.tau32_wta = tree.track_jet_Tau32_wta
  jet_track.tau1_wta  = tree.track_jet_Tau1_wta
  jet_track.tau2_wta  = tree.track_jet_Tau2_wta
  jet_track.tau3_wta  = tree.track_jet_Tau3_wta
  jet_track.sum_pt    = tree.track_sum_pt
  jet_track.sum_m     = tree.track_sum_m

  
  jet_nocalib.m_over_pt = jet_nocalib.M() / jet_nocalib.Pt()
  jet_nocalib.mTA       = jet_nocalib.M() / jet_track.M()

  jet_nocalib.Nconstit        = tree.n_constituents
  jet_nocalib.Nconstit_over_m = tree.n_constituents / ( jet_nocalib.M()/GeV )
  jet_nocalib.width           = tree.width
  jet_nocalib.width_over_m    = 1e6 * jet_nocalib.width / jet_nocalib.M()


  jet_nocalib.ECF1    = tree.ECF1
  jet_nocalib.ECF2    = tree.ECF2
  jet_nocalib.ECF3    = tree.ECF3
  jet_nocalib.D2      = tree.D2
  jet_nocalib.C2      = tree.C2

  jet_nocalib.Tau1_wta   = tree.Tau1_wta
  jet_nocalib.Tau2_wta   = tree.Tau2_wta
  jet_nocalib.Tau3_wta   = tree.Tau3_wta
  jet_nocalib.Tau21_wta  = tree.Tau21_wta
  jet_nocalib.Tau32_wta  = tree.Tau32_wta

  jet_nocalib.Angularity = tree.angularity 
  jet_nocalib.Aplanarity = tree.aplanarity
  jet_nocalib.PlanarFlow = tree.planarflow
  jet_nocalib.Sphericity = tree.sphericity
  # fjet1_ThrustMaj = tree.fjet1_ThrustMaj
  # fjet1_ThrustMin = tree.fjet1_ThrustMin

  i+=1

  csvwriter.writerow( (
           "%4.1f" % mc_Weight, #no eventNumber, mu or prw\
           "%4.1f" % jet_truth.Pt()/GeV,   "%4.3f" % jet_truth.Rapidity(),   "%4.1f" % jet_truth.E()/GeV,   "%4.1f" % jet_truth.P()/GeV,   "%4.1f" % jet_truth.M()/GeV, \
           "%4.1f" % jet_nocalib.Pt()/GeV, "%4.3f" % jet_nocalib.Rapidity(), "%4.1f" % jet_nocalib.E()/GeV, "%4.1f" % jet_nocalib.P()/GeV, "%4.1f" % jet_nocalib.M()/GeV, \
           "%4.1f" % jet_track.Pt()/GeV,   "%4.3f" % jet_track.Rapidity(),   "%4.1f" % jet_track.E()/GeV,   "%4.1f" % jet_track.P()/GeV,   "%4.1f" % jet_track.M()/GeV, \
           "%.3f"  % jet_nocalib.m_over_pt, "%.3f" % jet_track.m_over_pt, "%.3f" % jet_nocalib.mTA, \
           "%i"    % jet_nocalib.Nconstit, "%.3f" % jet_nocalib.Nconstit_over_m, "%.3f" % jet_nocalib.width, "%.3f" % jet_nocalib.width_over_m, \
           # "%f" % jet_nocalib.ECF1, "%f" % jet_nocalib.ECF2, "%f" % jet_nocalib.ECF3, \
           "%4.3f" % jet_nocalib.D2, "%4.3f" % jet_nocalib.C2, \
           "%4.3f" % jet_nocalib.Tau1_wta, "%4.3f" % jet_nocalib.Tau2_wta, "%4.3f" % jet_nocalib.Tau3_wta, "%4.3f" % jet_nocalib.Tau21_wta, "%4.3f" %  jet_nocalib.Tau32_wta, \
           "%4.3f" % jet_nocalib.Angularity, "%4.3f" % jet_nocalib.Aplanarity, "%4.3f" % jet_nocalib.PlanarFlow, "%4.3f" % jet_nocalib.Sphericity, \
           "%4.1f" % jet_calib.Pt()/GeV,   "%4.3f" % jet_calib.Rapidity(),   "%4.1f" % jet_calib.E()/GeV,   "%4.1f" % jet_calib.P()/GeV,   "%4.1f" % jet_calib.M()/GeV, \
           ) )

outfile.close()

print "INFO: file %s done" % outfilename

