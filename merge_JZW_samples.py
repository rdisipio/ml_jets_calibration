#!/usr/bin/env python

import os, sys
from ROOT import *

sf = {
  '361021' : 52696671.6 / 2000000.,
  '361022' : 809379.648 / 1999000.,
  '361023' : 8453.64024 / 15882500.,
  '361024' : 134.9920945 / 15983500.,
  '361025' : 4.19814486 / 15994500.,
  '361026' : 0.242119405 / 1997000.,
  '361027' : 0.006359523 / 3994500.,
}

input_file_names = {
 'largeR' : {
     '361023' :  "user.tnitta.361023.DAOD_JETM8_p2666.dnn.largeR.pT_M.histograms.root",
     '361024' :  "user.tnitta.361024.DAOD_JETM8_p2666.dnn.largeR.pT_M.histograms.root",
     '361025' :  "user.tnitta.361025.DAOD_JETM8_p2666.dnn.largeR.pT_M.histograms.root",
     '361026' :  "user.tnitta.361026.DAOD_JETM8_p2666.dnn.largeR.pT_M.histograms.root",
     '361027' :  "user.tnitta.361027.DAOD_JETM8_p2666.dnn.largeR.pT_M.histograms.root",
  },
  'smallR' : {
     '361021' : "",
     '361022' : "",
     '361023' : "",
     '361024' : "",
     '361025' : "",
     '361026' : "",
     '361027' : "",
  }
}

collection = "largeR"
if len(sys.argv) > 1:
   collection = sys.argv

files = {}
for dsid, fname in input_file_names[collection].iteritems():
   files[dsid] = TFile.Open( fname )

histograms = {}
for dsid, file in files.iteritems():
   print "INFO: DSID: %s :: file: %s" % ( dsid, file.GetName() ) 
   next = TIter( file.GetListOfKeys() )
   key = TKey()
   doRead = True
   while doRead:
      key = next()
      if key == None: break

      h = key.ReadObj()
      if not h.Class() in [ TH2F.Class(), TH2D.Class(), TH1F.Class(), TH1D.Class() ]: continue

      hname = h.GetName()
      h.Scale( sf[dsid] )

      if histograms.has_key( hname ):
         histograms[hname].Add( h ) 
      else:
         histograms[hname] = h.Clone()

#      print h.GetName(), h.IsA()

outfilename = "user.tnitta.all.DAOD_JETM8_p2666.dnn.largeR.pT_M.histograms.root"
outfile = TFile.Open( outfilename, "RECREATE" )
outfile.cd()
for h in histograms.values(): 
   h.Write()
outfile.Close()

print "INFO: created output file %s" % ( outfilename )
