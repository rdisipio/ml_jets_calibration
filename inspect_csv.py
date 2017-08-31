#!/usr/bin/env python

import sys, os

from ROOT import *
from features import header

infilename = sys.argv[1]

branches = ":".join(header)

print branches

tree = TTree("tree", "tree" )
tree.ReadFile( infilename, branches )

ofilename = infilename.replace(".csv", ".tree.root")
ofile = TFile.Open( ofilename, "RECREATE" )

tree.Write()

ofile.Close()
print "INFO: output file create", ofile.GetName()
