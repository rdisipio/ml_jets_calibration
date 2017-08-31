#!/usr/bin/env python

from ROOT import * 
from array import array 

gROOT.SetBatch(1)
gROOT.Macro("./rootlogon.C")
gROOT.LoadMacro("./AtlasUtils.C")

f = TFile.Open( "covariance.root" )

h = f.Get( "h2_correlation" )

c = TCanvas( "C", "C", 1600, 1400 )

gPad.SetBottomMargin(0.20)
gPad.SetLeftMargin(0.15)
gPad.SetRightMargin(0.02)
gPad.SetTopMargin(0.02)

h.GetXaxis().SetLabelSize( 0.02 )
h.GetXaxis().LabelsOption("v")
h.GetYaxis().SetLabelSize( 0.02 )
h.GetXaxis().SetLabelOffset( 0.002 )
h.GetYaxis().SetLabelOffset( 0.002 )

stops = array( 'd', [ 0.00, 0.05, 0.50, 0.95, 1.00 ] )
# 286CP / 284CP / x / 177CP / 7627CP
red   = array( 'd', [ 0.13, 0.45, 1.00, 0.96, 0.74 ] )
green = array( 'd', [ 0.31, 0.67, 1.00, 0.69, 0.29 ] )
blue  = array( 'd', [ 0.60, 0.85, 1.00, 0.72, 0.14 ] )

TColor.CreateGradientColorTable( len(stops), stops, red, green, blue, 100 )
gStyle.SetPaintTextFormat( "3.2f%" )
#gStyle.SetHistMinimumZero()

def DrawGrid( h ):
    line = TLine()
    line.SetLineWidth( 1 )
    line.SetLineColor( kWhite )
    
    xaxis = h.GetXaxis()
    yaxis = h.GetYaxis()
    
    #xaxis.SetMoreLogLabels()
    #yaxis.SetMoreLogLabels()
    
    for binx in range( 0, h.GetNbinsX() ):
        xmin = xaxis.GetBinLowEdge( binx + 1 )
        
        if not xmin == xaxis.GetXmin():
            line.DrawLine( xmin, yaxis.GetXmin(), xmin, yaxis.GetXmax() )
    
    for biny in range( 0, h.GetNbinsY() ):
        ymin = yaxis.GetBinLowEdge( biny + 1 )
        
        if not ymin == yaxis.GetXmin():
            line.DrawLine( xaxis.GetXmin(), ymin, xaxis.GetXmax(), ymin )

h.SetMarkerSize(0.5)
h.Draw("col text")
DrawGrid( h )
h.GetXaxis().SetTickLength(0)
h.GetYaxis().SetTickLength(0)
gPad.RedrawAxis()

c.SaveAs( "img/correlations.pdf" )
