#!/usr/bin/env python

import os,sys

from ROOT import *

gROOT.LoadMacro("AtlasStyle.C")
#gROOT.LoadMacro( "AtlasUtils.C" )
SetAtlasStyle()

gROOT.SetBatch(1)

etaregions = [ [0.0,0.1], [0.1,0.2], [0.2,0.3], [0.3,0.4], [0.4,0.5], [0.5,0.6], [0.6,0.7], [0.7,0.8], [0.8,0.9], [0.9,1.0], [1.0,2.5] ]

def SetTH1FStyle( h, color = kBlack, linewidth = 1, fillcolor = 0, fillstyle = 0, markerstyle = 21, markersize = 1.3, linestyle=kSolid ):
    '''Set the style with a long list of parameters'''
    
    h.SetLineColor( color )
    h.SetLineWidth( linewidth )
    h.SetLineStyle( linestyle )
    h.SetFillColor( fillcolor )
    h.SetFillStyle( fillstyle )
    h.SetMarkerStyle( markerstyle )
    h.SetMarkerColor( h.GetLineColor() )
    h.SetMarkerSize( markersize )


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def MakeLegend( params ):
    leg = TLegend( params['xoffset'], params['yoffset'], params['xoffset'] + params['width'], params['yoffset'] )
    leg.SetNColumns(1)
    leg.SetFillColor(0)
    leg.SetFillStyle(0)
    leg.SetBorderSize(0)
    leg.SetTextFont(72)
    leg.SetTextSize(0.06)
    return leg


def PrintATLASLabel( x = 0.5, y = 0.87, lumi = 0. ):
  l = TLatex()  #l.SetTextAlign(12); l.SetTextSize(tsize); 
  l.SetNDC()
  l.SetTextFont(72)
  l.SetTextColor(kBlack)
  l.DrawLatex(x,y,"ATLAS");
  l.SetTextFont(42);
  l.DrawLatex(x+0.13,y,"Simulation Internal");
  #l.DrawLatex(x+0.14,y,"Preliminary")
  l.SetTextSize(0.04)
#  s = "#sqrt{s} = 13 TeV, %2.1f fb^{-1}" % (lumi/1000.)
  s = "#sqrt{s} = 13 TeV"
  l.DrawLatex(x, y-0.05, s )


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

obs = "pT"
if len( sys.argv) > 1:
  obs = sys.argv[1]

#etaregion = "0"
#if len( sys.argv) > 2:
#  etaregion = sys.argv[2]

infilename = "dnn.root"
if len( sys.argv ) > 2:
   infilename = sys.argv[2]
infile = TFile.Open( infilename )
print "INFO: histograms from file", infilename

#h_calib    = infile.Get( "%s_resolution_calib" % ( obs ) )
#h_nocalib  = infile.Get( "%s_resolution_nocalib" % ( obs ) )
#h_dnncalib = infile.Get( "%s_resolution_dnncalib" % ( obs ) )

# Calculate inter-quantile range IQR68 / 2
h_response_calib    = infile.Get( "%s_response_calib" % ( obs ) )
h_response_nocalib  = infile.Get( "%s_response_nocalib" % ( obs ) )
h_response_dnncalib = infile.Get( "%s_response_dnncalib" % ( obs ) )

h_calib_q16x    = h_response_calib.QuantilesX(    0.16, "%s_calib_q16x" % obs )
h_nocalib_q16x  = h_response_nocalib.QuantilesX(  0.16, "%s_nocalib_q16x" % obs )
h_dnncalib_q16x = h_response_dnncalib.QuantilesX( 0.16, "%s_dncalib_q16x" % obs )

h_calib_q84x    = h_response_calib.QuantilesX(    0.84, "%s_calib_q84x" % obs )
h_nocalib_q84x  = h_response_nocalib.QuantilesX(  0.84, "%s_nocalib_q84x" % obs )
h_dnncalib_q84x = h_response_dnncalib.QuantilesX( 0.84, "%s_dncalib_q84x" % obs )

h_calib_q84x.Add(    h_calib_q16x,    -1 )
h_nocalib_q84x.Add(  h_nocalib_q16x,  -1 )
h_dnncalib_q84x.Add( h_dnncalib_q16x, -1 )

h_calib_q84x.Scale( 0.5 )
h_nocalib_q84x.Scale( 0.5 )
h_dnncalib_q84x.Scale( 0.5 )

p_calib    = h_calib_q84x
p_nocalib  = h_nocalib_q84x
p_dnncalib = h_dnncalib_q84x

SetTH1FStyle( p_calib,    color=kRed, linewidth=2, markerstyle=22 )
SetTH1FStyle( p_nocalib,  color=kBlack, linewidth=2, markerstyle=20 )
SetTH1FStyle( p_dnncalib, color=kGreen-2, linewidth=2, markerstyle=23 )

c = TCanvas( "C", "C", 1000, 800 )

p_nocalib.SetMinimum(0.03)
p_nocalib.SetMaximum(0.80)

gPad.SetLogy()
#gPad.SetLogx()
gPad.SetGrid()
p_nocalib.GetYaxis().SetMoreLogLabels() ; p_nocalib.GetYaxis().SetNoExponent()
p_nocalib.GetYaxis().SetTitleOffset( 1.2 )
#p_nocalib.GetYaxis().SetTitle( "(%s^{reco} - %s^{truth} ) / %s^{truth}" % ( obs, obs, obs ))
p_nocalib.GetYaxis().SetTitle( "IQR68 / 2" )
p_nocalib.GetXaxis().SetMoreLogLabels()
p_nocalib.GetXaxis().SetNoExponent()
p_nocalib.GetXaxis().SetTitle( "%s^{reco} [GeV]" % obs )

p_nocalib.Draw()
p_calib.Draw( "same" )
p_dnncalib.Draw( "same" )

# make legend
lparams = {
  'xoffset' : 0.65,
  'yoffset' : 0.83,
  'width'   : 0.3,
  'height'  : 0.048
}
leg = MakeLegend( lparams )
leg.SetTextFont( 42 )
leg.SetTextSize( 0.03 )
leg.AddEntry( p_nocalib,   "Uncalibrated", "lp" )
leg.AddEntry( p_calib,     "ATLAS-Calibrated",   "lp" )
leg.AddEntry( p_dnncalib,  "DNN-calibrated", "lp" )
leg.Draw()
leg.SetY1( leg.GetY1() - lparams['height'] * leg.GetNRows() )

PrintATLASLabel( 0.15, 0.90 )

txt = TLatex()
txt.SetNDC()
txt.SetTextSize(0.03)
txt.SetTextFont(42)

#etamin = etaregions[int(etaregion)][0]
#etamax = etaregions[int(etaregion)][1]
#txt.DrawLatex( 0.65, 0.90, "C/A R=0.4 jets p_{T} > 20 GeV" )
#txt.DrawLatex( 0.65, 0.85, "%2.1f #leq |#eta| #leq %2.1f" % (etamin,etamax) )

#imgname = "img/ca4_%s_resolution_etaregion%s.png" % ( obs, etaregion )
imgname = "img/ca4_%s_resolution.png" % ( obs )
c.SaveAs( imgname )
