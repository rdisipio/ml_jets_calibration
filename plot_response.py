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

obs = "E"
if len( sys.argv) > 1:
  obs = sys.argv[1]

etaregion = "0"
if len( sys.argv) > 2:
  etaregion = sys.argv[2]

infilename = "dnn.root"
if len( sys.argv ) > 3:
   infilename = sys.argv[3]
infile = TFile.Open( infilename )


h_calib    = infile.Get( "%s_response_calib_%s" % ( obs, etaregion ) )
h_nocalib  = infile.Get( "%s_response_nocalib_%s" % ( obs, etaregion ) )
h_dnncalib = infile.Get( "%s_response_dnncalib_%s" % ( obs, etaregion ) )

#h_calib.Rebin2D(2,2)
#h_nocalib.Rebin2D(2,2)
#h_dnncalib.Rebin2D(2,2)

p_calib    = h_calib.ProfileX()
p_nocalib  = h_nocalib.ProfileX()
p_dnncalib = h_dnncalib.ProfileX()

SetTH1FStyle( p_calib,    color=kRed, linewidth=2, markerstyle=22 )
SetTH1FStyle( p_nocalib,  color=kBlack, linewidth=2, markerstyle=21 )
SetTH1FStyle( p_dnncalib, color=kGreen-2, linewidth=2, markerstyle=26 )

c = TCanvas( "C", "C", 1000, 800 )

p_nocalib.SetMinimum(0.9)
p_nocalib.SetMaximum(1.1)
p_nocalib.GetYaxis().SetTitleOffset( 1.2 )
p_nocalib.GetYaxis().SetTitle( "%s / %s_{truth}" % ( obs, obs ))

p_nocalib.Draw()
p_calib.Draw( "same" )
#p_dnncalib.Draw( "same" )

l = TLine()
l.SetLineStyle( kDashed )
l.DrawLine( 0., 1., 1000, 1. ) 

gPad.SetLogx()
p_nocalib.GetXaxis().SetMoreLogLabels()
p_nocalib.GetXaxis().SetNoExponent()

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
leg.AddEntry( p_calib,     "Calibrated",   "lp" )
leg.AddEntry( p_dnncalib,  "DNN-calibrated", "lp" )
leg.Draw()
leg.SetY1( leg.GetY1() - lparams['height'] * leg.GetNRows() )

PrintATLASLabel( 0.15, 0.90 )

txt = TLatex()
txt.SetNDC()
txt.SetTextSize(0.03)
txt.SetTextFont(42)

etamin = etaregions[int(etaregion)][0]
etamax = etaregions[int(etaregion)][1]
txt.DrawLatex( 0.65, 0.90, "C/A R=0.4 jets p_{T} > 20 GeV" )
txt.DrawLatex( 0.65, 0.85, "%2.1f #leq |#eta| #leq %2.1f" % (etamin,etamax) )

imgname = "img/ca4_%s_response_etaregion%s.png" % ( obs, etaregion )
c.SaveAs( imgname )
