#!/usr/bin/env python

import os,sys

from ROOT import *

gROOT.LoadMacro("AtlasStyle.C")
#gROOT.LoadMacro( "AtlasUtils.C" )
SetAtlasStyle()

gROOT.SetBatch(1)

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

obs = "M"
if len( sys.argv) > 1:
  obs = sys.argv[1]

infilename = "dnn.largeR.pT_M.histograms.root"
if len( sys.argv ) > 2:
   infilename = sys.argv[2]
infile = TFile.Open( infilename )


h_calib    = infile.Get( "%s_response_calib" % ( obs ) )
h_nocalib  = infile.Get( "%s_response_nocalib" % ( obs ) )
h_dnncalib = infile.Get( "%s_response_dnncalib" % ( obs ) )

#h_calib.Rebin2D(2,2)
#h_nocalib.Rebin2D(2,2)
#h_dnncalib.Rebin2D(2,2)

p_calib    = h_calib.ProfileX() # "pfx_calib", 1, -1, "s" )
p_nocalib  = h_nocalib.ProfileX() # "pfx_nocalib", 1, -1, "s" )
p_dnncalib = h_dnncalib.ProfileX() # "pfx_dnn", 1, -1, "s" )

SetTH1FStyle( p_calib,    color=kRed, linewidth=2, markerstyle=22 )
SetTH1FStyle( p_nocalib,  color=kBlack, linewidth=2, markerstyle=20 )
SetTH1FStyle( p_dnncalib, color=kCyan+2, linewidth=2, markerstyle=23 )

c = TCanvas( "C", "C", 1000, 800 )

p_nocalib.SetMinimum(0.9)
p_nocalib.SetMaximum(1.2)
p_nocalib.GetYaxis().SetTitleOffset( 1.2 )
p_nocalib.GetYaxis().SetTitle( "%s / %s_{truth}" % ( obs, obs ))

p_nocalib.Draw()
p_calib.Draw( "same" )
p_dnncalib.Draw( "same" )

l = TLine()
l.SetLineStyle( kDashed )
l.DrawLine( p_nocalib.GetXaxis().GetXmin(), 1., p_nocalib.GetXaxis().GetXmax(), 1. )
l2 = TLine()
l2.SetLineStyle( kDashed )
l2.SetLineColor( kCyan+2 )
l2.DrawLine( p_nocalib.GetXaxis().GetXmin(), 1.01, p_nocalib.GetXaxis().GetXmax(), 1.01 )
l2.DrawLine( p_nocalib.GetXaxis().GetXmin(), 0.99, p_nocalib.GetXaxis().GetXmax(), 0.99 )

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

txt.DrawLatex( 0.65, 0.90, "Anti-k_{T} R=1.0 jets p_{T} > 250 GeV" )
txt.DrawLatex( 0.65, 0.85, "|#eta| < 2.0" )
imgname = "img/%s_response.png" % ( obs )
c.SaveAs( imgname )
