#!/usr/bin/env python

import os,sys

from ROOT import *

gROOT.LoadMacro("AtlasStyle.C")
#gROOT.LoadMacro( "AtlasUtils.C" )
SetAtlasStyle()

gROOT.SetBatch(1)

# change this to increase the number of eta slices
#etabins  = [ [0.0, 0.1], [0.1,0.2], [0.2,0.3], [0.3,0.4], [0.4,0.5], [0.5,0.6], [0.6,0.7], [0.7,0.8], [0.8,0.9], [0.9,1.0], [1.0,1.5],[1.5,2.0] ]
etabins  = [ [0.0, 0.2], [0.2, 0.5], [0.5, 1.0], [1.0,2.0] ]
ptbins   = [ [250., 350.], [350., 500.], [500.,600.], [600.,800.], [800.,1000.], [1000., 2000.] ]
Ebins    = [ [0., 200.], [200.,400.], [400.,600.], [600.,800.], [800.,1000.],[1000.,2000.] ]
massbins = [ [30., 50.], [ 50., 110.], [110.,140], [140,200], [200,300] ]
#                LOW          W/Z            H          t       QCD          

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
  l.DrawLatex(x+0.14,y,"Simulation Internal");
  #l.DrawLatex(x+0.14,y,"Preliminary")
  l.SetTextSize(0.04)
#  s = "#sqrt{s} = 13 TeV, %2.1f fb^{-1}" % (lumi/1000.)
  s = "#sqrt{s} = 13 TeV"
  l.DrawLatex(x, y-0.05, s )


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

obs = "E"
if len( sys.argv) > 1:
  obs = sys.argv[1]

region = "etabin_0"
if len( sys.argv) > 2:
  region = sys.argv[2]

infilename = "dnn.root"
if len( sys.argv ) > 3:
   infilename = sys.argv[3]
infile = TFile.Open( infilename )


h_calib    = infile.Get( "%s_response_calib_%s" % ( obs, region ) )
h_nocalib  = infile.Get( "%s_response_nocalib_%s" % ( obs, region ) )
h_dnncalib = infile.Get( "%s_response_dnncalib_%s" % ( obs, region ) )

if h_calib == None: print "ERROR: invalid response histogram for", obs, region

#h_calib.Rebin2D(2,2)
#h_nocalib.Rebin2D(2,2)
#h_dnncalib.Rebin2D(2,2)

p_calib    = h_calib.ProfileX( "pfx_atlascalib" )
p_nocalib  = h_nocalib.ProfileX( "pfx_nocalib" )
p_dnncalib = h_dnncalib.ProfileX( "pfx_dnncalib" )

SetTH1FStyle( p_calib,    color=kRed, linewidth=2, markerstyle=22 )
SetTH1FStyle( p_nocalib,  color=kBlack, linewidth=2, markerstyle=20 )
SetTH1FStyle( p_dnncalib, color=kCyan+2, linewidth=2, markerstyle=23 )

c = TCanvas( "C", "C", 1000, 800 )

p_nocalib.SetMinimum(0.85)
p_nocalib.SetMaximum(1.15)
if obs == "M": p_nocalib.SetMaximum(1.4)

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
l2.SetLineColor( kGreen-2 )
l2.DrawLine( p_nocalib.GetXaxis().GetXmin(), 1.01, p_nocalib.GetXaxis().GetXmax(), 1.01 )
l2.DrawLine( p_nocalib.GetXaxis().GetXmin(), 0.99, p_nocalib.GetXaxis().GetXmax(), 0.99 )


#gPad.SetLogx()
#p_nocalib.GetXaxis().SetMoreLogLabels()
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
leg.AddEntry( p_calib,     "ATLAS-calibrated",   "lp" )
leg.AddEntry( p_dnncalib,  "DNN-calibrated", "lp" )
leg.Draw()
leg.SetY1( leg.GetY1() - lparams['height'] * leg.GetNRows() )

PrintATLASLabel( 0.15, 0.90 )

txt = TLatex()
txt.SetNDC()
txt.SetTextSize(0.03)
txt.SetTextFont(42)

#txt.DrawLatex( 0.65, 0.90, "C/A R=0.4 jets p_{T} > 20 GeV" )
txt.DrawLatex( 0.65, 0.90, "Anti-k_{T} R=1.0 jets p_{T} > 250 GeV" )

if region.split("_")[0] == "etabin":
  ibin = int( region.split("_")[1] )
  etamin = etabins[ibin][0]
  etamax = etabins[ibin][1]
  txt.DrawLatex( 0.65, 0.85, "%2.1f #leq |#eta| #leq %2.1f" % (etamin,etamax) )
elif region.split("_")[0] == "ptbin":
  ibin = int( region.split("_")[1] )
  ptmin = ptbins[ibin][0]
  ptmax = ptbins[ibin][1]
  txt.DrawLatex( 0.65, 0.85, "%4i < p_{T} < %4i" % ( ptmin, ptmax ) )
elif region.split("_")[0] == "massbin":
  ibin = int( region.split("_")[1] )
  Mmin = massbins[ibin][0]
  Mmax = massbins[ibin][1]
  txt.DrawLatex( 0.65, 0.85, "%4i < M < %4i" % ( Mmin, Mmax ) )
else:
  pass

imgname = "img/%s_response_%s.png" % ( obs, region )
c.SaveAs( imgname )
