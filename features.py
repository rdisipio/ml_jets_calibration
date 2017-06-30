# Sana's / Joe Ennis'

header = [
 "mc_Weight",
 "jet_truth_Pt", "jet_truth_Eta", "jet_truth_E", "jet_truth_P", "jet_truth_M",
 "jet_nocalib_Pt", "jet_nocalib_Eta", "jet_nocalib_E", "jet_nocalib_P", "jet_nocalib_M",
 "jet_track_Pt", "jet_track_Eta", "jet_track_E", "jet_track_P", "jet_track_M",
 "jet_nocalib_m_over_pt", "jet_track_m_over_pt", "jet_nocalib_mTA",
 "jet_nocalib_Nconstit", "jet_nocalib_Nconstit_over_m", "jet_nocalib_width", "jet_nocalib_width_over_m",
 "jet_nocalib_D2", "jet_nocalib_C2",
 "jet_nocalib_Tau1_wta", "jet_nocalib_Tau2_wta", "jet_nocalib_Tau3_wta", "jet_nocalib_Tau21_wta", "jet_nocalib_Tau32_wta",
# "jet_nocalib_Angularity", "jet_nocalib_Aplanarity", "jet_nocalib_PlanarFlow", "jet_nocalib_Sphericity",
 "jet_track_width", "jet_track_width_over_m",
 "jet_track_D2", "jet_track_C2",
 "jet_track_Tau1_wta", "jet_track_Tau2_wta", "jet_track_Tau3_wta", "jet_track_Tau21_wta", "jet_track_Tau32_wta",
 "jet_calib_Pt", "jet_calib_Eta", "jet_calib_E", "jet_calib_P", "jet_calib_M",
]

#features_all = [
# "jet_nocalib_Pt", "jet_nocalib_Eta", "jet_nocalib_E", "jet_nocalib_P", "jet_nocalib_M",
# "jet_track_Pt", "jet_track_Eta", "jet_track_E", "jet_track_P", "jet_track_M",
# "jet_nocalib_m_over_pt", "jet_track_m_over_pt", "jet_nocalib_mTA",
# "jet_nocalib_Nconstit", "jet_nocalib_Nconstit_over_m", "jet_nocalib_width", "jet_nocalib_width_over_m",
# "jet_nocalib_D2", "jet_nocalib_C2",
# "jet_nocalib_Tau1_wta", "jet_nocalib_Tau2_wta", "jet_nocalib_Tau3_wta", "jet_nocalib_Tau21_wta", "jet_nocalib_Tau32_wta",
# "jet_nocalib_Angularity", "jet_nocalib_Aplanarity", "jet_nocalib_PlanarFlow", "jet_nocalib_Sphericity",
#]

features_all = [
 "jet_nocalib_Pt", "jet_nocalib_Eta", "jet_nocalib_E", "jet_nocalib_P", "jet_nocalib_M",
 "jet_nocalib_m_over_pt", 
# "jet_track_Pt", "jet_track_Eta", "jet_track_E", "jet_track_P", "jet_track_M",
  "jet_track_Eta", "jet_track_M",
# "jet_track_Eta", 
 "jet_track_m_over_pt",  "jet_nocalib_mTA",
 "jet_nocalib_Nconstit",        "jet_nocalib_width",
 "jet_nocalib_Nconstit_over_m", "jet_nocalib_width_over_m",
 "jet_nocalib_D2", "jet_nocalib_C2",
 "jet_nocalib_Tau1_wta", "jet_nocalib_Tau2_wta", "jet_nocalib_Tau3_wta",
 "jet_nocalib_Tau21_wta", "jet_nocalib_Tau32_wta",
 "jet_track_width", "jet_track_width_over_m",
 "jet_track_D2", "jet_track_C2",
 "jet_track_Tau1_wta", "jet_track_Tau2_wta", "jet_track_Tau3_wta", "jet_track_Tau21_wta", "jet_track_Tau32_wta",
]

# transverse momentum
features_pT = [
 "jet_nocalib_Pt", "jet_nocalib_Eta", "jet_nocalib_E", "jet_nocalib_P", "jet_nocalib_M",
# "jet_nocalib_Pt", "jet_nocalib_Eta",
# "jet_nocalib_D2",
# "jet_nocalib_C2",
# "jet_nocalib_Tau1_wta", "jet_nocalib_Tau3_wta",
# "jet_nocalib_Tau21_wta", "jet_nocalib_Tau32_wta",
# "jet_track_Pt", "jet_track_Eta",
  "jet_nocalib_Nconstit_over_m", "jet_nocalib_width_over_m",
# "jet_track_Pt", "jet_track_Eta", "jet_track_E", "jet_track_P", "jet_track_M",
 "jet_nocalib_m_over_pt", 
#"jet_track_m_over_pt", "jet_nocalib_mTA",
  ]

# (pseudo)rapidity
features_eta = [
 "jet_nocalib_Pt", "jet_nocalib_Eta", "jet_nocalib_E", "jet_nocalib_P", "jet_nocalib_M",
#"jet_nocalib_Pt", "jet_nocalib_Eta",
# "jet_nocalib_D2", "jet_nocalib_C2",
# "jet_nocalib_Tau21_wta", "jet_nocalib_Tau32_wta",
#"jet_track_Pt", "jet_track_Eta", "jet_track_E",
# "jet_track_Pt", "jet_track_Eta", "jet_track_E", "jet_track_P", "jet_track_M",
  "jet_track_Eta", 
#  "jet_track_Pt", "jet_track_Eta", 
# "jet_nocalib_m_over_pt", "jet_track_m_over_pt", 
#"jet_nocalib_mTA",
]

# energy
features_E  = [
 "jet_nocalib_Pt", "jet_nocalib_Eta", "jet_nocalib_E", "jet_nocalib_P", "jet_nocalib_M",
# "jet_nocalib_Pt", "jet_nocalib_Eta",
 "jet_nocalib_D2",
 "jet_nocalib_C2",
# "jet_nocalib_Tau1_wta", "jet_nocalib_Tau3_wta",
# "jet_nocalib_Tau21_wta", "jet_nocalib_Tau32_wta",
# "jet_nocalib_Angularity", "jet_nocalib_Aplanarity", "jet_nocalib_PlanarFlow", "jet_nocalib_Sphericity",
# "jet_track_Pt", "jet_track_Eta",
  "jet_nocalib_Nconstit",
  "jet_track_Pt", "jet_track_Eta", "jet_track_E",
# "jet_track_Pt", "jet_track_Eta", "jet_track_E", #"jet_track_P", "jet_track_M",
 "jet_nocalib_m_over_pt", 
# "jet_track_m_over_pt", "jet_nocalib_mTA",
 ]

# mass
features_M  = [
 "jet_nocalib_Pt", "jet_nocalib_Eta", "jet_nocalib_E", "jet_nocalib_P", "jet_nocalib_M",
# "jet_nocalib_Pt", "jet_nocalib_Eta",
 "jet_nocalib_D2",
 "jet_nocalib_C2",
 "jet_nocalib_Tau1_wta", "jet_nocalib_Tau3_wta",
# "jet_nocalib_Tau21_wta", "jet_nocalib_Tau32_wta",
# "jet_nocalib_Angularity", "jet_nocalib_Aplanarity", "jet_nocalib_PlanarFlow", "jet_nocalib_Sphericity",
# "jet_track_Pt", #"jet_track_Eta",
# "jet_nocalib_Nconstit", "jet_nocalib_width",
  "jet_nocalib_Nconstit_over_m", "jet_nocalib_width_over_m",
# "jet_nocalib_width", "jet_nocalib_width_over_m", "jet_nocalib_m_over_pt", "jet_nocalib_mTA",
  "jet_nocalib_mTA",
# "jet_track_Pt", "jet_track_Eta", "jet_track_E", "jet_track_P", "jet_track_M",
# "jet_track_m_over_pt", 
 ]

print "INFO: all input features:"
print features_all

y_features_nocalib = [ "jet_nocalib_Pt", "jet_nocalib_Eta", "jet_nocalib_E", "jet_nocalib_M" ]
y_features_truth   = [ "jet_truth_Pt", "jet_truth_Eta", "jet_truth_E", "jet_truth_M" ] 
y_features_calib   = [ "jet_calib_Pt", "jet_calib_Eta", "jet_calib_E", "jet_calib_M" ]

n_input_all = len( features_all )
n_input_pT  = len( features_pT )
n_input_eta = len( features_eta )
n_input_E   = len( features_E )
n_input_M   = len( features_M )

print "INFO: N inputs all:", n_input_all
