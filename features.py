# Sana's
header = [
 "jet_Weight",
 "jet_truth_Pt", "jet_truth_Eta", "jet_truth_E", "jet_truth_P", "jet_truth_M",
 "jet_nocalib_Pt", "jet_nocalib_Eta", "jet_nocalib_E", "jet_nocalib_P", "jet_nocalib_M",
 "jet_Nconstit",
 "jet_D2", "jet_C2",
 "jet_Tau1_wta", "jet_Tau2_wta", "jet_Tau3_wta", "jet_Tau21_wta", "jet_Tau32_wta",
 "jet_Angularity", "jet_Aplanarity", "jet_PlanarFlow", "jet_Sphericity",
 "jet_Width",
# "jet_ECF1", "jet_ECF2", "jet_ECF3",
 "jet_calib_Pt", "jet_calib_Eta", "jet_calib_E", "jet_calib_P", "jet_calib_M",
]

features_all = [
 "jet_nocalib_Pt", "jet_nocalib_Eta", "jet_nocalib_E", "jet_nocalib_P", "jet_nocalib_M",
 "jet_Tau1_wta", "jet_Tau2_wta", "jet_Tau3_wta",
# "jet_Tau1_wta", "jet_Tau2_wta", "jet_Tau3_wta", "jet_Tau21_wta", "jet_Tau32_wta",
# "jet_Nconstit",
 "jet_D2", "jet_C2",
# "jet_Angularity", "jet_Aplanarity", "jet_PlanarFlow", "jet_Sphericity",
 "jet_Width",
]

# transverse momentum
features_pT = [
 "jet_nocalib_Pt", "jet_nocalib_Eta", "jet_nocalib_E", "jet_nocalib_P", "jet_nocalib_M",
# "jet_D2", "jet_C2",
# "jet_Tau1_wta", "jet_Tau2_wta", "jet_Tau3_wta",
# "jet_Tau21_wta", "jet_Tau32_wta",
# "jet_Angularity", "jet_Aplanarity", "jet_PlanarFlow", "jet_Sphericity",
# "jet_Width",
  ]

# (pseudo)rapidity
features_eta = [
 "jet_nocalib_Pt", "jet_nocalib_Eta", "jet_nocalib_E", "jet_nocalib_P", "jet_nocalib_M",
 "jet_Nconstit",
 "jet_Width",
]

# energy
features_E  = [
 "jet_nocalib_Pt", "jet_nocalib_Eta", "jet_nocalib_E", "jet_nocalib_P", "jet_nocalib_M",
# "jet_D2", "jet_C2",
# "jet_Tau1_wta", "jet_Tau2_wta", "jet_Tau3_wta", "jet_Tau21_wta", "jet_Tau32_wta",
# "jet_Angularity", "jet_Aplanarity", "jet_PlanarFlow", "jet_Sphericity",
# "jet_Width",
 ]

# mass
features_M  = [
 "jet_nocalib_Pt", "jet_nocalib_Eta", "jet_nocalib_E", "jet_nocalib_P", "jet_nocalib_M",
 "jet_Nconstit",
 "jet_D2", "jet_C2",
 "jet_Tau1_wta", "jet_Tau2_wta", "jet_Tau3_wta", "jet_Tau21_wta", "jet_Tau32_wta",
 "jet_Width",
 ]

n_input_all = len( features_all )
n_input_pT  = len( features_pT )
n_input_eta = len( features_eta )
n_input_E   = len( features_E )
n_input_M   = len( features_M )

print "INFO: N inputs all:", n_input_all
print "INFO: N inputs pT: ", n_input_pT
print "INFO: N inputs eta:", n_input_eta
print "INFO: N inputs E:  ", n_input_E
print "INFO: N inputs M:  ", n_input_M
