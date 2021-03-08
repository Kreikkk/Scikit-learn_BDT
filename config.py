SFILENAME	= "ZgEWK.root"
BFILENAMES	= ["ZgQCD.root", "ttgamma.root", "SinglePhoton.root", "WenuDataDriven.root",
			   "Wgam.root", "WgamEWK.root", "Zllgam.root", "ZnunuFromQcd.root"]

TREENAME = "TMVA_input"

from ROOT import TString as string

METHODNAME = string("BDTgrad")
WEIGHTSPATH = string("dataloader/weights/TMVAClassification_BDTG.weights.xml")

BDTSTEP = 0.005
