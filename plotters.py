import atlasplots as aplt
import numpy as np
import pandas as pd
import ROOT as root

from array import array

from config import *
from dataloader import extract_from_output
from helpers import get_hist_max, error
import math

def BDT_output_hist_plot(SDataframe, BDataframe, model_id="", output_name="output"):
	root.TMVA.Tools.Instance()
	aplt.set_atlas_style()
	Min = np.max((np.min(SDataframe[output_name]), np.min(BDataframe[output_name])))
	Max = np.min((np.max(SDataframe[output_name]), np.max(BDataframe[output_name])))

	# Min =-math.ceil(abs(Min))
	# Max = math.ceil(Max)

	SHist = root.TH1F("", "", 50, Min, Max)
	BHist = root.TH1F("", "", 50, Min, Max)

	SHist.SetStats(False)
	BHist.SetStats(False)
	SHist.SetMinimum(0)
	BHist.SetMinimum(0)
	SHist.SetLineWidth(2)
	BHist.SetLineWidth(2)
	SHist.SetLineColor(4)
	BHist.SetLineColor(2)
	SHist.SetMarkerSize(0)
	BHist.SetMarkerSize(0)
	
	SHist.SetFillColorAlpha(4, 0.2)
	BHist.SetFillColor(2)
	BHist.SetFillStyle(3004)

	for out, weight in zip(SDataframe[output_name], SDataframe["weightModified"]):
		SHist.Fill(out, weight)
	for out, weight in zip(BDataframe[output_name], BDataframe["weightModified"]):
		BHist.Fill(out, weight)

	hists_max = np.max((get_hist_max(SHist, 50),
						get_hist_max(BHist, 50)))
	margins = [Min, Max, 0, hists_max]

	fig, ax = aplt.subplots(1, 1, name="", figsize=(800, 600))
	ax.plot(SHist, margins, "E1")
	ax.plot(BHist, margins, "E1")
	ax.add_margins(top=0.1)

	ax.set_xlabel(f"BDTgrad_{model_id} classifier response")
	ax.set_ylabel("Fraction of events")
	ax.text(0.2, 0.92, "#sqrt{s} = 13 TeV, 139 fb^{-1}", size=27, align=13)

	legend = root.TLegend(0.65, 0.8, 0.95, 0.92)
	legend.SetFillColorAlpha(0, 0)
	legend.AddEntry(SHist, "Signal", "F")
	legend.AddEntry(BHist, "Background", "F")
	legend.Draw()

	fig.savefig(f"BDTgrad_{model_id}_output.pdf")


def significance_plot(SDataframe, BDataframe, ratio, ndots=1000, model_id="", output_name="output", ROC=True):
	root.TMVA.Tools.Instance()
	aplt.set_atlas_style()
	Min = np.max((np.min(SDataframe[output_name]), np.min(BDataframe[output_name])))
	Max = np.min((np.max(SDataframe[output_name]), np.max(BDataframe[output_name])))
	SWSum = np.sum(SDataframe["weightModified"])
	BWSum = np.sum(BDataframe["weightModified"])

	XData = np.linspace(Min, Max, ndots)
	SData, BData = np.array([]), np.array([])
	for cursor in XData:
		S = np.sum(SDataframe[SDataframe[output_name] >= cursor]["weightModified"])
		B = np.sum(BDataframe[BDataframe[output_name] >= cursor]["weightModified"])
		SData = np.append(SData, S)
		BData = np.append(BData, B)

	YData = SData/np.sqrt(ratio*(SData+BData))
	SEff = SData/SWSum
	BRej = 1 - BData/BWSum

	XPlot, YPlot = array("d"), array("d")
	for x, y in zip(XData, YData):
		XPlot.append(x)
		YPlot.append(y)

	curve = root.TGraph(ndots, XPlot, YPlot)
	curve.SetLineColor(2)

	curve.SetLineWidth(2)
	curve.SetMarkerColor(2)
	curve.SetMarkerSize(0)
	curve.GetXaxis().SetRangeUser(Min, Max)
	curve.GetXaxis().SetTitle(f"Cut value applied on BDTgrad{model_id} output")
	curve.GetYaxis().SetTitle('Significance')

	fig, ax = aplt.subplots(1, 1, name="", figsize=(800, 600))
	ax.plot(curve)

	ax.add_margins(top=0.16)

	peak_index = YData.argmax()
	cut = XData[peak_index]
	sig_max = YData[peak_index]

	ax.text(0.2, 0.92, "#sqrt{s} = 13 TeV, 139 fb^{-1}", size=27, align=13)

	text1 = "For {} signal and {} background".\
		format(round(np.sum(SDataframe["weightModified"])),
			   round(np.sum(BDataframe["weightModified"])))

	text2 = "events the maximum {} is".format("S/#sqrt{S+B}")
	text3 = "{} when cutting at {}".format(round(sig_max, 3),
										   round(cut, 3))

	ax.text(0.2, 0.3, text1, size=20, align=13)
	ax.text(0.2, 0.27, text2, size=20, align=13)
	ax.text(0.2, 0.22, text3, size=20, align=13)

	line = root.TLine(cut, 0, cut, 4)
	line.SetLineStyle(10)
	line.SetLineColor(6)
	ax.plot(line)

	fig.savefig(f"BDTgrad_{model_id}_outputCut.pdf")

	SWCut = np.array(SDataframe[SDataframe[output_name] > cut]["weightModified"])
	BWCut = np.array(BDataframe[BDataframe[output_name] > cut]["weightModified"])
	print("Significance error:")
	print(error(SWCut, BWCut)/np.sqrt(ratio))	### Нужно ли делить на sqrt(ratio)?

	if ROC:
		ROC_plot(SEff, BRej, model_id=model_id)


def ROC_plot(SEff, BRej, model_id=""):
	root.TMVA.Tools.Instance()
	aplt.set_atlas_style()
	area = 0
	for index, xval in enumerate(SEff[:-1]):
		delta = SEff[index+1] - xval
		area -= delta*BRej[index]

	XPlot, YPlot = array("d"), array("d")
	for x, y in zip(SEff, BRej):
		XPlot.append(x)
		YPlot.append(y)

	ndots = len(SEff)

	fig, ax = aplt.subplots(1, 1, name="fig", figsize=(800, 800))

	curve = root.TGraph(ndots, XPlot, YPlot)
	curve.SetLineColor(4)
	curve.SetLineWidth(2)
	curve.GetXaxis().SetTitle(f"BDTgrad_{model_id} signal efficiency")
	curve.GetYaxis().SetTitle(f"BDTgrad_{model_id} background rejection")
	curve.SetMarkerSize(0)

	ax.plot(curve)

	ax.text(0.19, 0.92, "#sqrt{s} = 13 TeV, 139 fb^{-1}", size=27, align=13)
	ax.text(0.195, 0.84, f"Area under ROC-curve: {round(area, 3)}")

	fig.savefig(f"BDTgrad_{model_id}_ROC_curve.pdf")