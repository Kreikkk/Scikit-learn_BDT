import numpy as np


def normalized_hist_to_array(hist, nbins, include_error=True):
	arr = []
	if include_error:
		for index in range(1, nbins+1):
			arr.append(hist.GetBinContent(index) + hist.GetBinErrorUp(index))
	else:
		for index in range(1, nbins+1):
			arr.append(hist.GetBinContent(index))

	sum_of_weights = hist.GetSumOfWeights()

	return np.array(arr)/sum_of_weights


def get_hist_max(hist, nbins, include_error=True):
	arr = normalized_hist_to_array(hist, nbins, include_error)
	return np.max(arr)


def error(signal_events, bg_events):
	S = np.sum(signal_events)
	B = np.sum(bg_events)

	SErr = np.sum(signal_events**2)**0.5
	BErr = np.sum(bg_events**2)**0.5

	SPart = (S + B)**(-0.5) - 0.5*S*((S + B)**(-1.5))
	BPart = -0.5*S*((S + B)**(-1.5))

	return ((SPart*SErr)**2 + (BPart*BErr)**2)**0.5