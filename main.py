from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

import numpy as np

from dataloader import extract

import time
import pickle

from plotters import BDT_output_hist_plot, significance_plot


def dataset_gen(ratio=0.5):
	SDataframe, BDataframe = extract()
	SDataframe = SDataframe.sample(frac=1).reset_index(drop=True)
	BDataframe = BDataframe.sample(frac=1).reset_index(drop=True)

	ratio = 0.5
	STrainLen, BTrainLen = round(ratio*len(SDataframe)), round(ratio*len(BDataframe))

	STrainDF = SDataframe.iloc[:STrainLen]
	BTrainDF = BDataframe.iloc[:BTrainLen]

	STrainSubset = STrainDF.values
	BTrainSubset = BTrainDF.values

	STestDF = SDataframe.iloc[STrainLen:]
	BTestDF = BDataframe.iloc[BTrainLen:]

	STestSubset = STestDF.values
	BTestSubset = BTestDF.values

	TrainSubset = np.vstack((STrainSubset, BTrainSubset))

	return TrainSubset, STestDF, BTestDF, STrainDF, BTrainDF


def train(train_set, filename="output", dump=True):
	train_data = np.array(train_set[:,:11], dtype="float64")
	labels = np.array(train_set[:,14], dtype="float64")

	tree = DecisionTreeClassifier()
	grad = GradientBoostingClassifier(n_estimators=500, random_state=1, max_depth=3)

	t = time.time()
	grad.fit(train_data, labels)
	print(time.time() - t)

	if dump:
		with open(f"{filename}.pickle", "wb") as file:
			pickle.dump(grad, file)


def load(STestDF, BTestDF, filename="output"):
	with open(f"{filename}.pickle", "rb") as file:
		grad = pickle.load(file)

	SResponse = grad.decision_function(STestDF.values[:,:11])
	BResponse = grad.decision_function(BTestDF.values[:,:11])

	STestDF["output"] = SResponse
	BTestDF["output"] = BResponse

	return STestDF, BTestDF


if __name__ == "__main__":
	TrainSubset, STestDF, BTestDF, STrainDF, BTrainDF = dataset_gen()
	train(TrainSubset, filename="output500_bsh")

	STestDF, BTestDF = load(STestDF, BTestDF, filename="output500_bsh")
	BDT_output_hist_plot(STestDF, BTestDF, model_id="test500")
	significance_plot(STestDF, BTestDF, 0.5, model_id="test500")

	STrainDF, BTrainDF = load(STrainDF, BTrainDF, filename="output500_bsh")
	BDT_output_hist_plot(STrainDF, BTrainDF, model_id="train500")
	significance_plot(STrainDF, BTrainDF, 0.5, model_id="train500")
