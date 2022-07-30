"""
Define and fine tune a ByT5 model which takes a bunch of strings and outputs the keywords for each document.
Input is a CSV where each row has a title, document, and additional triplet of comma-separated keywords.
The comma-separated keywords should be embedded as a single CSV column and should not be a series of additional columns.

Uses transformers and ByT5 as a basis, then fine-tunes on top.

Stored models should be pickled and saved in this directory so that it can be copied and loaded into the app.
"""
import csv
import json
from glob import glob
from random import shuffle
from typing import List, Tuple

import torch
from transformers import AutoTokenizer
from transformers import T5ForConditionalGeneration
from torch.optim import AdamW

# Some slightly gross global-state things.
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')


def build_model(restore: bool = False) -> T5ForConditionalGeneration:
	"""
	Restore our model from an automatically detected checkpoint or start fresh.
	:return: A model for conditional generation.
	"""
	model = T5ForConditionalGeneration.from_pretrained('google/byt5-small')
	# TODO: Restore from checkpoint.
	return model


def transform_and_preprocess(input_batch: List[str]) -> torch.Tensor:
	"""
	Perform the required preprocessing on an input example and return a tensor for use in the model.
	Strictly speaking for T5, this isn't really a requirement, but for models with other kinds of tokenizers this is
	more important.
	"""
	# If we weren't using the auto-tokenizer to pad, we would do this:
	# return torch.tensor([list(txt.encode("utf-8")) for txt in input_batch]) + 3
	# The +3 broadcasts and offsets all tokens by three.  The values 0, 1, and 2 are reserved.
	return tokenizer(input_batch, padding="longest", return_tensors="pt").input_ids


def train_model(
		model,
		training_data_pairs: List[Tuple[str, str]],
		batch_size: int,
		num_epochs: int,
		learning_rate: float,
		**kwargs,  # Leave this in so we have a sink for extra arguments that might show up in our config.
):
	"""
	We're using a more traditional Torch-style training system rather than the built-in HuggingFace trainer because we
	want a finer-grained control over the learning rate and curriculum.  We can also
	:param model:
	:param training_data_pairs:
	:param batch_size:
	:param num_epochs:
	:param learning_rate:
	:return:
	"""
	model.to(device)
	model.train()
	# Shuffle our training data and make a validation set from it.
	optimizer = AdamW(model.parameters(), lr=learning_rate)
	for epoch in range(num_epochs):
		shuffle(training_data_pairs)
		for offset in range(0, len(training_data_pairs), batch_size):
			train_x = [x[0] for x in training_data_pairs[offset:offset+batch_size]]
			train_y = [x[1] for x in training_data_pairs[offset:offset+batch_size]]
			train_x = transform_and_preprocess(train_x).to(device)
			train_y = transform_and_preprocess(train_y).to(device)
			loss = model(train_x, labels=train_y).loss
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()


def load_data(dataset: str) -> List[Tuple[str, str]]:
	# Small disclaimer: the huffpo data has these fields: url,topics,title,text,date,text_len,num_topics,text_ending,extractive
	# We expect that topics, title, and text will be present.
	# For now we use only text and topics, but we could try title or others.

	# TODO: We should use the dataset loader and stream the data, but I'm being a little lazy.
	# It's only 50 megs.  What's the worst that could happen?
	# Narrator: he had no idea what was about to happen.
	data = list()
	source_header = "text"
	label_header = "topics"
	with open(dataset, 'rt', encoding="utf-8", errors="ignore") as fin:
		cin = csv.DictReader(fin)
		for row in cin:
			if all((source_header in row, label_header in row, row[source_header], row[label_header])):
				data.append((row[source_header], row[label_header]))
	return data


def main():
	training_configuration = {
		"dataset": "../data/topic_data.csv",
		"batch_size": 16,
		"num_epochs": 10,
		"learning_rate": 1e-5,
	}

	# Log the last run before we start:
	run_log = open(f"run_log_{len(glob('run_log*.log'))}.log", 'wt')  # Hacky auto-increment of run_log_0, run_log_1...
	run_log.write(json.dumps(training_configuration) + "\n")

	data = load_data(training_configuration["dataset"])
	model = build_model()
	train_model(model, data, **training_configuration)
	model.save_pretrained("./trained_model")

if __name__=="__main__":
	main()