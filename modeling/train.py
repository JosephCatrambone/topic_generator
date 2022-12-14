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
MAX_INPUT_LENGTH = 400  # Characters, not tokens.
PROMPT_PREFIX = "keywords for text: "
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')  # If this changes, be sure to update the Dockerfile.
_serving_model = None  # A lazy global which should, in general, be used only by predict.


def predict(input_batch: List[str]) -> List[str]:
	"""
	Perform a full inference pass, including the tokenization and preprocessing of inputs,
	pass all examples through the model, and produce an output.
	This is a convenience method that we want to export for use in the main application.
	:param model:
	:param input_batch:
	:return:
	"""
	global _serving_model
	if _serving_model is None:
		_serving_model = T5ForConditionalGeneration.from_pretrained("./modeling/trained_model", local_files_only=True)
	inputs = transform_and_preprocess(input_batch, add_prompt=True)
	outputs = _serving_model.generate(
		input_ids=inputs.input_ids,
		attention_mask=inputs.attention_mask,
	)
	#return [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
	return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def build_model(restore: bool = False) -> T5ForConditionalGeneration:
	"""
	Restore our model from an automatically detected checkpoint or start fresh.
	:return: A model for conditional generation.
	"""
	model = T5ForConditionalGeneration.from_pretrained('google/byt5-small')
	# TODO: Restore from checkpoint.
	return model


def transform_and_preprocess(input_batch: List[str], add_prompt: bool) -> torch.Tensor:
	"""
	Perform the required preprocessing on an input example and return a tensor for use in the model.
	Strictly speaking for T5, this isn't really a requirement, but for models with other kinds of tokenizers this is
	more important.
	"""
	# If we weren't using the auto-tokenizer to pad, we would do this:
	# return torch.tensor([list(txt.encode("utf-8")) for txt in input_batch]) + 3
	# The +3 broadcasts and offsets all tokens by three.  The values 0, 1, and 2 are reserved.
	prefix = ""
	if add_prompt:
		prefix = PROMPT_PREFIX
	input_batch = [prefix + x[:MAX_INPUT_LENGTH] for x in input_batch]  # Trim everything to a capped length to avoid OOM.
	return tokenizer(input_batch, padding="longest", return_tensors="pt")


def train_model(
		model,
		training_data_pairs: List[Tuple[str, str]],
		batch_size: int,
		num_epochs: int,
		learning_rate: float,
		validation_percent: float,
		log_out,
		**kwargs,  # Leave this in so we have a sink for extra arguments that might show up in our config.
) -> None:
	"""
	We're using a more traditional Torch-style training system rather than the built-in HuggingFace trainer because we
	want a finer-grained control over the learning rate and curriculum.  We can also
	:return:
	"""
	model.to(device)
	shuffle(training_data_pairs)
	train_data = training_data_pairs[:int(len(training_data_pairs)*validation_percent)]
	validation_data = training_data_pairs[-int(len(training_data_pairs) * validation_percent):]
	# Shuffle our training data and make a validation set from it.
	optimizer = AdamW(model.parameters(), lr=learning_rate)
	for epoch in range(num_epochs):
		shuffle(train_data)
		model.train()
		for offset in range(0, len(train_data), batch_size):
			text = [x[0] for x in train_data[offset:offset+batch_size]]
			target = [x[1] for x in train_data[offset:offset+batch_size]]
			text = transform_and_preprocess(text, add_prompt=True).input_ids.to(device)
			target = transform_and_preprocess(target, add_prompt=False).input_ids.to(device)
			loss = model(text, labels=target).loss
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()
		model.save_pretrained(f"./modeling/checkpoint_{epoch}")
		model.eval()
		loss_accumulator = 0.0
		for offset in range(0, len(validation_data), batch_size):
			text = [x[0] for x in train_data[offset:offset + batch_size]]
			target = [x[1] for x in train_data[offset:offset + batch_size]]
			text = transform_and_preprocess(text, add_prompt=True).input_ids.to(device)
			target = transform_and_preprocess(target, add_prompt=False).input_ids.to(device)
			loss = model(text, labels=target).loss
			loss_accumulator += loss.data
		loss_accumulator /= (len(validation_data)/float(batch_size))
		print(f"Epoch {epoch}: {loss_accumulator}")
		log_out.write(f"Epoch {epoch}: {loss_accumulator}\n")


def load_data(dataset: str) -> List[Tuple[str, str]]:
	# Small disclaimer: the data has these fields: url,topics,title,text,date,text_len,num_topics,text_ending,extractive
	# We expect that topics, title, and text will be present.
	# For now we use only text and topics, but we could try title or others.

	# TODO: We should use the dataset loader and stream the data, but I'm being a little lazy.
	# It's only 50 megs.  What's the worst that could happen?
	# Narrator: he had no idea what was about to happen.
	data = list()
	source_header = "title"
	label_header = "topics"
	with open(dataset, 'rt', encoding="utf-8", errors="ignore") as fin:
		cin = csv.DictReader(fin)
		for row in cin:
			if all((source_header in row, label_header in row, row[source_header], row[label_header])):
				data.append((row[source_header], row[label_header]))
	return data


def main():
	training_configuration = {
		"dataset": "./data/topic_data.csv",
		"batch_size": 8,
		"num_epochs": 20,
		"learning_rate": 1e-3,
		"prompt": PROMPT_PREFIX,
		"max_input_length": MAX_INPUT_LENGTH,
		"validation_percent": 0.1,
		"notes": "Had to decrease the batch size because I keep getting OOM.  Going from 16->8."
	}

	# Log the last run before we start:
	run_log = open(f"run_log_{len(glob('run_log*.log'))}.log", 'wt')  # Hacky auto-increment of run_log_0, run_log_1...
	run_log.write(json.dumps(training_configuration) + "\n")

	data = load_data(training_configuration["dataset"])
	model = build_model()
	train_model(model, data, log_out=run_log, **training_configuration)
	model.save_pretrained("./modeling/trained_model")


if __name__=="__main__":
	main()
