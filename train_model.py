# Source of help: https://spacy.io/usage/training/#ner
import spacy
import pprint
import convert_dataturks_to_spacy
import random
from spacy.util import minibatch, compounding
from pathlib import Path

def train_model(iterations):
    train_data = convert_dataturks_to_spacy.convert_dataturks_to_spacy("mail1.json")
    path = Path("ner_model_sm")
    if path.exists():
        nlp = spacy.load(path)
    else:
        nlp = spacy.load('en_core_web_sm')
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    else:
        ner = nlp.get_pipe("ner")

    # add labels
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        for i in range(iterations):
            random.shuffle(train_data)
            losses = {}
            batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, drop=0.5, losses=losses)
            if i % 100 == 0:
                print(i)

    for text, _ in train_data:
        doc = nlp(text)
        print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
        print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])

    if not path.exists():
        path.mkdir()
    nlp.to_disk(path)


if __name__ == "__main__":
    train_model(200)
