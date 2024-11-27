import nltk
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')
nltk.download('punkt')

import spacy
nlp = spacy.load("en_core_web_trf")

import os
import argparse
import wikidata
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import numpy as np
from llama_cpp import Llama
from timeit import default_timer as timer
model_path = "models/llama-2-7b.Q4_K_M.gguf"

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="Model path", required=False)
parser.add_argument("--gpu", action="store_true", help="Run with nvidia GPU", required=False)
parser.add_argument("--input", type=str, help="Input question. Either a string or a file path", required=True)
parser.add_argument("--debug", action="store_true", help="Debug mode", required=False)

args = parser.parse_args()

if args.model is not None and len(args.model) > 0:
    model_path = args.model

if os.path.exists(args.input):
    with open(args.input, "r") as f:
        args.input = f.read()

start = timer()
llm = Llama(model_path=model_path, verbose=False, n_gpu_layers=-1 if args.gpu else 0)
output = llm(args.input, max_tokens=32, stop=["Q:", "Question:", "Context:"], echo=False)
end = timer()

output_txt = output['choices'][0]['text']
print(output_txt)
if args.debug:
    print(f"Question LLM time: {end - start}")

doc = nlp(output_txt)
base_entities = [(ent.text, ent.label_) for ent in doc.ents]
entity_names = [entity[0] for entity in base_entities]
entities = []

ENTITY_LABELS = ["GPE", "NNP", "ORGANIZATION", "PERSON", "LOCATION"]

for entity in base_entities:
    if entity[1] in ENTITY_LABELS:
        entities.append(entity)

ne_tree = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(output_txt)))
for tree in ne_tree:
    if isinstance(tree, nltk.tree.Tree):
        name = " ".join([t[0] for t in tree])
        if tree[0] in ENTITY_LABELS and name not in entity_names:
            base_entities.append((name, tree.label()))
    elif tree[1] in ENTITY_LABELS and tree[0] not in entity_names:
        base_entities.append((tree[0], tree[1]))

if args.debug:
    print("Entities:", entities)

def disambiguate(context, ambiguous_term, labels):
    doc = nlp(context)
    for entity in doc.ents:
        if entity.text == ambiguous_term:
            scores = {label: doc.similarity(nlp(label)) for label in labels}
            return max(scores, key=scores.get)
for entity in entities:
    wikidata_entities = wikidata.get_wikidata_id(entity[0])
    # for item in wikidata_entities:
    #     if item[1].lower() == entity[0].lower():
    #         print(f"Entity: {entity[0]} - Wikidata ID: {item[0]} {item[2]}")
    #     else:
    #         print(f"Entity: {entity[0]} - Wikidata ID: {item[0]} {item[2]} (Name mismatch)")

    # print(list(map(lambda x: x[1], wikidata_entities)))
    # ranked_entities = rank_candidates(args.input + " " + output_txt, list(map(lambda x: x[2], wikidata_entities)))
    # print(ranked_entities)
    # print("Ranked Candidates:")
    # for candidate, score in ranked_entities:
    #     print(f"{candidate}: {score}")

    disambiguated_entity = disambiguate(output_txt, entity[0], list(map(lambda x: x[1], wikidata_entities)))

    print(f"Disambiguated Entity: {disambiguated_entity}")

