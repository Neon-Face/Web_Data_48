import re
import requests
from difflib import SequenceMatcher
from timeit import default_timer as timer

import nltk
import spacy
nlp = spacy.load("en_core_web_sm")

from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L3-v2')

def search_wikidata(entity):
	url = "https://www.wikidata.org/w/api.php"
	params = {
		"action": "wbsearchentities",
		"language": "en",
		"format": "json",
		"search": entity
	}

	response = requests.get(url, params=params)
	if response.status_code == 200:
		data = response.json()
		if data["search"]:
			return [(d["id"], d["label"], "" if d.get("description") is None else d["description"]) for d in data["search"]]
	return None

def extract_questions(data: str) -> list[str]:
	questions = []
	for line in data.split('\n'):
		parts = line.split('\t')
		if len(parts) > 1:
			question_text = parts[1].strip()

			question_text = re.sub(r'^Question:\s*', '', question_text)
			question_text = re.sub(r'Answer:\s*$', '', question_text)

			if question_text:
				questions.append((parts[0].strip(), question_text))

	return questions

def extract_entities(text: str, debug: bool = False) -> list[tuple[str, str]]:
	start = timer()
	doc = nlp(text)
	base_entities = [(ent.text, ent.label_) for ent in doc.ents]
	entity_names = [entity[0] for entity in base_entities]
	entities = []

	base_entities = list({t[0]: t for t in base_entities}.values())
	ENTITY_LABELS = ["GPE", "ORGANIZATION", "ORG", "PERSON", "LOCATION"]

	for entity in base_entities:
		if entity[1] in ENTITY_LABELS:
			entities.append(entity)

	ne_tree = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(text)))
	for tree in ne_tree:
		if isinstance(tree, nltk.tree.Tree):
			name = " ".join([t[0] for t in tree])
			if tree[0] in ENTITY_LABELS and name not in entity_names:
				entities.append((name, tree.label()))
		elif tree[1] in ENTITY_LABELS and tree[0] not in entity_names:
			entities.append((tree[0], tree[1]))
	end = timer()

	if debug:
		print(f"NER time: {end - start}")
		print("Entities:", entities)
	return entities

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def disambiguate(entity: str, question: str, debug: bool = False) -> tuple[str, str, str]:
	start = timer()
	wikidata_entities = search_wikidata(entity)

	if wikidata_entities is None:
		raise ValueError(f"Entity {entity} not found in Wikidata")

	inputs = []
	if debug:
		print("Wikidata Entities:", wikidata_entities)

	for e in wikidata_entities:
		inputs.append(f"Entity: {e[1]}. Description: {e[2]}")	

	entity_embeddings = model.encode(inputs, convert_to_tensor=True)
	query_embedding = model.encode(question, convert_to_tensor=True)

	cosine_scores = util.cos_sim(query_embedding, entity_embeddings)
	entity_ids = [int(e[0][1:]) for e in wikidata_entities]
	normalized_scores = [1 / (float(i)/sum(entity_ids)) for i in entity_ids]
	normalized_scores = [float(i)/sum(normalized_scores) for i in normalized_scores]

	cosine_scores = cosine_scores[0].tolist()	
	for i in range(len(cosine_scores)):
		cosine_scores[i] = (cosine_scores[i] * 0.8) + (normalized_scores[i] * 0.1) + similarity(wikidata_entities[i][2], question) * 0.1
	best_match_idx = cosine_scores.index(max(cosine_scores))

	end = timer()
	if debug:
		print(f"Disambigation time: {end - start}")

	return wikidata_entities[best_match_idx]

def get_wikidata_document(wikidata_id):
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{wikidata_id}.json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data
    return None

def get_wikipedia_url(entity_id: str, debug: bool = False) -> str:
	wikidata_doc = get_wikidata_document(entity_id)
	entity_data = wikidata_doc['entities'][entity_id]
	return entity_data.get('sitelinks', {}).get('enwiki', {}).get('url', '')