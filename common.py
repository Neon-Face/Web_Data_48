from collections import defaultdict
import re
import requests
from difflib import SequenceMatcher
from timeit import default_timer as timer
from sklearn.feature_extraction.text import TfidfVectorizer
from num2words import num2words
from spellchecker import SpellChecker
from thefuzz import fuzz

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
			res = []
			sitelinks = get_sitelinks([d["id"] for d in data["search"]])
			for d in data["search"]:
				res.append({
					"id": d["id"],
					"label": d["label"],
					"description": "" if d.get("description") is None else d["description"],
					"aliases": [] if d.get("aliases") is None else d["aliases"],
					"sitelinks": sitelinks[d["id"]]
				})
			return res
	return None

def get_sitelinks(ids: list[str]):
	url = "https://www.wikidata.org/w/api.php"
	params = {
		"action": "wbgetentities",
		"format": "json",
		"ids": "|".join(ids),
		"props": "sitelinks/urls"
	}

	response = requests.get(url, params=params)
	if response.status_code == 200:
		data = response.json()
		return data["entities"]
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
	entities = []

	base_entities = list({t[0]: t for t in base_entities}.values())
	ENTITY_LABELS = ["GPE", "NNP", "ORGANIZATION", "ORG", "PERSON", "LOCATION"]

	entity_names = []
	for entity in base_entities:
		if entity[1] in ENTITY_LABELS:
			entities.append(entity)
			entity_names.append(entity[0])

	ne_tree = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(text)))
	for tree in ne_tree:
		if isinstance(tree, nltk.tree.Tree):
			name = " ".join([t[0] for t in tree])
			if tree[0][1] in ENTITY_LABELS and name not in entity_names:
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

def disambiguate_with_model(entity: str, question: str, wikidata):
	start = timer()
	inputs = []
	for e in wikidata:
		label = e["label"]
		description = e["description"]
		inputs.append(f"Entity: {label}. Description: {description}")	

	entity_embeddings = model.encode(inputs, convert_to_tensor=True)
	query_embedding = model.encode(entity, convert_to_tensor=True)
	cosine_scores = util.cos_sim(query_embedding, entity_embeddings)

	cosine_scores = cosine_scores[0].tolist()	
	for i in range(len(cosine_scores)):
		cosine_scores[i] = (cosine_scores[i] * 0.8) + (len(wikidata[i]["sitelinks"]["sitelinks"]) * 0.1) + similarity(wikidata[i]["description"], question) * 0.1
	best_match_idx = cosine_scores.index(max(cosine_scores))

	return wikidata[best_match_idx]

def get_wikidata_document(wikidata_id):
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{wikidata_id}.json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data
    return None

def get_wikipedia_url(entity_id: str, wikidata) -> str:
	wikipedia_url = ""
	for w in wikidata:
		if w["id"] == entity_id:
			try:
				wikipedia_url = w["sitelinks"]["sitelinks"]["enwiki"]["url"]
			except:
				wikipedia_url = ""
			return wikipedia_url
	return ""

spell = SpellChecker()

CONTRACTION_MAP = {
    "can't": "can not",
    "won't": "will not",
    "i'm": "i am",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "they're": "they are",
    "we're": "we are",
    "we've":"we have",
    "i've":"i have",
    "you've":"you have"
}

keep_words = {"not","cannot"}

def expand_contractions(text):
    for contraction, expanded in CONTRACTION_MAP.items():
        text = re.sub(r'\b' + re.escape(contraction) + r'\b', expanded, text, flags=re.IGNORECASE)
    return text

def n2w(token_text):
    try:
        if re.match(r'^\d+(st|nd|rd|th)$', token_text):
            number = re.sub(r'(st|nd|rd|th)$', '', token_text)
            return num2words(int(number), to='ordinal')
        return num2words(float(token_text))
    except (ValueError, TypeError):
        return token_text

def correct_spelling(text):
    corrected_words = []
    doc = nlp(text)
    for token in doc:
        if token.ent_type_ or token.pos_ == "PROPN":
            corrected_words.append(token.text)
        else:
            corrected_word = spell.correction(token.text) 
            corrected_words.append(corrected_word if corrected_word else token.text)
    return " ".join(corrected_words)

def preprocess_text(text):
	text = text.replace("‘", "'").replace("’", "'").replace("%"," percentage")
	text = expand_contractions(text)
	text = correct_spelling(text)
	doc = nlp(text)
	processed_tokens = []
	for token in doc:
        # token_text = token.lemma_
		token_text = n2w(token.text) if token.like_num else token.lemma_
		if (not token.is_stop or token.text.lower() in keep_words
            and not token.is_punct
            and not token.is_space
            and len(token_text) >= 1):
			processed_tokens.append(token_text)

	processed_text = " ".join(processed_tokens)
	processed_text = re.sub(r'[^a-zA-Z0-9\s]', '', processed_text)
	processed_text = processed_text.strip()
	return processed_text

def compute_string_similarity(mention, candidate):
    scores = [fuzz.ratio(mention.lower(), candidate["label"].lower())]
    scores += [fuzz.ratio(mention.lower(), alias.lower()) for alias in candidate.get("aliases", [])]
    return max(scores)

def compute_context_similarity(context, candidate):
    documents = [context, candidate["description"]]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    similarity = (tfidf_matrix[0] * tfidf_matrix[1].T).data.sum()
    return similarity

def rank_candidates_basic(mention, context, candidates):
	scores = defaultdict(float)

	for candidate in candidates:
		string_score = compute_string_similarity(mention, candidate)
		context_score = compute_context_similarity(context, candidate)
		scores[candidate["id"]] = 0.5 * string_score + 0.3 * context_score + 0.2 * len(candidate["sitelinks"]["sitelinks"])

	ranked_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)
	return ranked_candidates