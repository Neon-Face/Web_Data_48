import re
import requests
from difflib import SequenceMatcher
import wikipediaapi
import rake_nltk

import nltk
import spacy
nlp = spacy.load("en_core_web_sm")

from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
model_embedding = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

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

def search_nltk_tree(ne_tree, ENTITY_LABELS: list[str], entity_names: list[str]):
	entities = []
	for tree in ne_tree:
		if isinstance(tree, nltk.tree.Tree):
			name = " ".join([t[0] for t in tree])
			if tree.label() in ENTITY_LABELS and name not in entity_names:
				entities.append((name, tree.label()))
		elif tree[1] in ENTITY_LABELS and tree[0] not in entity_names:
			entities.append((tree[0], tree[1]))
	return entities

def extract_keywords(question: str) -> str:
	r = rake_nltk.Rake()
	r.extract_keywords_from_text(question)
	return list(dict.fromkeys(r.get_ranked_phrases()))

def extract_entities(text: str, debug: bool = False) -> list[tuple[str, str]]:
	doc = nlp(text)
	base_entities = [(ent.text, ent.label_) for ent in doc.ents]
	entities = []

	base_entities = list({t[0]: t for t in base_entities}.values())
	ENTITY_LABELS = ["GPE", "ORGANIZATION", "ORG", "PERSON", "LOCATION", "WORK_OF_ART", "EVENT", "NORP", "FAC", "LOC", "PRODUCT"]

	if debug:
		print("Initial entities:", base_entities)

	entity_names = []
	for entity in base_entities:
		if entity[1] in ENTITY_LABELS:
			entities.append(entity)
			entity_names.append(entity[0])

	if debug:
		print("Filtered names:", entities)

	ne_tree = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(text)))
	if debug:
		print("NER tree:", ne_tree)

	entities += search_nltk_tree(ne_tree, ENTITY_LABELS, entity_names)
	if len(entities) == 0:
		ENTITY_LABELS += ["NN"]
		entities += search_nltk_tree(ne_tree, ENTITY_LABELS, entity_names)

	if debug:
		print("Entities:", entities)
	return entities

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def disambiguate_with_model(entity: str, question: str, wikidata, gpu: bool, debug = False):
	inputs = []
	for e in wikidata:
		label = e["label"]
		description = e["description"]
		inputs.append(f"Entity: {label}. Description: {description}")	

	entity_embeddings = model_embedding.encode(inputs, device="cuda" if gpu else "cpu")
	query_embedding = model_embedding.encode(entity, device="cuda" if gpu else "cpu")
	cosine_scores = util.cos_sim(query_embedding, entity_embeddings)

	raw = list(map(lambda x: len(x["sitelinks"]["sitelinks"]), wikidata))
	norm = [float(i)/sum(raw) for i in raw]
	cosine_scores = cosine_scores[0].tolist()
	for i in range(len(cosine_scores)):
		cosine_scores[i] = (cosine_scores[i] * 0.8) + (norm[i] * 0.1) + similarity(wikidata[i]["description"], question) * 0.1
	best_match_idx = cosine_scores.index(max(cosine_scores))

	return wikidata[best_match_idx]

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

def remove_non_text(data: str) -> str:
	data = re.sub(r'[^a-zA-Z0-9\s]', '', data)
	return data.strip()

def extract_text_from_wikipedia(wikipedia_url):
	wiki = wikipediaapi.Wikipedia('Web_data_48 (Web_data_48@example.com)', 'en')
	title = wikipedia_url.split('/')[-1]
	page = wiki.page(title)	

	if page.exists():
		return page.text
	return None

def extract_relevant_sentences(large_text, query_text, context_size):
    sentences = nltk.sent_tokenize(large_text)
    query_tokens = set(query_text.lower().split())
    extracted_contexts = []
    for i, sentence in enumerate(sentences):
        sentence_tokens = set(remove_non_text(sentence.lower()).split())
        overlap = query_tokens.intersection(sentence_tokens)
        if overlap:
            start_index = max(i - context_size, 0)
            end_index = min(i + context_size + 1, len(sentences))
            context = sentences[start_index:end_index]
            joined_context = " ".join(context)

            relevance_score = len(overlap)
            extracted_contexts.append((joined_context, relevance_score))

    ranked_contexts = sorted(extracted_contexts, key=lambda x: x[1], reverse=True)
    return ranked_contexts

def classify_response(question, answer):
	yes_words = {"yes", "yeah", "yep", "affirmative", "correct", "true", "valid"}
	no_words = {"no", "nah", "nope", "negative", "incorrect", "false", "invalid"}
	negation_words = {"not", "no", "n't", "never", "none", "false", "incorrect"}

	tokens = set(answer.lower().split())
	if tokens.intersection(yes_words) and (tokens.intersection(no_words) or any(word in tokens for word in negation_words)):
		for item in answer.lower().split():
			if item in yes_words:
				return "Yes"
			elif item in no_words:
				return "No"
			elif any(word in item for word in negation_words):
				return "No"
	elif tokens.intersection(yes_words):
		return "Yes"
	elif tokens.intersection(no_words):
		return "No"

	if any(word in tokens for word in negation_words):
		return "No"

	question_embedding = model.encode(question)
	answer_embedding = model.encode(answer)
	similarity_score = util.cos_sim(question_embedding, answer_embedding)

	if similarity_score > 0.9:
		return "Yes"

	return None

def extract_relevant_entity(question: str, answer: str, debug = False):
	entities = extract_entities(answer, debug)
	entity_texts = [entity for entity, _ in entities]

	if not entities:
		return fallback_relevant_entity(question, entities)

	expected_type = None
	if "who" in question.lower():
		expected_type = "PERSON"
	elif "where" in question.lower():
		expected_type = "GPE"
	elif "when" in question.lower():
		expected_type = "DATE"
	elif "what" in question.lower() or "which" in question.lower():
		expected_type = "ORG" 

	answer_doc = nlp(answer)
	relevant_entities = []
	for token in answer_doc:
		if token.pos_ == "VERB" or token.dep_ in {"nsubj", "ROOT"}:
			for entity, label in entities:
				if token.text in answer_doc.text and entity in token.sent.text:
					relevant_entities.append((entity, label))

	filtered_entities = [ent for ent, label in relevant_entities if label == expected_type]
	if filtered_entities:
		return filtered_entities[0]
	return entity_texts[0] if entity_texts else fallback_relevant_entity(question, entities)

def fallback_relevant_entity(question: str, entities):
	question_embedding = model_embedding.encode(question, convert_to_tensor=True)
	entity_embeddings = {entity: model_embedding.encode(entity, convert_to_tensor=True) for entity, _ in entities}

	best_entity = "unknown"
	best_score = -1

	for entity, embedding in entity_embeddings.items():
		similarity = util.cos_sim(question_embedding, embedding)
		if similarity > best_score:
			best_score = similarity
			best_entity = entity

	return best_entity

def remove_prefix(data: str) -> str:
	return re.sub(r'Answer:\s*$', '', data)