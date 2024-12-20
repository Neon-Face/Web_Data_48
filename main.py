import torch
import psutil
import os
import argparse
import common
import numpy as np
import faiss
from concurrent.futures import ThreadPoolExecutor
from llama_cpp import Llama
from timeit import default_timer as timer
model_path = "models/llama-2-7b.Q4_K_M.gguf"

def disambiguate_entity(entity, question, keywords):
    try:
        wikidata = common.search_wikidata(entity.strip())
        if not wikidata:
            return (entity, "unknown", None, [])

        e = common.disambiguate_with_model(entity, question, wikidata, args.gpu, args.debug)
        wikipedia_url = common.get_wikipedia_url(e["id"], wikidata)

        context_docs = common.extract_relevant_sentences(
            common.extract_text_from_wikipedia(wikipedia_url), keywords, 1
        )[:25]
        return (entity, e["label"], wikipedia_url, list(map(lambda x: x[0], context_docs)))
    except Exception as e:
        if args.debug:
            print(f"Error in disambiguating {entity}: {e}")
        return (entity, "unknown", None, [])

def process_entities(entities, question):
    keywords = common.extract_keywords(question)

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda entity: disambiguate_entity(entity[0], question, " ".join(keywords)), entities))
    return results

if os.getenv('INIT') == "1":
    import init

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="Model path", required=False)
parser.add_argument("--gpu", action="store_true", help="Run with nvidia GPU", required=False)
parser.add_argument("--input", type=str, help="Input question. Either a single quesiton as a string, or a file path in the expected format", required=False)
parser.add_argument("--debug", action="store_true", help="Debug mode", required=False)
parser.add_argument("--runtime", action="store_true", help="Debug mode", required=False)
# parser.add_argument("--basic-d", action="store_true", help="Basic entity disambiguation mode", required=False)
parser.add_argument("--two-stage", action="store_true", help="Two stage RAG mode", required=False)
parser.add_argument("--two-stage-entity-count", type=int, help="Maximum entities to extract in second RAG stage", required=False, default=3)
parser.add_argument("--no-ask", type=str, help="The string given here is assumed to be initial the LLM output", required=False)
parser.add_argument("--answer", action="store_true", help="Print the verification answer", required=False)

args = parser.parse_args()

if args.model is not None and len(args.model) > 0:
    model_path = args.model

if args.input is not None and len(args.input) > 0 and os.path.exists(args.input):
    with open(args.input, "r") as f:
        args.input = common.extract_questions(f.read())
else:
    args.input = common.extract_questions(f"question-001\t{args.input}")

if args.debug:
    print(f"Input questions: {args.input}")

start = timer()
ctx_size = 1024
if args.gpu:
    _, total = torch.cuda.mem_get_info(torch.device("cuda" if args.gpu else "cpu"))
    if total / (1024 * 1024) > 8000:
        ctx_size = 4096
else:
    print(psutil.virtual_memory().total)
    if psutil.virtual_memory().total / (1024 * 1024) > 8000:
        ctx_size = 4096
llm = Llama(model_path=model_path, verbose=False, n_gpu_layers=-1 if args.gpu else 0, n_ctx=ctx_size)
end = timer()

if args.runtime:
    print(f"LLM loading time: {end - start}")

question_idx = 0

def run_question(question_id, question):
    total_start = timer()
    if args.debug:
        print(f"Question: {question}")

    start = timer()
    if args.no_ask is not None and len(args.no_ask) > 0:
        output = {"choices": [{"text": args.no_ask }]}
    else:
        output = llm(question, max_tokens=32, stop=["Q:", "Question:", "Context:", "?"], echo=False)
    end = timer()

    output_txt = output['choices'][0]['text'].strip()
    print(f"{question_id}\tR\"{output_txt}\"")
    if args.runtime:
        print(f"Question LLM time: {end - start}")

    question_entities = common.extract_entities(question, args.debug)
    answer_entities = common.extract_entities(common.remove_prefix(output_txt), args.debug)
    entities = list({t[0]: t for t in question_entities + answer_entities}.values())

    documents = []
    entity_prints = []

    start = timer()
    seen = {}
    entity_disambiguations = []
    for item in process_entities(entities, question):
        if item[2] is not None and item[2] not in seen:
            entity_disambiguations.append(item)
            seen[item[2]] = True
    end = timer()

    if args.runtime:
        print(f"Entity disambiguation time: {end - start}")

    for _, label, wikipedia_url, docs in entity_disambiguations:
        entity_prints.append(format(f"{question_id}\tE\"{label}\"\t\"{wikipedia_url}\""))
        documents += docs

    if args.debug:
        print("Documents: ", len(documents))

    start = timer()
    document_embeddings = np.array(common.model_embedding.encode(documents, device="cuda" if args.gpu else "cpu"))
    index = faiss.IndexFlatL2(384)
    if len(documents) > 0:
        index.add(document_embeddings)

    query_embedding = np.array(common.model_embedding.encode([question], device="cuda" if args.gpu else "cpu"))
    _, indices = index.search(query_embedding, 6)

    top_results = [documents[i] for i in indices[0]]

    if args.two_stage:
        new_entites = []
        for item in top_results:
            for e in common.extract_entities(item):
                if e[0] not in list(map(lambda x: x[0], entity_disambiguations)) and e[0] not in list(map(lambda x: x[1], entity_disambiguations)) and e[0] not in list(map(lambda x: x[0], new_entites)):
                    new_entites.append(e)

        encode_items = []
        if args.debug:
            print("New entities: ", len(new_entites), new_entites, entities)
        entity_start = timer()
        for item in process_entities(new_entites[:args.two_stage_entity_count], question):
            entity_disambiguations.append(item)
            for doc in item[3]:
                if doc not in documents:
                    encode_items.append(doc)
        entity_end = timer()
        if args.runtime:
            print(f"RAG entity disambiguation time: {entity_end - entity_start}")

        documents += encode_items
        new_document_embeddings = np.array(common.model_embedding.encode(encode_items, device="cuda" if args.gpu else "cpu"))
        if len(encode_items) > 0:
            index.add(new_document_embeddings)

        _, indices = index.search(query_embedding, 3)
        top_results = [documents[i] for i in indices[0]]

    end = timer()

    if args.runtime:
        print(f"Embedding time: {end - start}")

    context = ". ".join(top_results)
    if args.debug:
        print(context)

    question_final = f"You are a helpful assistant trained to answer questions based on the given context. \nContext:\n{context}.\nQuestion:\n{question}.\nAnswer:"
    start_ask = timer()
    output = llm(
        question_final,
        max_tokens=32,
        stop=["Q:", "Question:", "Context:", "?"],
        echo=False
    )
    end_ask = timer()
    if args.runtime:
        print(f"Answer LLM time: {end_ask - start_ask}")
    verification_output = output['choices'][0]['text']

    answer_simmilarity = common.similarity(verification_output.strip().lower(), question.strip().lower())
    if answer_simmilarity > 0.95:
        if args.debug:
            print(f"Answer similarity: {answer_simmilarity}, so rerunning with more tokens")
        start_ask = timer()
        output = llm(
            question_final,
            max_tokens=128,
            echo=False
        )
        end_ask = timer()
        if args.runtime:
            print(f"Answer LLM time: {end_ask - start_ask}")
        verification_output = output['choices'][0]['text']

    if args.debug:
        print('Verification answer:', verification_output)
    start = timer()
    answer = common.classify_response(question, common.remove_prefix(common.remove_non_text(output_txt)))
    answer_entity = common.extract_relevant_entity(question, output_txt, args.debug)

    verification_answer = common.classify_response(question, common.remove_prefix(common.remove_non_text(verification_output)))
    verification_answer_entity = common.extract_relevant_entity(question, verification_output, args.debug)

    if args.debug:
        print(f"answer: {answer} answer_entity: {answer_entity} verification_answer: {verification_answer} verification_answer_entity: {verification_answer_entity}")

    if verification_answer is None and verification_answer_entity not in list(map(lambda x: x[0], entity_disambiguations)):
        res = disambiguate_entity(verification_answer_entity, question, " ".join(common.extract_keywords(question)))
        entity_disambiguations.append(res)
        entity_prints.append(format(f"{question_id}\tE\"{res[1]}\"\t\"{res[2]}\""))

    for entity, _, wikipedia_url, _ in entity_disambiguations:
        if answer is None and entity.lower() == answer_entity.lower():
            answer = wikipedia_url

        if verification_answer is None and entity.lower() == verification_answer_entity.lower():
            verification_answer = wikipedia_url

    if answer is None:
        answer = "unknown"
    if verification_answer is None:
        verification_answer = "unknown"

    print(f"{question_id}\tA\"{answer}\"")
    if args.answer:
        print(f"{question_id}\tVA\"{verification_answer}\"")

    correctness = "unknown"
    if answer == verification_answer and answer != "unknown":
        correctness = "correct"
    else:
        correctness = "incorrect"
    print(f"{question_id}\tC\"{correctness}\"")

    for entity_print in entity_prints:
        print(entity_print)

    end = timer()
    if args.runtime:
        print(f"Verification time: {end - start}")
        print(f"Total time: {end - total_start}")

for question_id, question in args.input:
    try: 
        run_question(question_id, question)
    except Exception as e:
        print(f"Error processing question {question_id}: {e}")
    question_idx += 1

