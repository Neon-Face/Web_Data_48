import os
import argparse
import common
from llama_cpp import Llama
from timeit import default_timer as timer
model_path = "models/llama-2-7b.Q4_K_M.gguf"

if os.getenv('INIT') == "1":
    import init

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="Model path", required=False)
parser.add_argument("--gpu", action="store_true", help="Run with nvidia GPU", required=False)
parser.add_argument("--input", type=str, help="Input question. Either a single quesiton as a string, or a file path in the expected format", required=True)
parser.add_argument("--debug", action="store_true", help="Debug mode", required=False)

args = parser.parse_args()

if args.model is not None and len(args.model) > 0:
    model_path = args.model

if os.path.exists(args.input):
    with open(args.input, "r") as f:
        args.input = common.extract_questions(f.read())
else:
    args.input = [('question-001', args.input)]

start = timer()
llm = Llama(model_path=model_path, verbose=False, n_gpu_layers=-1 if args.gpu else 0)
end = timer()

if args.debug:
    print(f"LLM loading time: {end - start}")

for question_id, question in args.input:
    if args.debug:
        print(f"Question: {question}")

    start = timer()
    output = llm(question, max_tokens=64, stop=["Q:", "Question:", "Context:"], echo=False)
    end = timer()

    output_txt = output['choices'][0]['text'].strip()
    print(f"{question_id}\tR\"{output_txt}\"")
    print(f"{question_id}\tA\"unknown\"")
    print(f"{question_id}\tC\"unknown\"")
    if args.debug:
        print(f"Question LLM time: {end - start}")

    question_entities = common.extract_entities(question, args.debug)
    answer_entities = common.extract_entities(output_txt, args.debug)
    entities = list({t[0]: t for t in question_entities + answer_entities}.values())
    if args.debug:
        print("Final entities:", entities)

    wikipedia_urls = []
    for entity in entities:
        try:
            disambiguated_entity = common.disambiguate(entity[0], question, args.debug)
            wikipedia_url = common.get_wikipedia_url(disambiguated_entity[0], args.debug)

            if wikipedia_url is not None and wikipedia_url not in wikipedia_urls:
                wikipedia_urls.append(wikipedia_url)
                print(f"{question_id}\tE\"{disambiguated_entity[1]}\"\t\"{wikipedia_url}\"")
        except:
            print(f"{question_id}\tE\"{entity[0]}\"\t\"unknown\"")
