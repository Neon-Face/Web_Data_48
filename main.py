import argparse
from llama_cpp import Llama
from timeit import default_timer as timer
model_path = "models/llama-2-7b.Q4_K_M.gguf"

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="Model path", required=False)
parser.add_argument("--gpu", action="store_true", help="Run with nvidia GPU", required=False)
parser.add_argument("--input", type=str, help="Input question", required=True)
parser.add_argument("--debug", action="store_true", help="Debug mode", required=False)

args = parser.parse_args()

if args.model is not None and len(args.model) > 0:
    model_path = args.model

start = timer()
llm = Llama(model_path=model_path, verbose=False, n_gpu_layers=-1 if args.gpu else 0)
output = llm(args.input, max_tokens=32, stop=["Q:", "Question:", "Context:"], echo=False)
end = timer()

print(output['choices'][0]['text'])
if args.debug:
    print(f"Time: {end - start}")