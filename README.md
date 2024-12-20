# WDPS

## Docker
Example with llama-2-7b.Q4_K_M.gguf model and input.txt in the CWD, attach them as volumes. Pass `--gpu` to run the model on a GPU.

CPU image:
```
docker run --rm -v ./llama-2-7b.Q4_K_M.gguf:/llama-2-7b.Q4_K_M.gguf -v ./input.txt:/input.txt t348575/web_data_48:v0.1.1-cpu --model /llama-2-7b.Q4_K_M.gguf --input /input.txt
```

GPU image (nvidia):
```
docker run --gpus=all --rm -v ./llama-2-7b.Q4_K_M.gguf:/llama-2-7b.Q4_K_M.gguf -v ./input.txt:/input.txt t348575/web_data_48:v0.1.1-gpu --model /llama-2-7b.Q4_K_M.gguf --input /input.txt --gpu
```

## Direct
* Clone the repository, and install using the provided `requirements.txt`.
* Run init.py once to perform some other initialization activities.
* Use `requirements-cpu.txt` to install the CPU only version of pytorch.
* Installing `llama_cpp_python` with gpu requires running `CMAKE_ARGS="-DGGML_CUDA=ON" FORCE_CMAKE=1 pip install llama_cpp_python`. (Consult the repo [here](https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file#supported-backends) for other backend support).
* Pass `--gpu` to run the model on a GPU.
```
pip install -r requirements.txt
python init.py
```
Running:
```
python main.py --model ./llama-2-7b.Q4_K_M.gguf --input ./input.txt
```

## Important runtime arguments
* `--disable-two-stage` By default it runs with the two stage RAG, where the output from first embedding stage is used to extract entities to fetch and insert more embeddings. **NOTE**: Because of the second stage, it is signifcantly slower than a single stage, but is far less likely to makes mistakes, especially in questions where the answer entity is not very easily linked to the question entity. Our [example_input.txt](example_input.txt) file took 3m5s with two stage, and only 2m14s with single stage.
* `--answer` Apart from the R,A,C,E outputs, also prints a "VA" answer, the "Verified" answer.
* `--two-stage-entity-count` The number of entities to extract in the second stage, default 3.
* Use `--help` to view all other arguments