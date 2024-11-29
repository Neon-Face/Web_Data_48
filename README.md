# WDPS

## Docker
Example with llama-2-7b.Q4_K_M.gguf model and input.txt in the CWD, attach them as volumes. Pass `--gpu` to run the model on an nvidia GPU.

CPU image:
```
docker run --rm -v ./llama-2-7b.Q4_K_M.gguf:/llama-2-7b.Q4_K_M.gguf -v ./input.txt:/input.txt t348575/web_data_48:v0.1.0-cpu --model /llama-2-7b.Q4_K_M.gguf --input /input.txt
```

GPU image (nvidia):
```
docker run --gpus=all --rm -v ./llama-2-7b.Q4_K_M.gguf:/llama-2-7b.Q4_K_M.gguf -v ./input.txt:/input.txt t348575/web_data_48:v0.1.0-gpu --model /llama-2-7b.Q4_K_M.gguf --input /input.txt --gpu
```

## Direct
* Clone the repository, and install using the provided `requirements.txt`.
* Use `requirements-cpu.txt` to install the CPU only version of pytorch.
* Installing `llama_cpp_python` with gpu requires running `CMAKE_ARGS="-DGGML_CUDA=ON" FORCE_CMAKE=1 pip install llama_cpp_python`. (Consult the repo [here](https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file#supported-backends) for other backend support).
* Pass `--gpu` to run the model on an nvidia GPU.
```
pip install -r requirements.txt
```
Running:
```
python main.py --model ./llama-2-7b.Q4_K_M.gguf --input ./input.txt
```