FROM python:3.10 AS base

WORKDIR /app
RUN pip install uv
RUN uv venv
RUN uv pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN uv pip install llama_cpp_python sentence-transformers spacy nltk requests
RUN uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl#sha256=1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85

FROM python:3.10-slim-bookworm
WORKDIR /app
COPY --from=base /app/.venv .venv
COPY init.py init.py
RUN .venv/bin/python init.py

COPY main.py main.py
COPY common.py common.py

ENTRYPOINT [".venv/bin/python", "main.py"]