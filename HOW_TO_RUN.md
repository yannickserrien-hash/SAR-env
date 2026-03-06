# Terminal 1 (default port 11434 — used by EnginePlanner)
ollama serve

# Terminal 2 (port 11435 — used by Agent 1)
OLLAMA_HOST=0.0.0.0:11435 ollama serve
# Terminal 3 (port 11435 — used by Agent 2)
OLLAMA_HOST=0.0.0.0:11436 ollama serve

python main.py