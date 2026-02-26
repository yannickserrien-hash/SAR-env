# Terminal 1 (default port 11434 — used by Agent 0 + EnginePlanner)
ollama serve

# Terminal 2 (port 11435 — used by Agent 1)
OLLAMA_HOST=0.0.0.0:11435 ollama serve

# Terminal 3
python main.py

You need to start as many ollama instances as agents to make sure they do not run sequentially.