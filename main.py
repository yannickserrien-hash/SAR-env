import os, requests
import sys
import json
import signal
import pathlib
import threading
from worlds1.WorldBuilder import create_builder
from loggers.OutputLogger import output_logger
from engine.engine_planner import EnginePlanner
from agents1.async_model_prompting import init_marble_pool, shutdown_marble_pool

# Event set by SIGTERM/SIGINT handler to request graceful shutdown.
# Passed into run_with_planner() so the main loop can check it each tick.
shutdown_event = threading.Event()

def _handle_sigterm(signum, frame):
    print(f"\n[main] Received signal {signum} — requesting graceful shutdown...")
    shutdown_event.set()

if __name__ == "__main__":
    fld = os.getcwd()

    # Register signal handlers for SLURM preemption / job timeout
    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    # Configuration
    condition = "normal"
    name = "humanagent"
    agent_type='marble'              # 'llm' | 'langgraph' | 'baseline'
    ticks_per_iteration = 1200  # 1200 ticks * 0.1s/tick = 120 seconds = 2 minutes
    num_rescue_agents = 2       # Number of LLM-based RescueAgents (1-5)
    include_human = False        # Whether to add a keyboard-controlled human agent
    enable_gui = True            # Set to False for headless runs (e.g. Delft Blue)
    planner_model = 'qwen3:8b'  # Larger model for the EnginePlanner (main brain)
    agent_model = 'qwen3:8b'    # Model for individual rescue agents

    # LLM server configuration
    # Local Ollama: api_base = "http://localhost:11434"
    # HPC (vLLM):   api_base = "http://localhost:8000", llm_backend = 'requests',
    #               model names must match vLLM's --served-model-name
    api_base = "http://localhost:11434"

    # Server ports (change to avoid conflicts when running multiple jobs on one node)
    api_port = 3001         # MATRX API server port
    vis_port = 3000         # Visualization server port

    planning_mode = 'dag'        # 'simple' (flat list) or 'dag' (task graph with conditionals)

    # Agent capability presets — one per agent (cycles if fewer than num_rescue_agents)
    # Available: 'scout', 'medic', 'heavy_lifter', 'generalist', or custom dicts
    agent_presets = ['generalist', 'generalist']
    # Whether agents know their capabilities upfront or discover through failure
    capability_knowledge = 'informed'  # 'informed' | 'discovery'

    # Communication strategies — one per agent (cycles if fewer than num_rescue_agents)
    # Available: 'always_respond', 'busy_aware'
    comm_strategies = ['always_respond', 'always_respond']

    # LLM backend: 'ollama_sdk' (Ollama Python SDK) or 'requests' (direct HTTP)
    llm_backend = 'ollama_sdk'

    # World preset: 'static' (current world), 'preset2' (2 houses), 'preset3' (2 big houses), 'random'
    world_preset = 'preset2'
    world_seed = None            # int for reproducibility, None for random each run

    # Set to a YAML file path to override LLM task/plan generation with manual inputs.
    # See manual_plans.yaml for the expected format. Set to None to use LLM mode.
    manual_plans_file = None  # e.g. "manual_plans.yaml"

    # Log directory (override with SAR_LOG_DIR env var for HPC)
    log_dir = os.environ.get('SAR_LOG_DIR', os.path.join(fld, 'logs'))

    # Scale LLM thread pool for the number of agents
    init_marble_pool(num_rescue_agents, backend=llm_backend)

    builder = None
    vis_thread = None
    planner = None
    iteration_history = []

    try:
        builder, agents, total_victims = create_builder(
            condition=condition, name=name, agent_type=agent_type, folder=fld,
            num_rescue_agents=num_rescue_agents, include_human=include_human,
            api_base=api_base, agent_model=agent_model,
            planning_mode=planning_mode,
            agent_presets=agent_presets, capability_knowledge=capability_knowledge,
            comm_strategies=comm_strategies,
            world_preset=world_preset, world_seed=world_seed,
            enable_gui=enable_gui,
        )

        # Configure MATRX API port before startup
        from matrx.api import api as matrx_api
        matrx_api.set_api_port(api_port)

        # Start overarching MATRX scripts and threads
        media_folder = pathlib.Path().resolve()
        builder.startup(media_folder=media_folder)
        if enable_gui:
            from SaR_gui import visualization_server
            print("Starting custom visualizer")
            vis_thread = visualization_server.run_matrx_visualizer(
                verbose=False, media_folder=media_folder, vis_port=vis_port
            )
        world = builder.get_world()
        print("Started world...")

        # Initialize score.json with defaults
        os.makedirs(log_dir, exist_ok=True)
        score_file = os.path.join(log_dir, 'score.json')
        with open(score_file, 'w') as f:
            json.dump({
                'score': 0,
                'block_hit_rate': 0.0,
                'victims_rescued': 0,
                'total_victims': total_victims
            }, f, indent=2)

        # Initialize EnginePlanner with LLM
        planner = EnginePlanner(
            max_iterations=50,
            score_file=score_file,
            llm_model=planner_model,
            ticks_per_iteration=ticks_per_iteration,
            include_human=include_human,
            api_url=api_base,
            manual_plans_file=manual_plans_file,
        )

        # Run with MARBLE-style planning loop
        builder.api_info['matrx_paused'] = False
        iteration_history = world.run_with_planner(
            builder.api_info, planner, agents,
            ticks_per_iteration=ticks_per_iteration,
            include_human=include_human,
            shutdown_event=shutdown_event,
        )

        print("DONE!")

    except Exception as e:
        print(f"[main] Simulation error: {e}", file=sys.stderr)

    finally:
        # Save iteration history (from planner if available, else from run_with_planner return)
        history = iteration_history
        if not history and planner is not None:
            history = planner.iteration_history

        if history:
            try:
                os.makedirs(log_dir, exist_ok=True)
                history_file = os.path.join(log_dir, 'iteration_history.json')
                with open(history_file, 'w') as f:
                    json.dump([{
                        'iteration': d.iteration,
                        'task_assignments': d.task_assignments,
                        'summary': d.summary,
                        'score': d.score,
                        'block_hit_rate': d.block_hit_rate
                    } for d in history], f, indent=2)
                print(f"Saved iteration history to {history_file}")
            except Exception as e:
                print(f"Failed to save iteration history: {e}", file=sys.stderr)

        # Shut down visualization
        if enable_gui and vis_thread is not None:
            try:
                print("Shutting down custom visualizer")
                requests.get(
                    f"http://localhost:{vis_port}/shutdown_visualizer",
                    timeout=5,
                )
                vis_thread.join(timeout=5)
            except Exception:
                pass  # daemon thread will exit with process

        # Shut down LLM thread pool
        shutdown_marble_pool()

        # Run output logger and stop builder
        try:
            output_logger(fld)
        except Exception as e:
            print(f"[main] Output logger error: {e}", file=sys.stderr)

        if builder is not None:
            try:
                builder.stop()
            except Exception as e:
                print(f"[main] Builder stop error: {e}", file=sys.stderr)

        print("[main] Cleanup complete.")
