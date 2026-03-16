import os, requests
import sys
import json
import pathlib
from SaR_gui import visualization_server
from worlds1.WorldBuilder import create_builder
from loggers.OutputLogger import output_logger
from engine.engine_planner import EnginePlanner
from agents1.async_model_prompting import init_marble_pool

if __name__ == "__main__":
    fld = os.getcwd()

    # Configuration
    condition = "normal"
    name = "humanagent"
    agent_type='marble'              # 'llm' | 'langgraph' | 'baseline'
    ticks_per_iteration = 1200  # 1200 ticks * 0.1s/tick = 120 seconds = 2 minutes
    num_rescue_agents = 2       # Number of LLM-based RescueAgents (1-5)
    include_human = False        # Whether to add a keyboard-controlled human agent
    ollama_base_port = 11434    # Each agent uses its own Ollama instance: agent N -> port base+N
    planner_model = 'qwen3:8b'  # Larger model for the EnginePlanner (main brain)

    planning_mode = 'dag'        # 'simple' (flat list) or 'dag' (task graph with conditionals)

    # Agent capability presets — one per agent (cycles if fewer than num_rescue_agents)
    # Available: 'scout', 'medic', 'heavy_lifter', 'generalist', or custom dicts
    agent_presets = ['scout', 'medic']
    # Whether agents know their capabilities upfront or discover through failure
    capability_knowledge = 'informed'  # 'informed' | 'discovery'

    # Set to a YAML file path to override LLM task/plan generation with manual inputs.
    # See manual_plans.yaml for the expected format. Set to None to use LLM mode.
    manual_plans_file = "manual_plans.yaml"  # e.g. "manual_plans.yaml"

    # Scale LLM thread pool for the number of agents
    init_marble_pool(num_rescue_agents)

    builder, agents = create_builder(
        condition=condition, name=name, agent_type=agent_type, folder=fld,
        num_rescue_agents=num_rescue_agents, include_human=include_human,
        ollama_base_port=ollama_base_port, planning_mode=planning_mode,
        agent_presets=agent_presets, capability_knowledge=capability_knowledge
    )

    # Start overarching MATRX scripts and threads
    media_folder = pathlib.Path().resolve()
    builder.startup(media_folder=media_folder)
    print("Starting custom visualizer")
    vis_thread = visualization_server.run_matrx_visualizer(verbose=False, media_folder=media_folder)
    world = builder.get_world()
    print("Started world...")

    # Initialize score.json with defaults
    os.makedirs('logs', exist_ok=True)
    score_file = os.path.join('logs', 'score.json')
    with open(score_file, 'w') as f:
        json.dump({
            'score': 0,
            'block_hit_rate': 0.0,
            'victims_rescued': 0,
            'total_victims': 8
        }, f, indent=2)

    # Initialize EnginePlanner with LLM
    planner = EnginePlanner(
        max_iterations=50,
        score_file=score_file,
        llm_model=planner_model,
        ticks_per_iteration=ticks_per_iteration,
        include_human=include_human,
        api_url=f"http://localhost:{ollama_base_port}",
        manual_plans_file=manual_plans_file,
    )
    
    # Run with MARBLE-style planning loop
    builder.api_info['matrx_paused'] = False
    iteration_history = world.run_with_planner(
        builder.api_info, planner, agents,
        ticks_per_iteration=ticks_per_iteration,
        include_human=include_human
    )

    print("DONE!")

    # Save iteration history
    history_file = os.path.join('logs', 'iteration_history.json')
    try:
        with open(history_file, 'w') as f:
            json.dump([{
                'iteration': d.iteration,
                'task_assignments': d.task_assignments,
                'summary': d.summary,
                'score': d.score,
                'block_hit_rate': d.block_hit_rate
            } for d in iteration_history], f, indent=2)
        print(f"Saved iteration history to {history_file}")
    except Exception as e:
        print(f"Failed to save iteration history: {e}")

    print("Shutting down custom visualizer")
    r = requests.get("http://localhost:" + str(visualization_server.port) + "/shutdown_visualizer")
    vis_thread.join()
    output_logger(fld)
    builder.stop()
