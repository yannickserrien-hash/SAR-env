import os, requests
import sys
import json
import pathlib
from SaR_gui import visualization_server
from worlds1.WorldBuilder import create_builder
from loggers.OutputLogger import output_logger
from engine.engine_planner import EnginePlanner

if __name__ == "__main__":
    fld = os.getcwd()

    # Configuration
    condition = "normal"
    name = "ale"
    agent_type = 'llm'
    ticks_per_iteration = 100  # Configurable: MATRX ticks per planning iteration

    builder, agents = create_builder(
        condition=condition, name=name, agent_type=agent_type, folder=fld
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
        llm_model='llama3:8b',
        # TODO: Customize this task_description based on your project needs
        task_description='',  # Uses DEFAULT_TASK_DESCRIPTION from engine_planner.py
        ticks_per_iteration=ticks_per_iteration
    )

    # Run with MARBLE-style planning loop
    builder.api_info['matrx_paused'] = False
    iteration_history = world.run_with_planner(
        builder.api_info, planner, agents,
        ticks_per_iteration=ticks_per_iteration
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
