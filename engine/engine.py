"""
Engine: Main orchestrator for MARBLE-style multi-agent coordination in MATRX.

Based on MARBLE's Engine pattern adapted for MATRX's tick-based execution.
"""

import logging
import pathlib
import json
from typing import List, Optional
from matrx import WorldBuilder
from engine.engine_planner import EnginePlanner
from engine.iteration_data import IterationData
from worlds1.WorldBuilder import create_builder
from SaR_gui import visualization_server

class Engine:
    """
    Main orchestrator for MARBLE-style multi-agent coordination in MATRX.

    Responsibilities:
    - Initialize MATRX world and agents
    - Run decentralized planning loop via EnginePlanner
    - Manage simulation lifecycle
    - Handle termination and cleanup
    """

    def __init__(self,
                 condition: str = 'normal',
                 name: str = 'HumanAgent',
                 folder: str = '.',
                 agent_type: str = 'llm',
                 max_iterations: int = 100):
        """
        Initialize Engine.

        Args:
            condition: Human capability condition ('normal', 'strong', 'weak')
            name: Human agent name
            folder: Working folder path
            agent_type: Type of agent ('llm' for RescueAgent)
            max_iterations: Maximum planning iterations
        """
        # Configure logging
        self.logger = logging.getLogger('Engine')
        self.logger.setLevel(logging.INFO)

        # Add console handler if not already present
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter('[%(name)s] %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Configuration
        self.condition = condition
        self.name = name
        self.folder = folder
        self.agent_type = agent_type
        self.max_iterations = max_iterations

        # Components (initialized in start())
        self.builder: Optional[WorldBuilder] = None
        self.world = None
        self.agents: List = []
        self.planner: Optional[EnginePlanner] = None
        self.vis_thread = None

        self.logger.info("Engine initialized")

    def start(self):
        """Main entry point: Initialize and run simulation."""
        self.logger.info("="*60)
        self.logger.info("Starting MARBLE-style Engine for MATRX")
        self.logger.info("="*60)

        # Step 1: Initialize MATRX world and agents
        self._initialize_world()

        # Step 2: Initialize EnginePlanner
        self._initialize_planner()

        # Step 3: Start visualization
        self._start_visualization()

        # Step 4: Initialize score.json
        self._initialize_score()

        # Step 5: Run decentralized planning loop
        self.logger.info("\n[4/5] Starting decentralized planning loop...")
        iteration_history = self.planner.decentralize_planning(self.agents, self.world)

        # Step 6: Finalize
        self._finalize(iteration_history)

        self.logger.info("\n" + "="*60)
        self.logger.info("Engine completed successfully")
        self.logger.info("="*60)

    def _initialize_world(self):
        """Initialize MATRX world and agents."""
        self.logger.info("\n[1/5] Initializing MATRX world...")

        # Create world builder
        self.builder, agents = create_builder(
            condition=self.condition,
            name=self.name,
            folder=self.folder,
            agent_type=self.agent_type
        )
        
        self.agents = agents
        
        # Start MATRX
        media_folder = pathlib.Path().resolve()
        self.builder.startup(media_folder=media_folder)
        self.world = self.builder.get_world()

        # Extract RescueAgent instances
        # MATRX stores agents in world.registered_agents

        self.logger.info(f"  Total planning agents: {len(self.agents)}")

    def _initialize_planner(self):
        """Initialize EnginePlanner."""
        self.logger.info("\n[2/5] Initializing EnginePlanner...")

        score_file = f"{self.folder}/logs/score.json"
        self.planner = EnginePlanner(
            max_iterations=self.max_iterations,
            score_file=score_file,
            llm_model='llama3:8b',
            # TODO: Customize this task_description based on your project needs
            task_description=''
        )

        self.logger.info(f"  Max iterations: {self.max_iterations}")
        self.logger.info(f"  Score file: {score_file}")

    def _start_visualization(self):
        """Start Flask visualization server."""
        self.logger.info("\n[3/5] Starting visualization server...")

        media_folder = pathlib.Path().resolve()
        self.vis_thread = visualization_server.run_matrx_visualizer(
            verbose=False,
            media_folder=media_folder
        )

        self.logger.info("  Server started")
        self.logger.info("  Open browser: http://localhost:3000")

    def _initialize_score(self):
        """Initialize score.json with default values if it doesn't exist."""
        import os
        score_file = f"{self.folder}/logs/score.json"

        # Ensure logs directory exists
        os.makedirs(f"{self.folder}/logs", exist_ok=True)

        # Create score.json if it doesn't exist
        if not os.path.exists(score_file):
            default_score = {
                'score': 0,
                'block_hit_rate': 0.0,
                'victims_rescued': 0,
                'total_victims': 16
            }
            with open(score_file, 'w') as f:
                json.dump(default_score, f, indent=2)
            self.logger.info(f"  ✓ Created {score_file} with default values")

    def _finalize(self, iteration_history: List[IterationData]):
        """Finalize simulation and save outputs."""
        self.logger.info("\n[5/5] Finalizing simulation...")

        # Save iteration history
        output_file = f"{self.folder}/logs/iteration_history.json"
        try:
            with open(output_file, 'w') as f:
                json.dump([{
                    'iteration': d.iteration,
                    'task_assignments': d.task_assignments,
                    'summary': d.summary,
                    'score': d.score,
                    'block_hit_rate': d.block_hit_rate
                } for d in iteration_history], f, indent=2)
            self.logger.info(f"  ✓ Saved iteration history: {output_file}")
        except Exception as e:
            self.logger.error(f"  ✗ Failed to save iteration history: {e}")

        # Shutdown visualization
        self.logger.info("  Shutting down visualization server...")
        import requests
        try:
            requests.get(f"http://localhost:{visualization_server.port}/shutdown_visualizer",
                        timeout=2)
            if self.vis_thread:
                self.vis_thread.join(timeout=5)
            self.logger.info("  Server stopped")
        except Exception as e:
            self.logger.warning(f"  Error shutting down visualizer: {e}")

        # Stop MATRX
        if self.builder:
            try:
                self.builder.stop()
                self.logger.info("  ✓ MATRX world stopped")
            except Exception as e:
                self.logger.warning(f"  ⚠ Error stopping MATRX: {e}")

        # Print summary
        self.logger.info("\n" + "-"*60)
        self.logger.info("SIMULATION SUMMARY")
        self.logger.info("-"*60)
        self.logger.info(f"Total iterations: {len(iteration_history)}")
        if iteration_history:
            final = iteration_history[-1]
            self.logger.info(f"Final score: {final.score:.2f}")
            self.logger.info(f"Block hit rate: {final.block_hit_rate:.2f}")
            self.logger.info(f"Status: {'COMPLETED' if final.block_hit_rate >= 1.0 else 'INCOMPLETE'}")
        self.logger.info("-"*60)
