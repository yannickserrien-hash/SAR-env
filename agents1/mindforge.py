import asyncio
import copy
import json
import logging
import os
import pprint
import requests
import time
import dspy
from typing import Dict
import mindforge.utils as U
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from termcolor import colored
from .agents import BeliefCurriculumAgent, SkillManager, EpisodicManager
from .agents.action import ActionAgent
from .agents.original_critic import CriticAgent
from .agents.social import SocialAgent
from .env import VoyagerEnv
from mindforge.control_primitives_context import load_control_primitives_context

class MindForge:
    def __init__(self, agent_config, env_config, resume: bool = False, enable_async_polling: bool = True, evaluate: bool = False, instructive: bool = True, agent_status: dict = None):
        """Implementation of the MindForge agent."""
        self.agent_config = agent_config
        self.env_config = env_config
        self.resume = resume
        self.enable_async_polling = enable_async_polling
        self.evaluate = evaluate
        self.instructive = instructive
        self.env_wait_ticks = 20
        self.agent_status = agent_status

        self.env = VoyagerEnv(bot_username=agent_config["username"], mc_port=env_config["minecraft_port"], server_port=agent_config["port"], request_timeout=env_config["env_request_timeout"])
        self.logger = self.create_logger(agent_config["username"])

        # init agents
        self.action_agent = ActionAgent(name="action", llm=agent_config["agents"]["action"]["parameters"]["llm"], temperature=agent_config["agents"]["action"]["parameters"]["temperature"], request_timeout=agent_config["agents"]["action"]["parameters"]["task_max_retries"], resume=self.resume, ckpt_dir=agent_config["ckpt_dir"], chat_log=agent_config["agents"]["action"]["parameters"]["show_chat_log"], execution_error=agent_config["agents"]["action"]["parameters"]["show_execution_error"], logger=self.logger)
        self.curriculum_agent = BeliefCurriculumAgent(agent_config["agents"]["curriculum"] | {"ckpt_dir": agent_config['ckpt_dir']}, agent_config["username"], self.logger, resume=self.resume)
        self.critic_agent = CriticAgent(llm=agent_config["agents"]["critic"]["parameters"]["llm"], temperature=agent_config["agents"]["critic"]["parameters"]["temperature"], mode=agent_config["agents"]["critic"]["parameters"]["mode"])
        self.social_agent = SocialAgent(name="social", llm=agent_config["agents"]["social"]["parameters"]["llm"], temperature=agent_config["agents"]["social"]["parameters"]["temperature"], request_timeout=env_config["api_request_timeout"], resume=agent_config["resume"], ckpt_dir=agent_config["ckpt_dir"], chat_log=agent_config["agents"]["social"]["parameters"]["show_chat_log"], execution_error=agent_config["agents"]["social"]["parameters"]["show_execution_error"], logger=self.logger)
        self.skill_manager = SkillManager(agent_config["agents"]["procedural"] | {"ckpt_dir": agent_config['ckpt_dir']}, agent_config["username"], self.logger, resume=self.resume)
        self.episodic_manager = EpisodicManager(agent_config["agents"]["episodic"] | {"ckpt_dir": agent_config['ckpt_dir']}, agent_config["username"], self.logger, resume=self.resume)
        self.recorder = U.EventRecorder(ckpt_dir=agent_config["ckpt_dir"], resume=resume)

        # misc
        self.reset_placed_if_failed = False # whether to reset placed blocks if failed, useful for building task

        # init variables for rollout
        self.task = None
        self.last_critique = ""
        self.conversations = []
        self.last_events = None
        self.code_from_last_round = None
        self.latest_interaction_beliefs = None
        self.partner_name = "strong" if self.agent_config["username"] == "weak" else "weak"
        self.partner_beliefs = {f"{self.partner_name}": ""} # hardcoded for two players

    def create_logger(self, username, debug_level=logging.INFO):
        """Log all the information of the agent to a file."""
        logger = logging.getLogger(username)
        logger.setLevel(debug_level)
        file_handler = logging.FileHandler(f'./logs/{time.time()}_{username}.log')
        file_handler.setLevel(debug_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger

    def reset(self, task, context="", reset_env=True):
        self.task = task
        self.last_code = ""
        self.last_critique = ""
        self.code_from_last_round = None
        self.latest_interaction_beliefs = None
        if reset_env:
            self.env.reset(options={"mode": "soft","wait_ticks": self.env_wait_ticks})
        difficulty = "easy" if len(self.curriculum_agent.completed_tasks) > 15 else "peaceful"
        events = self.env.step("bot.chat(`/time set ${getNextTime()}`);\n" + f"bot.chat('/difficulty {difficulty}');")
        self.last_events = events

    def close(self):
        self.env.close()

    def step(self, context: dict):
        action = self.action_agent(context)
        parsed_result = self.action_agent.parse_action(action.code)
        success = False
        if isinstance(parsed_result, dict):
            code = parsed_result["program_code"] + "\n" + parsed_result["exec_code"]
            self.last_code = code
            events = self.env.step(code, programs=self.skill_manager.programs)
            self.recorder.record(events, self.task)
            self.action_agent.update_chest_memory(events[-1][1]["nearbyChests"]) # pyright: ignore
            critic_context = {
                "biome": events[-1][1]["status"]["biome"], # pyright: ignore
                "timeOfDay": events[-1][1]["status"]["timeOfDay"], # pyright: ignore
                "voxels": events[-1][1]["voxels"], # pyright: ignore
                "health": events[-1][1]["status"]["health"], # pyright: ignore
                "food": events[-1][1]["status"]["food"], # pyright: ignore
                "position": events[-1][1]["status"]["position"], # pyright: ignore
                "equipment": ["" if x is None else x for x in events[-1][1]["status"]["equipment"]], # pyright: ignore
                "inventory": events[-1][1]["inventory"], # pyright: ignore
                "task": self.task,
                "context": context["task_beliefs"],
                "chest_observation": self.action_agent.render_chest_observation()
            }
            success, critique = self.critic_agent.check_task_success(events=events, task=self.task, context=context["task_beliefs"], chest_observation=self.action_agent.render_chest_observation(), max_retries=5)

            # add failures to episodic memory
            if not success:
                self.episodic_manager.add_new_episode(info={"episode": f"Task: {self.task}\n\nContext used to generate the action code: {context}\n\nAction: {code}\n\nCritique: {critique}"})
            if self.reset_placed_if_failed and not success:
                blocks, positions = [], []
                for event_type, event in events: # pyright: ignore
                    if event_type == "onSave" and event["onSave"].endswith("_placed"): # pyright: ignore
                        block = event["onSave"].split("_placed")[0] # pyright: ignore
                        position = event["status"]["position"] # pyright: ignore
                        blocks.append(block)
                        positions.append(position)
                new_events = self.env.step(f"await givePlacedItemBack(bot, {U.json_dumps(blocks)}, {U.json_dumps(positions)})", programs=self.skill_manager.programs)
                events[-1][1]["inventory"] = new_events[-1][1]["inventory"] # pyright: ignore
                events[-1][1]["voxels"] = new_events[-1][1]["voxels"] # pyright: ignore
            self.last_events = copy.deepcopy(events)
            self.last_critique = "" if success else critique
        else:
            self.last_code = "No code in the last round."
            self.last_critique = ""
            assert isinstance(parsed_result, str)
            self.recorder.record([], self.task)

        done = success
        info = {"task": self.task, "success": success} # pyright: ignore
        if isinstance(parsed_result, dict):
            info["code"] = parsed_result["program_code"]
        else:
            info["code"] = parsed_result
        if success:
            assert "program_code" in parsed_result and "program_name" in parsed_result, "program and program_name must be returned when success"
            info["program_code"] = parsed_result["program_code"] # pyright: ignore
            info["program_name"] = parsed_result["program_name"] # pyright: ignore
        return [], 0, done, info

    def rollout(self, *, task, context, reset_env=True, cooperation=False):
        self.reset(task=task, context=context, reset_env=reset_env)
        num_steps = 0
        messages, reward, done, info = None, 0, False, {}

        def _initiate_communication(context: dict):
            res = requests.post(
                    "http://localhost:3002/ready-for-communication",
                    json={
                        "agent": self.agent_config["username"],
                        "task": self.task,
                        "info": context | {"name": self.agent_config["username"], "llm": self.agent_config["agents"]["action"]["parameters"]["llm"], "temperature": 0.7, "internal": {}, "partners": {}, "template": "default"}
                    }
            )
            if res.status_code != 200:
                raise Exception("Failed to connect to communication server.")

            response = requests.get("http://localhost:3002/conversation")
            while response.json().get("agents") !=  0:
                time.sleep(1)
                response = requests.get("http://localhost:3002/conversation")

            while response.json().get("active") is True:
                time.sleep(1)
                response = requests.get("http://localhost:3002/conversation")

        def _post_process_communication():
            response = requests.get("http://localhost:3002/latest-conversation")
            return response.json().get("conversation")

        def _update_context(context: str, task: str, interaction_beliefs: str):
            generate_new_context = dspy.Predict(ContextUpdate)
            new_context = generate_new_context(previous_context=context, task=task, interaction_beliefs=interaction_beliefs)
            return new_context

        def _create_perception_beliefs(perceptions: dict):
            generate_perceptions = dspy.Predict(Perceptions)
            perception_beliefs = generate_perceptions(
                biome=perceptions["biome"],
                time_of_day=perceptions["time_of_day"],
                nearby_blocks=perceptions["voxels"],
                health=perceptions["health"],
                hunger=perceptions["food"],
                position=perceptions["position"],
                equipment=perceptions["equipment"],
                inventory=perceptions["inventory"],
                chest_observation=self.action_agent.render_chest_observation(),
                execution_errors=perceptions["execution_errors"],
                chat_log=perceptions["chat_log"]
            )
            return perception_beliefs

        interaction_beliefs = None
        while num_steps < self.env_config["max_num_steps"]:
            error_messages = []
            chat_messages = []
            for i, (event_type, event) in enumerate(self.last_events): # pyright: ignore
                        if event_type == "onChat":
                            chat_messages.append(event["onChat"]) # pyright: ignore
                        elif event_type == "onError":
                            error_messages.append(event["onError"]) # pyright: ignore

            perception_context = {
                "biome": self.last_events[-1][1]["status"]["biome"], # pyright: ignore
                "time_of_day": self.last_events[-1][1]["status"]["timeOfDay"], # pyright: ignore
                "voxels": self.last_events[-1][1]["voxels"], # pyright: ignore
                "health": self.last_events[-1][1]["status"]["health"], # pyright: ignore
                "food": self.last_events[-1][1]["status"]["food"], # pyright: ignore
                "position": self.last_events[-1][1]["status"]["position"], # pyright: ignore
                "equipment": ["" if x is None else x for x in self.last_events[-1][1]["status"]["equipment"]], # pyright: ignore
                "inventory": self.last_events[-1][1]["inventory"], # pyright: ignore
                "chest_observation": self.action_agent.render_chest_observation(),
                "execution_errors": "Execution error: " + "\n".join(error_messages) if error_messages else "Execution error: No error.",
                "chat_log": "Chat log: " + "\n".join(chat_messages) if chat_messages else "Chat log: None."
            }
            perception_beliefs = _create_perception_beliefs(perception_context) # pyright: ignore

            if (num_steps > 0 and cooperation): 
                communication_context = {
                    "code_from_last_round": self.last_code,
                    "execution_error": "Execution error: " + "\n".join(error_messages) if error_messages else "Execution error: No error.",
                    "task": self.task,
                    "task_beliefs": context,
                    "perception_beliefs": perception_beliefs.perception_beliefs,
                    "interaction_beliefs": "There are no interaction beliefs" if self.latest_interaction_beliefs  is None else self.latest_interaction_beliefs,
                    "partner_beliefs": self.partner_beliefs[f"{self.partner_name}"],
                    "critique": self.last_critique,
                }
                _initiate_communication(communication_context | perception_context)
                conversation = _post_process_communication()
                social_context = {
                    "task": self.task,
                    "latest_conversation": conversation,
                    "previous_partner_beliefs": self.partner_beliefs[f"{self.partner_name}"],
                    "previous_interaction_beliefs": self.latest_interaction_beliefs if self.latest_interaction_beliefs is not None else "There are no past interactions for this task."
                }
                partner_beliefs, interaction_beliefs = self.social_agent(social_context)
                self.latest_interaction_beliefs = interaction_beliefs.interaction_beliefs
                self.partner_beliefs[f"{self.partner_name}"] = partner_beliefs.partner_beliefs
                context = _update_context(context, self.task, self.latest_interaction_beliefs) # pyright: ignore
                context = context.new_context
                self.logger.info(f"Updated context: {context}")

            if num_steps == 0:
                skills = self.skill_manager.retrieve_skills(query=context) # pyright: ignore
            else:
                skills = self.skill_manager.retrieve_skills(query=context + "\n\n" + self.action_agent.summarize_chatlog(self.last_events)) # pyright: ignore

            self.logger.info(f"Skills retrieved for the task {task} and context {context}: {skills}")

            # build step context
            action_context = {
                "task": self.task,
                "episodic_memory": self.episodic_manager.generate_summary(self.episodic_manager.retrieve_episodes(query=f"Task: {self.task}\n\nContext: {context}")), # pyright: ignore
                "procedural_memory": "\n\n".join(load_control_primitives_context(["exploreUntil", "mineBlock", "craftItem", "placeItem", "smeltItem", "killMob", "useChest", "mineflayer"]) + skills), # pyright: ignore
                "task_beliefs": context,
                "perception_beliefs": perception_beliefs,
                "interaction_beliefs": "There are no interation beliefs" if interaction_beliefs is None else interaction_beliefs,
                "partner_beliefs": self.partner_beliefs[f"{self.partner_name}"],
                "critique": self.last_critique,

            }
            messages, reward, done, info = self.step(action_context)
            num_steps += 1
            if done:
                if cooperation and self.agent_status:
                    self.agent_status[self.agent_config["username"]] = "done"
                    while self.agent_status.get(self.partner_name) != "done":
                        communication_context = {
                            "code_from_last_round": self.last_code,
                            "execution_error": "Execution error: " + "\n".join(error_messages) if error_messages else "Execution error: No error.",
                            "task": self.task,
                            "task_beliefs": context,
                            "perception_beliefs": perception_beliefs.perception_beliefs,
                            "interaction_beliefs": "There are no interaction beliefs" if self.latest_interaction_beliefs  is None else self.latest_interaction_beliefs,
                            "partner_beliefs": self.partner_beliefs[f"{self.partner_name}"],
                            "critique": self.last_critique,
                        }
                        _initiate_communication(communication_context | perception_context)
                        _ = _post_process_communication()
                        time.sleep(1)

                return messages, reward, done, info

        return messages, reward, done, info

    def learn(self, max_iterations: int, reset_env=True, cooperate: bool = False):

        if self.resume:
            self.env.reset(options={"mode": "soft", "wait_ticks": self.env_wait_ticks})
        else:
            self.env.reset(options={"mode": "hard", "wait_ticks": self.env_wait_ticks})
            self.resume = True
        self.last_events = self.env.step("")

        while True:
            if self.recorder.iteration > max_iterations:
                break
            task, context = self.curriculum_agent.propose_next_task(events=self.last_events, chest_observation=self.action_agent.render_chest_observation(), max_retries=5)
            if (task is None) and (context is None):
                break
            self.logger.info(colored(f"{self.agent_config['username']} Starting task {task} for at most {self.env_config['max_num_steps']} times", "magenta"))
            messages, reward, done, info = self.rollout(task=task, context=context, reset_env=reset_env, cooperation=cooperate)
            if info["success"]:
                self.skill_manager.add_new_skill(info)
            self.curriculum_agent.update_exploration_progress(info)
            self.logger.info(colored(f"Completed tasks: {', '.join(self.curriculum_agent.completed_tasks)}", "magenta"))
            self.logger.info(colored(f"Failed tasks: {', '.join(self.curriculum_agent.failed_tasks)}", "magenta"))
        return {"completed_tasks": self.curriculum_agent.completed_tasks, "failed_tasks": self.curriculum_agent.failed_tasks, "skills": self.skill_manager.skills}

    def decompose_task(self, task):
        if not self.last_events:
            self.last_events = self.env.reset(options={"mode": "hard", "wait_ticks": self.env_wait_ticks})
        return self.curriculum_agent.decompose_task(task, self.last_events)

    def inference(self, task=None, sub_goals=[], reset_mode="hard", reset_env=True, evaluate=False, cooperation=False):
        if not task and not sub_goals:
            raise ValueError("Either task or sub_goals must be provided")
        if not sub_goals:
            sub_goals = self.decompose_task(task)
        self.env.reset(options={"mode": reset_mode, "wait_ticks": self.env_config["env_wait_ticks"]})
        self.curriculum_agent.completed_tasks = []
        self.curriculum_agent.failed_tasks = []
        self.last_events = self.env.step("")

        while self.curriculum_agent.progress < len(sub_goals):
            next_task = sub_goals[self.curriculum_agent.progress]
            context = self.curriculum_agent.get_task_context(next_task)
            messages, reward, done, info = self.rollout(task=next_task, context=context, reset_env=reset_env, cooperation=cooperation)
            self.curriculum_agent.update_exploration_progress(info)

        self.logger.info(colored(f"{self.agent_config['username']} Completed tasks: {', '.join(self.curriculum_agent.completed_tasks)}", "magenta"))
        self.logger.info(colored(f"{self.agent_config['username']} Failed tasks: {', '.join(self.curriculum_agent.failed_tasks)}", "magenta"))
        sucess_rate = len(self.curriculum_agent.completed_tasks) / (len(self.curriculum_agent.completed_tasks) + len(self.curriculum_agent.failed_tasks))
        self.logger.info(f"Success rate for Agent {self.agent_config['username']}: {sucess_rate}.")


class ContextUpdate(dspy.Signature):
    """Update the context based on the new information from the other agent."""
    previous_context: str  = dspy.InputField(desc="Previous context of the agent.")
    task: str = dspy.InputField(desc="The task that the agent is currently working on and needs to be completed.")
    interaction_beliefs: str = dspy.InputField(desc="Interaction beliefs from a conversation with a more knowledgeable agent.")

    new_context: str = dspy.OutputField(desc="Updated context of the agent to solve the task.", prefix="Create a new context based on the task and the interaction beliefs. If necessary, update the context to include the new information. If the previous context and interaction beliefs contradict, assume the interaction beliefs are true. Keep the new context concise and informative.")


class Perceptions(dspy.Signature):
    """Generate perception beliefs."""
    biome: str = dspy.InputField(desc="The biome after the task execution.")
    time_of_day: str = dspy.InputField(desc="The current time.")
    nearby_blocks: list = dspy.InputField(desc="Nearby blocks.")
    health: float = dspy.InputField(desc="Current health.")
    hunger: float = dspy.InputField(desc="Current hunger level.")
    position: dict = dspy.InputField(desc="Current position.")
    equipment: str = dspy.InputField(desc="Final equipment.")
    inventory: dict = dspy.InputField(desc="Final inventory.")
    chest_observation: str = dspy.InputField(desc="Observation of chests")
    execution_errors: str = dspy.InputField(desc="Execution errors that occurred during past attempt at the task.")
    chat_log: str = dspy.InputField(desc="Chat log from the past attempt at the task.")


    perception_beliefs: str = dspy.OutputField(desc="Perception beliefs about the state of the agent given the input observations.", prefix="Set of beliefs that encapsulate the agent's perception of the environment. Include information about the biome, time of day, nearby blocks, health, hunger, position, equipment, inventory, and any execution errors that occurred during the past attempt at the task as well as relevant chat logs.")