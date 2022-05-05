import random
import numpy as np
import pandas as pd
import os
from absl import app
from pysc2.agents import base_agent
from pysc2.lib import actions,features,units
from pysc2.env import sc2_env, run_loop
from setuptools import Command
from QLearningTable import QLearningTable
from BaseAgent import Agent
from RandomAgent import RandomAgent
from SmartAgent import SmartAgent

def main(unused_argv):
  agent = SmartAgent()
  agent2 = RandomAgent()
  try:
    while True:
      with sc2_env.SC2Env(
          map_name="AbyssalReef",
          players=[sc2_env.Agent(sc2_env.Race.terran), 
                   sc2_env.Bot(sc2_env.Race.terran, sc2_env.Difficulty.very_easy)],
          agent_interface_format=features.AgentInterfaceFormat(
              action_space=actions.ActionSpace.RAW,
              use_raw_units=True,
              raw_resolution=64,       
          ),
          step_mul=32,
      ) as env:
        run_loop.run_loop([agent], env)
  except KeyboardInterrupt:
    pass


if __name__ == "__main__":
  app.run(main)