import random
import numpy as np
import pandas as pd
import os
from absl import app
from pysc2.agents import base_agent
from pysc2.lib import actions,features,units
from pysc2.env import sc2_env, run_loop

# Reinforcment Learning Algorithm
class QLearningTable: 

  path_to_model_save = "ModelSave.txt"

  def __init__(self, actions, learning_rate=0.01, reward_decay=0.9):
    self.actions = actions
    self.learning_rate = learning_rate
    self.reward_decay = reward_decay
    self.count = 0
    self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
    self.load_model()

  # 90% Chooses preferred action and 10% randomly for extra possibilities
  def choose_action(self, observation, e_greedy=0.9):
    self.check_state_exist(observation)
    if np.random.uniform() < e_greedy:
      state_action = self.q_table.loc[observation, :]
      action = np.random.choice(
          state_action[state_action == np.max(state_action)].index)
    else:
      action = np.random.choice(self.actions)
    return action

  # Takes the state and action and update table accordingly to learn over time
  def learn(self, s, a, r, s_):
    self.check_state_exist(s_)
    q_predict = self.q_table.loc[s, a] # Get the value that was given for taking the action when we were first in the state
    # Determine the maximum possible value across all actions in the current state
    # and then discount it by the decay rate (0.9) and add the reward we received (can be terminal or not)
    if s_ != 'terminal':
      q_target = r + self.reward_decay * self.q_table.loc[s_, :].max()
    else: # Reward from last step of game is better
      q_target = r 
    self.q_table.loc[s, a] += self.learning_rate * (q_target - q_predict)
    self.save_model()

  def check_state_exist(self, state): # Check to see if the state is in the QTable already, and if not it will add it with a value of 0 for all possible actions.
    if state not in self.q_table.index:
      self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))

  def save_model(self):
    self.q_table.to_pickle(self.path_to_model_save)

  def load_model(self):
    if os.path.exists(self.path_to_model_save) and len(pd.read_pickle(self.path_to_model_save)) > 0:
      self.q_table = pd.read_pickle(self.path_to_model_save)