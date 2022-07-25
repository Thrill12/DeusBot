from BaseAgent import Agent
import random

# Takes the list of actions of the base agent, chooses one at random and then execute it
class RandomAgent(Agent):
  def step(self, obs):
    super(RandomAgent, self).step(obs)
    action = random.choice(self.actions)
    print("yes this is so different oh wow")
    return getattr(self, action)(obs)
