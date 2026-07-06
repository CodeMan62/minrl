from minrl.envs.env import env
from minrl.types import Rollout, Step
from minrl.agents.agent import BaseAgent


# interaction between agent and environment
def rollout(agent: BaseAgent, env: env, num_steps) -> Rollout:
  rollout = Rollout(index=0, steps=[], total_reward=0, terminated=False, truncated=False, info={})
  obs, info = env.reset()
  for step in range(num_steps):
    action = agent.act(obs)
    out = env.step(action)
    rollout.steps.append(Step(
      index=step,
      prev_obs=obs,
      action=action,
      next_obs=out.obs,
      reward=out.reward,
      terminated=out.terminated,
      truncated=out.truncated,
      info=out.info,
      token_ids=getattr(agent, "last_token_ids", None),
      logprobs=getattr(agent, "last_logprobs", None),
      action_mask=getattr(agent, "last_action_mask", None)))
    rollout.total_reward += out.reward
    rollout.terminated = out.terminated
    rollout.truncated = out.truncated
    rollout.info = out.info
    obs = out.obs
    if out.terminated or out.truncated:
      obs, info = env.reset()
  rollout.index = len(rollout.steps)
  return rollout
