import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id                = 'Goddard-v0',
    entry_point       = 'gym_goddard.envs:GoddardEnv',
    max_episode_steps = None,
    reward_threshold  = None,
    nondeterministic  = False,
)
