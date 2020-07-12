import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id                = 'Goddard-v0',
    entry_point       = 'gym_goddard.envs:GoddardDefaultEnv',
    max_episode_steps = 400,
    reward_threshold  = None,
    nondeterministic  = False,
)

register(
    id                = 'GoddardSaturnV-v0',
    entry_point       = 'gym_goddard.envs:GoddardSaturnEnv',
    max_episode_steps = 800,
    reward_threshold  = None,
    nondeterministic  = False,
)
