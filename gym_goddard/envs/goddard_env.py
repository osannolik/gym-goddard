import gym
import numpy as np

class Rocket(object):

    '''
        Expected parameters and methods of the rocket environment being simulated.

        V0/H0/M0:   Initial velocity/height/weight
        M1:         Final weight (set equal to dry mass if all fuel should be used)
        U_MAX:      Maximum possible force of thrust [N]
        GAMMA:      Fuel consumption [kg/N/s]
        DT:         Assumed time [s] between calls to step()

    '''

    V0 = H0 = M0 = M1 = U_MAX = GAMMA = DT = None

    def drag(self, v, h):
        raise NotImplementedError

    def g(self, h):
        raise NotImplementedError

class Default(Rocket):

    '''
        Models the surface-to-air missile (SA-2 Guideline) described in
        http://dcsl.gatech.edu/papers/jgcd92.pdf
        https://www.mcs.anl.gov/~more/cops/bcops/rocket.html

        The equations of motion is made dimensionless by scaling and choosing
        the model parameters in terms of initial height, mass and gravity.
    '''

    DT = 0.001

    V0 = 0.0
    H0 = M0 = G0 = 1.0
    
    HC = 500.0
    MC = 0.6
    VC = 620.0

    M1 = MC * M0

    thrust_to_weight_ratio = 3.5
    U_MAX = thrust_to_weight_ratio * G0 * M0
    DC = 0.5 * VC * M0 / G0
    GAMMA = 1.0 / (0.5*np.sqrt(G0*H0))

    def drag(self, v, h):
        return self.DC * v * abs(v) * np.exp(-self.HC*(h-self.H0)/self.H0)

    def g(self, h):
        return self.G0 * (self.H0/h)**2

class SaturnV(Rocket):

    '''
        Throttled first stage burn of Saturn V rocket

        Specifications taken from:
        1. https://en.wikipedia.org/wiki/Saturn_V
        2. https://web.archive.org/web/20170313142729/http://www.braeunig.us/apollo/saturnV.htm

    '''

    DT = 0.05
    G = 9.81
    V0 = 0.0
    H0 = 0.0

    # Vehicle mass + fuel: 2.97e6 kg
    # Thrust of first stage: 35.1e6 N
    M0 = 2.97e6
    U_MAX = 35.1e6

    # First stage gross and dry weight: 2.29e6 kg, 130e3 kg
    # => Saturn V gross weight immediately before first stage separation
    M1 = M0 - (2.29e6-130e3) # = 810e3 kg

    # Specific pulse: 263 s
    GAMMA = 1.0/(G*263.0) # = 387e-6 kg/N/s

    D = 0.35 * 0.27 * 113.0 / 2.0

    def drag(self, v, h):
        return self.D * v * abs(v)

    def g(self, h):
        return self.G

class GoddardEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, rocket=Rocket()):
        super(GoddardEnv, self).__init__()

        self._r = rocket

        # Thrust force
        self.action_space = gym.spaces.Box(
            low   = np.array([0.0]),
            high  = np.array([self._r.U_MAX]),
            dtype = np.float
        )

        # 0: velocity
        # 1: altitude/height
        # 2: rocket mass
        self.observation_space = gym.spaces.Box(
            low   = np.array([np.finfo(np.float).min, 0.0, self._r.M1]),
            high  = np.array([np.finfo(np.float).max, np.finfo(np.float).max, self._r.M0]),
            dtype = np.float
        )

        self.reset()

    def step(self, action):        
        v, h, m = self._state

        is_tank_empty = (m <= self._r.M1)

        u = 0.0 if is_tank_empty else action

        drag = self._r.drag(v,h)
        g = self._r.g(h)

        # Forward Euler
        self._state = (
            0.0 if h==self._r.H0 and v!=0.0 else (v + self._r.DT * ((u-drag)/m - g)),
            max(h + self._r.DT * v, self._r.H0),
            max(m - self._r.DT * self._r.GAMMA * u, self._r.M1)
        )

        self._h_max = max(self._h_max, self._state[1])

        reward = 0.0
        is_done = False

        return self._observation(), reward, is_done, {'u': u, 'drag': drag, 'g': g}

    def _observation(self):
        return np.array(self._state)

    def reset(self):
        self._state = (self._r.V0, self._r.H0, self._r.M0)
        self._h_max = self._r.H0
        return self._observation()

    def render(self, mode='human', close=False):
        print('v, h, m = {}'.format(self._state))

