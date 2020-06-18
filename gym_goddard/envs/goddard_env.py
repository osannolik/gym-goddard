import gym
import numpy as np

class Drag(object):

    def __call__(self,v,h):
        return max(0.0, (1.0-h/400.0) * v)

class DragExtras(Drag):

    @staticmethod
    def dv(v,h):
        return 1.0 - h/400.0

    @staticmethod
    def dh(v,h):
        return -v/400.0

    @staticmethod
    def tilde_dv(v,h,gamma):
        return gamma * (1.0-h/400.0)

    @staticmethod
    def tilde_dh(v,h,gamma):
        return -1.0/400.0 - gamma * v / 400.0

PARAMS = {
    'v0': 0.0,      # Initial velocity
    'h0': 0.0,      # Initial height
    'm0': 10.0,     # Initial weight (dry mass + fuel)
    'm1': 1.0,      # Final weight (could be equal to dry mass if all fuel should be used)
    'u_max': 196.2, # Maximum possible force of thrust [N]
    'gamma': 0.01,  # Fuel consumption [kg/N/s]
    'dt': 0.05,     # Assumed time [s] between calls to step()
    'g': 9.81       # Gravitational acceleration [m/s/s] (assumed constant as function of height)
}

class GoddardEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, parameters, drag_fn=Drag()):
        super(GoddardEnv, self).__init__()

        self._p = parameters
        self._drag = drag_fn

        self._dt, self._m1, self._g, self._gamma = \
            self._p['dt'], self._p['m1'], self._p['g'], self._p['gamma']

        # Thrust force
        self.action_space = gym.spaces.Box(
            low   = np.array([0.0]),
            high  = np.array([self._p['u_max']]),
            dtype = np.float
        )

        # 0: velocity
        # 1: altitude/height
        # 2: rocket mass
        self.observation_space = gym.spaces.Box(
            low   = np.array([np.finfo(np.float).min, 0.0, self._p['m1']]),
            high  = np.array([np.finfo(np.float).max, np.finfo(np.float).max, self._p['m0']]),
            dtype = np.float
        )

        self.reset()

    def step(self, action):        
        v, h, m = self._state

        is_tank_empty = (m <= self._m1)

        u = 0.0 if is_tank_empty else action

        # Forward Euler
        self._state = (
            v + self._dt * ((u-self._drag(v,h))/m - self._g),
            h + self._dt * v,
            max(m + self._dt * (-self._gamma * u), self._m1)
        )

        self._h_max = max(self._h_max, self._state[1])

        reward = 0.0
        is_done = False

        return self._observation(), reward, is_done, {}

    def _observation(self):
        return np.array(self._state)

    def reset(self):
        self._state = (self._p['v0'], self._p['h0'], self._p['m0'])
        self._h_max = self._p['h0']
        return self._observation()

    def render(self, mode='human', close=False):
        print('v, h, m = {}'.format(self._state))

class OptimalController(object):

    def __init__(self, p, drag_fn):
        self._drag = drag_fn
        self._gamma, self._g, self._u_max = p['gamma'], p['g'], p['u_max']
        self._trig = False

    def control(self, v, h, m):
        D = self._drag(v,h)
        Dtilde = self._drag.dv(v,h) + self._gamma*D

        # u_max condition
        trig_threshold = 1.0
        self._trig = self._trig | (abs(v * Dtilde - (D + m*self._g)) <= trig_threshold)

        if self._trig:
            # singular trajectory
            gdt = self._gamma*Dtilde
            numerator = self._drag.dh(v,h) - gdt*self._g - v*self._drag.tilde_dh(v,h,self._gamma)
            u = m*self._g + D + m*(numerator/(gdt+self._drag.tilde_dv(v,h,self._gamma)))
        else:
            u = self._u_max

        return max(0.0, min(self._u_max, u))

if __name__ == "__main__":

    godd = GoddardEnv(PARAMS, Drag())

    oc = OptimalController(PARAMS, DragExtras())

    (v, h, m) = godd.reset()

    v_log, h_log, m_log = [v], [h], [m]
    u_log = []

    hmax = h

    time = np.array(range(500)) * PARAMS['dt']

    for _ in time:
        u = oc.control(v,h,m)

        u_log.append(u)

        ((v, h, m), _, _, _) = godd.step(action = u)

        v_log.append(v)
        h_log.append(h)
        m_log.append(m)

        hmax = max(hmax, h)

        godd.render()

    print("h_max = {}".format(hmax))

    import matplotlib.pyplot as plt

    f, (ax1, ax2, ax3, ax4) = plt.subplots(4)

    ax1.plot(time, h_log[:-1])
    ax1.grid(True)
    ax1.set(xlabel='time [s]', ylabel='height [m]')

    ax2.plot(time, v_log[:-1])
    ax2.grid(True)
    ax2.set(xlabel='time [s]', ylabel='velocity [m/s]')

    ax3.plot(time, m_log[:-1])
    ax3.grid(True)
    ax3.set(xlabel='time [s]', ylabel='rocket mass [kg]')

    ax4.plot(time, u_log)
    ax4.grid(True)
    ax4.set(xlabel='time [s]', ylabel='thrust [N]')

    plt.show()
