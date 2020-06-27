import goddard_env as env
import argparse
import numpy as np

class DefaultControlled(env.Default):

    def dv(self,v,h):
        return 2.0 * self.D * v * np.exp(-self.H * h)

    def dh(self,v,h):
        return -self.H * self.drag(v,h)

    # Dtilde = dv(v,h) + gamma * drag(v,h)

    def Dtilde_dv(self,v,h):
        return 2.0 * self.D * np.exp(-self.H * h) + self.GAMMA * self.dv(v,h)

    def Dtilde_dh(self,v,h):
        return -2.0 * self.D * self.H * v * np.exp(-self.H * h) + self.GAMMA * self.dh(v,h)


class SaturnVControlled(env.SaturnV):

    def dv(self,v,h):
        return 2.0 * self.D * v

    def dh(self,v,h):
        return 0.0

    def Dtilde_dv(self,v,h):
        return 2.0 * self.D * (1.0 + self.GAMMA * v)

    def Dtilde_dh(self,v,h):
        return 0.0

class OptimalController(object):

    '''
        An optimal controller that solves the continuous time Goddard rocket problem according to
        Pontryagin's maximum principle. 
    '''

    def __init__(self, rocket):
        self._r = rocket
        self._trig = False
        self._prev_sing_traj = None

    def control(self, v, h, m):
        D = self._r.drag(v,h)
        Dtilde = self._r.dv(v,h) + self._r.GAMMA*D

        if self._trig:
            # singular trajectory
            gdt = self._r.GAMMA*Dtilde
            numerator = self._r.dh(v,h) - gdt*self._r.G - v*self._r.Dtilde_dh(v,h)
            u = m*self._r.G + D + m*(numerator/(gdt+self._r.Dtilde_dv(v,h)))
        else:
            # detect singular trajectory condition, i.e. == 0 or crosses 0 between samples
            sing_traj = v * Dtilde - (D + m*self._r.G)
            self._trig = self._prev_sing_traj is not None and (sing_traj * self._prev_sing_traj <= 0.0)
            self._prev_sing_traj = sing_traj
            u = self._r.U_MAX

        return max(0.0, min(self._r.U_MAX, u))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a rocket simulation with an optimal controller.')
    parser.add_argument('-r', '--rocket', default='default', choices=['default', 'saturn'], help='The rocket model to use')
    parser.add_argument('-t', '--time', default=40, type=int, help='Simulation duration time [s]')

    args = parser.parse_args()

    if args.rocket == 'default':
        rocket = DefaultControlled()
    elif args.rocket == 'saturn':
        rocket = SaturnVControlled()

    godd = env.GoddardEnv(rocket)

    oc = OptimalController(rocket)

    state = godd.reset()

    state_log = [state]
    extra_log = []

    v, h, m = state
    hmax = h

    time = np.arange(0, args.time, rocket.DT)

    for _ in time:
        (state, _, _, extra) = godd.step(action=oc.control(v, h, m))
        v, h, m = state
        state_log.append(state)
        extra_log.append([extra['u'], extra['drag']])

        hmax = max(hmax, h)

        godd.render()

    print("h_max = {}".format(hmax))

    import matplotlib.pyplot as plt

    state_log = np.array(state_log)

    f, (ax1, ax2, ax3, ax4) = plt.subplots(4)

    ax1.plot(time, state_log[:-1,1])
    ax1.grid(True)
    ax1.set(xlabel='time [s]', ylabel='height [m]')

    ax2.plot(time, state_log[:-1,0])
    ax2.grid(True)
    ax2.set(xlabel='time [s]', ylabel='velocity [m/s]')

    ax3.plot(time, state_log[:-1,2])
    ax3.grid(True)
    ax3.set(xlabel='time [s]', ylabel='rocket mass [kg]')

    ax4.plot(time, extra_log)
    ax4.grid(True)
    ax4.set(xlabel='time [s]', ylabel='thrust [N]')

    plt.show()
