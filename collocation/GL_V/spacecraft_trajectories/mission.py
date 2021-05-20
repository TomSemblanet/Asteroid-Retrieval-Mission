import matplotlib.pyplot as plt
import numpy as np
import cppad_py
import math

from src.optimal_control.GL_V.src.problem import Problem
from src.optimal_control.GL_V.src import utils
from src.lowthrust.utils.spacecraft import Spacecraft
from src.lowthrust.dynamics.cr3bp_dynamics_adapted import Cr3bpDynamics
from src.init.cr3bp import Cr3bp
from src.init.primary import Primary


class Mission(Problem):
    """ `Mission` class manages the general frame common to every transfers in the cis-lunar
        environment. It instantiates the `spacecraft`, `cr3bp` objects and the dynamic of the environment.

        Inherits from the `Problem` class.

        Parameters
        ----------
        spacecraft_prm : dict
            Spacecraft parameters dictionnary

        Attributs
        ---------
        spacecraft : Spacecraft
            Spacecraft object 
        cr3bp : Cr3bp
            Cr3bp containing all the constants relatives to the CR3BP in the Earth / Moon system
        cr3bp_dyna : Cr3bpDynamics
            Dynamics of the Cr3bp in the Earth / Moon system
        l_ref : float
            Characteristic length
        t_ref : float
            Characteristic time
        v_ref : float
            Characteristic velocity
        a_ref : float
            Characteristic acceleration

    """

    def __init__(self, spacecraft_prm, **kwargs):
        """ Initialization of the `Mission` class """
        # Instantiation of the problem
        Problem.__init__(self, **kwargs)

        # Instantiation of the spacecraft
        self.spacecraft = Spacecraft(**spacecraft_prm)

        # Instantiation of a CR3BP object
        self.cr3bp = Cr3bp(Primary.EARTH, Primary.MOON)

        # Instantiation of a Cr3bpDynamics object used to compute
        # dynamics equations
        self.cr3bp_dyna = Cr3bpDynamics(
            self.cr3bp, Cr3bpDynamics.Eqm.dimensions6)

        # Definition of cr3bp specific values
        self.set_problem_values()

    def dynamics(self, states, controls, f_par, expl_int=False):
        """ Sets the dynamics equations in the cr3bp frame (synodic ref.), 
                common for both LLO and NRHO missions """
      
        if expl_int == False:
            dynamics = np.ndarray(
                (states.shape[0], states.shape[1]), dtype=cppad_py.a_double)
        else:
            dynamics = np.zeros(len(states))
        
        # Mass [-]
        m = states[6]

        # Thrust [-]
        T = controls[0]

        # Thrust direction [-]
        ux, uy, uz = controls[1:]

        # Computation of states derivatives (cr3bp)
        x_dot, y_dot, z_dot, vx_dot, vy_dot, vz_dot = self.cr3bp_dyna.eqm_6_synodic(
            states, self.cr3bp.mu)

        # Computation of the acceleration [-]
        acc = T / m / self.a_ref

        dynamics[0] = x_dot
        dynamics[1] = y_dot
        dynamics[2] = z_dot

        dynamics[3] = vx_dot + acc*ux
        dynamics[4] = vy_dot + acc*uy
        dynamics[5] = vz_dot + acc*uz

        dynamics[6] = - self.t_ref * T / self.spacecraft.isp / self.g0

        return dynamics

    def set_problem_values(self):
        """ Setting of the cr3bp problem characteristic values """
        # Characteristic length       [km]
        self.l_ref = self.cr3bp.L

        # Characteristic time         [s]
        self.t_ref = self.cr3bp.T / (2 * np.pi)

        # Characteristic velocity     [km/s]
        self.v_ref = self.l_ref / self.t_ref

        # Characteristic acceleration [m/s^2]
        self.a_ref = 1000 * self.l_ref / (self.t_ref ** 2)

        # Sea level gravity on Earth  [m/s^2]
        self.g0 = 9.80665
