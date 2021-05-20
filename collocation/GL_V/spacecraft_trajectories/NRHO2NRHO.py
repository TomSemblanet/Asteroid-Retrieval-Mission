import numpy as np
import cppad_py

from src.optimal_control.GL_V.spacecraft_trajectories.mission import Mission
from src.optimal_control.GL_V.src.transcription import Transcription
from src.optimal_control.GL_V.src.solver import IPOPT
from src.optimal_control.GL_V.src.optimization import Optimization
from src.optimal_control.GL_V.src import utils

from src.lowthrust.guess.trajectory_stacking import TrajectoryStacking
from src.lowthrust.utils.utils import InitNRHOs


class NRHO2NRHO(Mission):
    """ `NRHO2NRHO` class implements a low-thrust transfer between two NRHO orbits
        using trajectory stacking method to generate the initial guess.
            
        cf. Robert Pritchett, Kathleen Howell, and Daniel Grebow. “Low-Thrust Transfer Design Based 
        on Collocation Techniques: Applications in the Restricted Three-Body Problem”. 

        Parameters
        ----------
        spacecraft_prm : dict
            Spacecraft parameters dictionnary
        orbits_prm : dict
            Dictionnary containing informations characterizing the departure and arrival NRHI
            orbits

        Attributs
        ---------
        orbits_prm : dict
            Dictionnary containing informations characterizing the departure and arrival NRHI
            orbits

        """

    def __init__(self, spacecraft_prm, orbits_prm, **kwargs):
        """ Initialization of the NRHO2NRHO class."""

        Mission.__init__(self, spacecraft_prm, n_states=7, n_controls=4, n_st_path_con=0, n_ct_path_con=1, n_event_con=13,
                         n_f_par=0, **kwargs)

        # Orbits parameters as an attribut
        self.orbits_prm = orbits_prm

    def set_boundaries(self):
        """ Setting of the states, controls, free-parameters, initial and final times
                boundaries """

        # X (syn. fram) [-]
        self.low_bnd.states[0] = 1 - self.cr3bp.mu - 1.5
        self.upp_bnd.states[0] = 1 - self.cr3bp.mu + 1.5

        # Y (syn. fram) [-]
        self.low_bnd.states[1] = - 0.1
        self.upp_bnd.states[1] = 0.1

        # Z (syn. fram) [-]
        self.low_bnd.states[2] = - 0.5
        self.upp_bnd.states[2] = 0.5

        # Vx (syn. fram) [-]
        self.low_bnd.states[3] = - 2
        self.upp_bnd.states[3] = 2

        # Vy (syn. fram) [-]
        self.low_bnd.states[4] = - 2
        self.upp_bnd.states[4] = 2

        # Vz (syn. fram) [-]
        self.low_bnd.states[5] = - 2
        self.upp_bnd.states[5] = 2

        # Set the mass limits [-]
        self.low_bnd.states[6] = 0
        self.upp_bnd.states[6] = self.spacecraft.mass0

        # Set the thrust limits [-]
        self.low_bnd.controls[0] = self.spacecraft.thrust_min 
        self.upp_bnd.controls[0] = self.spacecraft.thrust_max 

        # Set the limits for controls ux, uy, uz
        for i in [1, 2, 3]:
            self.low_bnd.controls[i] = -1
            self.upp_bnd.controls[i] = 1


        # Set the times limits
        t_init  = self.initial_guess.time[0]
        t_final = self.initial_guess.time[-1]

        self.low_bnd.ti = self.upp_bnd.ti = t_init

        self.low_bnd.tf = 0.5 * t_final
        self.upp_bnd.tf = 1.5 * t_final

    def end_point_cost(self, ti, xi, tf, xf, f_prm):
        """ Computation of the end point cost (Mayer term) 

            Parameters
            ----------
            ti : float
                Initial time value
            xi : array
                Initial states array
            tf : float
                Final time value
            xf : array
                Final states array
            f_prm : array
                Free parameters array

            Returns
            -------
            float
                Mayer term value

        """
        mf = xf[6]
        return - mf / self.spacecraft.mass0

    def path_constraints(self, states, controls, states_add, controls_add, controls_col, f_par):
        """ Computation of the path constraints 

            Parameters
            ----------
            states : ndarray
                Matrix of the states
            controls : ndarray 
                Matrix of the controls
            states_add : ndarray
                Matrix of the states at additional nodes
            controls_add : ndarray
                Matrix of the controls at additional nodes
            controls_col : ndarray
                Matrix of the controls at collocation nodes
            f_par : array
                Array of the free parameters

            Returns
            -------
            constraints_st : ndarray
                States path constraints matrix
            constraints_ct : ndarray
                Controls path constraints matrix
        """
        st_path = np.ndarray((self.prm['n_st_path_con'],
                            2*self.prm['n_nodes']-1), dtype=cppad_py.a_double)
        ct_path = np.ndarray((self.prm['n_ct_path_con'],
                            4*self.prm['n_nodes']-3), dtype=cppad_py.a_double)

        ux = np.concatenate((controls[1], controls_add[1], controls_col[1]))
        uy = np.concatenate((controls[2], controls_add[2], controls_col[2]))
        uz = np.concatenate((controls[3], controls_add[3], controls_col[3]))

        u2 = ux*ux + uy*uy + uz*uz

        ct_path[0] = u2

        return st_path, ct_path

    def set_path_constraints_boundaries(self):
        """ Setting of the path constraints boundaries """

        self.low_bnd.ct_path[0] = self.upp_bnd.ct_path[0] = 1

    def event_constraints(self, xi, ui, xf, uf, ti, tf, f_prm):
        """ Computation of the events constraints 

            Parameters
            ----------
            xi : array
                Array of states at initial time
            ui : array
                Array of controls at initial time
            xf : array
                Array of states at final time
            uf : array
                Array of controls at final time
            ti : float
                Value of initial time
            tf : float
                Value of final time
            f_prm : array
                Free parameters array

            Returns
            -------
            constraints : array
                Array of the event constraints

        """
        constraints = np.ndarray((self.prm['n_event_con'], 1),
                                 dtype=cppad_py.a_double)

        # xi, yi, zi [-]
        constraints[0] = xi[0]
        constraints[1] = xi[1]
        constraints[2] = xi[2]

        # vxi, vyi, vzi [-]
        constraints[3] = xi[3]
        constraints[4] = xi[4]
        constraints[5] = xi[5]

        # mi [-]
        constraints[6] = xi[6]

        # xf, yf, zf [-]
        constraints[7] = xf[0]
        constraints[8] = xf[1]
        constraints[9] = xf[2]

        # vxf, vyf, vzf [-]
        constraints[10] = xf[3]
        constraints[11] = xf[4]
        constraints[12] = xf[5]

        return constraints

    def set_events_constraints_boundaries(self):
        """ Setting of the events constraints boundaries """

        # Initial states
        xi, yi, zi, vxi, vyi, vzi, mi = self.initial_guess.states[:, 0]

        # Final states
        xf, yf, zf, vxf, vyf, vzf, mf = self.initial_guess.states[:, -1]

        # Position (init) [-]
        self.low_bnd.event[0] = self.upp_bnd.event[0] = xi
        self.low_bnd.event[1] = self.upp_bnd.event[1] = yi
        self.low_bnd.event[2] = self.upp_bnd.event[2] = zi

        # Velocity (init) [-]
        self.low_bnd.event[3] = self.upp_bnd.event[3] = vxi
        self.low_bnd.event[4] = self.upp_bnd.event[4] = vyi
        self.low_bnd.event[5] = self.upp_bnd.event[5] = vzi

        # Mass (init) [-]
        self.low_bnd.event[6] = self.upp_bnd.event[6] = mi

        # Position (final) [-]
        self.low_bnd.event[7] = self.upp_bnd.event[7] = xf
        self.low_bnd.event[8] = self.upp_bnd.event[8] = yf
        self.low_bnd.event[9] = self.upp_bnd.event[9] = zf

        # Velocity (final) [-]
        self.low_bnd.event[10] = self.upp_bnd.event[10] = vxf
        self.low_bnd.event[11] = self.upp_bnd.event[11] = vyf
        self.low_bnd.event[12] = self.upp_bnd.event[12] = vzf

    def set_initial_guess(self):
        """ Setting of the initial guess for the states, controls, free-parameters
            and time grid """

    	# Creation of the initial and final NRHO orbits objects
        if ('alti' in self.orbits_prm and 'altf' in self.orbits_prm):
            init_orbs = InitNRHOs(
                self.orbits_prm['family'], alti=self.orbits_prm['alti'], altf=self.orbits_prm['altf'])
        elif ('Azi' in self.orbits_prm and 'Azf' in self.orbits_prm):
            init_orbs = InitNRHOs(
                self.orbits_prm['family'], Azi=self.orbits_prm['Azi'], Azf=self.orbits_prm['Azf'])
        else:
            raise ValueError(
                'Initial and target orbits not correctly initialized')

        # Initial, final and flight time computation
        t_i = init_orbs.nrho_dep.period
        tof = init_orbs.nrho_arr.period / 2 + \
            (init_orbs.nrho_arr.period - init_orbs.nrho_dep.period)
        t_f = t_i + tof
        time = np.linspace(t_i, t_f, self.prm['n_nodes'])

        # Creation of the `TrajectoryStacking` object used to produce
        # the initial guess
        if ('alti' in self.orbits_prm and 'altf' in self.orbits_prm):
            traj_stck = TrajectoryStacking(
                t_i, tof, time, self.orbits_prm['family'], alti=self.orbits_prm['alti'], altf=self.orbits_prm['altf'])
        elif ('Azi' in self.orbits_prm and 'Azf' in self.orbits_prm):
            traj_stck = TrajectoryStacking(
                t_i, tof, time, self.orbits_prm['family'], Azi=self.orbits_prm['Azi'], Azf=self.orbits_prm['Azf'])

        # Production of the initial guess for x, y, z, vx, vy, vz
        traj_stck.propagate('arc_length')

        # Transposition of the states matrix to stuck to the  
        # representation standard of the transcription algorithm
        initial_states = np.transpose(traj_stck.state0)

        # States initial guess (x [-], y [-], z [-], vx [-], vy [-], vz [-], m [kg])
        self.initial_guess.states[:6] = initial_states
        self.initial_guess.states[6] = self.spacecraft.mass0

        # Velocity norm [-]
        v_norm = np.sqrt(
            initial_states[3]**2 + initial_states[4]**2 + initial_states[5]**2)

        # Controls initial guess (T [N], ux [-], uy [-], uz [-])
        self.initial_guess.controls[0] = np.zeros(
            self.prm['n_nodes']) 
        self.initial_guess.controls[1:] = initial_states[3:6] / v_norm

        # Time initial guess (t [-])
        self.initial_guess.time = time

if __name__ == '__main__':

    # Spacecraft properties
    m0 = 1000  # initial mass [kg]
    m_dry = 10  # dry mass [kg]
    thrust_max = 2  # maximum thrust [N]
    thrust_min = 0  # minimum thrust [N]
    Isp = 2000  # specific impulse [s]

    spacecraft_prm = {'mass0': m0, 'mass_dry': m_dry, 'thrust_max': thrust_max, 'thrust_min': thrust_min, 'isp': Isp}
                
    # Departure and arrival orbits properties
    family = 'southern' # NRHOs family
    Azi = 72000         # Initial NRHO periselene altitude [km]
    Azf = 73000         # Final NRHO periselene altitude   [km]

    orbits_prm = {'Azi': Azi, 'Azf': Azf, 'family': family}

    # Transcription options
    options = {'linear_solver': 'ma86', 'pickle_results': False}

    # Number of nodes
    n_nodes = 300

    # Instantiation of the problem
    problem = NRHO2NRHO(spacecraft_prm, orbits_prm, n_nodes=n_nodes)

    # Instantiation of the optimization
    optimization = Optimization(problem=problem, **options)

    # Launch of the optimization
    optimization.run()