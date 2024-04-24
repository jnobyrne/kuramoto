import numpy as np
from scipy.integrate import odeint


class Kuramoto:

    def __init__(self, coupling=1, model='kuramoto', dt=0.01, T=10, n_nodes=None, 
                 natfreqs=None, coupling2=None, offset=0, noise_strength=0):
        '''
        coupling: float or 2D ndarray
            Coupling strength. Default = 1. Typical values range between 0.4-2
        model : str
            Selects the model type. 'kuramoto', 'hmm', 'hmm_modified', 'ghkb' or 'experimental'
        dt: float
            Delta t for integration of equations.
        T: float
            Total time of simulated activity.
            From that the number of integration steps is T/dt.
        n_nodes: int, optional
            Number of oscillators.
            If None, it is inferred from len of natfreqs.
            Must be specified if natfreqs is not given.
        natfreqs: 1D ndarray, optional
            Natural oscillation frequencies.
            If None, then new random values will be generated and kept fixed
            for the object instance.
            Must be specified if n_nodes is not given.
            If given, it overrides the n_nodes argument.
        coupling2 : float, optional
            Coupling strength for the second harmonic in the HMM model
        offset : float, optional
            Dephasing offset for the HMM model.
        noise_strength : float, optional
            Standard deviation of the Gaussian noise term.
        '''
        if n_nodes is None and natfreqs is None:
            raise ValueError("n_nodes or natfreqs must be specified")

        self.dt = dt
        self.T = T
        self.coupling = coupling
        self.model = model
        self.coupling2 = coupling2
        self.offset = offset
        self.noise_strength = noise_strength

        if natfreqs is not None:
            self.natfreqs = natfreqs
            self.n_nodes = len(natfreqs)
        else:
            self.n_nodes = n_nodes
            self.natfreqs = np.random.normal(size=self.n_nodes)

    def init_angles(self):
        '''
        Random initial random angles (position, "theta").
        '''
        return 2 * np.pi * np.random.random(size=self.n_nodes)

    def derivative(self, angles_vec, t, adj_mat, coupling, model, coupling2, offset, noise_strength):
        '''
        Compute derivative of all nodes for current state, defined as

        dx_i    natfreq_i + k  sum_j ( Aij* sin (angle_j - angle_i) )
        ---- =             ---
         dt                M_i

        t: for compatibility with scipy.odeint
        '''
        assert len(angles_vec) == len(self.natfreqs) == len(adj_mat), \
            'Input dimensions do not match, check lengths'

        angles_i, angles_j = np.meshgrid(angles_vec, angles_vec)
        if model == 'kuramoto':
            interactions = adj_mat * np.sin(angles_j - angles_i) # Aij * sin(j-i)
            dxdt = self.natfreqs + (coupling * interactions).sum(axis=0)  # sum over incoming interactions
        elif model == 'hmm':
            rel_phase = angles_j - angles_i
            interactions = adj_mat * coupling * ( -np.sin(rel_phase + offset) 
                            + coupling2 * np.sin(2 * rel_phase) )
            dxdt = self.natfreqs + (interactions).sum(axis=0)  # sum over incoming interactions
        elif model == 'hmm2':
            rel_phase = angles_j - angles_i
            interactions = ( adj_mat * coupling * -np.sin(rel_phase + offset) 
                            + adj_mat * coupling2 * np.sin(2 * rel_phase) )
            dxdt = self.natfreqs + (interactions).sum(axis=0)  # sum over incoming interactions
        elif model == 'hmm_mod1':
            positive_coupling = np.zeros(np.shape(adj_mat))
            positive_coupling[coupling > 0] = coupling[coupling > 0]
            negative_coupling = np.zeros(np.shape(adj_mat))
            negative_coupling[coupling < 0] = np.abs(coupling[coupling < 0])
            rel_phase = angles_j - angles_i
            dxdt = ( self.natfreqs - (positive_coupling * adj_mat * np.sin(rel_phase - offset)).sum(axis=0)
                    + (negative_coupling * adj_mat * np.sin(2 * rel_phase)).sum(axis=0) )
        elif model == 'hmm_mod2':
            # no phase transition between -10,10 ; -10,10 and coupling2= -3
            rel_phase = angles_j - angles_i
            interactions = ( adj_mat * coupling2 * (-np.sin(rel_phase) 
                                 + coupling * np.sin(2 * rel_phase)) )
            dxdt = self.natfreqs + (interactions).sum(axis=0)  # sum over incoming interactions
        elif model == 'hmm_mod3':
            # same as mod1 except for a plus instead of a minus for the first interaction term
            noise = np.random.normal(0, noise_strength, adj_mat.shape[0])
            positive_coupling = np.zeros(np.shape(adj_mat))
            positive_coupling[coupling > 0] = coupling[coupling > 0]
            negative_coupling = np.zeros(np.shape(adj_mat))
            negative_coupling[coupling < 0] = np.abs(coupling[coupling < 0])
            rel_phase = angles_j - angles_i
            dxdt = ( self.natfreqs + (positive_coupling * adj_mat * np.sin(rel_phase - offset)).sum(axis=0)
                    + (negative_coupling * adj_mat * np.sin(2 * rel_phase)).sum(axis=0) ) + noise
        elif model == 'hmm_mod4':
            rel_phase = angles_j - angles_i
            interactions = adj_mat * (-np.sin(rel_phase + coupling) + coupling2 * np.sin(2 * rel_phase))
            dxdt = self.natfreqs + interactions.sum(axis=0)
        elif model == 'ghkb':
            rel_phase = angles_j - angles_i
            interactions = ( adj_mat * coupling * np.sin(rel_phase) 
                            - adj_mat * coupling2 * np.sin(2 * rel_phase) )
            dxdt = self.natfreqs + (interactions).sum(axis=0)  # sum over incoming interactions
        elif model == 'experimental':
            offsets = np.ones(np.shape(adj_mat)) * np.pi/2
            offsets[coupling < 0] = 3/2 * np.pi
            interactions = adj_mat * (np.sin(angles_j - angles_i - offsets) + 1 ) / 2
            dxdt = self.natfreqs + (np.abs(coupling) * interactions).sum(axis=0)  # sum over incoming interactions
        return dxdt

    def integrate(self, angles_vec, adj_mat):
        '''Updates all states by integrating state of all nodes'''
        # Coupling term (k / Mj) is constant in the integrated time window.
        # Compute it only once here and pass it to the derivative function
        n_interactions = (adj_mat != 0).sum(axis=0)  # number of incoming interactions
        coupling = self.coupling / n_interactions  # normalize coupling by number of interactions
        model = self.model
        coupling2 = self.coupling2
        offset = self.offset
        noise_strength = self.noise_strength

        t = np.linspace(0, self.T, int(self.T/self.dt))
        timeseries = odeint(self.derivative, angles_vec, t, args=(adj_mat, coupling, model, coupling2, offset, noise_strength))
        return timeseries.T  # transpose for consistency (act_mat:node vs time)

    def run(self, adj_mat=None, angles_vec=None):
        '''
        adj_mat: 2D nd array
            Adjacency matrix representing connectivity.
        angles_vec: 1D ndarray, optional
            States vector of nodes representing the position in radians.
            If not specified, random initialization [0, 2pi].

        Returns
        -------
        act_mat: 2D ndarray
            Activity matrix: node vs time matrix with the time series of all
            the nodes.
        '''
        if angles_vec is None:
            angles_vec = self.init_angles()

        return self.integrate(angles_vec, adj_mat)

    @staticmethod
    def phase_coherence(angles_vec):
        '''
        Compute global order parameter R_t - mean length of resultant vector
        '''
        suma = sum([(np.e ** (1j * i)) for i in angles_vec])
        return abs(suma / len(angles_vec))

    def mean_frequency(self, act_mat, adj_mat):
        '''
        Compute average frequency within the time window (self.T) for all nodes
        '''
        assert len(adj_mat) == act_mat.shape[0], 'adj_mat does not match act_mat'
        _, n_steps = act_mat.shape

        # Compute derivative for all nodes for all time steps
        dxdt = np.zeros_like(act_mat)
        for time in range(n_steps):
            dxdt[:, time] = self.derivative(act_mat[:, time], None, adj_mat)

        # Integrate all nodes over the time window T
        integral = np.sum(dxdt * self.dt, axis=1)
        # Average across complete time window - mean angular velocity (freq.)
        meanfreq = integral / self.T
        return meanfreq


