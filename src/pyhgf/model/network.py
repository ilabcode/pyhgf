# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from typing import Callable, Dict, Optional, Tuple

import jax.numpy as jnp
import numpy as np
from jax.lax import scan, switch
from jax.tree_util import Partial

from pyhgf.networks import beliefs_propagation, get_update_sequence
from pyhgf.typing import Attributes, Edges, Inputs, NetworkParameters, UpdateSequence


class Network:
    """A neural network for predictive coding applications.

    This is the core class to define and manipulate neural networks, that consists in
    1. attributes, 2. structure and 3. update sequences.

    Attributes
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.AdjacencyLists`. The tuple has the same length as the
        node number. For each node, the index lists the value/volatility
        parents/children.
    inputs :
        Information on the input nodes.
    node_trajectories :
        The dynamic of the node's beliefs after updating.
    update_sequence :
        The sequence of update functions that are applied during the belief propagation
        step.
    scan_fn :
        The function that is passed to :py:func:`jax.lax.scan`. This is a pre-
        parametrized version of :py:func:`pyhgf.networks.beliefs_propagation`.

    """

    def __init__(self) -> None:
        """Initialize an empty neural network."""
        self.edges: Edges = ()
        self.node_trajectories: Dict = {}
        self.attributes: Attributes = {}
        self.update_sequence: Optional[UpdateSequence] = None
        self.scan_fn: Optional[Callable] = None
        self.inputs: Inputs
        self.verbose: bool = False

    def create_belief_propagation_fn(self, overwrite: bool = True) -> "Network":
        """Create the belief propagation function.

        .. note:
        This step is called by default when using py:meth:`input_data`.

        Parameters
        ----------
        overwrite :
            If `True` (default), create a new belief propagation function and ignore
            preexisting values. Otherwise, do not create a new function if the attribute
            `scan_fn` is already defined.

        """
        # create the network structure from edges and inputs
        self.inputs = Inputs(self.inputs.idx, self.inputs.kind)
        self.structure = (self.inputs, self.edges)

        # create the update sequence if it does not already exist
        if self.update_sequence is None:
            self.set_update_sequence()
            if self.verbose:
                print("... Create the update sequence from the network structure.")

        # create the belief propagation function
        # this function is used by scan to loop over observations
        if self.scan_fn is None:
            self.scan_fn = Partial(
                beliefs_propagation,
                update_sequence=self.update_sequence,
                structure=self.structure,
            )
            if self.verbose:
                print("... Create the belief propagation function.")
        else:
            if overwrite:
                self.scan_fn = Partial(
                    beliefs_propagation,
                    update_sequence=self.update_sequence,
                    structure=self.structure,
                )
                if self.verbose:
                    print("... Create the belief propagation function (overwrite).")
            else:
                if self.verbose:
                    print("... The belief propagation function is already defined.")

        return self

    def cache_belief_propagation_fn(self) -> "Network":
        """Blank call to the belief propagation function.

        .. note:
           This step is called by default when using py:meth:`input_data`. It can
           sometimes be convenient to call this step independently to chache the JITed
           function before fitting the model.

        """
        if self.scan_fn is None:
            self = self.create_belief_propagation_fn()

        # blanck call to cache the JIT-ed functions
        _ = scan(
            self.scan_fn,
            self.attributes,
            (
                jnp.ones((1, len(self.inputs.idx))),
                jnp.ones((1, 1)),
                jnp.ones((1, 1)),
            ),
        )
        if self.verbose:
            print("... Cache the belief propagation function.")

        return self

    def input_data(
        self,
        input_data: np.ndarray,
        time_steps: Optional[np.ndarray] = None,
        observed: Optional[np.ndarray] = None,
    ) -> "Network":
        """Add new observations.

        Parameters
        ----------
        input_data :
            2d array of new observations (time x features).
        time_steps :
            Time vector (optional). If `None`, the time vector will default to
            `np.ones(len(input_data))`. This vector is automatically transformed
            into a time steps vector.
        observed :
            A 2d boolean array masking `input_data`. In case of missing inputs, (i.e.
            `observed` is `0`), the input node will have value and volatility set to
            `0.0`. If the parent(s) of this input receive prediction error from other
            children, they simply ignore this one. If they are not receiving other
            prediction errors, they are updated by keeping the same mean by decreasing
            the precision as a function of time to reflect the evolution of the
            underlying Gaussian Random Walk.
        .. warning::
            Missing inputs are missing observations from the agent's perspective and
            should not be used to handle missing data points that were observed (e.g.
            missing in the event log, or rejected trials).

        """
        if self.scan_fn is None:
            self = self.create_belief_propagation_fn()
        if self.verbose:
            print((f"Adding {len(input_data)} new observations."))
        if time_steps is None:
            time_steps = np.ones((len(input_data), 1))  # time steps vector
        else:
            time_steps = time_steps[..., jnp.newaxis]
        if input_data.ndim == 1:
            input_data = input_data[..., jnp.newaxis]

        # is it observation or missing inputs
        if observed is None:
            observed = np.ones(input_data.shape, dtype=int)

        # this is where the model loops over the whole input time series
        # at each time point, the node structure is traversed and beliefs are updated
        # using precision-weighted prediction errors
        _, node_trajectories = scan(
            self.scan_fn, self.attributes, (input_data, time_steps, observed)
        )

        # trajectories of the network attributes a each time point
        self.node_trajectories = node_trajectories

        return self

    def input_custom_sequence(
        self,
        update_branches: Tuple[UpdateSequence],
        branches_idx: np.array,
        input_data: np.ndarray,
        time_steps: Optional[np.ndarray] = None,
        observed: Optional[np.ndarray] = None,
    ):
        """Add new observations with custom update sequences.

        This method should be used when the update sequence should be adapted to the
        input data. (e.g. in the case of missing/null observations that should not
        trigger node update).

        .. note::
           When the dynamic adaptation of the update sequence is not required, it is
           recommended to use :py:meth:`pyhgf.model.HGF.input_data` instead as this
           might result in performance improvement.

        Parameters
        ----------
        update_branches :
            A tuple of UpdateSequence listing the possible update sequences.
        branches_idx :
            The branches indexes (integers). Should have the same length as the input
            data.
        input_data :
            2d array of new observations (time x features).
        time_steps :
            Time vector (optional). If `None`, the time vector will default to
            `np.ones(len(input_data))`. This vector is automatically transformed
            into a time steps vector.
        observed :
            A 2d boolean array masking `input_data`. In case of missing inputs, (i.e.
            `observed` is `0`), the input node will have value and volatility set to
            `0.0`. If the parent(s) of this input receive prediction error from other
            children, they simply ignore this one. If they are not receiving other
            prediction errors, they are updated by keeping the same mean be decreasing
            the precision as a function of time to reflect the evolution of the
            underlying Gaussian Random Walk.
        .. warning::
            Missing inputs are missing observations from the agent's perspective and
            should not be used to handle missing data points that were observed (e.g.
            missing in the event log, or rejected trials).

        """
        if self.verbose:
            print((f"Adding {len(input_data)} new observations."))
        if time_steps is None:
            time_steps = np.ones(len(input_data))  # time steps vector

        # concatenate data and time
        if time_steps is None:
            time_steps = np.ones((len(input_data), 1))  # time steps vector
        else:
            time_steps = time_steps[..., jnp.newaxis]
        if input_data.ndim == 1:
            input_data = input_data[..., jnp.newaxis]

        # is it observation or missing inputs
        if observed is None:
            observed = np.ones(input_data.shape, dtype=int)

        # create the update functions that will be scanned
        branches_fn = [
            Partial(
                beliefs_propagation,
                update_sequence=seq,
                structure=self.structure,
            )
            for seq in update_branches
        ]

        # create the function that will be scanned
        def switching_propagation(attributes, scan_input):
            data, idx = scan_input
            return switch(idx, branches_fn, attributes, data)

        # wrap the inputs
        scan_input = (input_data, time_steps, observed), branches_idx

        # scan over the input data and apply the switching belief propagation functions
        _, node_trajectories = scan(switching_propagation, self.attributes, scan_input)

        # the node structure at each value updates
        self.node_trajectories = node_trajectories

        # because some of the input nodes might not have been updated, here we manually
        # insert the input data to the input node (without triggering updates)
        for idx, inp in zip(self.inputs.idx, range(input_data.shape[1])):
            self.node_trajectories[idx]["value"] = input_data[inp]

        return self

    def get_network(self) -> NetworkParameters:
        """Return the attributes, structure and update sequence defining the network."""
        if self.scan_fn is None:
            self = self.create_belief_propagation_fn()

        assert self.update_sequence is not None

        return self.attributes, self.structure, self.update_sequence

    def set_update_sequence(self, update_type: str = "eHGF") -> "Network":
        """Generate an update sequence from the network's structure.

        See :py:func:`pyhgf.networks.get_update_sequence` for more details.

        Parameters
        ----------
        update_type :
            Only relevant if the neural network is an instance of generalised
            Hierarchical Gaussian Filter. The default prediction update to use for
            continuous nodes (`"eHGF"` or `"standard"`). Defaults to `"eHGF"`.

        """
        self.update_sequence = tuple(
            get_update_sequence(network=self, update_type=update_type)
        )

        return self
