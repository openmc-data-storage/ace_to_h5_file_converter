from __future__ import annotations

import copy
import enum
import os
import struct
import typing  # required to prevent typing.Union namespace overwriting Union
from abc import ABC, abstractmethod
from collections.abc import Iterable
from io import StringIO
from numbers import Integral, Real
from pathlib import Path
from warnings import warn

import numpy as np



# Type for arguments that accept file paths
PathLike = typing.Union[str, os.PathLike]

import io
import itertools
import json
import os
import re
from math import log, sqrt
from pathlib import Path, PurePath
from typing import Dict
from warnings import warn

import numpy as np

_CYTHON = False

import copy
import io
import os
import shutil
import tempfile
import warnings
from abc import ABC, abstractmethod
from collections import namedtuple
from collections.abc import (Callable, Iterable, Mapping, MutableMapping,
                             MutableSequence)
from copy import deepcopy
from functools import reduce
from io import StringIO
from itertools import zip_longest
from math import exp, log, log10
from numbers import Integral, Real
from pathlib import Path
from subprocess import PIPE, STDOUT, CalledProcessError, Popen
from warnings import warn

import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial



try:
    from .econstruct import (penetration_shift, reconstruct_mlbw,
                             reconstruct_rm, reconstruct_slbw, wave_number)
    _reconstruct = True
except ImportError:
    _reconstruct = False


import math
import typing
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable
from copy import deepcopy
from numbers import Integral, Real
from warnings import warn

import lxml.etree as ET
import numpy as np
import pandas as pd


class EqualityMixin:
    """A Class which provides a generic __eq__ method that can be inherited
    by downstream classes.
    """

    def __eq__(self, other):
        if isinstance(other, type(self)):
            for key, value in self.__dict__.items():
                if isinstance(value, np.ndarray):
                    if not np.array_equal(value, other.__dict__.get(key)):
                        return False
                else:
                    return value == other.__dict__.get(key)
        else:
            return False

        return True


class AngleEnergy(EqualityMixin, ABC):
    """Distribution in angle and energy of a secondary particle."""
    @abstractmethod
    def to_hdf5(self, group):
        pass

    @staticmethod
    def from_hdf5(group):
        """Generate angle-energy distribution from HDF5 data

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to read from

        Returns
        -------
        openmc.data.AngleEnergy
            Angle-energy distribution

        """
        dist_type = group.attrs['type'].decode()
        if dist_type == 'uncorrelated':
            return data.UncorrelatedAngleEnergy.from_hdf5(group)
        elif dist_type == 'correlated':
            return CorrelatedAngleEnergy.from_hdf5(group)
        elif dist_type == 'kalbach-mann':
            return data.KalbachMann.from_hdf5(group)
        elif dist_type == 'nbody':
            return data.NBodyPhaseSpace.from_hdf5(group)
        elif dist_type == 'coherent_elastic':
            return CoherentElasticAE.from_hdf5(group)
        elif dist_type == 'incoherent_elastic':
            return data.IncoherentElasticAE.from_hdf5(group)
        elif dist_type == 'incoherent_elastic_discrete':
            return data.IncoherentElasticAEDiscrete.from_hdf5(group)
        elif dist_type == 'incoherent_inelastic_discrete':
            return data.IncoherentInelasticAEDiscrete.from_hdf5(group)
        elif dist_type == 'incoherent_inelastic':
            return data.IncoherentInelasticAE.from_hdf5(group)
        elif dist_type == 'mixed_elastic':
            return data.MixedElasticAE.from_hdf5(group)

    @staticmethod
    def from_ace(ace, location_dist, location_start, rx=None):
        """Generate an angle-energy distribution from ACE data

        Parameters
        ----------
        ace : openmc.data.ace.Table
            ACE table to read from
        location_dist : int
            Index in the XSS array corresponding to the start of a block,
            e.g. JXS(11) for the the DLW block.
        location_start : int
            Index in the XSS array corresponding to the start of an energy
            distribution array
        rx : Reaction
            Reaction this energy distribution will be associated with

        Returns
        -------
        distribution : openmc.data.AngleEnergy
            Secondary angle-energy distribution

        """
        # Set starting index for energy distribution
        idx = location_dist + location_start - 1

        law = int(ace.xss[idx + 1])
        location_data = int(ace.xss[idx + 2])

        # Position index for reading law data
        idx = location_dist + location_data - 1

        # Parse energy distribution data
        if law == 2:
            distribution = UncorrelatedAngleEnergy()
            distribution.energy = DiscretePhoton.from_ace(ace, idx)
        elif law in (3, 33):
            distribution = UncorrelatedAngleEnergy()
            distribution.energy = LevelInelastic.from_ace(ace, idx)
        elif law == 4:
            distribution = UncorrelatedAngleEnergy()
            distribution.energy = ContinuousTabular.from_ace(
                ace, idx, location_dist)
        elif law == 5:
            distribution = UncorrelatedAngleEnergy()
            distribution.energy = GeneralEvaporation.from_ace(ace, idx)
        elif law == 7:
            distribution = UncorrelatedAngleEnergy()
            distribution.energy = MaxwellEnergy.from_ace(ace, idx)
        elif law == 9:
            distribution = UncorrelatedAngleEnergy()
            distribution.energy = Evaporation.from_ace(ace, idx)
        elif law == 11:
            distribution = UncorrelatedAngleEnergy()
            distribution.energy = WattEnergy.from_ace(ace, idx)
        elif law == 44:
            distribution = KalbachMann.from_ace(
                ace, idx, location_dist)
        elif law == 61:
            distribution = CorrelatedAngleEnergy.from_ace(
                ace, idx, location_dist)
        elif law == 66:
            distribution = NBodyPhaseSpace.from_ace(
                ace, idx, rx.q_value)
        else:
            raise ValueError("Unsupported ACE secondary energy "
                             "distribution law {}".format(law))

        return distribution


class Table(EqualityMixin):
    """ACE cross section table

    Parameters
    ----------
    name : str
        ZAID identifier of the table, e.g. '92235.70c'.
    atomic_weight_ratio : float
        Atomic mass ratio of the target nuclide.
    temperature : float
        Temperature of the target nuclide in MeV.
    pairs : list of tuple
        16 pairs of ZAIDs and atomic weight ratios. Used for thermal scattering
        tables to indicate what isotopes scattering is applied to.
    nxs : numpy.ndarray
        Array that defines various lengths with in the table
    jxs : numpy.ndarray
        Array that gives locations in the ``xss`` array for various blocks of
        data
    xss : numpy.ndarray
        Raw data for the ACE table

    Attributes
    ----------
    data_type : TableType
        Type of the ACE data

    """
    def __init__(self, name, atomic_weight_ratio, temperature, pairs,
                 nxs, jxs, xss):
        self.name = name
        self.atomic_weight_ratio = atomic_weight_ratio
        self.temperature = temperature
        self.pairs = pairs
        self.nxs = nxs
        self.jxs = jxs
        self.xss = xss

    @property
    def zaid(self):
        return self.name.split('.')[0]

    @property
    def data_type(self):
        xs = self.name.split('.')[1]
        return TableType.from_suffix(xs[-1])

    def __repr__(self):
        return "<ACE Table: {}>".format(self.name)




def clean_indentation(element, level=0, spaces_per_level=2, trailing_indent=True):
    """Set indentation of XML element and its sub-elements.
    Copied and pasted from https://effbot.org/zone/element-lib.htm#prettyprint.
    It walks your tree and adds spaces and newlines so the tree is
    printed in a nice way.

    Parameters
    ----------
    level : int
        Indentation level for the element passed in (default 0)
    spaces_per_level : int
        Number of spaces per indentation level (default 2)
    trailing_indent : bool
        Whether or not to add indentation after closing the element

    """
    i = "\n" + level*spaces_per_level*" "

    # ensure there's always some tail for the element passed in
    if not element.tail:
        element.tail = ""

    if len(element):
        if not element.text or not element.text.strip():
            element.text = i + spaces_per_level*" "
        if trailing_indent and (not element.tail or not element.tail.strip()):
            element.tail = i
        for sub_element in element:
            # `trailing_indent` is intentionally not forwarded to the recursive
            # call. Any child element of the topmost element should add
            # indentation at the end to ensure its parent's indentation is
            # correct.
            clean_indentation(sub_element, level+1, spaces_per_level)
        if not sub_element.tail or not sub_element.tail.strip():
            sub_element.tail = i
    else:
        if trailing_indent and level and (not element.tail or not element.tail.strip()):
            element.tail = i


def get_text(elem, name, default=None):
    """Retrieve text of an attribute or subelement.

    Parameters
    ----------
    elem : lxml.etree._Element
        Element from which to search
    name : str
        Name of attribute/subelement
    default : object
        A defult value to return if matching attribute/subelement exists

    Returns
    -------
    str
        Text of attribute or subelement

    """
    if name in elem.attrib:
        return elem.get(name, default)
    else:
        child = elem.find(name)
        return child.text if child is not None else default


def reorder_attributes(root):
    """Sort attributes in XML to preserve pre-Python 3.8 behavior

    Parameters
    ----------
    root : lxml.etree._Element
        Root element

    """
    for el in root.iter():
        attrib = el.attrib
        if len(attrib) > 1:
            # adjust attribute order, e.g. by sorting
            attribs = sorted(attrib.items())
            attrib.clear()
            attrib.update(attribs)


def get_elem_tuple(elem, name, dtype=int):
    '''Helper function to get a tuple of values from an elem

    Parameters
    ----------
    elem : lxml.etree._Element
        XML element that should contain a tuple
    name : str
        Name of the subelement to obtain tuple from
    dtype : data-type
        The type of each element in the tuple

    Returns
    -------
    tuple of dtype
        Data read from the tuple
    '''
    subelem = elem.find(name)
    if subelem is not None:
        return tuple([dtype(x) for x in subelem.text.split()])


class ProbabilityTables(EqualityMixin):
    r"""Unresolved resonance region probability tables.

    Parameters
    ----------
    energy : Iterable of float
        Energies in eV at which probability tables exist
    table : numpy.ndarray
        Probability tables for each energy. This array is of shape (N, 6, M)
        where N is the number of energies and M is the number of bands. The
        second dimension indicates whether the value is for the cumulative
        probability (0), total (1), elastic (2), fission (3), :math:`(n,\gamma)`
        (4), or heating number (5).
    interpolation : {2, 5}
        Interpolation scheme between tables
    inelastic_flag : int
        A value less than zero indicates that the inelastic cross section is
        zero within the unresolved energy range. A value greater than zero
        indicates the MT number for a reaction whose cross section is to be used
        in the unresolved range.
    absorption_flag : int
        A value less than zero indicates that the "other absorption" cross
        section is zero within the unresolved energy range. A value greater than
        zero indicates the MT number for a reaction whose cross section is to be
        used in the unresolved range.
    multiply_smooth : bool
        Indicate whether probability table values are cross sections (False) or
        whether they must be multiply by the corresponding "smooth" cross
        sections (True).

    Attributes
    ----------
    energy : Iterable of float
        Energies in eV at which probability tables exist
    table : numpy.ndarray
        Probability tables for each energy. This array is of shape (N, 6, M)
        where N is the number of energies and M is the number of bands. The
        second dimension indicates whether the value is for the cumulative
        probability (0), total (1), elastic (2), fission (3), :math:`(n,\gamma)`
        (4), or heating number (5).
    interpolation : {2, 5}
        Interpolation scheme between tables
    inelastic_flag : int
        A value less than zero indicates that the inelastic cross section is
        zero within the unresolved energy range. A value greater than zero
        indicates the MT number for a reaction whose cross section is to be used
        in the unresolved range.
    absorption_flag : int
        A value less than zero indicates that the "other absorption" cross
        section is zero within the unresolved energy range. A value greater than
        zero indicates the MT number for a reaction whose cross section is to be
        used in the unresolved range.
    multiply_smooth : bool
        Indicate whether probability table values are cross sections (False) or
        whether they must be multiply by the corresponding "smooth" cross
        sections (True).
    """

    def __init__(self, energy, table, interpolation, inelastic_flag=-1,
                 absorption_flag=-1, multiply_smooth=False):
        self.energy = energy
        self.table = table
        self.interpolation = interpolation
        self.inelastic_flag = inelastic_flag
        self.absorption_flag = absorption_flag
        self.multiply_smooth = multiply_smooth

    @property
    def absorption_flag(self):
        return self._absorption_flag

    @absorption_flag.setter
    def absorption_flag(self, absorption_flag):
        check_type('absorption flag', absorption_flag, Integral)
        self._absorption_flag = absorption_flag

    @property
    def energy(self):
        return self._energy

    @energy.setter
    def energy(self, energy):
        check_type('probability table energies', energy, Iterable, Real)
        self._energy = energy

    @property
    def inelastic_flag(self):
        return self._inelastic_flag

    @inelastic_flag.setter
    def inelastic_flag(self, inelastic_flag):
        check_type('inelastic flag', inelastic_flag, Integral)
        self._inelastic_flag = inelastic_flag

    @property
    def interpolation(self):
        return self._interpolation

    @interpolation.setter
    def interpolation(self, interpolation):
        check_value('interpolation', interpolation, [2, 5])
        self._interpolation = interpolation

    @property
    def multiply_smooth(self):
        return self._multiply_smooth

    @multiply_smooth.setter
    def multiply_smooth(self, multiply_smooth):
        check_type('multiply by smooth', multiply_smooth, bool)
        self._multiply_smooth = multiply_smooth

    @property
    def table(self):
        return self._table

    @table.setter
    def table(self, table):
        check_type('probability tables', table, np.ndarray)
        self._table = table

    def to_hdf5(self, group):
        """Write probability tables to an HDF5 group

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to write to

        """
        group.attrs['interpolation'] = self.interpolation
        group.attrs['inelastic'] = self.inelastic_flag
        group.attrs['absorption'] = self.absorption_flag
        group.attrs['multiply_smooth'] = int(self.multiply_smooth)

        group.create_dataset('energy', data=self.energy)
        group.create_dataset('table', data=self.table)

    @classmethod
    def from_hdf5(cls, group):
        """Generate probability tables from HDF5 data

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to read from

        Returns
        -------
        openmc.ProbabilityTables
            Probability tables

        """
        interpolation = group.attrs['interpolation']
        inelastic_flag = group.attrs['inelastic']
        absorption_flag = group.attrs['absorption']
        multiply_smooth = bool(group.attrs['multiply_smooth'])

        energy = group['energy'][()]
        table = group['table'][()]

        return cls(energy, table, interpolation, inelastic_flag,
                   absorption_flag, multiply_smooth)

    @classmethod
    def from_ace(cls, ace):
        """Generate probability tables from an ACE table

        Parameters
        ----------
        ace : openmc.data.ace.Table
            ACE table to read from

        Returns
        -------
        openmc.data.ProbabilityTables
            Unresolved resonance region probability tables

        """
        # Check if URR probability tables are present
        idx = ace.jxs[23]
        if idx == 0:
            return None

        N = int(ace.xss[idx])      # Number of incident energies
        M = int(ace.xss[idx+1])    # Length of probability table
        interpolation = int(ace.xss[idx+2])
        inelastic_flag = int(ace.xss[idx+3])
        absorption_flag = int(ace.xss[idx+4])
        multiply_smooth = (int(ace.xss[idx+5]) == 1)
        idx += 6

        # Get energies at which tables exist
        energy = ace.xss[idx : idx+N]*EV_PER_MEV
        idx += N

        # Get probability tables
        table = ace.xss[idx : idx+N*6*M].copy()
        table.shape = (N, 6, M)

        # Convert units on heating numbers
        table[:,5,:] *= EV_PER_MEV

        return cls(energy, table, interpolation, inelastic_flag,
                   absorption_flag, multiply_smooth)



_INTERPOLATION_SCHEMES = [
    'histogram',
    'linear-linear',
    'linear-log',
    'log-linear',
    'log-log'
]


class Univariate(EqualityMixin, ABC):
    """Probability distribution of a single random variable.

    The Univariate class is an abstract class that can be derived to implement a
    specific probability distribution.

    """
    @abstractmethod
    def to_xml_element(self, element_name):
        return ''

    @abstractmethod
    def __len__(self):
        return 0

    @classmethod
    @abstractmethod
    def from_xml_element(cls, elem):
        distribution = get_text(elem, 'type')
        if distribution == 'discrete':
            return Discrete.from_xml_element(elem)
        elif distribution == 'uniform':
            return Uniform.from_xml_element(elem)
        elif distribution == 'powerlaw':
            return PowerLaw.from_xml_element(elem)
        elif distribution == 'maxwell':
            return Maxwell.from_xml_element(elem)
        elif distribution == 'watt':
            return Watt.from_xml_element(elem)
        elif distribution == 'normal':
            return Normal.from_xml_element(elem)
        elif distribution == 'muir':
            # Support older files where Muir had its own class
            params = [float(x) for x in get_text(elem, 'parameters').split()]
            return muir(*params)
        elif distribution == 'tabular':
            return Tabular.from_xml_element(elem)
        elif distribution == 'legendre':
            return Legendre.from_xml_element(elem)
        elif distribution == 'mixture':
            return Mixture.from_xml_element(elem)

    @abstractmethod
    def sample(n_samples: int = 1, seed: typing.Optional[int] = None):
        """Sample the univariate distribution

        Parameters
        ----------
        n_samples : int
            Number of sampled values to generate
        seed : int or None
            Initial random number seed.

        Returns
        -------
        numpy.ndarray
            A 1-D array of sampled values
        """
        pass

    def integral(self):
        """Return integral of distribution

        .. versionadded:: 0.13.1

        Returns
        -------
        float
            Integral of distribution
        """
        return 1.0


class Discrete(Univariate):
    """Distribution characterized by a probability mass function.

    The Discrete distribution assigns probability values to discrete values of a
    random variable, rather than expressing the distribution as a continuous
    random variable.

    Parameters
    ----------
    x : Iterable of float
        Values of the random variable
    p : Iterable of float
        Discrete probability for each value

    Attributes
    ----------
    x : numpy.ndarray
        Values of the random variable
    p : numpy.ndarray
        Discrete probability for each value

    """

    def __init__(self, x, p):
        self.x = x
        self.p = p

    def __len__(self):
        return len(self.x)

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x):
        if isinstance(x, Real):
            x = [x]
        check_type('discrete values', x, Iterable, Real)
        self._x = np.array(x, dtype=float)

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, p):
        if isinstance(p, Real):
            p = [p]
        check_type('discrete probabilities', p, Iterable, Real)
        for pk in p:
            check_greater_than('discrete probability', pk, 0.0, True)
        self._p = np.array(p, dtype=float)

    def cdf(self):
        return np.insert(np.cumsum(self.p), 0, 0.0)

    def sample(self, n_samples=1, seed=None):
        np.random.seed(seed)
        p = self.p / self.p.sum()
        return np.random.choice(self.x, n_samples, p=p)

    def normalize(self):
        """Normalize the probabilities stored on the distribution"""
        norm = sum(self.p)
        self.p = [val / norm for val in self.p]

    def to_xml_element(self, element_name):
        """Return XML representation of the discrete distribution

        Parameters
        ----------
        element_name : str
            XML element name

        Returns
        -------
        element : lxml.etree._Element
            XML element containing discrete distribution data

        """
        element = ET.Element(element_name)
        element.set("type", "discrete")

        params = ET.SubElement(element, "parameters")
        params.text = ' '.join(map(str, self.x)) + ' ' + ' '.join(map(str, self.p))

        return element

    @classmethod
    def from_xml_element(cls, elem: ET.Element):
        """Generate discrete distribution from an XML element

        Parameters
        ----------
        elem : lxml.etree._Element
            XML element

        Returns
        -------
        openmc.stats.Discrete
            Discrete distribution generated from XML element

        """
        params = [float(x) for x in get_text(elem, 'parameters').split()]
        x = params[:len(params)//2]
        p = params[len(params)//2:]
        return cls(x, p)

    @classmethod
    def merge(
        cls,
        dists: typing.Sequence[Discrete],
        probs: typing.Sequence[int]
    ):
        """Merge multiple discrete distributions into a single distribution

        .. versionadded:: 0.13.1

        Parameters
        ----------
        dists : iterable of openmc.stats.Discrete
            Discrete distributions to combine
        probs : iterable of float
            Probability of each distribution

        Returns
        -------
        openmc.stats.Discrete
            Combined discrete distribution

        """
        if len(dists) != len(probs):
            raise ValueError("Number of distributions and probabilities must match.")

        # Combine distributions accounting for duplicate x values
        x_merged = set()
        p_merged = defaultdict(float)
        for dist, p_dist in zip(dists, probs):
            for x, p in zip(dist.x, dist.p):
                x_merged.add(x)
                p_merged[x] += p*p_dist

        # Create values and probabilities as arrays
        x_arr = np.array(sorted(x_merged))
        p_arr = np.array([p_merged[x] for x in x_arr])
        return cls(x_arr, p_arr)

    def integral(self):
        """Return integral of distribution

        .. versionadded:: 0.13.1

        Returns
        -------
        float
            Integral of discrete distribution
        """
        return np.sum(self.p)

    def clip(self, tolerance: float = 1e-6, inplace: bool = False) -> Discrete:
        r"""Remove low-importance points from discrete distribution.

        Given a probability mass function :math:`p(x)` with :math:`\{x_1, x_2,
        x_3, \dots\}` the possible values of the random variable with
        corresponding probabilities :math:`\{p_1, p_2, p_3, \dots\}`, this
        function will remove any low-importance points such that :math:`\sum_i
        x_i p_i` is preserved to within some threshold.

        .. versionadded:: 0.13.4

        Parameters
        ----------
        tolerance : float
            Maximum fraction of :math:`\sum_i x_i p_i` that will be discarded.
        inplace : bool
            Whether to modify the current object in-place or return a new one.

        Returns
        -------
        Discrete distribution with low-importance points removed

        """
        # Determine (reversed) sorted order of probabilities
        intensity = self.p * self.x
        index_sort = np.argsort(intensity)[::-1]

        # Get probabilities in above order
        sorted_intensity = intensity[index_sort]

        # Determine cumulative sum of probabilities
        cumsum = np.cumsum(sorted_intensity)
        cumsum /= cumsum[-1]

        # Find index which satisfies cutoff
        index_cutoff = np.searchsorted(cumsum, 1.0 - tolerance)

        # Now get indices up to cutoff
        new_indices = index_sort[:index_cutoff + 1]
        new_indices.sort()

        # Create new discrete distribution
        if inplace:
            self.x = self.x[new_indices]
            self.p = self.p[new_indices]
            return self
        else:
            new_x = self.x[new_indices]
            new_p = self.p[new_indices]
            return type(self)(new_x, new_p)


class Uniform(Univariate):
    """Distribution with constant probability over a finite interval [a,b]

    Parameters
    ----------
    a : float, optional
        Lower bound of the sampling interval. Defaults to zero.
    b : float, optional
        Upper bound of the sampling interval. Defaults to unity.

    Attributes
    ----------
    a : float
        Lower bound of the sampling interval
    b : float
        Upper bound of the sampling interval

    """

    def __init__(self, a: float = 0.0, b: float = 1.0):
        self.a = a
        self.b = b

    def __len__(self):
        return 2

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, a):
        check_type('Uniform a', a, Real)
        self._a = a

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, b):
        check_type('Uniform b', b, Real)
        self._b = b

    def to_tabular(self):
        prob = 1./(self.b - self.a)
        t = Tabular([self.a, self.b], [prob, prob], 'histogram')
        t.c = [0., 1.]
        return t

    def sample(self, n_samples=1, seed=None):
        np.random.seed(seed)
        return np.random.uniform(self.a, self.b, n_samples)

    def to_xml_element(self, element_name: str):
        """Return XML representation of the uniform distribution

        Parameters
        ----------
        element_name : str
            XML element name

        Returns
        -------
        element : lxml.etree._Element
            XML element containing uniform distribution data

        """
        element = ET.Element(element_name)
        element.set("type", "uniform")
        element.set("parameters", '{} {}'.format(self.a, self.b))
        return element

    @classmethod
    def from_xml_element(cls, elem: ET.Element):
        """Generate uniform distribution from an XML element

        Parameters
        ----------
        elem : lxml.etree._Element
            XML element

        Returns
        -------
        openmc.stats.Uniform
            Uniform distribution generated from XML element

        """
        params = get_text(elem, 'parameters').split()
        return cls(*map(float, params))


class PowerLaw(Univariate):
    """Distribution with power law probability over a finite interval [a,b]

    The power law distribution has density function :math:`p(x) dx = c x^n dx`.

    .. versionadded:: 0.13.0

    Parameters
    ----------
    a : float, optional
        Lower bound of the sampling interval. Defaults to zero.
    b : float, optional
        Upper bound of the sampling interval. Defaults to unity.
    n : float, optional
        Power law exponent. Defaults to zero, which is equivalent to a uniform
        distribution.

    Attributes
    ----------
    a : float
        Lower bound of the sampling interval
    b : float
        Upper bound of the sampling interval
    n : float
        Power law exponent

    """

    def __init__(self, a: float = 0.0, b: float = 1.0, n: float = 0.):
        self.a = a
        self.b = b
        self.n = n

    def __len__(self):
        return 3

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, a):
        check_type('interval lower bound', a, Real)
        self._a = a

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, b):
        check_type('interval upper bound', b, Real)
        self._b = b

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, n):
        check_type('power law exponent', n, Real)
        self._n = n

    def sample(self, n_samples=1, seed=None):
        np.random.seed(seed)
        xi = np.random.rand(n_samples)
        pwr = self.n + 1
        offset = self.a**pwr
        span = self.b**pwr - offset
        return np.power(offset + xi * span, 1/pwr)

    def to_xml_element(self, element_name: str):
        """Return XML representation of the power law distribution

        Parameters
        ----------
        element_name : str
            XML element name

        Returns
        -------
        element : lxml.etree._Element
            XML element containing distribution data

        """
        element = ET.Element(element_name)
        element.set("type", "powerlaw")
        element.set("parameters", f'{self.a} {self.b} {self.n}')
        return element

    @classmethod
    def from_xml_element(cls, elem: ET.Element):
        """Generate power law distribution from an XML element

        Parameters
        ----------
        elem : lxml.etree._Element
            XML element

        Returns
        -------
        openmc.stats.PowerLaw
            Distribution generated from XML element

        """
        params = get_text(elem, 'parameters').split()
        return cls(*map(float, params))


class Maxwell(Univariate):
    r"""Maxwellian distribution in energy.

    The Maxwellian distribution in energy is characterized by a single parameter
    :math:`\theta` and has a density function :math:`p(E) dE = c \sqrt{E}
    e^{-E/\theta} dE`.

    Parameters
    ----------
    theta : float
        Effective temperature for distribution in eV

    Attributes
    ----------
    theta : float
        Effective temperature for distribution in eV

    """

    def __init__(self, theta):
        self.theta = theta

    def __len__(self):
        return 1

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, theta):
        check_type('Maxwell temperature', theta, Real)
        check_greater_than('Maxwell temperature', theta, 0.0)
        self._theta = theta

    def sample(self, n_samples=1, seed=None):
        np.random.seed(seed)
        return self.sample_maxwell(self.theta, n_samples)

    @staticmethod
    def sample_maxwell(t, n_samples: int):
        r1, r2, r3 = np.random.rand(3, n_samples)
        c = np.cos(0.5 * np.pi * r3)
        return -t * (np.log(r1) + np.log(r2) * c * c)

    def to_xml_element(self, element_name: str):
        """Return XML representation of the Maxwellian distribution

        Parameters
        ----------
        element_name : str
            XML element name

        Returns
        -------
        element : lxml.etree._Element
            XML element containing Maxwellian distribution data

        """
        element = ET.Element(element_name)
        element.set("type", "maxwell")
        element.set("parameters", str(self.theta))
        return element

    @classmethod
    def from_xml_element(cls, elem: ET.Element):
        """Generate Maxwellian distribution from an XML element

        Parameters
        ----------
        elem : lxml.etree._Element
            XML element

        Returns
        -------
        openmc.stats.Maxwell
            Maxwellian distribution generated from XML element

        """
        theta = float(get_text(elem, 'parameters'))
        return cls(theta)


class Watt(Univariate):
    r"""Watt fission energy spectrum.

    The Watt fission energy spectrum is characterized by two parameters
    :math:`a` and :math:`b` and has density function :math:`p(E) dE = c e^{-E/a}
    \sinh \sqrt{b \, E} dE`.

    Parameters
    ----------
    a : float
        First parameter of distribution in units of eV
    b : float
        Second parameter of distribution in units of 1/eV

    Attributes
    ----------
    a : float
        First parameter of distribution in units of eV
    b : float
        Second parameter of distribution in units of 1/eV

    """

    def __init__(self, a=0.988e6, b=2.249e-6):
        self.a = a
        self.b = b

    def __len__(self):
        return 2

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, a):
        check_type('Watt a', a, Real)
        check_greater_than('Watt a', a, 0.0)
        self._a = a

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, b):
        check_type('Watt b', b, Real)
        check_greater_than('Watt b', b, 0.0)
        self._b = b

    def sample(self, n_samples=1, seed=None):
        np.random.seed(seed)
        w = Maxwell.sample_maxwell(self.a, n_samples)
        u = np.random.uniform(-1., 1., n_samples)
        aab = self.a * self.a * self.b
        return w + 0.25*aab + u*np.sqrt(aab*w)

    def to_xml_element(self, element_name: str):
        """Return XML representation of the Watt distribution

        Parameters
        ----------
        element_name : str
            XML element name

        Returns
        -------
        element : lxml.etree._Element
            XML element containing Watt distribution data

        """
        element = ET.Element(element_name)
        element.set("type", "watt")
        element.set("parameters", '{} {}'.format(self.a, self.b))
        return element

    @classmethod
    def from_xml_element(cls, elem: ET.Element):
        """Generate Watt distribution from an XML element

        Parameters
        ----------
        elem : lxml.etree._Element
            XML element

        Returns
        -------
        openmc.stats.Watt
            Watt distribution generated from XML element

        """
        params = get_text(elem, 'parameters').split()
        return cls(*map(float, params))


class Normal(Univariate):
    r"""Normally distributed sampling.

    The Normal Distribution is characterized by two parameters
    :math:`\mu` and :math:`\sigma` and has density function
    :math:`p(X) dX = 1/(\sqrt{2\pi}\sigma) e^{-(X-\mu)^2/(2\sigma^2)}`

    Parameters
    ----------
    mean_value : float
        Mean value of the  distribution
    std_dev : float
        Standard deviation of the Normal distribution

    Attributes
    ----------
    mean_value : float
        Mean of the Normal distribution
    std_dev : float
        Standard deviation of the Normal distribution
    """

    def __init__(self, mean_value, std_dev):
        self.mean_value = mean_value
        self.std_dev = std_dev

    def __len__(self):
        return 2

    @property
    def mean_value(self):
        return self._mean_value

    @mean_value.setter
    def mean_value(self, mean_value):
        check_type('Normal mean_value', mean_value, Real)
        self._mean_value = mean_value

    @property
    def std_dev(self):
        return self._std_dev

    @std_dev.setter
    def std_dev(self, std_dev):
        check_type('Normal std_dev', std_dev, Real)
        check_greater_than('Normal std_dev', std_dev, 0.0)
        self._std_dev = std_dev

    def sample(self, n_samples=1, seed=None):
        np.random.seed(seed)
        return np.random.normal(self.mean_value, self.std_dev, n_samples)

    def to_xml_element(self, element_name: str):
        """Return XML representation of the Normal distribution

        Parameters
        ----------
        element_name : str
            XML element name

        Returns
        -------
        element : lxml.etree._Element
            XML element containing Watt distribution data

        """
        element = ET.Element(element_name)
        element.set("type", "normal")
        element.set("parameters", '{} {}'.format(self.mean_value, self.std_dev))
        return element

    @classmethod
    def from_xml_element(cls, elem: ET.Element):
        """Generate Normal distribution from an XML element

        Parameters
        ----------
        elem : lxml.etree._Element
            XML element

        Returns
        -------
        openmc.stats.Normal
            Normal distribution generated from XML element

        """
        params = get_text(elem, 'parameters').split()
        return cls(*map(float, params))


def muir(e0: float, m_rat: float, kt: float):
    """Generate a Muir energy spectrum

    The Muir energy spectrum is a normal distribution, but for convenience
    reasons allows the user to specify three parameters to define the
    distribution: the mean energy of particles ``e0``, the mass of reactants
    ``m_rat``, and the ion temperature ``kt``.

    .. versionadded:: 0.13.2

    Parameters
    ----------
    e0 : float
        Mean of the Muir distribution in [eV]
    m_rat : float
        Ratio of the sum of the masses of the reaction inputs to 1 amu
    kt : float
         Ion temperature for the Muir distribution in [eV]

    Returns
    -------
    openmc.stats.Normal
        Corresponding normal distribution

    """
    # https://permalink.lanl.gov/object/tr?what=info:lanl-repo/lareport/LA-05411-MS
    std_dev = math.sqrt(2 * e0 * kt / m_rat)
    return Normal(e0, std_dev)


# Retain deprecated name for the time being
def Muir(*args, **kwargs):
    # warn of name change
    warn(
        "The Muir(...) class has been replaced by the muir(...) function and "
        "will be removed in a future version of OpenMC. Use muir(...) instead.",
        FutureWarning
    )
    return muir(*args, **kwargs)


class Tabular(Univariate):
    """Piecewise continuous probability distribution.

    This class is used to represent a probability distribution whose density
    function is tabulated at specific values with a specified interpolation
    scheme.

    Parameters
    ----------
    x : Iterable of float
        Tabulated values of the random variable
    p : Iterable of float
        Tabulated probabilities
    interpolation : {'histogram', 'linear-linear', 'linear-log', 'log-linear', 'log-log'}, optional
        Indicate whether the density function is constant between tabulated
        points or linearly-interpolated. Defaults to 'linear-linear'.
    ignore_negative : bool
        Ignore negative probabilities

    Attributes
    ----------
    x : numpy.ndarray
        Tabulated values of the random variable
    p : numpy.ndarray
        Tabulated probabilities
    interpolation : {'histogram', 'linear-linear', 'linear-log', 'log-linear', 'log-log'}, optional
        Indicate whether the density function is constant between tabulated
        points or linearly-interpolated.

    """

    def __init__(
            self,
            x: typing.Sequence[float],
            p: typing.Sequence[float],
            interpolation: str = 'linear-linear',
            ignore_negative: bool = False
        ):
        self._ignore_negative = ignore_negative
        self.x = x
        self.p = p
        self.interpolation = interpolation

    def __len__(self):
        return len(self.x)

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x):
        check_type('tabulated values', x, Iterable, Real)
        self._x = np.array(x, dtype=float)

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, p):
        check_type('tabulated probabilities', p, Iterable, Real)
        if not self._ignore_negative:
            for pk in p:
                check_greater_than('tabulated probability', pk, 0.0, True)
        self._p = np.array(p, dtype=float)

    @property
    def interpolation(self):
        return self._interpolation

    @interpolation.setter
    def interpolation(self, interpolation):
        check_value('interpolation', interpolation, _INTERPOLATION_SCHEMES)
        self._interpolation = interpolation

    def cdf(self):
        c = np.zeros_like(self.x)
        x = self.x
        p = self.p

        if self.interpolation == 'histogram':
            c[1:] = p[:-1] * np.diff(x)
        elif self.interpolation == 'linear-linear':
            c[1:] = 0.5 * (p[:-1] + p[1:]) * np.diff(x)
        else:
            raise NotImplementedError('Can only generate CDFs for tabular '
                                      'distributions using histogram or '
                                      'linear-linear interpolation')


        return np.cumsum(c)

    def mean(self):
        """Compute the mean of the tabular distribution"""
        if self.interpolation == 'linear-linear':
            mean = 0.0
            for i in range(1, len(self.x)):
                y_min = self.p[i-1]
                y_max = self.p[i]
                x_min = self.x[i-1]
                x_max = self.x[i]

                m = (y_max - y_min) / (x_max - x_min)

                exp_val = (1./3.) * m * (x_max**3 - x_min**3)
                exp_val += 0.5 * m * x_min * (x_min**2 - x_max**2)
                exp_val += 0.5 * y_min * (x_max**2 - x_min**2)
                mean += exp_val

        elif self.interpolation == 'histogram':
            x_l = self.x[:-1]
            x_r = self.x[1:]
            p_l = self.p[:-1]
            mean = (0.5 * (x_l + x_r) * (x_r - x_l) * p_l).sum()
        else:
            raise NotImplementedError('Can only compute mean for tabular '
                                      'distributions using histogram '
                                      'or linear-linear interpolation.')

        # Normalize for when integral of distribution is not 1
        mean /= self.integral()

        return mean

    def normalize(self):
        """Normalize the probabilities stored on the distribution"""
        self.p /= self.cdf().max()

    def sample(self, n_samples: int = 1, seed: typing.Optional[int] = None):
        np.random.seed(seed)
        xi = np.random.rand(n_samples)

        # always use normalized probabilities when sampling
        cdf = self.cdf()
        p = self.p / cdf.max()
        cdf /= cdf.max()

        # get CDF bins that are above the
        # sampled values
        c_i = np.full(n_samples, cdf[0])
        cdf_idx = np.zeros(n_samples, dtype=int)
        for i, val in enumerate(cdf[:-1]):
            mask = xi > val
            c_i[mask] = val
            cdf_idx[mask] = i

        # get table values at each index where
        # the random number is less than the next cdf
        # entry
        x_i = self.x[cdf_idx]
        p_i = p[cdf_idx]

        if self.interpolation == 'histogram':
            # mask where probability is greater than zero
            pos_mask = p_i > 0.0
            # probabilities greater than zero are set proportional to the
            # position of the random numebers in relation to the cdf value
            p_i[pos_mask] = x_i[pos_mask] + (xi[pos_mask] - c_i[pos_mask]) \
                           / p_i[pos_mask]
            # probabilities smaller than zero are set to the random number value
            p_i[~pos_mask] = x_i[~pos_mask]

            samples_out = p_i

        elif self.interpolation == 'linear-linear':
            # get variable and probability values for the
            # next entry
            x_i1 = self.x[cdf_idx + 1]
            p_i1 = p[cdf_idx + 1]
            # compute slope between entries
            m = (p_i1 - p_i) / (x_i1 - x_i)
            # set values for zero slope
            zero = m == 0.0
            m[zero] = x_i[zero] + (xi[zero] - c_i[zero]) / p_i[zero]
            # set values for non-zero slope
            non_zero = ~zero
            quad = np.power(p_i[non_zero], 2) + 2.0 * m[non_zero] * (xi[non_zero] - c_i[non_zero])
            quad[quad < 0.0] = 0.0
            m[non_zero] = x_i[non_zero] + (np.sqrt(quad) - p_i[non_zero]) / m[non_zero]
            samples_out = m

        else:
            raise NotImplementedError('Can only sample tabular distributions '
                                      'using histogram or '
                                      'linear-linear interpolation')

        assert all(samples_out < self.x[-1])
        return samples_out

    def to_xml_element(self, element_name: str):
        """Return XML representation of the tabular distribution

        Parameters
        ----------
        element_name : str
            XML element name

        Returns
        -------
        element : lxml.etree._Element
            XML element containing tabular distribution data

        """
        element = ET.Element(element_name)
        element.set("type", "tabular")
        element.set("interpolation", self.interpolation)

        params = ET.SubElement(element, "parameters")
        params.text = ' '.join(map(str, self.x)) + ' ' + ' '.join(map(str, self.p))

        return element

    @classmethod
    def from_xml_element(cls, elem: ET.Element):
        """Generate tabular distribution from an XML element

        Parameters
        ----------
        elem : lxml.etree._Element
            XML element

        Returns
        -------
        openmc.stats.Tabular
            Tabular distribution generated from XML element

        """
        interpolation = get_text(elem, 'interpolation')
        params = [float(x) for x in get_text(elem, 'parameters').split()]
        x = params[:len(params)//2]
        p = params[len(params)//2:]
        return cls(x, p, interpolation)

    def integral(self):
        """Return integral of distribution

        .. versionadded:: 0.13.1

        Returns
        -------
        float
            Integral of tabular distrbution
        """
        if self.interpolation == 'histogram':
            return np.sum(np.diff(self.x) * self.p[:-1])
        elif self.interpolation == 'linear-linear':
            return np.trapz(self.p, self.x)
        else:
            raise NotImplementedError(
                f'integral() not supported for {self.inteprolation} interpolation')


class Legendre(Univariate):
    r"""Probability density given by a Legendre polynomial expansion
    :math:`\sum\limits_{\ell=0}^N \frac{2\ell + 1}{2} a_\ell P_\ell(\mu)`.

    Parameters
    ----------
    coefficients : Iterable of Real
        Expansion coefficients :math:`a_\ell`. Note that the :math:`(2\ell +
        1)/2` factor should not be included.

    Attributes
    ----------
    coefficients : Iterable of Real
        Expansion coefficients :math:`a_\ell`. Note that the :math:`(2\ell +
        1)/2` factor should not be included.

    """

    def __init__(self, coefficients: typing.Sequence[float]):
        self.coefficients = coefficients
        self._legendre_poly = None

    def __call__(self, x):
        # Create Legendre polynomial if we haven't yet
        if self._legendre_poly is None:
            l = np.arange(len(self._coefficients))
            coeffs = (2.*l + 1.)/2. * self._coefficients
            self._legendre_poly = np.polynomial.Legendre(coeffs)

        return self._legendre_poly(x)

    def __len__(self):
        return len(self._coefficients)

    @property
    def coefficients(self):
        return self._coefficients

    @coefficients.setter
    def coefficients(self, coefficients):
        self._coefficients = np.asarray(coefficients)

    def sample(self, n_samples=1, seed=None):
        raise NotImplementedError

    def to_xml_element(self, element_name):
        raise NotImplementedError

    @classmethod
    def from_xml_element(cls, elem):
        raise NotImplementedError


class Mixture(Univariate):
    """Probability distribution characterized by a mixture of random variables.

    Parameters
    ----------
    probability : Iterable of Real
        Probability of selecting a particular distribution
    distribution : Iterable of Univariate
        List of distributions with corresponding probabilities

    Attributes
    ----------
    probability : Iterable of Real
        Probability of selecting a particular distribution
    distribution : Iterable of Univariate
        List of distributions with corresponding probabilities

    """

    def __init__(
        self,
        probability: typing.Sequence[float],
        distribution: typing.Sequence[Univariate]
    ):
        self.probability = probability
        self.distribution = distribution

    def __len__(self):
        return sum(len(d) for d in self.distribution)

    @property
    def probability(self):
        return self._probability

    @probability.setter
    def probability(self, probability):
        check_type('mixture distribution probabilities', probability,
                      Iterable, Real)
        for p in probability:
            check_greater_than('mixture distribution probabilities',
                                  p, 0.0, True)
        self._probability = probability

    @property
    def distribution(self):
        return self._distribution

    @distribution.setter
    def distribution(self, distribution):
        check_type('mixture distribution components', distribution,
                      Iterable, Univariate)
        self._distribution = distribution

    def cdf(self):
        return np.insert(np.cumsum(self.probability), 0, 0.0)

    def sample(self, n_samples=1, seed=None):
        np.random.seed(seed)

        # Get probability of each distribution accounting for its intensity
        p = np.array([prob*dist.integral() for prob, dist in
                      zip(self.probability, self.distribution)])
        p /= p.sum()

        # Sample from the distributions
        idx = np.random.choice(range(len(self.distribution)),
                               n_samples, p=p)

        # Draw samples from the distributions sampled above
        out = np.empty_like(idx, dtype=float)
        for i in np.unique(idx):
            n_dist_samples = np.count_nonzero(idx == i)
            samples = self.distribution[i].sample(n_dist_samples)
            out[idx == i] = samples
        return out

    def normalize(self):
        """Normalize the probabilities stored on the distribution"""
        norm = sum(self.probability)
        self.probability = [val / norm for val in self.probability]

    def to_xml_element(self, element_name: str):
        """Return XML representation of the mixture distribution

        .. versionadded:: 0.13.0

        Parameters
        ----------
        element_name : str
            XML element name

        Returns
        -------
        element : lxml.etree._Element
            XML element containing mixture distribution data

        """
        element = ET.Element(element_name)
        element.set("type", "mixture")

        for p, d in zip(self.probability, self.distribution):
          data = ET.SubElement(element, "pair")
          data.set("probability", str(p))
          data.append(d.to_xml_element("dist"))

        return element

    @classmethod
    def from_xml_element(cls, elem: ET.Element):
        """Generate mixture distribution from an XML element

        .. versionadded:: 0.13.0

        Parameters
        ----------
        elem : lxml.etree._Element
            XML element

        Returns
        -------
        openmc.stats.Mixture
            Mixture distribution generated from XML element

        """
        probability = []
        distribution = []
        for pair in elem.findall('pair'):
            probability.append(float(get_text(pair, 'probability')))
            distribution.append(Univariate.from_xml_element(pair.find("dist")))

        return cls(probability, distribution)

    def integral(self):
        """Return integral of the distribution

        .. versionadded:: 0.13.1

        Returns
        -------
        float
            Integral of the distribution
        """
        return sum([
            p*dist.integral()
            for p, dist in zip(self.probability, self.distribution)
        ])

    def clip(self, tolerance: float = 1e-6, inplace: bool = False) -> Mixture:
        r"""Remove low-importance points from contained discrete distributions.

        Given a probability mass function :math:`p(x)` with :math:`\{x_1, x_2,
        x_3, \dots\}` the possible values of the random variable with
        corresponding probabilities :math:`\{p_1, p_2, p_3, \dots\}`, this
        function will remove any low-importance points such that :math:`\sum_i
        x_i p_i` is preserved to within some threshold.

        .. versionadded:: 0.13.4

        Parameters
        ----------
        tolerance : float
            Maximum fraction of :math:`\sum_i x_i p_i` that will be discarded
            for any discrete distributions within the mixture distribution.
        inplace : bool
            Whether to modify the current object in-place or return a new one.

        Returns
        -------
        Discrete distribution with low-importance points removed

        """
        if inplace:
            for dist in self.distribution:
                if isinstance(dist, Discrete):
                    dist.clip(tolerance, inplace=True)
            return self
        else:
            distribution = [
                dist.clip(tolerance) if isinstance(dist, Discrete) else dist
                for dist in self.distribution
            ]
            return type(self)(self.probability, distribution)


def combine_distributions(
    dists: typing.Sequence[Univariate],
    probs: typing.Sequence[float]
):
    """Combine distributions with specified probabilities

    This function can be used to combine multiple instances of
    :class:`~openmc.stats.Discrete` and `~openmc.stats.Tabular`. Multiple
    discrete distributions are merged into a single distribution and the
    remainder of the distributions are put into a :class:`~openmc.stats.Mixture`
    distribution.

    .. versionadded:: 0.13.1

    Parameters
    ----------
    dists : iterable of openmc.stats.Univariate
        Distributions to combine
    probs : iterable of float
        Probability (or intensity) of each distribution

    """
    # Get copy of distribution list so as not to modify the argument
    dist_list = deepcopy(dists)

    # Get list of discrete/continuous distribution indices
    discrete_index = [i for i, d in enumerate(dist_list) if isinstance(d, Discrete)]
    cont_index = [i for i, d in enumerate(dist_list) if isinstance(d, Tabular)]

    # Apply probabilites to continuous distributions
    for i in cont_index:
        dist = dist_list[i]
        dist.p *= probs[i]

    if discrete_index:
        # Create combined discrete distribution
        dist_discrete = [dist_list[i] for i in discrete_index]
        discrete_probs = [probs[i] for i in discrete_index]
        combined_dist = Discrete.merge(dist_discrete, discrete_probs)

        # Replace multiple discrete distributions with merged
        for idx in reversed(discrete_index):
            dist_list.pop(idx)
        dist_list.append(combined_dist)

    # Combine discrete and continuous if present
    if len(dist_list) > 1:
        probs = [1.0]*len(dist_list)
        dist_list[:] = [Mixture(probs, dist_list.copy())]

    return dist_list[0]

class UncorrelatedAngleEnergy(AngleEnergy):
    """Uncorrelated angle-energy distribution

    Parameters
    ----------
    angle : openmc.data.AngleDistribution
        Distribution of outgoing angles represented as scattering cosines
    energy : openmc.data.EnergyDistribution
        Distribution of outgoing energies

    Attributes
    ----------
    angle : openmc.data.AngleDistribution
        Distribution of outgoing angles represented as scattering cosines
    energy : openmc.data.EnergyDistribution
        Distribution of outgoing energies

    """

    def __init__(self, angle=None, energy=None):
        self._angle = None
        self._energy = None

        if angle is not None:
            self.angle = angle
        if energy is not None:
            self.energy = energy

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, angle):
        check_type('uncorrelated angle distribution', angle,
                      AngleDistribution)
        self._angle = angle

    @property
    def energy(self):
        return self._energy

    @energy.setter
    def energy(self, energy):
        check_type('uncorrelated energy distribution', energy,
                      EnergyDistribution)
        self._energy = energy

    def to_hdf5(self, group):
        """Write distribution to an HDF5 group

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to write to

        """
        group.attrs['type'] = np.string_('uncorrelated')
        if self.angle is not None:
            angle_group = group.create_group('angle')
            self.angle.to_hdf5(angle_group)

        if self.energy is not None:
            energy_group = group.create_group('energy')
            self.energy.to_hdf5(energy_group)

    @classmethod
    def from_hdf5(cls, group):
        """Generate uncorrelated angle-energy distribution from HDF5 data

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to read from

        Returns
        -------
        openmc.data.UncorrelatedAngleEnergy
            Uncorrelated angle-energy distribution

        """
        dist = cls()
        if 'angle' in group:
            dist.angle = AngleDistribution.from_hdf5(group['angle'])
        if 'energy' in group:
            dist.energy = EnergyDistribution.from_hdf5(group['energy'])
        return dist


class Resonances:
    """Resolved and unresolved resonance data

    Parameters
    ----------
    ranges : list of openmc.data.ResonanceRange
        Distinct energy ranges for resonance data

    Attributes
    ----------
    ranges : list of openmc.data.ResonanceRange
        Distinct energy ranges for resonance data
    resolved : openmc.data.ResonanceRange or None
        Resolved resonance range
    unresolved : openmc.data.Unresolved or None
        Unresolved resonance range

    """

    def __init__(self, ranges):
        self.ranges = ranges

    def __iter__(self):
        for r in self.ranges:
            yield r

    @property
    def ranges(self):
        return self._ranges

    @ranges.setter
    def ranges(self, ranges):
        check_type('resonance ranges', ranges, MutableSequence)
        self._ranges = cv.CheckedList(ResonanceRange, 'resonance ranges',
                                      ranges)

    @property
    def resolved(self):
        resolved_ranges = [r for r in self.ranges
                           if not isinstance(r, Unresolved)]
        if len(resolved_ranges) > 1:
            raise ValueError('More than one resolved range present')
        elif len(resolved_ranges) == 0:
            return None
        else:
            return resolved_ranges[0]

    @property
    def unresolved(self):
        for r in self.ranges:
            if isinstance(r, Unresolved):
                return r
        else:
            return None

    @classmethod
    def from_endf(cls, ev):
        """Generate resonance data from an ENDF evaluation.

        Parameters
        ----------
        ev : openmc.data.endf.Evaluation
            ENDF evaluation

        Returns
        -------
        openmc.data.Resonances
            Resonance data

        """
        file_obj = io.StringIO(ev.section[2, 151])

        # Determine whether discrete or continuous representation
        items = get_head_record(file_obj)
        n_isotope = items[4]  # Number of isotopes

        ranges = []
        for _ in range(n_isotope):
            items = get_cont_record(file_obj)
            fission_widths = (items[3] == 1)  # fission widths are given?
            n_ranges = items[4]  # number of resonance energy ranges

            for j in range(n_ranges):
                items = get_cont_record(file_obj)
                resonance_flag = items[2]  # flag for resolved (1)/unresolved (2)
                formalism = items[3]  # resonance formalism

                if resonance_flag in (0, 1):
                    # resolved resonance region
                    erange = _FORMALISMS[formalism].from_endf(ev, file_obj, items)

                elif resonance_flag == 2:
                    # unresolved resonance region
                    erange = Unresolved.from_endf(file_obj, items, fission_widths)

                # erange.material = self
                ranges.append(erange)

        return cls(ranges)


class ResonanceRange:
    """Resolved resonance range

    Parameters
    ----------
    target_spin : float
        Intrinsic spin, :math:`I`, of the target nuclide
    energy_min : float
        Minimum energy of the resolved resonance range in eV
    energy_max : float
        Maximum energy of the resolved resonance range in eV
    channel : dict
        Dictionary whose keys are l-values and values are channel radii as a
        function of energy
    scattering : dict
        Dictionary whose keys are l-values and values are scattering radii as a
        function of energy

    Attributes
    ----------
    channel_radius : dict
        Dictionary whose keys are l-values and values are channel radii as a
        function of energy
    energy_max : float
        Maximum energy of the resolved resonance range in eV
    energy_min : float
        Minimum energy of the resolved resonance range in eV
    scattering_radius : dict
        Dictionary whose keys are l-values and values are scattering radii as a
        function of energ
    target_spin : float
        Intrinsic spin, :math:`I`, of the target nuclide

    """
    def __init__(self, target_spin, energy_min, energy_max, channel, scattering):
        self.target_spin = target_spin
        self.energy_min = energy_min
        self.energy_max = energy_max
        self.channel_radius = channel
        self.scattering_radius = scattering

        self._prepared = False
        self._parameter_matrix = {}

    def __copy__(self):
        cls = type(self)
        new_copy = cls.__new__(cls)
        new_copy.__dict__.update(self.__dict__)
        new_copy._prepared = False
        return new_copy

    @classmethod
    def from_endf(cls, ev, file_obj, items):
        """Create resonance range from an ENDF evaluation.

        This factory method is only used when LRU=0, indicating that only a
        scattering radius appears in MF=2, MT=151. All subclasses of
        ResonanceRange override this method with their own.

        Parameters
        ----------
        ev : openmc.data.endf.Evaluation
            ENDF evaluation
        file_obj : file-like object
            ENDF file positioned at the second record of a resonance range
            subsection in MF=2, MT=151
        items : list
            Items from the CONT record at the start of the resonance range
            subsection

        Returns
        -------
        openmc.data.ResonanceRange
            Resonance range data

        """
        energy_min, energy_max = items[0:2]

        # For scattering radius-only, NRO must be zero
        assert items[4] == 0

        # Get energy-independent scattering radius
        items = get_cont_record(file_obj)
        target_spin = items[0]
        ap = Polynomial((items[1],))

        # Calculate channel radius from ENDF-102 equation D.14
        a = Polynomial((0.123 * (NEUTRON_MASS*ev.target['mass'])**(1./3.) + 0.08,))

        return cls(target_spin, energy_min, energy_max, {0: a}, {0: ap})

    def reconstruct(self, energies):
        """Evaluate cross section at specified energies.

        Parameters
        ----------
        energies : float or Iterable of float
            Energies at which the cross section should be evaluated

        Returns
        -------
        3-tuple of float or numpy.ndarray
            Elastic, capture, and fission cross sections at the specified
            energies

        """
        if not _reconstruct:
            raise RuntimeError("Resonance reconstruction not available.")

        # Pre-calculate penetrations and shifts for resonances
        if not self._prepared:
            self._prepare_resonances()

        if isinstance(energies, Iterable):
            elastic = np.zeros_like(energies)
            capture = np.zeros_like(energies)
            fission = np.zeros_like(energies)

            for i, E in enumerate(energies):
                xse, xsg, xsf = self._reconstruct(self, E)
                elastic[i] = xse
                capture[i] = xsg
                fission[i] = xsf
        else:
            elastic, capture, fission = self._reconstruct(self, energies)

        return {2: elastic, 102: capture, 18: fission}


class MultiLevelBreitWigner(ResonanceRange):
    """Multi-level Breit-Wigner resolved resonance formalism data.

    Multi-level Breit-Wigner resolved resonance data is identified by LRF=2 in
    the ENDF-6 format.

    Parameters
    ----------
    target_spin : float
        Intrinsic spin, :math:`I`, of the target nuclide
    energy_min : float
        Minimum energy of the resolved resonance range in eV
    energy_max : float
        Maximum energy of the resolved resonance range in eV
    channel : dict
        Dictionary whose keys are l-values and values are channel radii as a
        function of energy
    scattering : dict
        Dictionary whose keys are l-values and values are scattering radii as a
        function of energy

    Attributes
    ----------
    atomic_weight_ratio : float
        Atomic weight ratio of the target nuclide given as a function of
        l-value. Note that this may be different than the value for the
        evaluation as a whole.
    channel_radius : dict
        Dictionary whose keys are l-values and values are channel radii as a
        function of energy
    energy_max : float
        Maximum energy of the resolved resonance range in eV
    energy_min : float
        Minimum energy of the resolved resonance range in eV
    parameters : pandas.DataFrame
        Energies, spins, and resonances widths for each resonance
    q_value : dict
        Q-value to be added to incident particle's center-of-mass energy to
        determine the channel energy for use in the penetrability factor. The
        keys of the dictionary are l-values.
    scattering_radius : dict
        Dictionary whose keys are l-values and values are scattering radii as a
        function of energy
    target_spin : float
        Intrinsic spin, :math:`I`, of the target nuclide

    """

    def __init__(self, target_spin, energy_min, energy_max, channel, scattering):
        super().__init__(target_spin, energy_min, energy_max, channel,
                         scattering)
        self.parameters = None
        self.q_value = {}
        self.atomic_weight_ratio = None

        # Set resonance reconstruction function
        if _reconstruct:
            self._reconstruct = reconstruct_mlbw
        else:
            self._reconstruct = None

    @classmethod
    def from_endf(cls, ev, file_obj, items):
        """Create MLBW data from an ENDF evaluation.

        Parameters
        ----------
        ev : openmc.data.endf.Evaluation
            ENDF evaluation
        file_obj : file-like object
            ENDF file positioned at the second record of a resonance range
            subsection in MF=2, MT=151
        items : list
            Items from the CONT record at the start of the resonance range
            subsection

        Returns
        -------
        openmc.data.MultiLevelBreitWigner
            Multi-level Breit-Wigner resonance parameters

        """

        # Read energy-dependent scattering radius if present
        energy_min, energy_max = items[0:2]
        nro, naps = items[4:6]
        if nro != 0:
            params, ape = get_tab1_record(file_obj)

        # Other scatter radius parameters
        items = get_cont_record(file_obj)
        target_spin = items[0]
        ap = Polynomial((items[1],))  # energy-independent scattering-radius
        NLS = items[4]  # number of l-values

        # Read resonance widths, J values, etc
        channel_radius = {}
        scattering_radius = {}
        q_value = {}
        records = []
        for l in range(NLS):
            items, values = get_list_record(file_obj)
            l_value = items[2]
            awri = items[0]
            q_value[l_value] = items[1]
            competitive = items[3]

            # Calculate channel radius from ENDF-102 equation D.14
            a = Polynomial((0.123 * (NEUTRON_MASS*awri)**(1./3.) + 0.08,))

            # Construct scattering and channel radius
            if nro == 0:
                scattering_radius[l_value] = ap
                if naps == 0:
                    channel_radius[l_value] = a
                elif naps == 1:
                    channel_radius[l_value] = ap
            elif nro == 1:
                scattering_radius[l_value] = ape
                if naps == 0:
                    channel_radius[l_value] = a
                elif naps == 1:
                    channel_radius[l_value] = ape
                elif naps == 2:
                    channel_radius[l_value] = ap

            energy = values[0::6]
            spin = values[1::6]
            gt = np.asarray(values[2::6])
            gn = np.asarray(values[3::6])
            gg = np.asarray(values[4::6])
            gf = np.asarray(values[5::6])
            if competitive > 0:
                gx = gt - (gn + gg + gf)
            else:
                gx = np.zeros_like(gt)

            for i, E in enumerate(energy):
                records.append([energy[i], l_value, spin[i], gt[i], gn[i],
                                gg[i], gf[i], gx[i]])

        columns = ['energy', 'L', 'J', 'totalWidth', 'neutronWidth',
                   'captureWidth', 'fissionWidth', 'competitiveWidth']
        parameters = pd.DataFrame.from_records(records, columns=columns)

        # Create instance of class
        mlbw = cls(target_spin, energy_min, energy_max,
                   channel_radius, scattering_radius)
        mlbw.q_value = q_value
        mlbw.atomic_weight_ratio = awri
        mlbw.parameters = parameters

        return mlbw

    def _prepare_resonances(self):
        df = self.parameters.copy()

        # Penetration and shift factors
        p = np.zeros(len(df))
        s = np.zeros(len(df))

        # Penetration and shift factors for competitive reaction
        px = np.zeros(len(df))
        sx = np.zeros(len(df))

        l_values = []
        competitive = []

        A = self.atomic_weight_ratio
        for i, E, l, J, gt, gn, gg, gf, gx in df.itertuples():
            if l not in l_values:
                l_values.append(l)
                competitive.append(gx > 0)

            # Determine penetration and shift corresponding to resonance energy
            k = wave_number(A, E)
            rho = k*self.channel_radius[l](E)
            p[i], s[i] = penetration_shift(l, rho)

            # Determine penetration at modified energy for competitive reaction
            if gx > 0:
                Ex = E + self.q_value[l]*(A + 1)/A
                rho = k*self.channel_radius[l](Ex)
                px[i], sx[i] = penetration_shift(l, rho)
            else:
                px[i] = sx[i] = 0.0

        df['p'] = p
        df['s'] = s
        df['px'] = px
        df['sx'] = sx

        self._l_values = np.array(l_values)
        self._competitive = np.array(competitive)
        for l in l_values:
            self._parameter_matrix[l] = df[df.L == l].values

        self._prepared = True


class SingleLevelBreitWigner(MultiLevelBreitWigner):
    """Single-level Breit-Wigner resolved resonance formalism data.

    Single-level Breit-Wigner resolved resonance data is is identified by LRF=1
    in the ENDF-6 format.

    Parameters
    ----------
    target_spin : float
        Intrinsic spin, :math:`I`, of the target nuclide
    energy_min : float
        Minimum energy of the resolved resonance range in eV
    energy_max : float
        Maximum energy of the resolved resonance range in eV
    channel : dict
        Dictionary whose keys are l-values and values are channel radii as a
        function of energy
    scattering : dict
        Dictionary whose keys are l-values and values are scattering radii as a
        function of energy

    Attributes
    ----------
    atomic_weight_ratio : float
        Atomic weight ratio of the target nuclide given as a function of
        l-value. Note that this may be different than the value for the
        evaluation as a whole.
    channel_radius : dict
        Dictionary whose keys are l-values and values are channel radii as a
        function of energy
    energy_max : float
        Maximum energy of the resolved resonance range in eV
    energy_min : float
        Minimum energy of the resolved resonance range in eV
    parameters : pandas.DataFrame
        Energies, spins, and resonances widths for each resonance
    q_value : dict
        Q-value to be added to incident particle's center-of-mass energy to
        determine the channel energy for use in the penetrability factor. The
        keys of the dictionary are l-values.
    scattering_radius : dict
        Dictionary whose keys are l-values and values are scattering radii as a
        function of energy
    target_spin : float
        Intrinsic spin, :math:`I`, of the target nuclide

    """

    def __init__(self, target_spin, energy_min, energy_max, channel, scattering):
        super().__init__(target_spin, energy_min, energy_max, channel,
                         scattering)

        # Set resonance reconstruction function
        if _reconstruct:
            self._reconstruct = reconstruct_slbw
        else:
            self._reconstruct = None


class ReichMoore(ResonanceRange):
    """Reich-Moore resolved resonance formalism data.

    Reich-Moore resolved resonance data is identified by LRF=3 in the ENDF-6
    format.

    Parameters
    ----------
    target_spin : float
        Intrinsic spin, :math:`I`, of the target nuclide
    energy_min : float
        Minimum energy of the resolved resonance range in eV
    energy_max : float
        Maximum energy of the resolved resonance range in eV
    channel : dict
        Dictionary whose keys are l-values and values are channel radii as a
        function of energy
    scattering : dict
        Dictionary whose keys are l-values and values are scattering radii as a
        function of energy

    Attributes
    ----------
    angle_distribution : bool
        Indicate whether parameters can be used to compute angular distributions
    atomic_weight_ratio : float
        Atomic weight ratio of the target nuclide given as a function of
        l-value. Note that this may be different than the value for the
        evaluation as a whole.
    channel_radius : dict
        Dictionary whose keys are l-values and values are channel radii as a
        function of energy
    energy_max : float
        Maximum energy of the resolved resonance range in eV
    energy_min : float
        Minimum energy of the resolved resonance range in eV
    num_l_convergence : int
        Number of l-values which must be used to converge the calculation
    scattering_radius : dict
        Dictionary whose keys are l-values and values are scattering radii as a
        function of energy
    parameters : pandas.DataFrame
        Energies, spins, and resonances widths for each resonance
    target_spin : float
        Intrinsic spin, :math:`I`, of the target nuclide

    """

    def __init__(self, target_spin, energy_min, energy_max, channel, scattering):
        super().__init__(target_spin, energy_min, energy_max, channel,
                         scattering)
        self.parameters = None
        self.angle_distribution = False
        self.num_l_convergence = 0

        # Set resonance reconstruction function
        if _reconstruct:
            self._reconstruct = reconstruct_rm
        else:
            self._reconstruct = None

    @classmethod
    def from_endf(cls, ev, file_obj, items):
        """Create Reich-Moore resonance data from an ENDF evaluation.

        Parameters
        ----------
        ev : openmc.data.endf.Evaluation
            ENDF evaluation
        file_obj : file-like object
            ENDF file positioned at the second record of a resonance range
            subsection in MF=2, MT=151
        items : list
            Items from the CONT record at the start of the resonance range
            subsection

        Returns
        -------
        openmc.data.ReichMoore
            Reich-Moore resonance parameters

        """
        # Read energy-dependent scattering radius if present
        energy_min, energy_max = items[0:2]
        nro, naps = items[4:6]
        if nro != 0:
            params, ape = get_tab1_record(file_obj)

        # Other scatter radius parameters
        items = get_cont_record(file_obj)
        target_spin = items[0]
        ap = Polynomial((items[1],))
        angle_distribution = (items[3] == 1)  # Flag for angular distribution
        NLS = items[4]  # Number of l-values
        num_l_convergence = items[5]  # Number of l-values for convergence

        # Read resonance widths, J values, etc
        channel_radius = {}
        scattering_radius = {}
        records = []
        for i in range(NLS):
            items, values = get_list_record(file_obj)
            apl = Polynomial((items[1],)) if items[1] != 0.0 else ap
            l_value = items[2]
            awri = items[0]

            # Calculate channel radius from ENDF-102 equation D.14
            a = Polynomial((0.123 * (NEUTRON_MASS*awri)**(1./3.) + 0.08,))

            # Construct scattering and channel radius
            if nro == 0:
                scattering_radius[l_value] = apl
                if naps == 0:
                    channel_radius[l_value] = a
                elif naps == 1:
                    channel_radius[l_value] = apl
            elif nro == 1:
                if naps == 0:
                    channel_radius[l_value] = a
                    scattering_radius[l_value] = ape
                elif naps == 1:
                    channel_radius[l_value] = scattering_radius[l_value] = ape
                elif naps == 2:
                    channel_radius[l_value] = apl
                    scattering_radius[l_value] = ape

            energy = values[0::6]
            spin = values[1::6]
            gn = values[2::6]
            gg = values[3::6]
            gfa = values[4::6]
            gfb = values[5::6]

            for i, E in enumerate(energy):
                records.append([energy[i], l_value, spin[i], gn[i], gg[i],
                                gfa[i], gfb[i]])

        # Create pandas DataFrame with resonance data
        columns = ['energy', 'L', 'J', 'neutronWidth', 'captureWidth',
                   'fissionWidthA', 'fissionWidthB']
        parameters = pd.DataFrame.from_records(records, columns=columns)

        # Create instance of ReichMoore
        rm = cls(target_spin, energy_min, energy_max,
                 channel_radius, scattering_radius)
        rm.parameters = parameters
        rm.angle_distribution = angle_distribution
        rm.num_l_convergence = num_l_convergence
        rm.atomic_weight_ratio = awri

        return rm

    def _prepare_resonances(self):
        df = self.parameters.copy()

        # Penetration and shift factors
        p = np.zeros(len(df))
        s = np.zeros(len(df))

        l_values = []
        lj_values = []

        A = self.atomic_weight_ratio
        for i, E, l, J, gn, gg, gfa, gfb in df.itertuples():
            if l not in l_values:
                l_values.append(l)
            if (l, abs(J)) not in lj_values:
                lj_values.append((l, abs(J)))

            # Determine penetration and shift corresponding to resonance energy
            k = wave_number(A, E)
            rho = k*self.channel_radius[l](E)
            p[i], s[i] = penetration_shift(l, rho)

        df['p'] = p
        df['s'] = s

        self._l_values = np.array(l_values)
        for (l, J) in lj_values:
            self._parameter_matrix[l, J] = df[(df.L == l) &
                                              (abs(df.J) == J)].values

        self._prepared = True


class RMatrixLimited(ResonanceRange):
    """R-matrix limited resolved resonance formalism data.

    R-matrix limited resolved resonance data is identified by LRF=7 in the
    ENDF-6 format.

    Parameters
    ----------
    energy_min : float
        Minimum energy of the resolved resonance range in eV
    energy_max : float
        Maximum energy of the resolved resonance range in eV
    particle_pairs : list of dict
        List of particle pairs. Each particle pair is represented by a
        dictionary that contains the mass, atomic number, spin, and parity of
        each particle as well as other characteristics.
    spin_groups : list of dict
        List of spin groups. Each spin group is characterized by channels,
        resonance energies, and resonance widths.

    Attributes
    ----------
    reduced_width : bool
        Flag indicating whether channel widths in eV or reduced-width amplitudes
        in eV^1/2 are given
    formalism : int
        Flag to specify which formulae for the R-matrix are to be used
    particle_pairs : list of dict
        List of particle pairs. Each particle pair is represented by a
        dictionary that contains the mass, atomic number, spin, and parity of
        each particle as well as other characteristics.
    spin_groups : list of dict
        List of spin groups. Each spin group is characterized by channels,
        resonance energies, and resonance widths.

    """

    def __init__(self, energy_min, energy_max, particle_pairs, spin_groups):
        super().__init__(0.0, energy_min, energy_max, None, None)
        self.reduced_width = False
        self.formalism = 3
        self.particle_pairs = particle_pairs
        self.spin_groups = spin_groups

    @classmethod
    def from_endf(cls, ev, file_obj, items):
        """Read R-Matrix limited resonance data from an ENDF evaluation.

        Parameters
        ----------
        ev : openmc.data.endf.Evaluation
            ENDF evaluation
        file_obj : file-like object
            ENDF file positioned at the second record of a resonance range
            subsection in MF=2, MT=151
        items : list
            Items from the CONT record at the start of the resonance range
            subsection

        Returns
        -------
        openmc.data.RMatrixLimited
            R-matrix limited resonance parameters

        """
        energy_min, energy_max = items[0:2]

        items = get_cont_record(file_obj)
        reduced_width = (items[2] == 1)  # reduced width amplitude?
        formalism = items[3]  # Specify which formulae are used
        n_spin_groups = items[4]  # Number of Jpi values (NJS)

        particle_pairs = []
        spin_groups = []

        items, values = get_list_record(file_obj)
        n_pairs = items[5]//2  # Number of particle pairs (NPP)
        for i in range(n_pairs):
            first = {'mass': values[12*i],
                     'z': int(values[12*i + 2]),
                     'spin': values[12*i + 4],
                     'parity': values[12*i + 10]}
            second = {'mass': values[12*i + 1],
                      'z': int(values[12*i + 3]),
                      'spin': values[12*i + 5],
                      'parity': values[12*i + 11]}

            q_value = values[12*i + 6]
            penetrability = values[12*i + 7]
            shift = values[12*i + 8]
            mt = int(values[12*i + 9])

            particle_pairs.append(ParticlePair(
                first, second, q_value, penetrability, shift, mt))

        # loop over spin groups
        for i in range(n_spin_groups):
            items, values = get_list_record(file_obj)
            J = items[0]
            if J == 0.0:
                parity = '+' if items[1] == 1.0 else '-'
            else:
                parity = '+' if J > 0. else '-'
                J = abs(J)
            kbk = items[2]
            kps = items[3]
            n_channels = items[5]
            channels = []
            for j in range(n_channels):
                channel = {}
                channel['particle_pair'] = particle_pairs[
                    int(values[6*j]) - 1]
                channel['l'] = values[6*j + 1]
                channel['spin'] = values[6*j + 2]
                channel['boundary'] = values[6*j + 3]
                channel['effective_radius'] = values[6*j + 4]
                channel['true_radius'] = values[6*j + 5]
                channels.append(channel)

            # Read resonance energies and widths
            items, values = get_list_record(file_obj)
            n_resonances = items[3]
            records = []
            m = n_channels//6 + 1
            for j in range(n_resonances):
                energy = values[6*m*j]
                records.append([energy] + [values[6*m*j + k + 1]
                                           for k in range(n_channels)])

            # Determine column names
            columns = ['energy']
            for channel in channels:
                mt = channel['particle_pair'].mt
                if mt == 2:
                    columns.append('neutronWidth')
                elif mt == 18:
                    columns.append('fissionWidth')
                elif mt == 102:
                    columns.append('captureWidth')
                else:
                    columns.append('width (MT={})'.format(mt))

            # Create Pandas dataframe with resonance parameters
            parameters = pd.DataFrame.from_records(records, columns=columns)

            # Construct SpinGroup instance and add to list
            sg = SpinGroup(J, parity, channels, parameters)
            spin_groups.append(sg)

            # Optional extension (Background R-Matrix)
            if kbk > 0:
                items, values = get_list_record(file_obj)
                lbk = items[4]
                if lbk == 1:
                    params, rbr = get_tab1_record(file_obj)
                    params, rbi = get_tab1_record(file_obj)

            # Optional extension (Tabulated phase shifts)
            if kps > 0:
                items, values = get_list_record(file_obj)
                lps = items[4]
                if lps == 1:
                    params, psr = get_tab1_record(file_obj)
                    params, psi = get_tab1_record(file_obj)

        rml = cls(energy_min, energy_max, particle_pairs, spin_groups)
        rml.reduced_width = reduced_width
        rml.formalism = formalism

        return rml


class ParticlePair:
    def __init__(self, first, second, q_value, penetrability,
                 shift, mt):
        self.first = first
        self.second = second
        self.q_value = q_value
        self.penetrability = penetrability
        self.shift = shift
        self.mt = mt


class SpinGroup:
    """Resonance spin group

    Attributes
    ----------
    spin : float
        Total angular momentum (nuclear spin)
    parity : {'+', '-'}
        Even (+) or odd(-) parity
    channels : list of openmc.data.Channel
        Available channels
    parameters : pandas.DataFrame
        Energies/widths for each resonance/channel

    """

    def __init__(self, spin, parity, channels, parameters):
        self.spin = spin
        self.parity = parity
        self.channels = channels
        self.parameters = parameters

    def __repr__(self):
        return '<SpinGroup: Jpi={}{}>'.format(self.spin, self.parity)


class Unresolved(ResonanceRange):
    """Unresolved resonance parameters as identified by LRU=2 in MF=2.

    Parameters
    ----------
    target_spin : float
        Intrinsic spin, :math:`I`, of the target nuclide
    energy_min : float
        Minimum energy of the unresolved resonance range in eV
    energy_max : float
        Maximum energy of the unresolved resonance range in eV
    channel : openmc.data.Function1D
        Channel radii as a function of energy
    scattering : openmc.data.Function1D
        Scattering radii as a function of energy

    Attributes
    ----------
    add_to_background : bool
        If True, file 3 contains partial cross sections to be added to the
        average unresolved cross sections calculated from parameters.
    atomic_weight_ratio : float
        Atomic weight ratio of the target nuclide
    channel_radius : openmc.data.Function1D
        Channel radii as a function of energy
    energies : Iterable of float
        Energies at which parameters are tabulated
    energy_max : float
        Maximum energy of the unresolved resonance range in eV
    energy_min : float
        Minimum energy of the unresolved resonance range in eV
    parameters : list of pandas.DataFrame
        Average resonance parameters at each energy
    scattering_radius : openmc.data.Function1D
        Scattering radii as a function of energy
    target_spin : float
        Intrinsic spin, :math:`I`, of the target nuclide

    """

    def __init__(self, target_spin, energy_min, energy_max, channel, scattering):
        super().__init__(target_spin, energy_min, energy_max, channel,
                         scattering)
        self.energies = None
        self.parameters = None
        self.add_to_background = False
        self.atomic_weight_ratio = None

    @classmethod
    def from_endf(cls, file_obj, items, fission_widths):
        """Read unresolved resonance data from an ENDF evaluation.

        Parameters
        ----------
        file_obj : file-like object
            ENDF file positioned at the second record of a resonance range
            subsection in MF=2, MT=151
        items : list
            Items from the CONT record at the start of the resonance range
            subsection
        fission_widths : bool
            Whether fission widths are given

        Returns
        -------
        openmc.data.Unresolved
            Unresolved resonance region parameters

        """
        # Read energy-dependent scattering radius if present
        energy_min, energy_max = items[0:2]
        nro, naps = items[4:6]
        if nro != 0:
            params, ape = get_tab1_record(file_obj)

        # Get SPI, AP, and LSSF
        formalism = items[3]
        if not (fission_widths and formalism == 1):
            items = get_cont_record(file_obj)
            target_spin = items[0]
            if nro == 0:
                ap = Polynomial((items[1],))
            add_to_background = (items[2] == 0)

        if not fission_widths and formalism == 1:
            # Case A -- fission widths not given, all parameters are
            # energy-independent
            NLS = items[4]
            columns = ['L', 'J', 'd', 'amun', 'gn0', 'gg']
            records = []
            for ls in range(NLS):
                items, values = get_list_record(file_obj)
                awri = items[0]
                l = items[2]
                NJS = items[5]
                for j in range(NJS):
                    d, j, amun, gn0, gg = values[6*j:6*j + 5]
                    records.append([l, j, d, amun, gn0, gg])
            parameters = pd.DataFrame.from_records(records, columns=columns)
            energies = None

        elif fission_widths and formalism == 1:
            # Case B -- fission widths given, only fission widths are
            # energy-dependent
            items, energies = get_list_record(file_obj)
            target_spin = items[0]
            if nro == 0:
                ap = Polynomial((items[1],))
            add_to_background = (items[2] == 0)
            NE, NLS = items[4:6]
            records = []
            columns = ['L', 'J', 'E', 'd', 'amun', 'amuf', 'gn0', 'gg', 'gf']
            for ls in range(NLS):
                items = get_cont_record(file_obj)
                awri = items[0]
                l = items[2]
                NJS = items[4]
                for j in range(NJS):
                    items, values = get_list_record(file_obj)
                    muf = items[3]
                    d = values[0]
                    j = values[1]
                    amun = values[2]
                    gn0 = values[3]
                    gg = values[4]
                    gfs = values[6:]
                    for E, gf in zip(energies, gfs):
                        records.append([l, j, E, d, amun, muf, gn0, gg, gf])
            parameters = pd.DataFrame.from_records(records, columns=columns)

        elif formalism == 2:
            # Case C -- all parameters are energy-dependent
            NLS = items[4]
            columns = ['L', 'J', 'E', 'd', 'amux', 'amun', 'amuf', 'gx', 'gn0',
                       'gg', 'gf']
            records = []
            for ls in range(NLS):
                items = get_cont_record(file_obj)
                awri = items[0]
                l = items[2]
                NJS = items[4]
                for j in range(NJS):
                    items, values = get_list_record(file_obj)
                    ne = items[5]
                    j = items[0]
                    amux = values[2]
                    amun = values[3]
                    amuf = values[5]
                    energies = []
                    for k in range(1, ne + 1):
                        E = values[6*k]
                        d = values[6*k + 1]
                        gx = values[6*k + 2]
                        gn0 = values[6*k + 3]
                        gg = values[6*k + 4]
                        gf = values[6*k + 5]
                        energies.append(E)
                        records.append([l, j, E, d, amux, amun, amuf, gx, gn0,
                                        gg, gf])
            parameters = pd.DataFrame.from_records(records, columns=columns)

        # Calculate channel radius from ENDF-102 equation D.14
        a = Polynomial((0.123 * (NEUTRON_MASS*awri)**(1./3.) + 0.08,))

        # Determine scattering and channel radius
        if nro == 0:
            scattering_radius = ap
            if naps == 0:
                channel_radius = a
            elif naps == 1:
                channel_radius = ap
        elif nro == 1:
            scattering_radius = ape
            if naps == 0:
                channel_radius = a
            elif naps == 1:
                channel_radius = ape
            elif naps == 2:
                channel_radius = ap

        urr = cls(target_spin, energy_min, energy_max, channel_radius,
                  scattering_radius)
        urr.parameters = parameters
        urr.add_to_background = add_to_background
        urr.atomic_weight_ratio = awri
        urr.energies = energies

        return urr


_FORMALISMS = {0: ResonanceRange,
               1: SingleLevelBreitWigner,
               2: MultiLevelBreitWigner,
               3: ReichMoore,
               7: RMatrixLimited}

_RESOLVED = (SingleLevelBreitWigner, MultiLevelBreitWigner,
             ReichMoore, RMatrixLimited)


def _add_file2_contributions(file32params, file2params):
    """Function for aiding in adding resonance parameters from File 2 that are
    not always present in File 32. Uses already imported resonance data.

    Paramaters
    ----------
    file32params : pandas.Dataframe
        Incomplete set of resonance parameters contained in File 32.
    file2params : pandas.Dataframe
        Resonance parameters from File 2. Ordered by energy.

    Returns
    -------
    parameters : pandas.Dataframe
        Complete set of parameters ordered by L-values and then energy

    """
    # Use l-values and competitiveWidth from File 2 data
    # Re-sort File 2 by energy to match File 32
    file2params = file2params.sort_values(by=['energy'])
    file2params.reset_index(drop=True, inplace=True)
    # Sort File 32 parameters by energy as well (maintaining index)
    file32params.sort_values(by=['energy'], inplace=True)
    # Add in values (.values converts to array first to ignore index)
    file32params['L'] = file2params['L'].values
    if 'competitiveWidth' in file2params.columns:
        file32params['competitiveWidth'] = file2params['competitiveWidth'].values
    # Resort to File 32 order (by L then by E) for use with covariance
    file32params.sort_index(inplace=True)
    return file32params


class ResonanceCovariances(Resonances):
    """Resolved resonance covariance data

    Parameters
    ----------
    ranges : list of openmc.data.ResonanceCovarianceRange
        Distinct energy ranges for resonance data

    Attributes
    ----------
    ranges : list of openmc.data.ResonanceCovarianceRange
        Distinct energy ranges for resonance data

    """

    @property
    def ranges(self):
        return self._ranges

    @ranges.setter
    def ranges(self, ranges):
        check_type('resonance ranges', ranges, MutableSequence)
        self._ranges = cv.CheckedList(ResonanceCovarianceRange,
                                      'resonance range', ranges)

    @classmethod
    def from_endf(cls, ev, resonances):
        """Generate resonance covariance data from an ENDF evaluation.

        Parameters
        ----------
        ev : openmc.data.endf.Evaluation
            ENDF evaluation
        resonances : openmc.data.Resonance object
            openmc.data.Resonanance object generated from the same evaluation
            used to import values not contained in File 32

        Returns
        -------
        openmc.data.ResonanceCovariances
            Resonance covariance data

        """
        file_obj = io.StringIO(ev.section[32, 151])

        # Determine whether discrete or continuous representation
        items = endf.get_head_record(file_obj)
        n_isotope = items[4]  # Number of isotopes

        ranges = []
        for _ in range(n_isotope):
            items = endf.get_cont_record(file_obj)
            n_ranges = items[4]  # Number of resonance energy ranges

            for j in range(n_ranges):
                items = endf.get_cont_record(file_obj)
                # Unresolved flags - 0: only scattering radius given
                #                    1: resolved parameters given
                #                    2: unresolved parameters given
                unresolved_flag = items[2]
                formalism = items[3]  # resonance formalism

                # Throw error for unsupported formalisms
                if formalism in [0, 7]:
                    error = 'LRF='+str(formalism)+' covariance not supported '\
                            'for this formalism'
                    raise NotImplementedError(error)

                if unresolved_flag in (0, 1):
                    # Resolved resonance region
                    resonance = resonances.ranges[j]
                    erange = _FORMALISMS[formalism].from_endf(ev, file_obj,
                                                              items, resonance)
                    ranges.append(erange)

                elif unresolved_flag == 2:
                    warn = 'Unresolved resonance not supported. Covariance '\
                           'values for the unresolved region not imported.'
                    warnings.warn(warn)

        return cls(ranges)


class ResonanceCovarianceRange:
    """Resonace covariance range. Base class for different formalisms.

    Parameters
    ----------
    energy_min : float
        Minimum energy of the resolved resonance range in eV
    energy_max : float
        Maximum energy of the resolved resonance range in eV

    Attributes
    ----------
    energy_min : float
        Minimum energy of the resolved resonance range in eV
    energy_max : float
        Maximum energy of the resolved resonance range in eV
    parameters : pandas.DataFrame
        Resonance parameters
    covariance : numpy.array
        The covariance matrix contained within the ENDF evaluation
    lcomp : int
        Flag indicating format of the covariance matrix within the ENDF file
    file2res : openmc.data.ResonanceRange object
        Corresponding resonance range with File 2 data.
    mpar : int
        Number of parameters in covariance matrix for each individual resonance
    formalism : str
        String descriptor of formalism
    """
    def __init__(self, energy_min, energy_max):
        self.energy_min = energy_min
        self.energy_max = energy_max

    def subset(self, parameter_str, bounds):
        """Produce a subset of resonance parameters and the corresponding
        covariance matrix to an IncidentNeutron object.

        Parameters
        ----------
        parameter_str : str
            parameter to be discriminated
            (i.e. 'energy', 'captureWidth', 'fissionWidthA'...)
        bounds : np.array
            [low numerical bound, high numerical bound]

        Returns
        -------
        res_cov_range : openmc.data.ResonanceCovarianceRange
            ResonanceCovarianceRange object that contains a subset of the
            covariance matrix (upper triangular) as well as a subset parameters
            within self.file2params

        """
        # Copy range and prevent change of original
        res_cov_range = copy.deepcopy(self)

        parameters = self.file2res.parameters
        cov = res_cov_range.covariance
        mpar = res_cov_range.mpar
        # Create mask
        mask1 = parameters[parameter_str] >= bounds[0]
        mask2 = parameters[parameter_str] <= bounds[1]
        mask = mask1 & mask2
        res_cov_range.parameters = parameters[mask]
        indices = res_cov_range.parameters.index.values
        # Build subset of covariance
        sub_cov_dim = len(indices)*mpar
        cov_subset_vals = []
        for index1 in indices:
            for i in range(mpar):
                for index2 in indices:
                    for j in range(mpar):
                        if index2*mpar+j >= index1*mpar+i:
                            cov_subset_vals.append(cov[index1*mpar+i,
                                                   index2*mpar+j])

        cov_subset = np.zeros([sub_cov_dim, sub_cov_dim])
        tri_indices = np.triu_indices(sub_cov_dim)
        cov_subset[tri_indices] = cov_subset_vals

        res_cov_range.file2res.parameters = parameters[mask]
        res_cov_range.covariance = cov_subset
        return res_cov_range

    def sample(self, n_samples):
        """Sample resonance parameters based on the covariances provided
        within an ENDF evaluation.

        Parameters
        ----------
        n_samples : int
            The number of samples to produce

        Returns
        -------
        samples : list of openmc.data.ResonanceCovarianceRange objects
            List of samples size `n_samples`

        """
        warn_str = 'Sampling routine does not guarantee positive values for '\
                   'parameters. This can lead to undefined behavior in the '\
                   'reconstruction routine.'
        warnings.warn(warn_str)
        parameters = self.parameters
        cov = self.covariance

        # Symmetrizing covariance matrix
        cov = cov + cov.T - np.diag(cov.diagonal())
        formalism = self.formalism
        mpar = self.mpar
        samples = []

        # Handling MLBW/SLBW sampling
        if formalism == 'mlbw' or formalism == 'slbw':
            params = ['energy', 'neutronWidth', 'captureWidth', 'fissionWidth',
                      'competitiveWidth']
            param_list = params[:mpar]
            mean_array = parameters[param_list].values
            mean = mean_array.flatten()
            par_samples = np.random.multivariate_normal(mean, cov,
                                                        size=n_samples)
            spin = parameters['J'].values
            l_value = parameters['L'].values
            for sample in par_samples:
                energy = sample[0::mpar]
                gn = sample[1::mpar]
                gg = sample[2::mpar]
                gf = sample[3::mpar] if mpar > 3 else parameters['fissionWidth'].values
                gx = sample[4::mpar] if mpar > 4 else parameters['competitiveWidth'].values
                gt = gn + gg + gf + gx

                records = []
                for j, E in enumerate(energy):
                    records.append([energy[j], l_value[j], spin[j], gt[j],
                                    gn[j], gg[j], gf[j], gx[j]])
                columns = ['energy', 'L', 'J', 'totalWidth', 'neutronWidth',
                           'captureWidth', 'fissionWidth', 'competitiveWidth']
                sample_params = pd.DataFrame.from_records(records,
                                                          columns=columns)
                # Copy ResonanceRange object
                res_range = copy.copy(self.file2res)
                res_range.parameters = sample_params
                samples.append(res_range)

        # Handling RM sampling
        elif formalism == 'rm':
            params = ['energy', 'neutronWidth', 'captureWidth',
                      'fissionWidthA', 'fissionWidthB']
            param_list = params[:mpar]
            mean_array = parameters[param_list].values
            mean = mean_array.flatten()
            par_samples = np.random.multivariate_normal(mean, cov,
                                                        size=n_samples)
            spin = parameters['J'].values
            l_value = parameters['L'].values
            for sample in par_samples:
                energy = sample[0::mpar]
                gn = sample[1::mpar]
                gg = sample[2::mpar]
                gfa = sample[3::mpar] if mpar > 3 else parameters['fissionWidthA'].values
                gfb = sample[4::mpar] if mpar > 3 else parameters['fissionWidthB'].values

                records = []
                for j, E in enumerate(energy):
                    records.append([energy[j], l_value[j], spin[j], gn[j],
                                    gg[j], gfa[j], gfb[j]])
                columns = ['energy', 'L', 'J', 'neutronWidth',
                           'captureWidth', 'fissionWidthA', 'fissionWidthB']
                sample_params = pd.DataFrame.from_records(records,
                                                          columns=columns)
                # Copy ResonanceRange object
                res_range = copy.copy(self.file2res)
                res_range.parameters = sample_params
                samples.append(res_range)

        return samples


class MultiLevelBreitWignerCovariance(ResonanceCovarianceRange):
    """Multi-level Breit-Wigner resolved resonance formalism covariance data.
    Parameters
    ----------
    energy_min : float
        Minimum energy of the resolved resonance range in eV
    energy_max : float
        Maximum energy of the resolved resonance range in eV

    Attributes
    ----------
    energy_min : float
        Minimum energy of the resolved resonance range in eV
    energy_max : float
        Maximum energy of the resolved resonance range in eV
    parameters : pandas.DataFrame
        Resonance parameters
    covariance : numpy.array
        The covariance matrix contained within the ENDF evaluation
    mpar : int
        Number of parameters in covariance matrix for each individual resonance
    lcomp : int
        Flag indicating format of the covariance matrix within the ENDF file
    file2res : openmc.data.ResonanceRange object
        Corresponding resonance range with File 2 data.
    formalism : str
        String descriptor of formalism

    """

    def __init__(self, energy_min, energy_max, parameters, covariance, mpar,
                 lcomp, file2res):
        super().__init__(energy_min, energy_max)
        self.parameters = parameters
        self.covariance = covariance
        self.mpar = mpar
        self.lcomp = lcomp
        self.file2res = copy.copy(file2res)
        self.formalism = 'mlbw'

    @classmethod
    def from_endf(cls, ev, file_obj, items, resonance):
        """Create MLBW covariance data from an ENDF evaluation.

        Parameters
        ----------
        ev : openmc.data.endf.Evaluation
            ENDF evaluation
        file_obj : file-like object
            ENDF file positioned at the second record of a resonance range
            subsection in MF=32, MT=151
        items : list
            Items from the CONT record at the start of the resonance range
            subsection
        resonance : openmc.data.ResonanceRange object
            Corresponding resonance range with File 2 data.

        Returns
        -------
        openmc.data.MultiLevelBreitWignerCovariance
            Multi-level Breit-Wigner resonance covariance parameters

        """

        # Read energy-dependent scattering radius if present
        energy_min, energy_max = items[0:2]
        nro, naps = items[4:6]
        if nro != 0:
            params, ape = endf.get_tab1_record(file_obj)

        # Other scatter radius parameters
        items = endf.get_cont_record(file_obj)
        lcomp = items[3]  # Flag for compatibility 0, 1, 2 - 2 is compact form
        nls = items[4]  # number of l-values

        # Build covariance matrix for General Resolved Resonance Formats
        if lcomp == 1:
            items = endf.get_cont_record(file_obj)
            # Number of short range type resonance covariances
            num_short_range = items[4]

            # Read resonance widths, J values, etc
            records = []
            for i in range(num_short_range):
                items, values = endf.get_list_record(file_obj)
                mpar = items[2]
                num_res = items[5]
                num_par_vals = num_res*6
                res_values = values[:num_par_vals]
                cov_values = values[num_par_vals:]

                energy = res_values[0::6]
                spin = res_values[1::6]
                gt = res_values[2::6]
                gn = res_values[3::6]
                gg = res_values[4::6]
                gf = res_values[5::6]

                for i, E in enumerate(energy):
                    records.append([energy[i], spin[i], gt[i], gn[i],
                                    gg[i], gf[i]])

                # Build the upper-triangular covariance matrix
                cov_dim = mpar*num_res
                cov = np.zeros([cov_dim, cov_dim])
                indices = np.triu_indices(cov_dim)
                cov[indices] = cov_values

        # Compact format - Resonances and individual uncertainties followed by
        # compact correlations
        elif lcomp == 2:
            items, values = endf.get_list_record(file_obj)
            num_res = items[5]
            energy = values[0::12]
            spin = values[1::12]
            gt = values[2::12]
            gn = values[3::12]
            gg = values[4::12]
            gf = values[5::12]
            par_unc = []
            for i in range(num_res):
                res_unc = values[i*12+6 : i*12+12]
                # Delete 0 values (not provided, no fission width)
                # DAJ/DGT always zero, DGF sometimes nonzero [1, 2, 5]
                res_unc_nonzero = []
                for j in range(6):
                    if j in [1, 2, 5] and res_unc[j] != 0.0:
                        res_unc_nonzero.append(res_unc[j])
                    elif j in [0, 3, 4]:
                        res_unc_nonzero.append(res_unc[j])
                par_unc.extend(res_unc_nonzero)

            records = []
            for i, E in enumerate(energy):
                records.append([energy[i], spin[i], gt[i], gn[i],
                                gg[i], gf[i]])

            corr = endf.get_intg_record(file_obj)
            cov = np.diag(par_unc).dot(corr).dot(np.diag(par_unc))

        # Compatible resolved resonance format
        elif lcomp == 0:
            cov = np.zeros([4, 4])
            records = []
            cov_index = 0
            for i in range(nls):
                items, values = endf.get_list_record(file_obj)
                num_res = items[5]
                for j in range(num_res):
                    one_res = values[18*j:18*(j+1)]
                    res_values = one_res[:6]
                    cov_values = one_res[6:]
                    records.append(list(res_values))

                    # Populate the coviariance matrix for this resonance
                    # There are no covariances between resonances in lcomp=0
                    cov[cov_index, cov_index] = cov_values[0]
                    cov[cov_index+1, cov_index+1 : cov_index+2] = cov_values[1:2]
                    cov[cov_index+1, cov_index+3] = cov_values[4]
                    cov[cov_index+2, cov_index+2] = cov_values[3]
                    cov[cov_index+2, cov_index+3] = cov_values[5]
                    cov[cov_index+3, cov_index+3] = cov_values[6]

                    cov_index += 4
                    if j < num_res-1:  # Pad matrix for additional values
                        cov = np.pad(cov, ((0, 4), (0, 4)), 'constant',
                                     constant_values=0)

        # Create pandas DataFrame with resonance data, currently
        # redundant with data.IncidentNeutron.resonance
        columns = ['energy', 'J', 'totalWidth', 'neutronWidth',
                   'captureWidth', 'fissionWidth']
        parameters = pd.DataFrame.from_records(records, columns=columns)
        # Determine mpar (number of parameters for each resonance in
        # covariance matrix)
        nparams, params = parameters.shape
        covsize = cov.shape[0]
        mpar = int(covsize/nparams)
        # Add parameters from File 2
        parameters = _add_file2_contributions(parameters,
                                              resonance.parameters)
        # Create instance of class
        mlbw = cls(energy_min, energy_max, parameters, cov, mpar, lcomp,
                   resonance)
        return mlbw


class SingleLevelBreitWignerCovariance(MultiLevelBreitWignerCovariance):
    """Single-level Breit-Wigner resolved resonance formalism covariance data.
    Single-level Breit-Wigner resolved resonance data is is identified by LRF=1
    in the ENDF-6 format.

    Parameters
    ----------
    energy_min : float
        Minimum energy of the resolved resonance range in eV
    energy_max : float
        Maximum energy of the resolved resonance range in eV

    Attributes
    ----------
    energy_min : float
        Minimum energy of the resolved resonance range in eV
    energy_max : float
        Maximum energy of the resolved resonance range in eV
    parameters : pandas.DataFrame
        Resonance parameters
    covariance : numpy.array
        The covariance matrix contained within the ENDF evaluation
    mpar : int
        Number of parameters in covariance matrix for each individual resonance
    formalism : str
        String descriptor of formalism
    lcomp : int
        Flag indicating format of the covariance matrix within the ENDF file
    file2res : openmc.data.ResonanceRange object
        Corresponding resonance range with File 2 data.
    """

    def __init__(self, energy_min, energy_max, parameters, covariance, mpar,
                 lcomp, file2res):
        super().__init__(energy_min, energy_max, parameters, covariance, mpar,
                         lcomp, file2res)
        self.formalism = 'slbw'


class ReichMooreCovariance(ResonanceCovarianceRange):
    """Reich-Moore resolved resonance formalism covariance data.

    Reich-Moore resolved resonance data is identified by LRF=3 in the ENDF-6
    format.

    Parameters
    ----------
    energy_min : float
        Minimum energy of the resolved resonance range in eV
    energy_max : float
        Maximum energy of the resolved resonance range in eV

    Attributes
    ----------
    energy_min : float
        Minimum energy of the resolved resonance range in eV
    energy_max : float
        Maximum energy of the resolved resonance range in eV
    parameters : pandas.DataFrame
        Resonance parameters
    covariance : numpy.array
        The covariance matrix contained within the ENDF evaluation
    lcomp : int
        Flag indicating format of the covariance matrix within the ENDF file
    mpar : int
        Number of parameters in covariance matrix for each individual resonance
    file2res : openmc.data.ResonanceRange object
        Corresponding resonance range with File 2 data.
    formalism : str
        String descriptor of formalism
    """

    def __init__(self, energy_min, energy_max, parameters, covariance, mpar,
                 lcomp, file2res):
        super().__init__(energy_min, energy_max)
        self.parameters = parameters
        self.covariance = covariance
        self.mpar = mpar
        self.lcomp = lcomp
        self.file2res = copy.copy(file2res)
        self.formalism = 'rm'

    @classmethod
    def from_endf(cls, ev, file_obj, items, resonance):
        """Create Reich-Moore resonance covariance data from an ENDF
        evaluation. Includes the resonance parameters contained separately in
        File 32.

        Parameters
        ----------
        ev : openmc.data.endf.Evaluation
            ENDF evaluation
        file_obj : file-like object
            ENDF file positioned at the second record of a resonance range
            subsection in MF=2, MT=151
        items : list
            Items from the CONT record at the start of the resonance range
            subsection
        resonance : openmc.data.Resonance object
            openmc.data.Resonanance object generated from the same evaluation
            used to import values not contained in File 32

        Returns
        -------
        openmc.data.ReichMooreCovariance
            Reich-Moore resonance covariance parameters

        """
        # Read energy-dependent scattering radius if present
        energy_min, energy_max = items[0:2]
        nro, naps = items[4:6]
        if nro != 0:
            params, ape = endf.get_tab1_record(file_obj)

        # Other scatter radius parameters
        items = endf.get_cont_record(file_obj)
        lcomp = items[3]  # Flag for compatibility 0, 1, 2 - 2 is compact form

        # Build covariance matrix for General Resolved Resonance Formats
        if lcomp == 1:
            items = endf.get_cont_record(file_obj)
            # Number of short range type resonance covariances
            num_short_range = items[4]
            # Read resonance widths, J values, etc
            records = []
            for i in range(num_short_range):
                items, values = endf.get_list_record(file_obj)
                mpar = items[2]
                num_res = items[5]
                num_par_vals = num_res*6
                res_values = values[:num_par_vals]
                cov_values = values[num_par_vals:]

                energy = res_values[0::6]
                spin = res_values[1::6]
                gn = res_values[2::6]
                gg = res_values[3::6]
                gfa = res_values[4::6]
                gfb = res_values[5::6]

                for i, E in enumerate(energy):
                    records.append([energy[i], spin[i], gn[i], gg[i],
                                    gfa[i], gfb[i]])

                # Build the upper-triangular covariance matrix
                cov_dim = mpar*num_res
                cov = np.zeros([cov_dim, cov_dim])
                indices = np.triu_indices(cov_dim)
                cov[indices] = cov_values

        # Compact format - Resonances and individual uncertainties followed by
        # compact correlations
        elif lcomp == 2:
            items, values = endf.get_list_record(file_obj)
            num_res = items[5]
            energy = values[0::12]
            spin = values[1::12]
            gn = values[2::12]
            gg = values[3::12]
            gfa = values[4::12]
            gfb = values[5::12]
            par_unc = []
            for i in range(num_res):
                res_unc = values[i*12+6 : i*12+12]
                # Delete 0 values (not provided in evaluation)
                res_unc = [x for x in res_unc if x != 0.0]
                par_unc.extend(res_unc)

            records = []
            for i, E in enumerate(energy):
                records.append([energy[i], spin[i], gn[i], gg[i],
                                gfa[i], gfb[i]])

            corr = get_intg_record(file_obj)
            cov = np.diag(par_unc).dot(corr).dot(np.diag(par_unc))

        # Create pandas DataFrame with resonacne data
        columns = ['energy', 'J', 'neutronWidth', 'captureWidth',
                   'fissionWidthA', 'fissionWidthB']
        parameters = pd.DataFrame.from_records(records, columns=columns)

        # Determine mpar (number of parameters for each resonance in
        # covariance matrix)
        nparams, params = parameters.shape
        covsize = cov.shape[0]
        mpar = int(covsize/nparams)

        # Add parameters from File 2
        parameters = _add_file2_contributions(parameters,
                                              resonance.parameters)
        # Create instance of ReichMooreCovariance
        rmc = cls(energy_min, energy_max, parameters, cov, mpar, lcomp,
                  resonance)
        return rmc


_FORMALISMS = {
    0: ResonanceCovarianceRange,
    1: SingleLevelBreitWignerCovariance,
    2: MultiLevelBreitWignerCovariance,
    3: ReichMooreCovariance
    # 7: RMatrixLimitedCovariance
}


REACTION_NAME = {1: '(n,total)', 2: '(n,elastic)', 4: '(n,level)',
                 5: '(n,misc)', 11: '(n,2nd)', 16: '(n,2n)', 17: '(n,3n)',
                 18: '(n,fission)', 19: '(n,f)', 20: '(n,nf)', 21: '(n,2nf)',
                 22: '(n,na)', 23: '(n,n3a)', 24: '(n,2na)', 25: '(n,3na)',
                 27: '(n,absorption)', 28: '(n,np)', 29: '(n,n2a)',
                 30: '(n,2n2a)', 32: '(n,nd)', 33: '(n,nt)', 34: '(n,n3He)',
                 35: '(n,nd2a)', 36: '(n,nt2a)', 37: '(n,4n)', 38: '(n,3nf)',
                 41: '(n,2np)', 42: '(n,3np)', 44: '(n,n2p)', 45: '(n,npa)',
                 91: '(n,nc)', 101: '(n,disappear)', 102: '(n,gamma)',
                 103: '(n,p)', 104: '(n,d)', 105: '(n,t)', 106: '(n,3He)',
                 107: '(n,a)', 108: '(n,2a)', 109: '(n,3a)', 111: '(n,2p)',
                 112: '(n,pa)', 113: '(n,t2a)', 114: '(n,d2a)', 115: '(n,pd)',
                 116: '(n,pt)', 117: '(n,da)', 152: '(n,5n)', 153: '(n,6n)',
                 154: '(n,2nt)', 155: '(n,ta)', 156: '(n,4np)', 157: '(n,3nd)',
                 158: '(n,nda)', 159: '(n,2npa)', 160: '(n,7n)', 161: '(n,8n)',
                 162: '(n,5np)', 163: '(n,6np)', 164: '(n,7np)', 165: '(n,4na)',
                 166: '(n,5na)', 167: '(n,6na)', 168: '(n,7na)', 169: '(n,4nd)',
                 170: '(n,5nd)', 171: '(n,6nd)', 172: '(n,3nt)', 173: '(n,4nt)',
                 174: '(n,5nt)', 175: '(n,6nt)', 176: '(n,2n3He)',
                 177: '(n,3n3He)', 178: '(n,4n3He)', 179: '(n,3n2p)',
                 180: '(n,3n2a)', 181: '(n,3npa)', 182: '(n,dt)',
                 183: '(n,npd)', 184: '(n,npt)', 185: '(n,ndt)',
                 186: '(n,np3He)', 187: '(n,nd3He)', 188: '(n,nt3He)',
                 189: '(n,nta)', 190: '(n,2n2p)', 191: '(n,p3He)',
                 192: '(n,d3He)', 193: '(n,3Hea)', 194: '(n,4n2p)',
                 195: '(n,4n2a)', 196: '(n,4npa)', 197: '(n,3p)',
                 198: '(n,n3p)', 199: '(n,3n2pa)', 200: '(n,5n2p)', 203: '(n,Xp)',
                 204: '(n,Xd)', 205: '(n,Xt)', 206: '(n,X3He)', 207: '(n,Xa)',
                 301: 'heating', 444: 'damage-energy',
                 649: '(n,pc)', 699: '(n,dc)', 749: '(n,tc)', 799: '(n,3Hec)',
                 849: '(n,ac)', 891: '(n,2nc)', 901: 'heating-local'}
REACTION_NAME.update({i: '(n,n{})'.format(i - 50) for i in range(51, 91)})
REACTION_NAME.update({i: '(n,p{})'.format(i - 600) for i in range(600, 649)})
REACTION_NAME.update({i: '(n,d{})'.format(i - 650) for i in range(650, 699)})
REACTION_NAME.update({i: '(n,t{})'.format(i - 700) for i in range(700, 749)})
REACTION_NAME.update({i: '(n,3He{})'.format(i - 750) for i in range(750, 799)})
REACTION_NAME.update({i: '(n,a{})'.format(i - 800) for i in range(800, 849)})
REACTION_NAME.update({i: '(n,2n{})'.format(i - 875) for i in range(875, 891)})

REACTION_MT = {name: mt for mt, name in REACTION_NAME.items()}
REACTION_MT['fission'] = 18

FISSION_MTS = (18, 19, 20, 21, 38)


def _get_products(ev, mt):
    """Generate products from MF=6 in an ENDF evaluation

    Parameters
    ----------
    ev : openmc.data.endf.Evaluation
        ENDF evaluation to read from
    mt : int
        The MT value of the reaction to get products for

    Raises
    ------
    IOError
        When the Kalbach-Mann systematics is used, but the product
        is not defined in the 'center-of-mass' system. The breakup logic
        is not implemented which can lead to this error being raised while
        the definition of the product is correct.

    Returns
    -------
    products : list of openmc.data.Product
        Products of the reaction

    """
    file_obj = StringIO(ev.section[6, mt])

    # Read HEAD record
    items = get_head_record(file_obj)
    reference_frame = {1: 'laboratory', 2: 'center-of-mass',
                       3: 'light-heavy', 4: 'breakup'}[items[3]]
    n_products = items[4]

    products = []
    for i in range(n_products):
        # Get yield for this product
        params, yield_ = get_tab1_record(file_obj)

        za = int(params[0])
        awr = params[1]
        law = params[3]

        if za == 0:
            p = Product('photon')
        elif za == 1:
            p = Product('neutron')
        elif za == 1000:
            p = Product('electron')
        else:
            Z, A = divmod(za, 1000)
            p = Product('{}{}'.format(ATOMIC_SYMBOL[Z], A))

        p.yield_ = yield_

        """
        # Set reference frame
        if reference_frame == 'laboratory':
            p.center_of_mass = False
        elif reference_frame == 'center-of-mass':
            p.center_of_mass = True
        elif reference_frame == 'light-heavy':
            p.center_of_mass = (awr <= 4.0)
        """

        if law == 0:
            # No distribution given
            pass
        if law == 1:
            # Continuum energy-angle distribution

            # Peak ahead to determine type of distribution
            position = file_obj.tell()
            params = get_cont_record(file_obj)
            file_obj.seek(position)

            lang = params[2]
            if lang == 1:
                p.distribution = [CorrelatedAngleEnergy.from_endf(file_obj)]
            elif lang == 2:
                # Products need to be described in the center-of-mass system
                product_center_of_mass = False
                if reference_frame == 'center-of-mass':
                    product_center_of_mass = True
                elif reference_frame == 'light-heavy':
                    product_center_of_mass = (awr <= 4.0)
                # TODO: 'breakup' logic not implemented

                if product_center_of_mass is False:
                    raise IOError(
                        "Kalbach-Mann representation must be defined in the "
                        "'center-of-mass' system"
                    )

                zat = ev.target["atomic_number"] * 1000 + ev.target["mass_number"]
                projectile_mass = ev.projectile["mass"]
                p.distribution = [KalbachMann.from_endf(file_obj,
                                                        za,
                                                        zat,
                                                        projectile_mass)]

        elif law == 2:
            # Discrete two-body scattering
            params, tab2 = get_tab2_record(file_obj)
            ne = params[5]
            energy = np.zeros(ne)
            mu = []
            for i in range(ne):
                items, values = get_list_record(file_obj)
                energy[i] = items[1]
                lang = items[2]
                if lang == 0:
                    mu.append(Legendre(values))
                elif lang == 12:
                    mu.append(Tabular(values[::2], values[1::2]))
                elif lang == 14:
                    mu.append(Tabular(values[::2], values[1::2],
                                      'log-linear'))

            angle_dist = AngleDistribution(energy, mu)
            dist = UncorrelatedAngleEnergy(angle_dist)
            p.distribution = [dist]
            # TODO: Add level-inelastic info?

        elif law == 3:
            # Isotropic discrete emission
            p.distribution = [UncorrelatedAngleEnergy()]
            # TODO: Add level-inelastic info?

        elif law == 4:
            # Discrete two-body recoil
            pass

        elif law == 5:
            # Charged particle elastic scattering
            pass

        elif law == 6:
            # N-body phase-space distribution
            p.distribution = [NBodyPhaseSpace.from_endf(file_obj)]

        elif law == 7:
            # Laboratory energy-angle distribution
            p.distribution = [LaboratoryAngleEnergy.from_endf(file_obj)]

        products.append(p)

    return products


def _get_fission_products_ace(ace):
    """Generate fission products from an ACE table

    Parameters
    ----------
    ace : openmc.data.ace.Table
        ACE table to read from

    Returns
    -------
    products : list of openmc.data.Product
        Prompt and delayed fission neutrons
    derived_products : list of openmc.data.Product
        "Total" fission neutron

    """
    # No NU block
    if ace.jxs[2] == 0:
        return None, None

    products = []
    derived_products = []

    # Either prompt nu or total nu is given
    if ace.xss[ace.jxs[2]] > 0:
        whichnu = 'prompt' if ace.jxs[24] > 0 else 'total'

        neutron = Product('neutron')
        neutron.emission_mode = whichnu

        idx = ace.jxs[2]
        LNU = int(ace.xss[idx])
        if LNU == 1:
            # Polynomial function form of nu
            NC = int(ace.xss[idx+1])
            coefficients = ace.xss[idx+2 : idx+2+NC].copy()
            for i in range(coefficients.size):
                coefficients[i] *= EV_PER_MEV**(-i)
            neutron.yield_ = Polynomial(coefficients)
        elif LNU == 2:
            # Tabular data form of nu
            neutron.yield_ = Tabulated1D.from_ace(ace, idx + 1)

        products.append(neutron)

    # Both prompt nu and total nu
    elif ace.xss[ace.jxs[2]] < 0:
        # Read prompt neutron yield
        prompt_neutron = Product('neutron')
        prompt_neutron.emission_mode = 'prompt'

        idx = ace.jxs[2] + 1
        LNU = int(ace.xss[idx])
        if LNU == 1:
            # Polynomial function form of nu
            NC = int(ace.xss[idx+1])
            coefficients = ace.xss[idx+2 : idx+2+NC].copy()
            for i in range(coefficients.size):
                coefficients[i] *= EV_PER_MEV**(-i)
            prompt_neutron.yield_ = Polynomial(coefficients)
        elif LNU == 2:
            # Tabular data form of nu
            prompt_neutron.yield_ = Tabulated1D.from_ace(ace, idx + 1)

        # Read total neutron yield
        total_neutron = Product('neutron')
        total_neutron.emission_mode = 'total'

        idx = ace.jxs[2] + int(abs(ace.xss[ace.jxs[2]])) + 1
        LNU = int(ace.xss[idx])

        if LNU == 1:
            # Polynomial function form of nu
            NC = int(ace.xss[idx+1])
            coefficients = ace.xss[idx+2 : idx+2+NC].copy()
            for i in range(coefficients.size):
                coefficients[i] *= EV_PER_MEV**(-i)
            total_neutron.yield_ = Polynomial(coefficients)
        elif LNU == 2:
            # Tabular data form of nu
            total_neutron.yield_ = Tabulated1D.from_ace(ace, idx + 1)

        products.append(prompt_neutron)
        derived_products.append(total_neutron)

    # Check for delayed nu data
    if ace.jxs[24] > 0:
        yield_delayed = Tabulated1D.from_ace(ace, ace.jxs[24] + 1)

        # Delayed neutron precursor distribution
        idx = ace.jxs[25]
        n_group = ace.nxs[8]
        total_group_probability = 0.
        for group in range(n_group):
            delayed_neutron = Product('neutron')
            delayed_neutron.emission_mode = 'delayed'

            # Convert units of inverse shakes to inverse seconds
            delayed_neutron.decay_rate = ace.xss[idx] * 1.e8

            group_probability = Tabulated1D.from_ace(ace, idx + 1)
            if np.all(group_probability.y == group_probability.y[0]):
                delayed_neutron.yield_ = deepcopy(yield_delayed)
                delayed_neutron.yield_.y *= group_probability.y[0]
                total_group_probability += group_probability.y[0]
            else:
                # Get union energy grid and ensure energies are within
                # interpolable range of both functions
                max_energy = min(yield_delayed.x[-1], group_probability.x[-1])
                energy = np.union1d(yield_delayed.x, group_probability.x)
                energy = energy[energy <= max_energy]

                # Calculate group yield
                group_yield = yield_delayed(energy) * group_probability(energy)
                delayed_neutron.yield_ = Tabulated1D(energy, group_yield)

            # Advance position
            nr = int(ace.xss[idx + 1])
            ne = int(ace.xss[idx + 2 + 2*nr])
            idx += 3 + 2*nr + 2*ne

            # Energy distribution for delayed fission neutrons
            location_start = int(ace.xss[ace.jxs[26] + group])
            delayed_neutron.distribution.append(
                AngleEnergy.from_ace(ace, ace.jxs[27], location_start))

            products.append(delayed_neutron)

        # Renormalize delayed neutron yields to reflect fact that in ACE
        # file, the sum of the group probabilities is not exactly one
        for product in products[1:]:
            if total_group_probability > 0.:
                product.yield_.y /= total_group_probability

    return products, derived_products


def _get_fission_products_endf(ev):
    """Generate fission products from an ENDF evaluation

    Parameters
    ----------
    ev : openmc.data.endf.Evaluation

    Returns
    -------
    products : list of openmc.data.Product
        Prompt and delayed fission neutrons
    derived_products : list of openmc.data.Product
        "Total" fission neutron

    """
    products = []
    derived_products = []

    if (1, 456) in ev.section:
        prompt_neutron = Product('neutron')
        prompt_neutron.emission_mode = 'prompt'

        # Prompt nu values
        file_obj = StringIO(ev.section[1, 456])
        lnu = get_head_record(file_obj)[3]
        if lnu == 1:
            # Polynomial representation
            items, coefficients = get_list_record(file_obj)
            prompt_neutron.yield_ = Polynomial(coefficients)
        elif lnu == 2:
            # Tabulated representation
            params, prompt_neutron.yield_ = get_tab1_record(file_obj)

        products.append(prompt_neutron)

    if (1, 452) in ev.section:
        total_neutron = Product('neutron')
        total_neutron.emission_mode = 'total'

        # Total nu values
        file_obj = StringIO(ev.section[1, 452])
        lnu = get_head_record(file_obj)[3]
        if lnu == 1:
            # Polynomial representation
            items, coefficients = get_list_record(file_obj)
            total_neutron.yield_ = Polynomial(coefficients)
        elif lnu == 2:
            # Tabulated representation
            params, total_neutron.yield_ = get_tab1_record(file_obj)

        if (1, 456) in ev.section:
            derived_products.append(total_neutron)
        else:
            products.append(total_neutron)

    if (1, 455) in ev.section:
        file_obj = StringIO(ev.section[1, 455])

        # Determine representation of delayed nu data
        items = get_head_record(file_obj)
        ldg = items[2]
        lnu = items[3]

        if ldg == 0:
            # Delayed-group constants energy independent
            items, decay_constants = get_list_record(file_obj)
            for constant in decay_constants:
                delayed_neutron = Product('neutron')
                delayed_neutron.emission_mode = 'delayed'
                delayed_neutron.decay_rate = constant
                products.append(delayed_neutron)
        elif ldg == 1:
            # Delayed-group constants energy dependent
            raise NotImplementedError('Delayed neutron with energy-dependent '
                                      'group constants.')

        # In MF=1, MT=455, the delayed-group abundances are actually not
        # specified if the group constants are energy-independent. In this case,
        # the abundances must be inferred from MF=5, MT=455 where multiple
        # energy distributions are given.
        if lnu == 1:
            # Nu represented as polynomial
            items, coefficients = get_list_record(file_obj)
            yield_ = Polynomial(coefficients)
            for neutron in products[-6:]:
                neutron.yield_ = deepcopy(yield_)
        elif lnu == 2:
            # Nu represented by tabulation
            params, yield_ = get_tab1_record(file_obj)
            for neutron in products[-6:]:
                neutron.yield_ = deepcopy(yield_)

        if (5, 455) in ev.section:
            file_obj = StringIO(ev.section[5, 455])
            items = get_head_record(file_obj)
            nk = items[4]
            if nk > 1 and len(decay_constants) == 1:
                # If only one precursor group is listed in MF=1, MT=455, use the
                # energy spectra from MF=5 to split them into different groups
                for _ in range(nk - 1):
                    products.append(deepcopy(products[1]))
            elif nk != len(decay_constants):
                raise ValueError(
                    'Number of delayed neutron fission spectra ({}) does not '
                    'match number of delayed neutron precursors ({}).'.format(
                        nk, len(decay_constants)))
            for i in range(nk):
                params, applicability = get_tab1_record(file_obj)
                dist = UncorrelatedAngleEnergy()
                dist.energy = EnergyDistribution.from_endf(file_obj, params)

                delayed_neutron = products[1 + i]
                yield_ = delayed_neutron.yield_

                # Here we handle the fact that the delayed neutron yield is the
                # product of the total delayed neutron yield and the
                # "applicability" of the energy distribution law in file 5.
                if isinstance(yield_, Tabulated1D):
                    if np.all(applicability.y == applicability.y[0]):
                        yield_.y *= applicability.y[0]
                    else:
                        # Get union energy grid and ensure energies are within
                        # interpolable range of both functions
                        max_energy = min(yield_.x[-1], applicability.x[-1])
                        energy = np.union1d(yield_.x, applicability.x)
                        energy = energy[energy <= max_energy]

                        # Calculate group yield
                        group_yield = yield_(energy) * applicability(energy)
                        delayed_neutron.yield_ = Tabulated1D(energy, group_yield)
                elif isinstance(yield_, Polynomial):
                    if len(yield_) == 1:
                        delayed_neutron.yield_ = deepcopy(applicability)
                        delayed_neutron.yield_.y *= yield_.coef[0]
                    else:
                        if np.all(applicability.y == applicability.y[0]):
                            yield_.coef[0] *= applicability.y[0]
                        else:
                            raise NotImplementedError(
                                'Total delayed neutron yield and delayed group '
                                'probability are both energy-dependent.')

                delayed_neutron.distribution.append(dist)

    return products, derived_products


def _get_activation_products(ev, rx):
    """Generate activation products from an ENDF evaluation

    Parameters
    ----------
    ev : openmc.data.endf.Evaluation
        The ENDF evaluation
    rx : openmc.data.Reaction
        Reaction which generates activation products

    Returns
    -------
    products : list of openmc.data.Product
        Activation products

    """
    file_obj = StringIO(ev.section[8, rx.mt])

    # Determine total number of states and whether decay chain is given in a
    # decay sublibrary
    items = get_head_record(file_obj)
    n_states = items[4]
    decay_sublib = (items[5] == 1)

    # Determine if file 9/10 are present
    present = {9: False, 10: False}
    for _ in range(n_states):
        if decay_sublib:
            items = get_cont_record(file_obj)
        else:
            items, values = get_list_record(file_obj)
        lmf = items[2]
        if lmf == 9:
            present[9] = True
        elif lmf == 10:
            present[10] = True

    products = []

    for mf in (9, 10):
        if not present[mf]:
            continue

        file_obj = StringIO(ev.section[mf, rx.mt])
        items = get_head_record(file_obj)
        n_states = items[4]
        for i in range(n_states):
            # Determine what the product is
            items, xs = get_tab1_record(file_obj)
            Z, A = divmod(items[2], 1000)
            excited_state = items[3]

            # Get GNDS name for product
            symbol = ATOMIC_SYMBOL[Z]
            if excited_state > 0:
                name = '{}{}_e{}'.format(symbol, A, excited_state)
            else:
                name = '{}{}'.format(symbol, A)

            p = Product(name)
            if mf == 9:
                p.yield_ = xs
            else:
                # Re-interpolate production cross section and neutron cross
                # section to union energy grid
                energy = np.union1d(xs.x, rx.xs['0K'].x)
                prod_xs = xs(energy)
                neutron_xs = rx.xs['0K'](energy)
                idx = np.where(neutron_xs > 0)

                # Calculate yield as ratio
                yield_ = np.zeros_like(energy)
                yield_[idx] = prod_xs[idx] / neutron_xs[idx]
                p.yield_ = Tabulated1D(energy, yield_)

            # Check if product already exists from MF=6 and if it does, just
            # overwrite the existing yield.
            for product in rx.products:
                if name == product.particle:
                    product.yield_ = p.yield_
                    break
            else:
                products.append(p)

    return products


def _get_photon_products_ace(ace, rx):
    """Generate photon products from an ACE table

    Parameters
    ----------
    ace : openmc.data.ace.Table
        ACE table to read from
    rx : openmc.data.Reaction
        Reaction that generates photons

    Returns
    -------
    photons : list of openmc.Products
        Photons produced from reaction with given MT

    """
    n_photon_reactions = ace.nxs[6]
    photon_mts = ace.xss[ace.jxs[13]:ace.jxs[13] +
                         n_photon_reactions].astype(int)

    photons = []
    for i in range(n_photon_reactions):
        # Determine corresponding reaction
        neutron_mt = photon_mts[i] // 1000

        if neutron_mt != rx.mt:
            continue

        # Create photon product and assign to reactions
        photon = Product('photon')

        # ==================================================================
        # Photon yield / production cross section

        loca = int(ace.xss[ace.jxs[14] + i])
        idx = ace.jxs[15] + loca - 1
        mftype = int(ace.xss[idx])
        idx += 1

        if mftype in (12, 16):
            # Yield data taken from ENDF File 12 or 6
            mtmult = int(ace.xss[idx])
            assert mtmult == neutron_mt

            # Read photon yield as function of energy
            photon.yield_ = Tabulated1D.from_ace(ace, idx + 1)

        elif mftype == 13:
            # Cross section data from ENDF File 13

            # Energy grid index at which data starts
            threshold_idx = int(ace.xss[idx]) - 1
            n_energy = int(ace.xss[idx + 1])
            energy = ace.xss[ace.jxs[1] + threshold_idx:
                             ace.jxs[1] + threshold_idx + n_energy]*EV_PER_MEV

            # Get photon production cross section
            photon_prod_xs = ace.xss[idx + 2:idx + 2 + n_energy]
            neutron_xs = list(rx.xs.values())[0](energy)
            idx = np.where(neutron_xs > 0.)

            # Calculate photon yield
            yield_ = np.zeros_like(photon_prod_xs)
            yield_[idx] = photon_prod_xs[idx] / neutron_xs[idx]
            photon.yield_ = Tabulated1D(energy, yield_)

        else:
            raise ValueError("MFTYPE must be 12, 13, 16. Got {0}".format(
                mftype))

        # ==================================================================
        # Photon energy distribution

        location_start = int(ace.xss[ace.jxs[18] + i])
        distribution = AngleEnergy.from_ace(ace, ace.jxs[19], location_start)
        assert isinstance(distribution, UncorrelatedAngleEnergy)

        # ==================================================================
        # Photon angular distribution
        loc = int(ace.xss[ace.jxs[16] + i])

        if loc == 0:
            # No angular distribution data are given for this reaction,
            # isotropic scattering is asssumed in LAB
            energy = np.array([photon.yield_.x[0], photon.yield_.x[-1]])
            mu_isotropic = Uniform(-1., 1.)
            distribution.angle = AngleDistribution(
                energy, [mu_isotropic, mu_isotropic])
        else:
            distribution.angle = AngleDistribution.from_ace(ace, ace.jxs[17], loc)

        # Add to list of distributions
        photon.distribution.append(distribution)
        photons.append(photon)

    return photons


def _get_photon_products_endf(ev, rx):
    """Generate photon products from an ENDF evaluation

    Parameters
    ----------
    ev : openmc.data.endf.Evaluation
        ENDF evaluation to read from
    rx : openmc.data.Reaction
        Reaction that generates photons

    Returns
    -------
    products : list of openmc.Products
        Photons produced from reaction with given MT

    """
    products = []

    if (12, rx.mt) in ev.section:
        file_obj = StringIO(ev.section[12, rx.mt])

        items = get_head_record(file_obj)
        option = items[2]

        if option == 1:
            # Multiplicities given
            n_discrete_photon = items[4]
            if n_discrete_photon > 1:
                items, total_yield = get_tab1_record(file_obj)
            for k in range(n_discrete_photon):
                photon = Product('photon')

                # Get photon yield
                items, photon.yield_ = get_tab1_record(file_obj)

                # Get photon energy distribution
                law = items[3]
                dist = UncorrelatedAngleEnergy()
                if law == 1:
                    # TODO: Get file 15 distribution
                    pass
                elif law == 2:
                    energy = items[0]
                    primary_flag = items[2]
                    dist.energy = DiscretePhoton(primary_flag, energy,
                                                 ev.target['mass'])

                photon.distribution.append(dist)
                products.append(photon)

        elif option == 2:
            # Transition probability arrays given
            ppyield = {}
            ppyield['type'] = 'transition'
            ppyield['transition'] = transition = {}

            # Determine whether simple (LG=1) or complex (LG=2) transitions
            lg = items[3]

            # Get transition data
            items, values = get_list_record(file_obj)
            transition['energy_start'] = items[0]
            transition['energies'] = np.array(values[::lg + 1])
            transition['direct_probability'] = np.array(values[1::lg + 1])
            if lg == 2:
                # Complex case
                transition['conditional_probability'] = np.array(
                    values[2::lg + 1])

    elif (13, rx.mt) in ev.section:
        file_obj = StringIO(ev.section[13, rx.mt])

        # Determine option
        items = get_head_record(file_obj)
        n_discrete_photon = items[4]
        if n_discrete_photon > 1:
            items, total_xs = get_tab1_record(file_obj)
        for k in range(n_discrete_photon):
            photon = Product('photon')
            items, xs = get_tab1_record(file_obj)

            # Re-interpolate photon production cross section and neutron cross
            # section to union energy grid
            energy = np.union1d(xs.x, rx.xs['0K'].x)
            photon_prod_xs = xs(energy)
            neutron_xs = rx.xs['0K'](energy)
            idx = np.where(neutron_xs > 0)

            # Calculate yield as ratio
            yield_ = np.zeros_like(energy)
            yield_[idx] = photon_prod_xs[idx] / neutron_xs[idx]
            photon.yield_ = Tabulated1D(energy, yield_)

            # Get photon energy distribution
            law = items[3]
            dist = UncorrelatedAngleEnergy()
            if law == 1:
                # TODO: Get file 15 distribution
                pass
            elif law == 2:
                energy = items[1]
                primary_flag = items[2]
                dist.energy = DiscretePhoton(primary_flag, energy,
                                             ev.target['mass'])

            photon.distribution.append(dist)
            products.append(photon)

    return products


class Reaction(EqualityMixin):
    """A nuclear reaction

    A Reaction object represents a single reaction channel for a nuclide with
    an associated cross section and, if present, a secondary angle and energy
    distribution.

    Parameters
    ----------
    mt : int
        The ENDF MT number for this reaction.

    Attributes
    ----------
    center_of_mass : bool
        Indicates whether scattering kinematics should be performed in the
        center-of-mass or laboratory reference frame.
        grid above the threshold value in barns.
    redundant : bool
        Indicates whether or not this is a redundant reaction
    mt : int
        The ENDF MT number for this reaction.
    q_value : float
        The Q-value of this reaction in eV.
    xs : dict of str to openmc.data.Function1D
        Microscopic cross section for this reaction as a function of incident
        energy; these cross sections are provided in a dictionary where the key
        is the temperature of the cross section set.
    products : Iterable of openmc.data.Product
        Reaction products
    derived_products : Iterable of openmc.data.Product
        Derived reaction products. Used for 'total' fission neutron data when
        prompt/delayed data also exists.

    """

    def __init__(self, mt):
        self._center_of_mass = True
        self._redundant = False
        self._q_value = 0.
        self._xs = {}
        self._products = []
        self._derived_products = []

        self.mt = mt

    def __repr__(self):
        if self.mt in REACTION_NAME:
            return "<Reaction: MT={} {}>".format(self.mt, REACTION_NAME[self.mt])
        else:
            return "<Reaction: MT={}>".format(self.mt)

    @property
    def center_of_mass(self):
        return self._center_of_mass

    @center_of_mass.setter
    def center_of_mass(self, center_of_mass):
        check_type('center of mass', center_of_mass, (bool, np.bool_))
        self._center_of_mass = center_of_mass

    @property
    def redundant(self):
        return self._redundant

    @redundant.setter
    def redundant(self, redundant):
        check_type('redundant', redundant, (bool, np.bool_))
        self._redundant = redundant

    @property
    def q_value(self):
        return self._q_value

    @q_value.setter
    def q_value(self, q_value):
        check_type('Q value', q_value, Real)
        self._q_value = q_value

    @property
    def products(self):
        return self._products

    @products.setter
    def products(self, products):
        check_type('reaction products', products, Iterable, Product)
        self._products = products

    @property
    def derived_products(self):
        return self._derived_products

    @derived_products.setter
    def derived_products(self, derived_products):
        check_type('reaction derived products', derived_products,
                      Iterable, Product)
        self._derived_products = derived_products

    @property
    def xs(self):
        return self._xs

    @xs.setter
    def xs(self, xs):
        check_type('reaction cross section dictionary', xs, MutableMapping)
        for key, value in xs.items():
            check_type('reaction cross section temperature', key, str)
            check_type('reaction cross section', value, Callable)
        self._xs = xs

    def to_hdf5(self, group):
        """Write reaction to an HDF5 group

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to write to

        """

        group.attrs['mt'] = self.mt
        if self.mt in REACTION_NAME:
            group.attrs['label'] = np.string_(REACTION_NAME[self.mt])
        else:
            group.attrs['label'] = np.string_(self.mt)
        group.attrs['Q_value'] = self.q_value
        group.attrs['center_of_mass'] = 1 if self.center_of_mass else 0
        group.attrs['redundant'] = 1 if self.redundant else 0
        for T in self.xs:
            Tgroup = group.create_group(T)
            if self.xs[T] is not None:
                dset = Tgroup.create_dataset('xs', data=self.xs[T].y)
                threshold_idx = getattr(self.xs[T], '_threshold_idx', 0)
                dset.attrs['threshold_idx'] = threshold_idx
        for i, p in enumerate(self.products):
            pgroup = group.create_group('product_{}'.format(i))
            p.to_hdf5(pgroup)

    @classmethod
    def from_hdf5(cls, group, energy):
        """Generate reaction from an HDF5 group

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to read from
        energy : dict
            Dictionary whose keys are temperatures (e.g., '300K') and values are
            arrays of energies at which cross sections are tabulated at.

        Returns
        -------
        openmc.data.Reaction
            Reaction data

        """

        mt = group.attrs['mt']
        rx = cls(mt)
        rx.q_value = group.attrs['Q_value']
        rx.center_of_mass = bool(group.attrs['center_of_mass'])
        rx.redundant = bool(group.attrs.get('redundant', False))

        # Read cross section at each temperature
        for T, Tgroup in group.items():
            if T.endswith('K'):
                if 'xs' in Tgroup:
                    # Make sure temperature has associated energy grid
                    if T not in energy:
                        raise ValueError(
                            'Could not create reaction cross section for MT={} '
                            'at T={} because no corresponding energy grid '
                            'exists.'.format(mt, T))
                    xs = Tgroup['xs'][()]
                    threshold_idx = Tgroup['xs'].attrs['threshold_idx']
                    tabulated_xs = Tabulated1D(energy[T][threshold_idx:], xs)
                    tabulated_xs._threshold_idx = threshold_idx
                    rx.xs[T] = tabulated_xs

        # Determine number of products
        n_product = 0
        for name in group:
            if name.startswith('product_'):
                n_product += 1

        # Read reaction products
        for i in range(n_product):
            pgroup = group['product_{}'.format(i)]
            rx.products.append(Product.from_hdf5(pgroup))

        return rx

    @classmethod
    def from_ace(cls, ace, i_reaction):
        # Get nuclide energy grid
        n_grid = ace.nxs[3]
        grid = ace.xss[ace.jxs[1]:ace.jxs[1] + n_grid]*EV_PER_MEV

        # Convert data temperature to a "300.0K" number for indexing
        # temperature data
        strT = str(int(round(ace.temperature*EV_PER_MEV / K_BOLTZMANN))) + "K"

        if i_reaction > 0:
            mt = int(ace.xss[ace.jxs[3] + i_reaction - 1])
            rx = cls(mt)

            # Get Q-value of reaction
            rx.q_value = ace.xss[ace.jxs[4] + i_reaction - 1]*EV_PER_MEV

            # ==================================================================
            # CROSS SECTION

            # Get locator for cross-section data
            loc = int(ace.xss[ace.jxs[6] + i_reaction - 1])

            # Determine starting index on energy grid
            threshold_idx = int(ace.xss[ace.jxs[7] + loc - 1]) - 1

            # Determine number of energies in reaction
            n_energy = int(ace.xss[ace.jxs[7] + loc])
            energy = grid[threshold_idx:threshold_idx + n_energy]

            # Read reaction cross section
            xs = ace.xss[ace.jxs[7] + loc + 1:ace.jxs[7] + loc + 1 + n_energy]

            # For damage energy production, convert to eV
            if mt == 444:
                xs *= EV_PER_MEV

            # Fix negatives -- known issue for Y89 in JEFF 3.2
            if np.any(xs < 0.0):
                warn("Negative cross sections found for MT={} in {}. Setting "
                     "to zero.".format(rx.mt, ace.name))
                xs[xs < 0.0] = 0.0

            tabulated_xs = Tabulated1D(energy, xs)
            tabulated_xs._threshold_idx = threshold_idx
            rx.xs[strT] = tabulated_xs

            # ==================================================================
            # YIELD AND ANGLE-ENERGY DISTRIBUTION

            # Determine multiplicity
            ty = int(ace.xss[ace.jxs[5] + i_reaction - 1])
            rx.center_of_mass = (ty < 0)
            if i_reaction < ace.nxs[5] + 1:
                if ty != 19:
                    if abs(ty) > 100:
                        # Energy-dependent neutron yield
                        idx = ace.jxs[11] + abs(ty) - 101
                        yield_ = Tabulated1D.from_ace(ace, idx)
                    else:
                        # 0-order polynomial i.e. a constant
                        yield_ = Polynomial((abs(ty),))

                    neutron = Product('neutron')
                    neutron.yield_ = yield_
                    rx.products.append(neutron)
                else:
                    assert mt in FISSION_MTS
                    rx.products, rx.derived_products = _get_fission_products_ace(ace)

                    for p in rx.products:
                        if p.emission_mode in ('prompt', 'total'):
                            neutron = p
                            break
                    else:
                        raise Exception("Couldn't find prompt/total fission neutron")

                # Determine locator for ith energy distribution
                lnw = int(ace.xss[ace.jxs[10] + i_reaction - 1])
                while lnw > 0:
                    # Applicability of this distribution
                    neutron.applicability.append(Tabulated1D.from_ace(
                        ace, ace.jxs[11] + lnw + 2))

                    # Read energy distribution data
                    neutron.distribution.append(AngleEnergy.from_ace(
                        ace, ace.jxs[11], lnw, rx))

                    lnw = int(ace.xss[ace.jxs[11] + lnw - 1])

        else:
            # Elastic scattering
            mt = 2
            rx = cls(mt)

            # Get elastic cross section values
            elastic_xs = ace.xss[ace.jxs[1] + 3*n_grid:ace.jxs[1] + 4*n_grid]

            # Fix negatives -- known issue for Ti46,49,50 in JEFF 3.2
            if np.any(elastic_xs < 0.0):
                warn("Negative elastic scattering cross section found for {}. "
                     "Setting to zero.".format(ace.name))
                elastic_xs[elastic_xs < 0.0] = 0.0

            tabulated_xs = Tabulated1D(grid, elastic_xs)
            tabulated_xs._threshold_idx = 0
            rx.xs[strT] = tabulated_xs

            # No energy distribution for elastic scattering
            neutron = Product('neutron')
            neutron.distribution.append(UncorrelatedAngleEnergy())
            rx.products.append(neutron)

        # ======================================================================
        # ANGLE DISTRIBUTION (FOR UNCORRELATED)

        if i_reaction < ace.nxs[5] + 1:
            # Check if angular distribution data exist
            loc = int(ace.xss[ace.jxs[8] + i_reaction])
            if loc < 0:
                # Angular distribution is given as part of a product
                # angle-energy distribution
                angle_dist = None
            elif loc == 0:
                # Angular distribution is isotropic
                energy = [0.0, grid[-1]]
                mu = Uniform(-1., 1.)
                angle_dist = AngleDistribution(energy, [mu, mu])
            else:
                angle_dist = AngleDistribution.from_ace(ace, ace.jxs[9], loc)

            # Apply angular distribution to each uncorrelated angle-energy
            # distribution
            if angle_dist is not None:
                for d in neutron.distribution:
                    d.angle = angle_dist

        # ======================================================================
        # PHOTON PRODUCTION

        rx.products += _get_photon_products_ace(ace, rx)

        return rx

    @classmethod
    def from_endf(cls, ev, mt):
        """Generate a reaction from an ENDF evaluation

        Parameters
        ----------
        ev : openmc.data.endf.Evaluation
            ENDF evaluation
        mt : int
            The MT value of the reaction to get data for

        Returns
        -------
        rx : openmc.data.Reaction
            Reaction data

        """
        rx = Reaction(mt)

        # Integrated cross section
        if (3, mt) in ev.section:
            file_obj = StringIO(ev.section[3, mt])
            get_head_record(file_obj)
            params, rx.xs['0K'] = get_tab1_record(file_obj)
            rx.q_value = params[1]

        # Get fission product yields (nu) as well as delayed neutron energy
        # distributions
        if mt in FISSION_MTS:
            rx.products, rx.derived_products = _get_fission_products_endf(ev)

        if (6, mt) in ev.section:
            # Product angle-energy distribution
            for product in _get_products(ev, mt):
                if mt in FISSION_MTS and product.particle == 'neutron':
                    rx.products[0].applicability = product.applicability
                    rx.products[0].distribution = product.distribution
                else:
                    rx.products.append(product)

        elif (4, mt) in ev.section or (5, mt) in ev.section:
            # Uncorrelated angle-energy distribution
            neutron = Product('neutron')

            # Note that the energy distribution for MT=455 is read in
            # _get_fission_products_endf rather than here
            if (5, mt) in ev.section:
                file_obj = StringIO(ev.section[5, mt])
                items = get_head_record(file_obj)
                nk = items[4]
                for i in range(nk):
                    params, applicability = get_tab1_record(file_obj)
                    dist = UncorrelatedAngleEnergy()
                    dist.energy = EnergyDistribution.from_endf(file_obj, params)

                    neutron.applicability.append(applicability)
                    neutron.distribution.append(dist)
            elif mt == 2:
                # Elastic scattering -- no energy distribution is given since it
                # can be calulcated analytically
                dist = UncorrelatedAngleEnergy()
                neutron.distribution.append(dist)
            elif mt >= 51 and mt < 91:
                # Level inelastic scattering -- no energy distribution is given
                # since it can be calculated analytically. Here we determine the
                # necessary parameters to create a LevelInelastic object
                dist = UncorrelatedAngleEnergy()

                A = ev.target['mass']
                threshold = (A + 1.)/A*abs(rx.q_value)
                mass_ratio = (A/(A + 1.))**2
                dist.energy = LevelInelastic(threshold, mass_ratio)

                neutron.distribution.append(dist)

            if (4, mt) in ev.section:
                for dist in neutron.distribution:
                    dist.angle = AngleDistribution.from_endf(ev, mt)

            if mt in FISSION_MTS and (5, mt) in ev.section:
                # For fission reactions,
                rx.products[0].applicability = neutron.applicability
                rx.products[0].distribution = neutron.distribution
            else:
                rx.products.append(neutron)

        if (8, mt) in ev.section:
            rx.products += _get_activation_products(ev, rx)

        if (12, mt) in ev.section or (13, mt) in ev.section:
            rx.products += _get_photon_products_endf(ev, rx)

        return rx


class Product(EqualityMixin):
    """Secondary particle emitted in a nuclear reaction

    Parameters
    ----------
    particle : str, optional
        The particle type of the reaction product. Defaults to 'neutron'.

    Attributes
    ----------
    applicability : Iterable of openmc.data.Tabulated1D
        Probability of sampling a given distribution for this product.
    decay_rate : float
        Decay rate in inverse seconds
    distribution : Iterable of openmc.data.AngleEnergy
        Distributions of energy and angle of product.
    emission_mode : {'prompt', 'delayed', 'total'}
        Indicate whether the particle is emitted immediately or whether it
        results from the decay of reaction product (e.g., neutron emitted from a
        delayed neutron precursor). A special value of 'total' is used when the
        yield represents particles from prompt and delayed sources.
    particle : str
        The particle type of the reaction product
    yield_ : openmc.data.Function1D
        Yield of secondary particle in the reaction.

    """

    def __init__(self, particle='neutron'):
        self.applicability = []
        self.decay_rate = 0.0
        self.distribution = []
        self.emission_mode = 'prompt'
        self.particle = particle
        self.yield_ = Polynomial((1,))  # 0-order polynomial, i.e., a constant

    def __repr__(self):
        if isinstance(self.yield_, Tabulated1D):
            if np.all(self.yield_.y == self.yield_.y[0]):
                return "<Product: {}, emission={}, yield={}>".format(
                    self.particle, self.emission_mode, self.yield_.y[0])
            else:
                return "<Product: {}, emission={}, yield=tabulated>".format(
                    self.particle, self.emission_mode)
        else:
            return "<Product: {}, emission={}, yield=polynomial>".format(
                self.particle, self.emission_mode)

    @property
    def applicability(self):
        return self._applicability

    @applicability.setter
    def applicability(self, applicability):
        check_type('product distribution applicability', applicability,
                      Iterable, Tabulated1D)
        self._applicability = applicability

    @property
    def decay_rate(self):
        return self._decay_rate

    @decay_rate.setter
    def decay_rate(self, decay_rate):
        check_type('product decay rate', decay_rate, Real)
        check_greater_than('product decay rate', decay_rate, 0.0, True)
        self._decay_rate = decay_rate

    @property
    def distribution(self):
        return self._distribution

    @distribution.setter
    def distribution(self, distribution):
        check_type('product angle-energy distribution', distribution,
                      Iterable, AngleEnergy)
        self._distribution = distribution

    @property
    def emission_mode(self):
        return self._emission_mode

    @emission_mode.setter
    def emission_mode(self, emission_mode):
        check_value('product emission mode', emission_mode,
                       ('prompt', 'delayed', 'total'))
        self._emission_mode = emission_mode

    @property
    def particle(self):
        return self._particle

    @particle.setter
    def particle(self, particle):
        check_type('product particle type', particle, str)
        self._particle = particle

    @property
    def yield_(self):
        return self._yield

    @yield_.setter
    def yield_(self, yield_):
        check_type('product yield', yield_, Function1D)
        self._yield = yield_

    def to_hdf5(self, group):
        """Write product to an HDF5 group

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to write to

        """
        group.attrs['particle'] = np.string_(self.particle)
        group.attrs['emission_mode'] = np.string_(self.emission_mode)
        if self.decay_rate > 0.0:
            group.attrs['decay_rate'] = self.decay_rate

        # Write yield
        self.yield_.to_hdf5(group, 'yield')

        # Write applicability/distribution
        group.attrs['n_distribution'] = len(self.distribution)
        for i, d in enumerate(self.distribution):
            dgroup = group.create_group('distribution_{}'.format(i))
            if self.applicability:
                self.applicability[i].to_hdf5(dgroup, 'applicability')
            d.to_hdf5(dgroup)

    @classmethod
    def from_hdf5(cls, group):
        """Generate reaction product from HDF5 data

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to read from

        Returns
        -------
        openmc.data.Product
            Reaction product

        """
        particle = group.attrs['particle'].decode()
        p = cls(particle)

        p.emission_mode = group.attrs['emission_mode'].decode()
        if 'decay_rate' in group.attrs:
            p.decay_rate = group.attrs['decay_rate']

        # Read yield
        p.yield_ = Function1D.from_hdf5(group['yield'])

        # Read applicability/distribution
        n_distribution = group.attrs['n_distribution']
        distribution = []
        applicability = []
        for i in range(n_distribution):
            dgroup = group['distribution_{}'.format(i)]
            if 'applicability' in dgroup:
                applicability.append(Tabulated1D.from_hdf5(
                    dgroup['applicability']))
            distribution.append(AngleEnergy.from_hdf5(dgroup))

        p.distribution = distribution
        p.applicability = applicability

        return p


# For a given material, give a name for the ACE table and a list of ZAID
# identifiers.
ThermalTuple = namedtuple('ThermalTuple', ['name', 'zaids', 'nmix'])
_THERMAL_DATA = {
    'c_Al27': ThermalTuple('al27', [13027], 1),
    'c_Al_in_Al2O3': ThermalTuple('asap00', [13027], 1),
    'c_Be': ThermalTuple('be', [4009], 1),
    'c_Be_in_BeO': ThermalTuple('bebeo', [4009], 1),
    'c_Be_in_Be2C': ThermalTuple('bebe2c', [4009], 1),
    'c_Be_in_FLiBe': ThermalTuple('beflib', [4009], 1),
    'c_C6H6': ThermalTuple('benz', [1001, 6000, 6012], 2),
    'c_C_in_SiC': ThermalTuple('csic', [6000, 6012, 6013], 1),
    'c_Ca_in_CaH2': ThermalTuple('cacah2', [20040, 20042, 20043, 20044, 20046, 20048], 1),
    'c_D_in_D2O': ThermalTuple('dd2o', [1002], 1),
    'c_D_in_D2O_solid': ThermalTuple('dice', [1002], 1),
    'c_F_in_FLiBe': ThermalTuple('fflibe', [9019], 1),
    'c_Fe56': ThermalTuple('fe56', [26056], 1),
    'c_Graphite': ThermalTuple('graph', [6000, 6012, 6013], 1),
    'c_Graphite_10p': ThermalTuple('grph10', [6000, 6012, 6013], 1),
    'c_Graphite_30p': ThermalTuple('grph30', [6000, 6012, 6013], 1),
    'c_H_in_C5O2H8': ThermalTuple('lucite', [1001], 1),
    'c_H_in_CaH2': ThermalTuple('hcah2', [1001], 1),
    'c_H_in_CH2': ThermalTuple('hch2', [1001], 1),
    'c_H_in_CH4_liquid': ThermalTuple('lch4', [1001], 1),
    'c_H_in_CH4_solid': ThermalTuple('sch4', [1001], 1),
    'c_H_in_CH4_solid_phase_II': ThermalTuple('sch4p2', [1001], 1),
    'c_H_in_H2O': ThermalTuple('hh2o', [1001], 1),
    'c_H_in_H2O_solid': ThermalTuple('hice', [1001], 1),
    'c_H_in_HF': ThermalTuple('hhf', [1001], 1),
    'c_H_in_Mesitylene': ThermalTuple('mesi00', [1001], 1),
    'c_H_in_ParaffinicOil': ThermalTuple('hparaf', [1001], 1),
    'c_H_in_Toluene': ThermalTuple('tol00', [1001], 1),
    'c_H_in_UH3': ThermalTuple('huh3', [1001], 1),
    'c_H_in_YH2': ThermalTuple('hyh2', [1001], 1),
    'c_H_in_ZrH': ThermalTuple('hzrh', [1001], 1),
    'c_H_in_ZrH2': ThermalTuple('hzrh2', [1001], 1),
    'c_H_in_ZrHx': ThermalTuple('hzrhx', [1001], 1),
    'c_Li_in_FLiBe': ThermalTuple('liflib', [3006, 3007], 1),
    'c_Mg24': ThermalTuple('mg24', [12024], 1),
    'c_N_in_UN': ThermalTuple('n-un', [7014, 7015], 1),
    'c_O_in_Al2O3': ThermalTuple('osap00', [8016, 8017, 8018], 1),
    'c_O_in_BeO': ThermalTuple('obeo', [8016, 8017, 8018], 1),
    'c_O_in_D2O': ThermalTuple('od2o', [8016, 8017, 8018], 1),
    'c_O_in_H2O_solid': ThermalTuple('oice', [8016, 8017, 8018], 1),
    'c_O_in_UO2': ThermalTuple('ouo2', [8016, 8017, 8018], 1),
    'c_ortho_D': ThermalTuple('orthod', [1002], 1),
    'c_ortho_H': ThermalTuple('orthoh', [1001], 1),
    'c_para_D': ThermalTuple('parad', [1002], 1),
    'c_para_H': ThermalTuple('parah', [1001], 1),
    'c_Si28': ThermalTuple('si00', [14028], 1),
    'c_Si_in_SiC': ThermalTuple('sisic', [14028, 14029, 14030], 1),
    'c_SiO2_alpha': ThermalTuple('sio2-a', [8016, 8017, 8018, 14028, 14029, 14030], 3),
    'c_SiO2_beta': ThermalTuple('sio2-b', [8016, 8017, 8018, 14028, 14029, 14030], 3),
    'c_U_in_UN': ThermalTuple('u-un', [92233, 92234, 92235, 92236, 92238], 1),
    'c_U_in_UO2': ThermalTuple('uuo2', [92233, 92234, 92235, 92236, 92238], 1),
    'c_Y_in_YH2': ThermalTuple('yyh2', [39089], 1),
    'c_Zr_in_ZrH': ThermalTuple('zrzrh', [40000, 40090, 40091, 40092, 40094, 40096], 1),
    'c_Zr_in_ZrH2': ThermalTuple('zrzrh2', [40000, 40090, 40091, 40092, 40094, 40096], 1),
    'c_Zr_in_ZrHx': ThermalTuple('zrzrhx', [40000, 40090, 40091, 40092, 40094, 40096], 1),
}


_TEMPLATE_RECONR = """
reconr / %%%%%%%%%%%%%%%%%%% Reconstruct XS for neutrons %%%%%%%%%%%%%%%%%%%%%%%
{nendf} {npendf}
'{library} PENDF for {zsymam}'/
{mat} 2/
{error}/ err
'{library}: {zsymam}'/
'Processed by NJOY'/
0/
"""

_TEMPLATE_BROADR = """
broadr / %%%%%%%%%%%%%%%%%%%%%%% Doppler broaden XS %%%%%%%%%%%%%%%%%%%%%%%%%%%%
{nendf} {npendf} {nbroadr}
{mat} {num_temp} 0 0 0. /
{error}/ errthn
{temps}
0/
"""

_TEMPLATE_HEATR = """
heatr / %%%%%%%%%%%%%%%%%%%%%%%%% Add heating kerma %%%%%%%%%%%%%%%%%%%%%%%%%%%%
{nendf} {nheatr_in} {nheatr} /
{mat} 4 0 0 0 /
302 318 402 444 /
"""

_TEMPLATE_HEATR_LOCAL = """
heatr / %%%%%%%%%%%%%%%%% Add heating kerma (local photons) %%%%%%%%%%%%%%%%%%%%
{nendf} {nheatr_in} {nheatr_local} /
{mat} 4 0 0 1 /
302 318 402 444 /
"""

_TEMPLATE_GASPR = """
gaspr / %%%%%%%%%%%%%%%%%%%%%%%%% Add gas production %%%%%%%%%%%%%%%%%%%%%%%%%%%
{nendf} {ngaspr_in} {ngaspr} /
"""

_TEMPLATE_PURR = """
purr / %%%%%%%%%%%%%%%%%%%%%%%% Add probability tables %%%%%%%%%%%%%%%%%%%%%%%%%
{nendf} {npurr_in} {npurr} /
{mat} {num_temp} 1 20 64 /
{temps}
1.e10
0/
"""

_TEMPLATE_ACER = """
acer / %%%%%%%%%%%%%%%%%%%%%%%% Write out in ACE format %%%%%%%%%%%%%%%%%%%%%%%%
{nendf} {nacer_in} 0 {nace} {ndir}
1 0 1 .{ext} /
'{library}: {zsymam} at {temperature}'/
{mat} {temperature}
1 1 {ismooth}/
/
"""

_THERMAL_TEMPLATE_THERMR = """
thermr / %%%%%%%%%%%%%%%% Add thermal scattering data (free gas) %%%%%%%%%%%%%%%
0 {nthermr1_in} {nthermr1}
0 {mat} 12 {num_temp} 1 0 {iform} 1 221 1/
{temps}
{error} {energy_max}
thermr / %%%%%%%%%%%%%%%% Add thermal scattering data (bound) %%%%%%%%%%%%%%%%%%
{nthermal_endf} {nthermr2_in} {nthermr2}
{mat_thermal} {mat} 16 {num_temp} {inelastic} {elastic} {iform} {natom} 222 1/
{temps}
{error} {energy_max}
"""

_THERMAL_TEMPLATE_ACER = """
acer / %%%%%%%%%%%%%%%%%%%%%%%% Write out in ACE format %%%%%%%%%%%%%%%%%%%%%%%%
{nendf} {nthermal_acer_in} 0 {nace} {ndir}
2 0 1 .{ext}/
'{library}: {zsymam_thermal} processed by NJOY'/
{mat} {temperature} '{data.name}' {nza} /
{zaids} /
222 64 {mt_elastic} {elastic_type} {data.nmix} {energy_max} {iwt}/
"""


def run(commands, tapein, tapeout, input_filename=None, stdout=False,
        njoy_exec='njoy'):
    """Run NJOY with given commands

    Parameters
    ----------
    commands : str
        Input commands for NJOY
    tapein : dict
        Dictionary mapping tape numbers to paths for any input files
    tapeout : dict
        Dictionary mapping tape numbers to paths for any output files
    input_filename : str, optional
        File name to write out NJOY input commands
    stdout : bool, optional
        Whether to display output when running NJOY
    njoy_exec : str, optional
        Path to NJOY executable

    Raises
    ------
    subprocess.CalledProcessError
        If the NJOY process returns with a non-zero status

    """

    if input_filename is not None:
        with open(str(input_filename), 'w') as f:
            f.write(commands)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy evaluations to appropriates 'tapes'
        for tape_num, filename in tapein.items():
            tmpfilename = os.path.join(tmpdir, 'tape{}'.format(tape_num))
            shutil.copy(str(filename), tmpfilename)

        # Start up NJOY process
        njoy = Popen([njoy_exec], cwd=tmpdir, stdin=PIPE, stdout=PIPE,
                     stderr=STDOUT, universal_newlines=True)

        njoy.stdin.write(commands)
        njoy.stdin.flush()
        lines = []
        while True:
            # If process is finished, break loop
            line = njoy.stdout.readline()
            if not line and njoy.poll() is not None:
                break

            lines.append(line)
            if stdout:
                # If user requested output, print to screen
                print(line, end='')

        # Check for error
        if njoy.returncode != 0:
            raise CalledProcessError(njoy.returncode, njoy_exec,
                                     ''.join(lines))

        # Copy output files back to original directory
        for tape_num, filename in tapeout.items():
            tmpfilename = os.path.join(tmpdir, 'tape{}'.format(tape_num))
            if os.path.isfile(tmpfilename):
                shutil.move(tmpfilename, str(filename))


def make_pendf(filename, pendf='pendf', error=0.001, stdout=False):
    """Generate pointwise ENDF file from an ENDF file

    Parameters
    ----------
    filename : str
        Path to ENDF file
    pendf : str, optional
        Path of pointwise ENDF file to write
    error : float, optional
        Fractional error tolerance for NJOY processing
    stdout : bool
        Whether to display NJOY standard output

    Raises
    ------
    subprocess.CalledProcessError
        If the NJOY process returns with a non-zero status

    """

    make_ace(filename, pendf=pendf, error=error, broadr=False,
             heatr=False, purr=False, acer=False, stdout=stdout)


def make_ace(filename, temperatures=None, acer=True, xsdir=None,
             output_dir=None, pendf=False, error=0.001, broadr=True,
             heatr=True, gaspr=True, purr=True, evaluation=None,
             smoothing=True, **kwargs):
    """Generate incident neutron ACE file from an ENDF file

    File names can be passed to
    ``[acer, xsdir, pendf, broadr, heatr, gaspr, purr]``
    to specify the exact output for the given module.
    Otherwise, the files will be writen to the current directory
    or directory specified by ``output_dir``. Default file
    names mirror the variable names, e.g. ``heatr`` output
    will be written to a file named ``heatr`` unless otherwise
    specified.

    Parameters
    ----------
    filename : str
        Path to ENDF file
    temperatures : iterable of float, optional
        Temperatures in Kelvin to produce ACE files at. If omitted, data is
        produced at room temperature (293.6 K).
    acer : bool or str, optional
        Flag indicating if acer should be run. If a string is give, write the
        resulting ``ace`` file to this location. Path of ACE file to write.
        Defaults to ``"ace"``
    xsdir : str, optional
        Path of xsdir file to write. Defaults to ``"xsdir"`` in the same
        directory as ``acer``
    output_dir : str, optional
        Directory to write output for requested modules. If not provided
        and at least one of ``[pendf, broadr, heatr, gaspr, purr, acer]``
        is ``True``, then write output files to current directory. If given,
        must be a path to a directory.
    pendf : str, optional
        Path of pendf file to write. If omitted, the pendf file is not saved.
    error : float, optional
        Fractional error tolerance for NJOY processing
    broadr : bool or str, optional
        Indicating whether to Doppler broaden XS when running NJOY. If string,
        write the output tape to this file.
    heatr : bool or str, optional
        Indicating whether to add heating kerma when running NJOY. If string,
        write the output tape to this file.
    gaspr : bool or str, optional
        Indicating whether to add gas production data when running NJOY.
        If string, write the output tape to this file.
    purr : bool or str, optional
        Indicating whether to add probability table when running NJOY.
        If string, write the output tape to this file.
    evaluation : openmc.data.endf.Evaluation, optional
        If the ENDF file contains multiple material evaluations, this argument
        indicates which evaluation should be used.
    smoothing : bool, optional
        If the smoothing option (ACER card 6) is on (True) or off (False).
    **kwargs
        Keyword arguments passed to :func:`openmc.data.njoy.run`

    Raises
    ------
    subprocess.CalledProcessError
        If the NJOY process returns with a non-zero status
    IOError
        If ``output_dir`` does not point to a directory

    """
    if output_dir is None:
        output_dir = Path()
    else:
        output_dir = Path(output_dir)
        if not output_dir.is_dir():
            raise IOError("{} is not a directory".format(output_dir))

    ev = evaluation if evaluation is not None else endf.Evaluation(filename)
    mat = ev.material
    zsymam = ev.target['zsymam']

    # Determine name of library
    library = '{}-{}.{}'.format(*ev.info['library'])

    if temperatures is None:
        temperatures = [293.6]
    num_temp = len(temperatures)
    temps = ' '.join(str(i) for i in temperatures)

    # Create njoy commands by modules
    commands = ""

    nendf, npendf = 20, 21
    tapein = {nendf: filename}
    tapeout = {}
    if pendf:
        tapeout[npendf] = (output_dir / "pendf") if pendf is True else pendf

    # reconr
    commands += _TEMPLATE_RECONR
    nlast = npendf

    # broadr
    if broadr:
        nbroadr = nlast + 1
        tapeout[nbroadr] = (output_dir / "broadr") if broadr is True else broadr
        commands += _TEMPLATE_BROADR
        nlast = nbroadr

    # heatr
    if heatr:
        nheatr_in = nlast
        nheatr_local = nheatr_in + 1
        tapeout[nheatr_local] = (output_dir / "heatr_local") if heatr is True \
            else heatr + '_local'
        commands += _TEMPLATE_HEATR_LOCAL
        nheatr = nheatr_local + 1
        tapeout[nheatr] = (output_dir / "heatr") if heatr is True else heatr
        commands += _TEMPLATE_HEATR
        nlast = nheatr

    # gaspr
    if gaspr:
        ngaspr_in = nlast
        ngaspr = ngaspr_in + 1
        tapeout[ngaspr] = (output_dir / "gaspr") if gaspr is True else gaspr
        commands += _TEMPLATE_GASPR
        nlast = ngaspr

    # purr
    if purr:
        npurr_in = nlast
        npurr = npurr_in + 1
        tapeout[npurr] = (output_dir / "purr") if purr is True else purr
        commands += _TEMPLATE_PURR
        nlast = npurr

    commands = commands.format(**locals())

    # acer
    if acer:
        ismooth = int(smoothing)
        nacer_in = nlast
        for i, temperature in enumerate(temperatures):
            # Extend input with an ACER run for each temperature
            nace = nacer_in + 1 + 2*i
            ndir = nace + 1
            ext = '{:02}'.format(i + 1)
            commands += _TEMPLATE_ACER.format(**locals())

            # Indicate tapes to save for each ACER run
            tapeout[nace] = output_dir / "ace_{:.1f}".format(temperature)
            tapeout[ndir] = output_dir / "xsdir_{:.1f}".format(temperature)
    commands += 'stop\n'
    run(commands, tapein, tapeout, **kwargs)

    if acer:
        ace = (output_dir / "ace") if acer is True else Path(acer)
        xsdir = (ace.parent / "xsdir") if xsdir is None else xsdir
        with ace.open('w') as ace_file, xsdir.open('w') as xsdir_file:
            for temperature in temperatures:
                # Get contents of ACE file
                text = (output_dir / "ace_{:.1f}".format(temperature)).read_text()

                # If the target is metastable, make sure that ZAID in the ACE
                # file reflects this by adding 400
                if ev.target['isomeric_state'] > 0:
                    mass_first_digit = int(text[3])
                    if mass_first_digit <= 2:
                        text = text[:3] + str(mass_first_digit + 4) + text[4:]

                # Concatenate into destination ACE file
                ace_file.write(text)

                # Concatenate into destination xsdir file
                xsdir_in = output_dir / "xsdir_{:.1f}".format(temperature)
                xsdir_file.write(xsdir_in.read_text())

        # Remove ACE/xsdir files for each temperature
        for temperature in temperatures:
            (output_dir / "ace_{:.1f}".format(temperature)).unlink()
            (output_dir / "xsdir_{:.1f}".format(temperature)).unlink()


def make_ace_thermal(filename, filename_thermal, temperatures=None,
                     ace='ace', xsdir=None, output_dir=None, error=0.001,
                     iwt=2, evaluation=None, evaluation_thermal=None,
                     table_name=None, zaids=None, nmix=None, **kwargs):
    """Generate thermal scattering ACE file from ENDF files

    Parameters
    ----------
    filename : str
        Path to ENDF neutron sublibrary file
    filename_thermal : str
        Path to ENDF thermal scattering sublibrary file
    temperatures : iterable of float, optional
        Temperatures in Kelvin to produce data at. If omitted, data is produced
        at all temperatures given in the ENDF thermal scattering sublibrary.
    ace : str, optional
        Path of ACE file to write
    xsdir : str, optional
        Path of xsdir file to write. Defaults to ``"xsdir"`` in the same
        directory as ``ace``
    output_dir : str, optional
        Directory to write ace and xsdir files. If not provided, then write
        output files to current directory. If given, must be a path to a
        directory.
    error : float, optional
        Fractional error tolerance for NJOY processing
    iwt : int
        `iwt` parameter used in NJOR/ACER card 9
    evaluation : openmc.data.endf.Evaluation, optional
        If the ENDF neutron sublibrary file contains multiple material
        evaluations, this argument indicates which evaluation to use.
    evaluation_thermal : openmc.data.endf.Evaluation, optional
        If the ENDF thermal scattering sublibrary file contains multiple
        material evaluations, this argument indicates which evaluation to use.
    table_name : str, optional
        Name to assign to ACE table
    zaids : list of int, optional
        ZAIDs that the thermal scattering data applies to
    nmix : int, optional
        Number of atom types in mixed moderator
    **kwargs
        Keyword arguments passed to :func:`openmc.data.njoy.run`

    Raises
    ------
    subprocess.CalledProcessError
        If the NJOY process returns with a non-zero status

    """
    if output_dir is None:
        output_dir = Path()
    else:
        output_dir = Path(output_dir)
        if not output_dir.is_dir():
            raise IOError("{} is not a directory".format(output_dir))

    ev = evaluation if evaluation is not None else endf.Evaluation(filename)
    mat = ev.material
    zsymam = ev.target['zsymam']

    ev_thermal = (evaluation_thermal if evaluation_thermal is not None
                  else endf.Evaluation(filename_thermal))
    mat_thermal = ev_thermal.material
    zsymam_thermal = ev_thermal.target['zsymam'].strip()

    # Determine name, isotopes, and number of atom types
    if table_name and zaids and nmix:
        data = ThermalTuple(table_name, zaids, nmix)
    else:
        with warnings.catch_warnings(record=True) as w:
            proper_name = openmc.data.get_thermal_name(zsymam_thermal)
            if w:
                raise RuntimeError(
                    f"Thermal scattering material {zsymam_thermal} not "
                    "recognized. Please contact OpenMC developers at "
                    "https://openmc.discourse.group.")
        data = _THERMAL_DATA[proper_name]

    zaids = ' '.join(str(zaid) for zaid in data.zaids)
    nza = len(data.zaids)

    # Determine name of library
    library = '{}-{}.{}'.format(*ev_thermal.info['library'])

    # Determine if thermal elastic is present
    if (7, 2) in ev_thermal.section:
        elastic = 1
        mt_elastic = 223

        # Determine whether elastic is incoherent (0) or coherent (1)
        file_obj = StringIO(ev_thermal.section[7, 2])
        elastic_type = endf.get_head_record(file_obj)[2] - 1
    else:
        elastic = 0
        mt_elastic = 0
        elastic_type = 0

    # Determine number of principal atoms
    file_obj = StringIO(ev_thermal.section[7, 4])
    items = endf.get_head_record(file_obj)
    items, values = endf.get_list_record(file_obj)
    energy_max = values[3]
    natom = int(values[5])

    # Note that the 'iform' parameter is omitted in NJOY 99. We assume that the
    # user is using NJOY 2012 or later.
    iform = 0
    inelastic = 2

    # Determine temperatures from MF=7, MT=4 if none were specified
    if temperatures is None:
        file_obj = StringIO(ev_thermal.section[7, 4])
        endf.get_head_record(file_obj)
        endf.get_list_record(file_obj)
        endf.get_tab2_record(file_obj)
        params = endf.get_tab1_record(file_obj)[0]
        temperatures = [params[0]]
        for i in range(params[2]):
            temperatures.append(endf.get_list_record(file_obj)[0][0])

    num_temp = len(temperatures)
    temps = ' '.join(str(i) for i in temperatures)

    # Create njoy commands by modules
    commands = ""

    nendf, nthermal_endf, npendf = 20, 21, 22
    tapein = {nendf: filename, nthermal_endf: filename_thermal}
    tapeout = {}

    # reconr
    commands += _TEMPLATE_RECONR
    nlast = npendf

    # broadr
    nbroadr = nlast + 1
    commands += _TEMPLATE_BROADR
    nlast = nbroadr

    # thermr
    nthermr1_in = nlast
    nthermr1 = nthermr1_in + 1
    nthermr2_in = nthermr1
    nthermr2 = nthermr2_in + 1
    commands += _THERMAL_TEMPLATE_THERMR
    nlast = nthermr2

    commands = commands.format(**locals())

    # acer
    nthermal_acer_in = nlast
    for i, temperature in enumerate(temperatures):
        # Extend input with an ACER run for each temperature
        nace = nthermal_acer_in + 1 + 2*i
        ndir = nace + 1
        ext = '{:02}'.format(i + 1)
        commands += _THERMAL_TEMPLATE_ACER.format(**locals())

        # Indicate tapes to save for each ACER run
        tapeout[nace] = output_dir / "ace_{:.1f}".format(temperature)
        tapeout[ndir] = output_dir / "xsdir_{:.1f}".format(temperature)
    commands += 'stop\n'
    run(commands, tapein, tapeout, **kwargs)

    ace = output_dir / ace
    xsdir = (ace.parent / "xsdir") if xsdir is None else Path(xsdir)
    with ace.open('w') as ace_file, xsdir.open('w') as xsdir_file:
        # Concatenate ACE and xsdir files together
        for temperature in temperatures:
            ace_in = output_dir / "ace_{:.1f}".format(temperature)
            ace_file.write(ace_in.read_text())

            xsdir_in = output_dir / "xsdir_{:.1f}".format(temperature)
            xsdir_file.write(xsdir_in.read_text())

    # Remove ACE/xsdir files for each temperature
    for temperature in temperatures:
        (output_dir / "ace_{:.1f}".format(temperature)).unlink()
        (output_dir / "xsdir_{:.1f}".format(temperature)).unlink()


# Fractions of resonance widths used for reconstructing resonances
_RESONANCE_ENERGY_GRID = np.logspace(-3, 3, 61)


class IncidentNeutron(EqualityMixin):
    """Continuous-energy neutron interaction data.

    This class stores data derived from an ENDF-6 format neutron interaction
    sublibrary. Instances of this class are not normally instantiated by the
    user but rather created using the factory methods
    :meth:`IncidentNeutron.from_hdf5`, :meth:`IncidentNeutron.from_ace`, and
    :meth:`IncidentNeutron.from_endf`.

    Parameters
    ----------
    name : str
        Name of the nuclide using the GNDS naming convention
    atomic_number : int
        Number of protons in the target nucleus
    mass_number : int
        Number of nucleons in the target nucleus
    metastable : int
        Metastable state of the target nucleus. A value of zero indicates ground
        state.
    atomic_weight_ratio : float
        Atomic mass ratio of the target nuclide.
    kTs : Iterable of float
        List of temperatures of the target nuclide in the data set.
        The temperatures have units of eV.

    Attributes
    ----------
    atomic_number : int
        Number of protons in the target nucleus
    atomic_symbol : str
        Atomic symbol of the nuclide, e.g., 'Zr'
    atomic_weight_ratio : float
        Atomic weight ratio of the target nuclide.
    fission_energy : None or openmc.data.FissionEnergyRelease
        The energy released by fission, tabulated by component (e.g. prompt
        neutrons or beta particles) and dependent on incident neutron energy
    mass_number : int
        Number of nucleons in the target nucleus
    metastable : int
        Metastable state of the target nucleus. A value of zero indicates ground
        state.
    name : str
        Name of the nuclide using the GNDS naming convention
    reactions : dict
        Contains the cross sections, secondary angle and energy distributions,
        and other associated data for each reaction. The keys are the MT values
        and the values are Reaction objects.
    resonances : openmc.data.Resonances or None
        Resonance parameters
    resonance_covariance : openmc.data.ResonanceCovariance or None
        Covariance for resonance parameters
    temperatures : list of str
        List of string representations of the temperatures of the target nuclide
        in the data set.  The temperatures are strings of the temperature,
        rounded to the nearest integer; e.g., '294K'
    kTs : Iterable of float
        List of temperatures of the target nuclide in the data set.
        The temperatures have units of eV.
    urr : dict
        Dictionary whose keys are temperatures (e.g., '294K') and values are
        unresolved resonance region probability tables.

    """

    def __init__(self, name, atomic_number, mass_number, metastable,
                 atomic_weight_ratio, kTs):
        self.name = name
        self.atomic_number = atomic_number
        self.mass_number = mass_number
        self.metastable = metastable
        self.atomic_weight_ratio = atomic_weight_ratio
        self.kTs = kTs
        self.energy = {}
        self._fission_energy = None
        self.reactions = {}
        self._urr = {}
        self._resonances = None

    def __contains__(self, mt):
        return mt in self.reactions

    def __getitem__(self, mt):
        if mt in self.reactions:
            return self.reactions[mt]
        else:
            # Try to create a redundant cross section
            mts = self.get_reaction_components(mt)
            if len(mts) > 0:
                return self._get_redundant_reaction(mt, mts)
            else:
                raise KeyError('No reaction with MT={}.'.format(mt))

    def __repr__(self):
        return "<IncidentNeutron: {}>".format(self.name)

    def __iter__(self):
        return iter(self.reactions.values())

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        check_type('name', name, str)
        self._name = name

    @property
    def atomic_number(self):
        return self._atomic_number

    @atomic_number.setter
    def atomic_number(self, atomic_number):
        check_type('atomic number', atomic_number, Integral)
        check_greater_than('atomic number', atomic_number, 0, True)
        self._atomic_number = atomic_number

    @property
    def mass_number(self):
        return self._mass_number

    @mass_number.setter
    def mass_number(self, mass_number):
        check_type('mass number', mass_number, Integral)
        check_greater_than('mass number', mass_number, 0, True)
        self._mass_number = mass_number

    @property
    def metastable(self):
        return self._metastable

    @metastable.setter
    def metastable(self, metastable):
        check_type('metastable', metastable, Integral)
        check_greater_than('metastable', metastable, 0, True)
        self._metastable = metastable

    @property
    def atomic_weight_ratio(self):
        return self._atomic_weight_ratio

    @atomic_weight_ratio.setter
    def atomic_weight_ratio(self, atomic_weight_ratio):
        check_type('atomic weight ratio', atomic_weight_ratio, Real)
        check_greater_than('atomic weight ratio', atomic_weight_ratio, 0.0)
        self._atomic_weight_ratio = atomic_weight_ratio

    @property
    def fission_energy(self):
        return self._fission_energy

    @fission_energy.setter
    def fission_energy(self, fission_energy):
        check_type('fission energy release', fission_energy,
                      FissionEnergyRelease)
        self._fission_energy = fission_energy

    @property
    def reactions(self):
        return self._reactions

    @reactions.setter
    def reactions(self, reactions):
        check_type('reactions', reactions, Mapping)
        self._reactions = reactions

    @property
    def resonances(self):
        return self._resonances

    @resonances.setter
    def resonances(self, resonances):
        check_type('resonances', resonances, res.Resonances)
        self._resonances = resonances

    @property
    def resonance_covariance(self):
        return self._resonance_covariance

    @resonance_covariance.setter
    def resonance_covariance(self, resonance_covariance):
        check_type('resonance covariance', resonance_covariance,
                      res_cov.ResonanceCovariances)
        self._resonance_covariance = resonance_covariance

    @property
    def urr(self):
        return self._urr

    @urr.setter
    def urr(self, urr):
        check_type('probability table dictionary', urr, MutableMapping)
        for key, value in urr:
            check_type('probability table temperature', key, str)
            check_type('probability tables', value, ProbabilityTables)
        self._urr = urr

    @property
    def temperatures(self):
        return ["{}K".format(int(round(kT / K_BOLTZMANN))) for kT in self.kTs]

    @property
    def atomic_symbol(self):
        return ATOMIC_SYMBOL[self.atomic_number]

    def add_temperature_from_ace(self, ace_or_filename, metastable_scheme='nndc'):
        """Append data from an ACE file at a different temperature.

        Parameters
        ----------
        ace_or_filename : openmc.data.ace.Table or str
            ACE table to read from. If given as a string, it is assumed to be
            the filename for the ACE file.
        metastable_scheme : {'nndc', 'mcnp'}
            Determine how ZAID identifiers are to be interpreted in the case of
            a metastable nuclide. Because the normal ZAID (=1000*Z + A) does not
            encode metastable information, different conventions are used among
            different libraries. In MCNP libraries, the convention is to add 400
            for a metastable nuclide except for Am242m, for which 95242 is
            metastable and 95642 (or 1095242 in newer libraries) is the ground
            state. For NNDC libraries, ZAID is given as 1000*Z + A + 100*m.

        """

        data = IncidentNeutron.from_ace(ace_or_filename, metastable_scheme)

        # Check if temprature already exists
        strT = data.temperatures[0]
        if strT in self.temperatures:
            warn('Cross sections at T={} already exist.'.format(strT))
            return

        # Check that name matches
        if data.name != self.name:
            raise ValueError('Data provided for an incorrect nuclide.')

        # Add temperature
        self.kTs += data.kTs

        # Add energy grid
        self.energy[strT] = data.energy[strT]

        # Add normal and redundant reactions
        for mt in data.reactions:
            if mt in self:
                self[mt].xs[strT] = data[mt].xs[strT]
            else:
                warn("Tried to add cross sections for MT={} at T={} but this "
                     "reaction doesn't exist.".format(mt, strT))

        # Add probability tables
        if strT in data.urr:
            self.urr[strT] = data.urr[strT]

    def add_elastic_0K_from_endf(self, filename, overwrite=False):
        """Append 0K elastic scattering cross section from an ENDF file.

        Parameters
        ----------
        filename : str
            Path to ENDF file
        overwrite : bool
            If existing 0 K data is present, this flag can be used to indicate
            that it should be overwritten. Otherwise, an exception will be
            thrown.

        Raises
        ------
        ValueError
            If 0 K data is already present and the `overwrite` parameter is
            False.

        """
        # Check for existing data
        if '0K' in self.energy and not overwrite:
            raise ValueError('0 K data already exists for this nuclide.')

        data = type(self).from_endf(filename)
        if data.resonances is not None:
            x = []
            y = []
            for rr in data.resonances:
                if isinstance(rr, res.RMatrixLimited):
                    raise TypeError('R-Matrix Limited not supported.')
                elif isinstance(rr, res.Unresolved):
                    continue

                # Get energies/widths for resonances
                e_peak = rr.parameters['energy'].values
                if isinstance(rr, res.MultiLevelBreitWigner):
                    gamma = rr.parameters['totalWidth'].values
                elif isinstance(rr, res.ReichMoore):
                    df = rr.parameters
                    gamma = (df['neutronWidth'] +
                             df['captureWidth'] +
                             abs(df['fissionWidthA']) +
                             abs(df['fissionWidthB'])).values

                # Determine peak energies and widths
                e_min, e_max = rr.energy_min, rr.energy_max
                in_range = (e_peak > e_min) & (e_peak < e_max)
                e_peak = e_peak[in_range]
                gamma = gamma[in_range]

                # Get midpoints between resonances (use min/max energy of
                # resolved region as absolute lower/upper bound)
                e_mid = np.concatenate(
                    ([e_min], (e_peak[1:] + e_peak[:-1])/2, [e_max]))

                # Add grid around each resonance that includes the peak +/- the
                # width times each value in _RESONANCE_ENERGY_GRID. Values are
                # constrained so that points around one resonance don't overlap
                # with points around another. This algorithm is from Fudge
                # (https://doi.org/10.1063/1.1945057).
                energies = []
                for e, g, e_lower, e_upper in zip(e_peak, gamma, e_mid[:-1],
                                                  e_mid[1:]):
                    e_left = e - g*_RESONANCE_ENERGY_GRID
                    energies.append(e_left[e_left > e_lower][::-1])
                    e_right = e + g*_RESONANCE_ENERGY_GRID[1:]
                    energies.append(e_right[e_right < e_upper])

                # Concatenate all points
                energies = np.concatenate(energies)

                # Create 1000 equal log-spaced energies over RRR, combine with
                # resonance peaks and half-height energies
                e_log = np.logspace(log10(e_min), log10(e_max), 1000)
                energies = np.union1d(e_log, energies)

                # Linearize and thin cross section
                xi, yi = linearize(energies, data[2].xs['0K'])
                xi, yi = thin(xi, yi)

                # If there are multiple resolved resonance ranges (e.g. Pu239 in
                # ENDF/B-VII.1), combine them
                x = np.concatenate((x, xi))
                y = np.concatenate((y, yi))
        else:
            energies = data[2].xs['0K'].x
            x, y = linearize(energies, data[2].xs['0K'])
            x, y = thin(x, y)

        # Set 0K energy grid and elastic scattering cross section
        self.energy['0K'] = x
        self[2].xs['0K'] = Tabulated1D(x, y)

    def get_reaction_components(self, mt):
        """Determine what reactions make up redundant reaction.

        Parameters
        ----------
        mt : int
            ENDF MT number of the reaction to find components of.

        Returns
        -------
        mts : list of int
            ENDF MT numbers of reactions that make up the redundant reaction and
            have cross sections provided.

        """
        mts = []
        if mt in SUM_RULES:
            for mt_i in SUM_RULES[mt]:
                mts += self.get_reaction_components(mt_i)
        if mts:
            return mts
        else:
            return [mt] if mt in self else []

    # def export_to_hdf5(self, path, mode='a', libver='earliest'):
    #     """Export incident neutron data to an HDF5 file.

    #     Parameters
    #     ----------
    #     path : str
    #         Path to write HDF5 file to
    #     mode : {'r+', 'w', 'x', 'a'}
    #         Mode that is used to open the HDF5 file. This is the second argument
    #         to the :class:`h5py.File` constructor.
    #     libver : {'earliest', 'latest'}
    #         Compatibility mode for the HDF5 file. 'latest' will produce files
    #         that are less backwards compatible but have performance benefits.

    #     """
    #     # If data come from ENDF, don't allow exporting to HDF5
    #     if hasattr(self, '_evaluation'):
    #         raise NotImplementedError('Cannot export incident neutron data that '
    #                                   'originated from an ENDF file.')

    #     # Open file and write version
    #     with h5py.File(str(path), mode, libver=libver) as f:
    #         f.attrs['filetype'] = np.string_('data_neutron')
    #         f.attrs['version'] = np.array(HDF5_VERSION)

    #         # Write basic data
    #         g = f.create_group(self.name)
    #         g.attrs['Z'] = self.atomic_number
    #         g.attrs['A'] = self.mass_number
    #         g.attrs['metastable'] = self.metastable
    #         g.attrs['atomic_weight_ratio'] = self.atomic_weight_ratio
    #         ktg = g.create_group('kTs')
    #         for i, temperature in enumerate(self.temperatures):
    #             ktg.create_dataset(temperature, data=self.kTs[i])

    #         # Write energy grid
    #         eg = g.create_group('energy')
    #         for temperature in self.temperatures:
    #             eg.create_dataset(temperature, data=self.energy[temperature])

    #         # Write 0K energy grid if needed
    #         if '0K' in self.energy and '0K' not in eg:
    #             eg.create_dataset('0K', data=self.energy['0K'])

    #         # Write reaction data
    #         rxs_group = g.create_group('reactions')
    #         for rx in self.reactions.values():
    #             # Skip writing redundant reaction if it doesn't have photon
    #             # production or is a summed transmutation reaction. MT=4 is also
    #             # sometimes needed for probability tables. Also write gas
    #             # production, heating, and damage energy production.
    #             if rx.redundant:
    #                 photon_rx = any(p.particle == 'photon' for p in rx.products)
    #                 keep_mts = (4, 16, 103, 104, 105, 106, 107,
    #                             203, 204, 205, 206, 207, 301, 444, 901)
    #                 if not (photon_rx or rx.mt in keep_mts):
    #                     continue

    #             rx_group = rxs_group.create_group('reaction_{:03}'.format(rx.mt))
    #             rx.to_hdf5(rx_group)

    #             # Write total nu data if available
    #             if len(rx.derived_products) > 0 and 'total_nu' not in g:
    #                 tgroup = g.create_group('total_nu')
    #                 rx.derived_products[0].to_hdf5(tgroup)

    #         # Write unresolved resonance probability tables
    #         if self.urr:
    #             urr_group = g.create_group('urr')
    #             for temperature, urr in self.urr.items():
    #                 tgroup = urr_group.create_group(temperature)
    #                 urr.to_hdf5(tgroup)

    #         # Write fission energy release data
    #         if self.fission_energy is not None:
    #             fer_group = g.create_group('fission_energy_release')
    #             self.fission_energy.to_hdf5(fer_group)

    # @classmethod
    # def from_hdf5(cls, group_or_filename):
    #     """Generate continuous-energy neutron interaction data from HDF5 group

    #     Parameters
    #     ----------
    #     group_or_filename : h5py.Group or str
    #         HDF5 group containing interaction data. If given as a string, it is
    #         assumed to be the filename for the HDF5 file, and the first group is
    #         used to read from.

    #     Returns
    #     -------
    #     openmc.data.IncidentNeutron
    #         Continuous-energy neutron interaction data

    #     """
    #     if isinstance(group_or_filename, h5py.Group):
    #         group = group_or_filename
    #     else:
    #         h5file = h5py.File(str(group_or_filename), 'r')

    #         # Make sure version matches
    #         if 'version' in h5file.attrs:
    #             major, minor = h5file.attrs['version']
    #             # For now all versions of HDF5 data can be read
    #         else:
    #             raise IOError(
    #                 'HDF5 data does not indicate a version. Your installation of '
    #                 'the OpenMC Python API expects version {}.x data.'
    #                 .format(HDF5_VERSION_MAJOR))

    #         group = list(h5file.values())[0]

    #     name = group.name[1:]
    #     atomic_number = group.attrs['Z']
    #     mass_number = group.attrs['A']
    #     metastable = group.attrs['metastable']
    #     atomic_weight_ratio = group.attrs['atomic_weight_ratio']
    #     kTg = group['kTs']
    #     kTs = []
    #     for temp in kTg:
    #         kTs.append(kTg[temp][()])

    #     data = cls(name, atomic_number, mass_number, metastable,
    #                atomic_weight_ratio, kTs)

    #     # Read energy grid
    #     e_group = group['energy']
    #     for temperature, dset in e_group.items():
    #         data.energy[temperature] = dset[()]

    #     # Read reaction data
    #     rxs_group = group['reactions']
    #     for name, obj in sorted(rxs_group.items()):
    #         if name.startswith('reaction_'):
    #             rx = Reaction.from_hdf5(obj, data.energy)
    #             data.reactions[rx.mt] = rx

    #             # Read total nu data if available
    #             if rx.mt in FISSION_MTS and 'total_nu' in group:
    #                 tgroup = group['total_nu']
    #                 rx.derived_products.append(Product.from_hdf5(tgroup))

    #     # Read unresolved resonance probability tables
    #     if 'urr' in group:
    #         urr_group = group['urr']
    #         for temperature, tgroup in urr_group.items():
    #             data.urr[temperature] = ProbabilityTables.from_hdf5(tgroup)

    #     # Read fission energy release data
    #     if 'fission_energy_release' in group:
    #         fer_group = group['fission_energy_release']
    #         data.fission_energy = FissionEnergyRelease.from_hdf5(fer_group)

    #     return data

    @classmethod
    def from_ace(cls, ace_or_filename, metastable_scheme='nndc'):
        """Generate incident neutron continuous-energy data from an ACE table

        Parameters
        ----------
        ace_or_filename : openmc.data.ace.Table or str
            ACE table to read from. If the value is a string, it is assumed to
            be the filename for the ACE file.
        metastable_scheme : {'nndc', 'mcnp'}
            Determine how ZAID identifiers are to be interpreted in the case of
            a metastable nuclide. Because the normal ZAID (=1000*Z + A) does not
            encode metastable information, different conventions are used among
            different libraries. In MCNP libraries, the convention is to add 400
            for a metastable nuclide except for Am242m, for which 95242 is
            metastable and 95642 (or 1095242 in newer libraries) is the ground
            state. For NNDC libraries, ZAID is given as 1000*Z + A + 100*m.

        Returns
        -------
        openmc.data.IncidentNeutron
            Incident neutron continuous-energy data

        """

        # First obtain the data for the first provided ACE table/file
        if isinstance(ace_or_filename, Table):
            ace = ace_or_filename
        else:
            ace = get_table(ace_or_filename)

        # If mass number hasn't been specified, make an educated guess
        zaid, xs = ace.name.split('.')
        if not xs.endswith('c'):
            raise TypeError(
                "{} is not a continuous-energy neutron ACE table.".format(ace))
        name, element, Z, mass_number, metastable = \
            get_metadata(int(zaid), metastable_scheme)

        # Assign temperature to the running list
        kTs = [ace.temperature*EV_PER_MEV]

        data = cls(name, Z, mass_number, metastable,
                   ace.atomic_weight_ratio, kTs)

        # Get string of temperature to use as a dictionary key
        strT = data.temperatures[0]

        # Read energy grid
        n_energy = ace.nxs[3]
        i = ace.jxs[1]
        energy = ace.xss[i : i + n_energy]*EV_PER_MEV
        data.energy[strT] = energy
        total_xs = ace.xss[i + n_energy : i + 2*n_energy]
        absorption_xs = ace.xss[i + 2*n_energy : i + 3*n_energy]
        heating_number = ace.xss[i + 4*n_energy : i + 5*n_energy]*EV_PER_MEV

        # Create redundant reaction for total (MT=1)
        total = Reaction(1)
        total.xs[strT] = Tabulated1D(energy, total_xs)
        total.redundant = True
        data.reactions[1] = total

        # Create redundant reaction for absorption (MT=101)
        if np.count_nonzero(absorption_xs) > 0:
            absorption = Reaction(101)
            absorption.xs[strT] = Tabulated1D(energy, absorption_xs)
            absorption.redundant = True
            data.reactions[101] = absorption

        # Create redundant reaction for heating (MT=301)
        heating = Reaction(301)
        heating.xs[strT] = Tabulated1D(energy, heating_number*total_xs)
        heating.redundant = True
        data.reactions[301] = heating

        # Read each reaction
        n_reaction = ace.nxs[4] + 1
        for i in range(n_reaction):
            rx = Reaction.from_ace(ace, i)
            data.reactions[rx.mt] = rx

        # Some photon production reactions may be assigned to MTs that don't
        # exist, usually MT=4. In this case, we create a new reaction and add
        # them
        n_photon_reactions = ace.nxs[6]
        photon_mts = ace.xss[ace.jxs[13]:ace.jxs[13] +
                             n_photon_reactions].astype(int)

        for mt in np.unique(photon_mts // 1000):
            if mt not in data:
                if mt not in SUM_RULES:
                    warn('Photon production is present for MT={} but no '
                         'cross section is given.'.format(mt))
                    continue

                # Create redundant reaction with appropriate cross section
                mts = data.get_reaction_components(mt)
                if len(mts) == 0:
                    warn('Photon production is present for MT={} but no '
                         'reaction components exist.'.format(mt))
                    continue

                # Determine redundant cross section
                rx = data._get_redundant_reaction(mt, mts)
                rx.products += _get_photon_products_ace(ace, rx)
                data.reactions[mt] = rx

        # For transmutation reactions, sometimes only individual levels are
        # present in an ACE file, e.g. MT=600-649 instead of the summation
        # MT=103. In this case, if a user wants to tally (n,p), OpenMC doesn't
        # know about the total cross section. Here, we explicitly create a
        # redundant reaction for this purpose.
        for mt in (16, 103, 104, 105, 106, 107):
            if mt not in data:
                # Determine if any individual levels are present
                mts = data.get_reaction_components(mt)
                if len(mts) == 0:
                    continue

                # Determine redundant cross section
                rx = data._get_redundant_reaction(mt, mts)
                data.reactions[mt] = rx

        # Make sure redundant cross sections that are present in an ACE file get
        # marked as such
        for rx in data:
            mts = data.get_reaction_components(rx.mt)
            if mts != [rx.mt]:
                rx.redundant = True
            if rx.mt in (203, 204, 205, 206, 207, 444):
                rx.redundant = True

        # Read unresolved resonance probability tables
        urr = ProbabilityTables.from_ace(ace)
        if urr is not None:
            data.urr[strT] = urr

        return data

    @classmethod
    def from_endf(cls, ev_or_filename, covariance=False):
        """Generate incident neutron continuous-energy data from an ENDF evaluation

        Parameters
        ----------
        ev_or_filename : openmc.data.endf.Evaluation or str
            ENDF evaluation to read from. If given as a string, it is assumed to
            be the filename for the ENDF file.

        covariance : bool
            Flag to indicate whether or not covariance data from File 32 should be
            retrieved

        Returns
        -------
        openmc.data.IncidentNeutron
            Incident neutron continuous-energy data

        """
        if isinstance(ev_or_filename, Evaluation):
            ev = ev_or_filename
        else:
            ev = Evaluation(ev_or_filename)

        atomic_number = ev.target['atomic_number']
        mass_number = ev.target['mass_number']
        metastable = ev.target['isomeric_state']
        atomic_weight_ratio = ev.target['mass']
        temperature = ev.target['temperature']

        # Determine name
        element = ATOMIC_SYMBOL[atomic_number]
        if metastable > 0:
            name = '{}{}_m{}'.format(element, mass_number, metastable)
        else:
            name = '{}{}'.format(element, mass_number)

        # Instantiate incident neutron data
        data = cls(name, atomic_number, mass_number, metastable,
                   atomic_weight_ratio, [temperature])

        if (2, 151) in ev.section:
            data.resonances = res.Resonances.from_endf(ev)

        if (32, 151) in ev.section and covariance:
            data.resonance_covariance = (
                res_cov.ResonanceCovariances.from_endf(ev, data.resonances)
            )

        # Read each reaction
        for mf, mt, nc, mod in ev.reaction_list:
            if mf == 3:
                data.reactions[mt] = Reaction.from_endf(ev, mt)

        # Replace cross sections for elastic, capture, fission
        try:
            if any(isinstance(r, res._RESOLVED) for r in data.resonances):
                for mt in (2, 102, 18):
                    if mt in data.reactions:
                        rx = data.reactions[mt]
                        rx.xs['0K'] = ResonancesWithBackground(
                            data.resonances, rx.xs['0K'], mt)
        except ValueError:
            # Thrown if multiple resolved ranges (e.g. Pu239 in ENDF/B-VII.1)
            pass

        # If first-chance, second-chance, etc. fission are present, check
        # whether energy distributions were specified in MF=5. If not, copy the
        # energy distribution from MT=18.
        for mt, rx in data.reactions.items():
            if mt in (19, 20, 21, 38):
                if (5, mt) not in ev.section:
                    if rx.products:
                        neutron = data.reactions[18].products[0]
                        rx.products[0].applicability = neutron.applicability
                        rx.products[0].distribution = neutron.distribution

        # Read fission energy release (requires that we already know nu for
        # fission)
        if (1, 458) in ev.section:
            data.fission_energy = FissionEnergyRelease.from_endf(ev, data)

        data._evaluation = ev
        return data

    @classmethod
    def from_njoy(cls, filename, temperatures=None, evaluation=None, **kwargs):
        """Generate incident neutron data by running NJOY.

        Parameters
        ----------
        filename : str
            Path to ENDF file
        temperatures : iterable of float
            Temperatures in Kelvin to produce data at. If omitted, data is
            produced at room temperature (293.6 K)
        evaluation : openmc.data.endf.Evaluation, optional
            If the ENDF file contains multiple material evaluations, this
            argument indicates which evaluation to use.
        **kwargs
            Keyword arguments passed to :func:`openmc.data.njoy.make_ace`

        Returns
        -------
        data : openmc.data.IncidentNeutron
            Incident neutron continuous-energy data

        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Run NJOY to create an ACE library
            kwargs.setdefault("output_dir", tmpdir)
            for key in ("acer", "pendf", "heatr", "broadr", "gaspr", "purr"):
                kwargs.setdefault(key, os.path.join(kwargs["output_dir"], key))
            kwargs['evaluation'] = evaluation
            make_ace(filename, temperatures, **kwargs)

            # Create instance from ACE tables within library
            lib = Library(kwargs['acer'])
            data = cls.from_ace(lib.tables[0])
            for table in lib.tables[1:]:
                data.add_temperature_from_ace(table)

            # Add 0K elastic scattering cross section
            if '0K' not in data.energy:
                pendf = Evaluation(kwargs['pendf'])
                file_obj = StringIO(pendf.section[3, 2])
                get_head_record(file_obj)
                params, xs = get_tab1_record(file_obj)
                data.energy['0K'] = xs.x
                data[2].xs['0K'] = xs

            # Add fission energy release data
            ev = evaluation if evaluation is not None else Evaluation(filename)
            if (1, 458) in ev.section:
                data.fission_energy = f = FissionEnergyRelease.from_endf(ev, data)
            else:
                f = None

            # For energy deposition, we want to store two different KERMAs:
            # one calculated assuming outgoing photons deposit their energy
            # locally, and one calculated assuming they carry their energy
            # away. This requires two HEATR runs (which make_ace does by
            # default). Here, we just need to correct for the fact that NJOY
            # uses a fission heating number of h = EFR, whereas we want:
            #
            # 1) h = EFR + EGP + EGD + EB (for local case)
            # 2) h = EFR + EB (for non-local case)
            #
            # The best way to handle this is to subtract off the fission
            # KERMA that NJOY calculates and add back exactly what we want.

            # If NJOY is not run with HEATR at all, skip everything below
            if not kwargs["heatr"]:
                return data

            # Helper function to get a cross section from an ENDF file on a
            # given energy grid
            def get_file3_xs(ev, mt, E):
                file_obj = StringIO(ev.section[3, mt])
                get_head_record(file_obj)
                _, xs = get_tab1_record(file_obj)
                return xs(E)

            heating_local = Reaction(901)
            heating_local.redundant = True

            heatr_evals = get_evaluations(kwargs["heatr"])
            heatr_local_evals = get_evaluations(kwargs["heatr"] + "_local")

            for ev, ev_local, temp in zip(heatr_evals, heatr_local_evals, data.temperatures):
                # Get total KERMA (originally from ACE file) and energy grid
                kerma = data.reactions[301].xs[temp]
                E = kerma.x

                if f is not None:
                    # Replace fission KERMA with (EFR + EB)*sigma_f
                    fission = data[18].xs[temp]
                    kerma_fission = get_file3_xs(ev, 318, E)
                    kerma.y = kerma.y - kerma_fission + (
                        f.fragments(E) + f.betas(E)) * fission(E)

                # For local KERMA, we first need to get the values from the
                # HEATR run with photon energy deposited locally and put
                # them on the same energy grid
                kerma_local = get_file3_xs(ev_local, 301, E)

                if f is not None:
                    # When photons deposit their energy locally, we replace the
                    # fission KERMA with (EFR + EGP + EGD + EB)*sigma_f
                    kerma_fission_local = get_file3_xs(ev_local, 318, E)
                    kerma_local = kerma_local - kerma_fission_local + (
                        f.fragments(E) + f.prompt_photons(E)
                        + f.delayed_photons(E) + f.betas(E))*fission(E)

                heating_local.xs[temp] = Tabulated1D(E, kerma_local)

            data.reactions[901] = heating_local

        return data

    def _get_redundant_reaction(self, mt, mts):
        """Create redundant reaction from its components

        Parameters
        ----------
        mt : int
            MT value of the desired reaction
        mts : iterable of int
            MT values of its components

        Returns
        -------
        openmc.Reaction
            Redundant reaction

        """

        rx = Reaction(mt)
        # Get energy grid
        for strT in self.temperatures:
            energy = self.energy[strT]
            xss = [self.reactions[mt_i].xs[strT] for mt_i in mts]
            idx = min([xs._threshold_idx if hasattr(xs, '_threshold_idx')
                       else 0 for xs in xss])
            rx.xs[strT] = Tabulated1D(energy[idx:], Sum(xss)(energy[idx:]))
            rx.xs[strT]._threshold_idx = idx

        rx.redundant = True

        return rx


class NBodyPhaseSpace(AngleEnergy):
    """N-body phase space distribution

    Parameters
    ----------
    total_mass : float
        Total mass of product particles
    n_particles : int
        Number of product particles
    atomic_weight_ratio : float
        Atomic weight ratio of target nuclide
    q_value : float
        Q value for reaction in eV

    Attributes
    ----------
    total_mass : float
        Total mass of product particles
    n_particles : int
        Number of product particles
    atomic_weight_ratio : float
        Atomic weight ratio of target nuclide
    q_value : float
        Q value for reaction in eV

    """

    def __init__(self, total_mass, n_particles, atomic_weight_ratio, q_value):
        self.total_mass = total_mass
        self.n_particles = n_particles
        self.atomic_weight_ratio = atomic_weight_ratio
        self.q_value = q_value

    @property
    def total_mass(self):
        return self._total_mass

    @total_mass.setter
    def total_mass(self, total_mass):
        name = 'N-body phase space total mass'
        check_type(name, total_mass, Real)
        check_greater_than(name, total_mass, 0.)
        self._total_mass = total_mass

    @property
    def n_particles(self):
        return self._n_particles

    @n_particles.setter
    def n_particles(self, n_particles):
        name = 'N-body phase space number of particles'
        check_type(name, n_particles, Integral)
        check_greater_than(name, n_particles, 0)
        self._n_particles = n_particles

    @property
    def atomic_weight_ratio(self):
        return self._atomic_weight_ratio

    @atomic_weight_ratio.setter
    def atomic_weight_ratio(self, atomic_weight_ratio):
        name = 'N-body phase space atomic weight ratio'
        check_type(name, atomic_weight_ratio, Real)
        check_greater_than(name, atomic_weight_ratio, 0.0)
        self._atomic_weight_ratio = atomic_weight_ratio

    @property
    def q_value(self):
        return self._q_value

    @q_value.setter
    def q_value(self, q_value):
        name = 'N-body phase space Q value'
        check_type(name, q_value, Real)
        self._q_value = q_value

    def to_hdf5(self, group):
        """Write distribution to an HDF5 group

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to write to

        """
        group.attrs['type'] = np.string_('nbody')
        group.attrs['total_mass'] = self.total_mass
        group.attrs['n_particles'] = self.n_particles
        group.attrs['atomic_weight_ratio'] = self.atomic_weight_ratio
        group.attrs['q_value'] = self.q_value

    @classmethod
    def from_hdf5(cls, group):
        """Generate N-body phase space distribution from HDF5 data

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to read from

        Returns
        -------
        openmc.data.NBodyPhaseSpace
            N-body phase space distribution

        """
        total_mass = group.attrs['total_mass']
        n_particles = group.attrs['n_particles']
        awr = group.attrs['atomic_weight_ratio']
        q_value = group.attrs['q_value']
        return cls(total_mass, n_particles, awr, q_value)

    @classmethod
    def from_ace(cls, ace, idx, q_value):
        """Generate N-body phase space distribution from ACE data

        Parameters
        ----------
        ace : openmc.data.ace.Table
            ACE table to read from
        idx : int
            Index in XSS array of the start of the energy distribution data
            (LDIS + LOCC - 1)
        q_value : float
            Q-value for reaction in eV

        Returns
        -------
        openmc.data.NBodyPhaseSpace
            N-body phase space distribution

        """
        n_particles = int(ace.xss[idx])
        total_mass = ace.xss[idx + 1]
        return cls(total_mass, n_particles, ace.atomic_weight_ratio, q_value)

    @classmethod
    def from_endf(cls, file_obj):
        """Generate N-body phase space distribution from an ENDF evaluation

        Parameters
        ----------
        file_obj : file-like object
            ENDF file positions at the start of the N-body phase space
            distribution

        Returns
        -------
        openmc.data.NBodyPhaseSpace
            N-body phase space distribution

        """
        items = get_cont_record(file_obj)
        total_mass = items[0]
        n_particles = items[5]
        # TODO: get awr and Q value
        return cls(total_mass, n_particles, 1.0, 0.0)


class IDWarning(UserWarning):
    pass


class IDManagerMixin:
    """A Class which automatically manages unique IDs.

    This mixin gives any subclass the ability to assign unique IDs through an
    'id' property and keeps track of which ones have already been
    assigned. Crucially, each subclass must define class variables 'next_id' and
    'used_ids' as they are used in the 'id' property that is supplied here.

    """

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, uid):
        # The first time this is called for a class, we search through the MRO
        # to determine which class actually holds next_id and used_ids. Since
        # next_id is an integer (immutable), we can't modify it directly through
        # the instance without just creating a new attribute
        try:
            cls = self._id_class
        except AttributeError:
            for cls in self.__class__.__mro__:
                if 'next_id' in cls.__dict__:
                    break

        if uid is None:
            while cls.next_id in cls.used_ids:
                cls.next_id += 1
            self._id = cls.next_id
            cls.used_ids.add(cls.next_id)
        else:
            name = cls.__name__
            check_type(f'{name} ID', uid, Integral)
            check_greater_than(f'{name} ID', uid, 0, equality=True)
            if uid in cls.used_ids:
                msg = f'Another {name} instance already exists with id={uid}.'
                warn(msg, IDWarning)
            else:
                cls.used_ids.add(uid)
            self._id = uid


def reset_auto_ids():
    """Reset counters for all auto-generated IDs"""
    for cls in IDManagerMixin.__subclasses__():
        cls.used_ids.clear()
        cls.next_id = 1


def reserve_ids(ids, cls=None):
    """Reserve a set of IDs that won't be used for auto-generated IDs.

    Parameters
    ----------
    ids : iterable of int
        IDs to reserve
    cls : type or None
        Class for which IDs should be reserved (e.g., :class:`openmc.Cell`). If
        None, all classes that have auto-generated IDs will be used.

    """
    if cls is None:
        for cls in IDManagerMixin.__subclasses__():
            cls.used_ids |= set(ids)
    else:
        cls.used_ids |= set(ids)


def set_auto_id(next_id):
    """Set the next ID for auto-generated IDs.

    Parameters
    ----------
    next_id : int
        The next ID to assign to objects with auto-generated IDs.

    """
    for cls in IDManagerMixin.__subclasses__():
        cls.next_id = next_id

class LaboratoryAngleEnergy(AngleEnergy):
    """Laboratory angle-energy distribution

    Parameters
    ----------
    breakpoints : Iterable of int
        Breakpoints defining interpolation regions
    interpolation : Iterable of int
        Interpolation codes
    energy : Iterable of float
        Incoming energies at which distributions exist
    mu : Iterable of openmc.stats.Univariate
        Distribution of scattering cosines for each incoming energy
    energy_out : Iterable of Iterable of openmc.stats.Univariate
        Distribution of outgoing energies for each incoming energy/scattering
        cosine

    Attributes
    ----------
    breakpoints : Iterable of int
        Breakpoints defining interpolation regions
    interpolation : Iterable of int
        Interpolation codes
    energy : Iterable of float
        Incoming energies at which distributions exist
    mu : Iterable of openmc.stats.Univariate
        Distribution of scattering cosines for each incoming energy
    energy_out : Iterable of Iterable of openmc.stats.Univariate
        Distribution of outgoing energies for each incoming energy/scattering
        cosine

    """

    def __init__(self, breakpoints, interpolation, energy, mu, energy_out):
        super().__init__()
        self.breakpoints = breakpoints
        self.interpolation = interpolation
        self.energy = energy
        self.mu = mu
        self.energy_out = energy_out

    @property
    def breakpoints(self):
        return self._breakpoints

    @breakpoints.setter
    def breakpoints(self, breakpoints):
        check_type('laboratory angle-energy breakpoints', breakpoints,
                      Iterable, Integral)
        self._breakpoints = breakpoints

    @property
    def interpolation(self):
        return self._interpolation

    @interpolation.setter
    def interpolation(self, interpolation):
        check_type('laboratory angle-energy interpolation', interpolation,
                      Iterable, Integral)
        self._interpolation = interpolation

    @property
    def energy(self):
        return self._energy

    @energy.setter
    def energy(self, energy):
        check_type('laboratory angle-energy incoming energy', energy,
                      Iterable, Real)
        self._energy = energy

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, mu):
        check_type('laboratory angle-energy outgoing cosine', mu,
                      Iterable, Univariate)
        self._mu = mu

    @property
    def energy_out(self):
        return self._energy_out

    @energy_out.setter
    def energy_out(self, energy_out):
        check_iterable_type('laboratory angle-energy outgoing energy',
                               energy_out, Univariate, 2, 2)
        self._energy_out = energy_out

    @classmethod
    def from_endf(cls, file_obj):
        """Generate laboratory angle-energy distribution from an ENDF evaluation

        Parameters
        ----------
        file_obj : file-like object
            ENDF file positioned at the start of a section for a correlated
            angle-energy distribution

        Returns
        -------
        openmc.data.LaboratoryAngleEnergy
            Laboratory angle-energy distribution

        """
        params, tab2 = get_tab2_record(file_obj)
        ne = params[5]
        energy = np.zeros(ne)
        mu = []
        energy_out = []
        for i in range(ne):
            params, _ = get_tab2_record(file_obj)
            energy[i] = params[1]
            n_mu = params[5]
            mu_i = np.zeros(n_mu)
            p_mu_i = np.zeros(n_mu)
            energy_out_i = []
            for j in range(n_mu):
                params, f = get_tab1_record(file_obj)
                mu_i[j] = params[1]
                p_mu_i[j] = sum(f.y)
                energy_out_i.append(Tabular(f.x, f.y))
            mu.append(Tabular(mu_i, p_mu_i))
            energy_out.append(energy_out_i)

        return cls(tab2.breakpoints, tab2.interpolation, energy, mu, energy_out)

    def to_hdf5(self, group):
        raise NotImplementedError


class _AtomicRepresentation(EqualityMixin):
    """Atomic representation of an isotope or a particle.

    Parameters
    ----------
    z : int
        Number of protons (atomic number)
    a : int
        Number of nucleons (mass number)

    Raises
    ------
    ValueError
        When the number of protons (z) declared is higher than the number
        of nucleons (a)

    Attributes
    ----------
    z : int
        Number of protons (atomic number)
    a : int
        Number of nucleons (mass number)
    n : int
        Number of neutrons
    za : int
        ZA identifier, 1000*Z + A, where Z is the atomic number and A the mass
        number

    """
    def __init__(self, z, a):
        # Sanity checks on values
        check_type('z', z, Integral)
        check_greater_than('z', z, 0, equality=True)
        check_type('a', a, Integral)
        check_greater_than('a', a, 0, equality=True)
        if z > a:
            raise ValueError(f"Number of protons ({z}) must be less than or "
                             f"equal to number of nucleons ({a}).")

        self._z = z
        self._a = a

    def __add__(self, other):
        """Add two _AtomicRepresentations"""
        z = self.z + other.z
        a = self.a + other.a
        return _AtomicRepresentation(z=z, a=a)

    def __sub__(self, other):
        """Substract two _AtomicRepresentations"""
        z = self.z - other.z
        a = self.a - other.a
        return _AtomicRepresentation(z=z, a=a)

    @property
    def a(self):
        return self._a

    @property
    def z(self):
        return self._z

    @property
    def n(self):
        return self.a - self.z

    @property
    def za(self):
        return self.z * 1000 + self.a

    @classmethod
    def from_za(cls, za):
        """Instantiate an _AtomicRepresentation from a ZA identifier.

        Parameters
        ----------
        za : int
            ZA identifier, 1000*Z + A, where Z is the atomic number and A the
            mass number

        Returns
        -------
        _AtomicRepresentation
            Atomic representation of the isotope/particle

        """
        z, a = divmod(za, 1000)
        return cls(z, a)


def _separation_energy(compound, nucleus, particle):
    """Calculates the separation energy as defined in ENDF-6 manual
    BNL-203218-2018-INRE, Revision 215, File 6 description for LAW=1
    and LANG=2. This function can be used for the incident or emitted
    particle of the following reaction: A + a -> C -> B + b

    Parameters
    ----------
    compound : _AtomicRepresentation
        Atomic representation of the compound (C)
    nucleus : _AtomicRepresentation
        Atomic representation of the nucleus (A or B)
    particle : _AtomicRepresentation
        Atomic representation of the particle (a or b)

    Returns
    -------
    separation_energy : float
        Separation energy in MeV

    """
    # Determine A, Z, and N for compound and nucleus
    A_c = compound.a
    Z_c = compound.z
    N_c = compound.n
    A_a = nucleus.a
    Z_a = nucleus.z
    N_a = nucleus.n

    # Determine breakup energy of incident particle (ENDF-6 Formats Manual,
    # Appendix H, Table 3) in MeV
    za_to_breaking_energy = {
        1: 0.0,
        1001: 0.0,
        1002: 2.224566,
        1003: 8.481798,
        2003: 7.718043,
        2004: 28.29566
    }
    I_a = za_to_breaking_energy[particle.za]

    # Eq. 4 in in doi:10.1103/PhysRevC.37.2350 or ENDF-6 Formats Manual section
    # 6.2.3.2
    return (
        15.68 * (A_c - A_a) -
        28.07 * ((N_c - Z_c)**2 / A_c - (N_a - Z_a)**2 / A_a) -
        18.56 * (A_c**(2./3.) - A_a**(2./3.)) +
        33.22 * ((N_c - Z_c)**2 / A_c**(4./3.) - (N_a - Z_a)**2 / A_a**(4./3.)) -
        0.717 * (Z_c**2 / A_c**(1./3.) - Z_a**2 / A_a**(1./3.)) +
        1.211 * (Z_c**2 / A_c - Z_a**2 / A_a) -
        I_a
    )


def kalbach_slope(energy_projectile, energy_emitted, za_projectile,
                  za_emitted, za_target):
    """Returns Kalbach-Mann slope from calculations.

    The associated reaction is defined as:
    A + a -> C -> B + b

    Where:

    - A is the targeted nucleus,
    - a is the projectile,
    - C is the compound,
    - B is the residual nucleus,
    - b is the emitted particle.

    The Kalbach-Mann slope calculation is done as defined in ENDF-6 manual
    BNL-203218-2018-INRE, Revision 215, File 6 description for LAW=1 and
    LANG=2. One exception to this, is that the entrance and emission channel
    energies are not calculated with the AWR number, but approximated with
    the number of mass instead.

    .. versionadded:: 0.13.1

    Parameters
    ----------
    energy_projectile : float
        Energy of the projectile in the laboratory system in eV
    energy_emitted : float
        Energy of the emitted particle in the center of mass system in eV
    za_projectile : int
        ZA identifier of the projectile
    za_emitted : int
        ZA identifier of the emitted particle
    za_target : int
        ZA identifier of the targeted nucleus

    Raises
    ------
    NotImplementedError
        When the projectile is not a neutron

    Returns
    -------
    slope : float
        Kalbach-Mann slope given with the same format as ACE file.

    """
    # TODO: develop for photons as projectile
    # TODO: test for other particles than neutron
    if za_projectile != 1:
        raise NotImplementedError(
            "Developed and tested for neutron projectile only."
        )

    # Special handling of elemental carbon
    if za_emitted == 6000:
        za_emitted = 6012
    if za_target == 6000:
        za_target = 6012

    projectile = _AtomicRepresentation.from_za(za_projectile)
    emitted = _AtomicRepresentation.from_za(za_emitted)
    target = _AtomicRepresentation.from_za(za_target)
    compound = projectile + target
    residual = compound - emitted

    # Calculate entrance and emission channel energy in MeV, defined in section
    # 6.2.3.2 in the ENDF-6 Formats Manual
    epsilon_a = energy_projectile * target.a / (target.a + projectile.a) / EV_PER_MEV
    epsilon_b = energy_emitted * (residual.a + emitted.a) \
        / (residual.a * EV_PER_MEV)

    # Calculate separation energies using Eq. 4 in doi:10.1103/PhysRevC.37.2350
    # or ENDF-6 Formats Manual section 6.2.3.2
    s_a = _separation_energy(compound, target, projectile)
    s_b = _separation_energy(compound, residual, emitted)

    # See Eq. 10 in doi:10.1103/PhysRevC.37.2350 or section 6.2.3.2 in the
    # ENDF-6 Formats Manual
    za_to_M = {1: 1.0, 1001: 1.0, 1002: 1.0, 2004: 0.0}
    za_to_m = {1: 0.5, 1001: 1.0, 1002: 1.0, 1003: 1.0, 2003: 1.0, 2004: 2.0}
    M = za_to_M[projectile.za]
    m = za_to_m[emitted.za]
    e_a = epsilon_a + s_a
    e_b = epsilon_b + s_b
    r_1 = min(e_a, 130.)
    r_3 = min(e_a, 41.)
    x_1 = r_1 * e_b / e_a
    x_3 = r_3 * e_b / e_a
    return 0.04 * x_1 + 1.8e-6 * x_1**3 + 6.7e-7 * M * m * x_3**4


class KalbachMann(AngleEnergy):
    """Kalbach-Mann distribution

    Parameters
    ----------
    breakpoints : Iterable of int
        Breakpoints defining interpolation regions
    interpolation : Iterable of int
        Interpolation codes
    energy : Iterable of float
        Incoming energies at which distributions exist
    energy_out : Iterable of openmc.stats.Univariate
        Distribution of outgoing energies corresponding to each incoming energy
    precompound : Iterable of openmc.data.Tabulated1D
        Precompound factor 'r' as a function of outgoing energy for each
        incoming energy
    slope : Iterable of openmc.data.Tabulated1D
        Kalbach-Chadwick angular distribution slope value 'a' as a function of
        outgoing energy for each incoming energy

    Attributes
    ----------
    breakpoints : Iterable of int
        Breakpoints defining interpolation regions
    interpolation : Iterable of int
        Interpolation codes
    energy : Iterable of float
        Incoming energies at which distributions exist
    energy_out : Iterable of openmc.stats.Univariate
        Distribution of outgoing energies corresponding to each incoming energy
    precompound : Iterable of openmc.data.Tabulated1D
        Precompound factor 'r' as a function of outgoing energy for each
        incoming energy
    slope : Iterable of openmc.data.Tabulated1D
        Kalbach-Chadwick angular distribution slope value 'a' as a function of
        outgoing energy for each incoming energy

    """

    def __init__(self, breakpoints, interpolation, energy, energy_out,
                 precompound, slope):
        super().__init__()
        self.breakpoints = breakpoints
        self.interpolation = interpolation
        self.energy = energy
        self.energy_out = energy_out
        self.precompound = precompound
        self.slope = slope

    @property
    def breakpoints(self):
        return self._breakpoints

    @breakpoints.setter
    def breakpoints(self, breakpoints):
        check_type('Kalbach-Mann breakpoints', breakpoints,
                      Iterable, Integral)
        self._breakpoints = breakpoints

    @property
    def interpolation(self):
        return self._interpolation

    @interpolation.setter
    def interpolation(self, interpolation):
        check_type('Kalbach-Mann interpolation', interpolation,
                      Iterable, Integral)
        self._interpolation = interpolation

    @property
    def energy(self):
        return self._energy

    @energy.setter
    def energy(self, energy):
        check_type('Kalbach-Mann incoming energy', energy,
                      Iterable, Real)
        self._energy = energy

    @property
    def energy_out(self):
        return self._energy_out

    @energy_out.setter
    def energy_out(self, energy_out):
        check_type('Kalbach-Mann distributions', energy_out,
                      Iterable, Univariate)
        self._energy_out = energy_out

    @property
    def precompound(self):
        return self._precompound

    @precompound.setter
    def precompound(self, precompound):
        check_type('Kalbach-Mann precompound factor', precompound,
                      Iterable, Tabulated1D)
        self._precompound = precompound

    @property
    def slope(self):
        return self._slope

    @slope.setter
    def slope(self, slope):
        check_type('Kalbach-Mann slope', slope, Iterable, Tabulated1D)
        self._slope = slope

    def to_hdf5(self, group):
        """Write distribution to an HDF5 group

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to write to

        """
        group.attrs['type'] = np.string_('kalbach-mann')

        dset = group.create_dataset('energy', data=self.energy)
        dset.attrs['interpolation'] = np.vstack((self.breakpoints,
                                                 self.interpolation))

        # Determine total number of (E,p,r,a) tuples and create array
        n_tuple = sum(len(d) for d in self.energy_out)
        distribution = np.empty((5, n_tuple))

        # Create array for offsets
        offsets = np.empty(len(self.energy_out), dtype=int)
        interpolation = np.empty(len(self.energy_out), dtype=int)
        n_discrete_lines = np.empty(len(self.energy_out), dtype=int)
        j = 0

        # Populate offsets and distribution array
        for i, (eout, km_r, km_a) in enumerate(zip(
                self.energy_out, self.precompound, self.slope)):
            n = len(eout)
            offsets[i] = j

            if isinstance(eout, Mixture):
                discrete, continuous = eout.distribution
                n_discrete_lines[i] = m = len(discrete)
                interpolation[i] = 1 if continuous.interpolation == 'histogram' else 2
                distribution[0, j:j+m] = discrete.x
                distribution[1, j:j+m] = discrete.p
                distribution[2, j:j+m] = discrete.c
                distribution[0, j+m:j+n] = continuous.x
                distribution[1, j+m:j+n] = continuous.p
                distribution[2, j+m:j+n] = continuous.c
            else:
                if isinstance(eout, Tabular):
                    n_discrete_lines[i] = 0
                    interpolation[i] = 1 if eout.interpolation == 'histogram' else 2
                elif isinstance(eout, Discrete):
                    n_discrete_lines[i] = n
                    interpolation[i] = 1
                distribution[0, j:j+n] = eout.x
                distribution[1, j:j+n] = eout.p
                distribution[2, j:j+n] = eout.c

            distribution[3, j:j+n] = km_r.y
            distribution[4, j:j+n] = km_a.y
            j += n

        # Create dataset for distributions
        dset = group.create_dataset('distribution', data=distribution)

        # Write interpolation as attribute
        dset.attrs['offsets'] = offsets
        dset.attrs['interpolation'] = interpolation
        dset.attrs['n_discrete_lines'] = n_discrete_lines

    @classmethod
    def from_hdf5(cls, group):
        """Generate Kalbach-Mann distribution from HDF5 data

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to read from

        Returns
        -------
        openmc.data.KalbachMann
            Kalbach-Mann energy distribution

        """
        interp_data = group['energy'].attrs['interpolation']
        energy_breakpoints = interp_data[0, :]
        energy_interpolation = interp_data[1, :]
        energy = group['energy'][()]

        data = group['distribution']
        offsets = data.attrs['offsets']
        interpolation = data.attrs['interpolation']
        n_discrete_lines = data.attrs['n_discrete_lines']

        energy_out = []
        precompound = []
        slope = []
        n_energy = len(energy)
        for i in range(n_energy):
            # Determine length of outgoing energy distribution and number of
            # discrete lines
            j = offsets[i]
            if i < n_energy - 1:
                n = offsets[i+1] - j
            else:
                n = data.shape[1] - j
            m = n_discrete_lines[i]

            # Create discrete distribution if lines are present
            if m > 0:
                eout_discrete = Discrete(data[0, j:j+m], data[1, j:j+m])
                eout_discrete.c = data[2, j:j+m]
                p_discrete = eout_discrete.c[-1]

            # Create continuous distribution
            if m < n:
                interp = INTERPOLATION_SCHEME[interpolation[i]]
                eout_continuous = Tabular(data[0, j+m:j+n], data[1, j+m:j+n], interp)
                eout_continuous.c = data[2, j+m:j+n]

            # If both continuous and discrete are present, create a mixture
            # distribution
            if m == 0:
                eout_i = eout_continuous
            elif m == n:
                eout_i = eout_discrete
            else:
                eout_i = Mixture([p_discrete, 1. - p_discrete],
                                 [eout_discrete, eout_continuous])

            # Precompound factor and slope are on rows 3 and 4, respectively
            km_r = Tabulated1D(data[0, j:j+n], data[3, j:j+n])
            km_a = Tabulated1D(data[0, j:j+n], data[4, j:j+n])

            energy_out.append(eout_i)
            precompound.append(km_r)
            slope.append(km_a)

        return cls(energy_breakpoints, energy_interpolation,
                   energy, energy_out, precompound, slope)

    @classmethod
    def from_ace(cls, ace, idx, ldis):
        """Generate Kalbach-Mann energy-angle distribution from ACE data

        Parameters
        ----------
        ace : openmc.data.ace.Table
            ACE table to read from
        idx : int
            Index in XSS array of the start of the energy distribution data
            (LDIS + LOCC - 1)
        ldis : int
            Index in XSS array of the start of the energy distribution block
            (e.g. JXS[11])

        Returns
        -------
        openmc.data.KalbachMann
            Kalbach-Mann energy-angle distribution

        """
        # Read number of interpolation regions and incoming energies
        n_regions = int(ace.xss[idx])
        n_energy_in = int(ace.xss[idx + 1 + 2*n_regions])

        # Get interpolation information
        idx += 1
        if n_regions > 0:
            breakpoints = ace.xss[idx:idx + n_regions].astype(int)
            interpolation = ace.xss[idx + n_regions:idx + 2*n_regions].astype(int)
        else:
            breakpoints = np.array([n_energy_in])
            interpolation = np.array([2])

        # Incoming energies at which distributions exist
        idx += 2*n_regions + 1
        energy = ace.xss[idx:idx + n_energy_in]*EV_PER_MEV

        # Location of distributions
        idx += n_energy_in
        loc_dist = ace.xss[idx:idx + n_energy_in].astype(int)

        # Initialize variables
        energy_out = []
        km_r = []
        km_a = []

        # Read each outgoing energy distribution
        for i in range(n_energy_in):
            idx = ldis + loc_dist[i] - 1

            # intt = interpolation scheme (1=hist, 2=lin-lin)
            INTTp = int(ace.xss[idx])
            intt = INTTp % 10
            n_discrete_lines = (INTTp - intt)//10
            if intt not in (1, 2):
                warn("Interpolation scheme for continuous tabular distribution "
                     "is not histogram or linear-linear.")
                intt = 2

            n_energy_out = int(ace.xss[idx + 1])
            data = ace.xss[idx + 2:idx + 2 + 5*n_energy_out].copy()
            data.shape = (5, n_energy_out)
            data[0, :] *= EV_PER_MEV

            # Create continuous distribution
            eout_continuous = Tabular(data[0][n_discrete_lines:],
                                      data[1][n_discrete_lines:]/EV_PER_MEV,
                                      INTERPOLATION_SCHEME[intt],
                                      ignore_negative=True)
            eout_continuous.c = data[2][n_discrete_lines:]
            if np.any(data[1][n_discrete_lines:] < 0.0):
                warn("Kalbach-Mann energy distribution has negative "
                     "probabilities.")

            # If discrete lines are present, create a mixture distribution
            if n_discrete_lines > 0:
                eout_discrete = Discrete(data[0][:n_discrete_lines],
                                         data[1][:n_discrete_lines])
                eout_discrete.c = data[2][:n_discrete_lines]
                if n_discrete_lines == n_energy_out:
                    eout_i = eout_discrete
                else:
                    p_discrete = min(sum(eout_discrete.p), 1.0)
                    eout_i = Mixture([p_discrete, 1. - p_discrete],
                                     [eout_discrete, eout_continuous])
            else:
                eout_i = eout_continuous

            energy_out.append(eout_i)
            km_r.append(Tabulated1D(data[0], data[3]))
            km_a.append(Tabulated1D(data[0], data[4]))

        return cls(breakpoints, interpolation, energy, energy_out, km_r, km_a)

    @classmethod
    def from_endf(cls, file_obj, za_emitted, za_target, projectile_mass):
        """Generate Kalbach-Mann distribution from an ENDF evaluation.

        If the projectile is a neutron, the slope is calculated when it is
        not given explicitly.

        .. versionchanged:: 0.13.1
            Arguments changed to accommodate slope calculation

        Parameters
        ----------
        file_obj : file-like object
            ENDF file positioned at the start of the Kalbach-Mann distribution
        za_emitted : int
            ZA identifier of the emitted particle
        za_target : int
            ZA identifier of the target
        projectile_mass : float
            Mass of the projectile

        Warns
        -----
        UserWarning
            If the mass of the projectile is not equal to 1 (other than
            a neutron), the slope is not calculated and set to 0 if missing.

        Returns
        -------
        openmc.data.KalbachMann
            Kalbach-Mann energy-angle distribution

        """
        params, tab2 = get_tab2_record(file_obj)
        lep = params[3]
        ne = params[5]
        energy = np.zeros(ne)
        n_discrete_energies = np.zeros(ne, dtype=int)
        energy_out = []
        precompound = []
        slope = []
        calculated_slope = []
        for i in range(ne):
            items, values = get_list_record(file_obj)
            energy[i] = items[1]
            n_discrete_energies[i] = items[2]
            # TODO: split out discrete energies
            n_angle = items[3]
            n_energy_out = items[5]
            values = np.asarray(values)
            values.shape = (n_energy_out, n_angle + 2)

            # Outgoing energy distribution at the i-th incoming energy
            eout_i = values[:, 0]
            eout_p_i = values[:, 1]
            energy_out_i = Tabular(eout_i, eout_p_i, INTERPOLATION_SCHEME[lep])
            energy_out.append(energy_out_i)

            # Precompound factors for Kalbach-Mann
            r_i = values[:, 2]

            # Slope factors for Kalbach-Mann
            if n_angle == 2:
                a_i = values[:, 3]
                calculated_slope.append(False)
            else:
                # Check if the projectile is not a neutron
                if not np.isclose(projectile_mass, 1.0, atol=1.0e-12, rtol=0.):
                    warn(
                        "Kalbach-Mann slope calculation is only available with "
                        "neutrons as projectile. Slope coefficients are set to 0."
                    )
                    a_i = np.zeros_like(r_i)
                    calculated_slope.append(False)

                else:
                    # TODO: retrieve ZA of the projectile
                    za_projectile = 1
                    a_i = [kalbach_slope(energy_projectile=energy[i],
                                         energy_emitted=e,
                                         za_projectile=za_projectile,
                                         za_emitted=za_emitted,
                                         za_target=za_target)
                           for e in eout_i]
                    calculated_slope.append(True)

            precompound.append(Tabulated1D(eout_i, r_i))
            slope.append(Tabulated1D(eout_i, a_i))

        km_distribution = cls(tab2.breakpoints, tab2.interpolation, energy,
                              energy_out, precompound, slope)

        # List of bool to indicate slope calculation by OpenMC
        km_distribution._calculated_slope = calculated_slope

        return km_distribution


def linearize(x, f, tolerance=0.001):
    """Return a tabulated representation of a one-variable function

    Parameters
    ----------
    x : Iterable of float
        Initial x values at which the function should be evaluated
    f : Callable
        Function of a single variable
    tolerance : float
        Tolerance on the interpolation error

    Returns
    -------
    numpy.ndarray
        Tabulated values of the independent variable
    numpy.ndarray
        Tabulated values of the dependent variable

    """
    # Make sure x is a numpy array
    x = np.asarray(x)

    # Initialize output arrays
    x_out = []
    y_out = []

    # Initialize stack
    x_stack = [x[0]]
    y_stack = [f(x[0])]

    for i in range(x.shape[0] - 1):
        x_stack.insert(0, x[i + 1])
        y_stack.insert(0, f(x[i + 1]))

        while True:
            # Get the bounding points currently on the stack
            x_high, x_low = x_stack[-2:]
            y_high, y_low = y_stack[-2:]

            # Evaluate the function at the midpoint
            x_mid = 0.5*(x_low + x_high)
            y_mid = f(x_mid)

            # Linearly interpolate between the bounding points
            y_interp = y_low + (y_high - y_low)/(x_high - x_low)*(x_mid - x_low)

            # Check the error on the interpolated point and compare to tolerance
            error = abs((y_interp - y_mid)/y_mid)
            if error > tolerance:
                x_stack.insert(-1, x_mid)
                y_stack.insert(-1, y_mid)
            else:
                x_out.append(x_stack.pop())
                y_out.append(y_stack.pop())
                if len(x_stack) == 1:
                    break

    x_out.append(x_stack.pop())
    y_out.append(y_stack.pop())

    return np.array(x_out), np.array(y_out)

def thin(x, y, tolerance=0.001):
    """Check for (x,y) points that can be removed.

    Parameters
    ----------
    x : numpy.ndarray
        Independent variable
    y : numpy.ndarray
        Dependent variable
    tolerance : float
        Tolerance on interpolation error

    Returns
    -------
    numpy.ndarray
        Tabulated values of the independent variable
    numpy.ndarray
        Tabulated values of the dependent variable

    """
    # Initialize output arrays
    x_out = x.copy()
    y_out = y.copy()

    N = x.shape[0]
    i_left = 0
    i_right = 2

    while i_left < N - 2 and i_right < N:
        m = (y[i_right] - y[i_left])/(x[i_right] - x[i_left])

        for i in range(i_left + 1, i_right):
            # Determine error in interpolated point
            y_interp = y[i_left] + m*(x[i] - x[i_left])
            if abs(y[i]) > 0.:
                error = abs((y_interp - y[i])/y[i])
            else:
                error = 2*tolerance

            if error > tolerance:
                for i_remove in range(i_left + 1, i_right - 1):
                    x_out[i_remove] = np.nan
                    y_out[i_remove] = np.nan
                i_left = i_right - 1
                i_right = i_left + 1
                break

        i_right += 1

    for i_remove in range(i_left + 1, i_right - 1):
        x_out[i_remove] = np.nan
        y_out[i_remove] = np.nan

    return x_out[np.isfinite(x_out)], y_out[np.isfinite(y_out)]


INTERPOLATION_SCHEME = {1: 'histogram', 2: 'linear-linear', 3: 'linear-log',
                        4: 'log-linear', 5: 'log-log'}


def sum_functions(funcs):
    """Add tabulated/polynomials functions together

    Parameters
    ----------
    funcs : list of Function1D
        Functions to add

    Returns
    -------
    Function1D
        Sum of polynomial/tabulated functions

    """
    # Copy so we can iterate multiple times
    funcs = list(funcs)

    # Get x values for all tabulated components
    xs = []
    for f in funcs:
        if isinstance(f, Tabulated1D):
            xs.append(f.x)
            if not np.all(f.interpolation == 2):
                raise ValueError('Only linear-linear tabulated functions '
                                 'can be combined')

    if xs:
        # Take the union of all energies (sorted)
        x = reduce(np.union1d, xs)

        # Evaluate each function and add together
        y = sum(f(x) for f in funcs)
        return Tabulated1D(x, y)
    else:
        # If no tabulated functions are present, we need to combine the
        # polynomials by adding their coefficients
        coeffs = [sum(x) for x in zip_longest(*funcs, fillvalue=0.0)]
        return Polynomial(coeffs)


class Function1D(EqualityMixin, ABC):
    """A function of one independent variable with HDF5 support."""
    @abstractmethod
    def __call__(self): pass

    @abstractmethod
    def to_hdf5(self, group, name='xy'):
        """Write function to an HDF5 group

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to write to
        name : str
            Name of the dataset to create

        """
        pass

    @classmethod
    def from_hdf5(cls, dataset):
        """Generate function from an HDF5 dataset

        Parameters
        ----------
        dataset : h5py.Dataset
            Dataset to read from

        Returns
        -------
        openmc.data.Function1D
            Function read from dataset

        """
        for subclass in cls.__subclasses__():
            if dataset.attrs['type'].decode() == subclass.__name__:
                return subclass.from_hdf5(dataset)
        raise ValueError("Unrecognized Function1D class: '"
                         + dataset.attrs['type'].decode() + "'")


class Tabulated1D(Function1D):
    """A one-dimensional tabulated function.

    This class mirrors the TAB1 type from the ENDF-6 format. A tabulated
    function is specified by tabulated (x,y) pairs along with interpolation
    rules that determine the values between tabulated pairs.

    Once an object has been created, it can be used as though it were an actual
    function, e.g.:

    >>> f = Tabulated1D([0, 10], [4, 5])
    >>> [f(xi) for xi in numpy.linspace(0, 10, 5)]
    [4.0, 4.25, 4.5, 4.75, 5.0]

    Parameters
    ----------
    x : Iterable of float
        Independent variable
    y : Iterable of float
        Dependent variable
    breakpoints : Iterable of int
        Breakpoints for interpolation regions
    interpolation : Iterable of int
        Interpolation scheme identification number, e.g., 3 means y is linear in
        ln(x).

    Attributes
    ----------
    x : Iterable of float
        Independent variable
    y : Iterable of float
        Dependent variable
    breakpoints : Iterable of int
        Breakpoints for interpolation regions
    interpolation : Iterable of int
        Interpolation scheme identification number, e.g., 3 means y is linear in
        ln(x).
    n_regions : int
        Number of interpolation regions
    n_pairs : int
        Number of tabulated (x,y) pairs

    """

    def __init__(self, x, y, breakpoints=None, interpolation=None):
        if breakpoints is None or interpolation is None:
            # Single linear-linear interpolation region by default
            self.breakpoints = np.array([len(x)])
            self.interpolation = np.array([2])
        else:
            self.breakpoints = np.asarray(breakpoints, dtype=int)
            self.interpolation = np.asarray(interpolation, dtype=int)

        self.x = np.asarray(x)
        self.y = np.asarray(y)

    def __call__(self, x):
        # Check if input is scalar
        if not isinstance(x, Iterable):
            return self._interpolate_scalar(x)

        x = np.array(x)

        # Create output array
        y = np.zeros_like(x)

        # Get indices for interpolation
        idx = np.searchsorted(self.x, x, side='right') - 1

        # Loop over interpolation regions
        for k in range(len(self.breakpoints)):
            # Get indices for the begining and ending of this region
            i_begin = self.breakpoints[k-1] - 1 if k > 0 else 0
            i_end = self.breakpoints[k] - 1

            # Figure out which idx values lie within this region
            contained = (idx >= i_begin) & (idx < i_end)

            xk = x[contained]                 # x values in this region
            xi = self.x[idx[contained]]       # low edge of corresponding bins
            xi1 = self.x[idx[contained] + 1]  # high edge of corresponding bins
            yi = self.y[idx[contained]]
            yi1 = self.y[idx[contained] + 1]

            if self.interpolation[k] == 1:
                # Histogram
                y[contained] = yi

            elif self.interpolation[k] == 2:
                # Linear-linear
                y[contained] = yi + (xk - xi)/(xi1 - xi)*(yi1 - yi)

            elif self.interpolation[k] == 3:
                # Linear-log
                y[contained] = yi + np.log(xk/xi)/np.log(xi1/xi)*(yi1 - yi)

            elif self.interpolation[k] == 4:
                # Log-linear
                y[contained] = yi*np.exp((xk - xi)/(xi1 - xi)*np.log(yi1/yi))

            elif self.interpolation[k] == 5:
                # Log-log
                y[contained] = (yi*np.exp(np.log(xk/xi)/np.log(xi1/xi)
                                *np.log(yi1/yi)))

        # In some cases, x values might be outside the tabulated region due only
        # to precision, so we check if they're close and set them equal if so.
        y[np.isclose(x, self.x[0], atol=1e-14)] = self.y[0]
        y[np.isclose(x, self.x[-1], atol=1e-14)] = self.y[-1]

        return y

    def _interpolate_scalar(self, x):
        if x <= self._x[0]:
            return self._y[0]
        elif x >= self._x[-1]:
            return self._y[-1]

        # Get the index for interpolation
        idx = np.searchsorted(self._x, x, side='right') - 1

        # Loop over interpolation regions
        for b, p in zip(self.breakpoints, self.interpolation):
            if idx < b - 1:
                break

        xi = self._x[idx]       # low edge of the corresponding bin
        xi1 = self._x[idx + 1]  # high edge of the corresponding bin
        yi = self._y[idx]
        yi1 = self._y[idx + 1]

        if p == 1:
            # Histogram
            return yi

        elif p == 2:
            # Linear-linear
            return yi + (x - xi)/(xi1 - xi)*(yi1 - yi)

        elif p == 3:
            # Linear-log
            return yi + log(x/xi)/log(xi1/xi)*(yi1 - yi)

        elif p == 4:
            # Log-linear
            return yi*exp((x - xi)/(xi1 - xi)*log(yi1/yi))

        elif p == 5:
            # Log-log
            return yi*exp(log(x/xi)/log(xi1/xi)*log(yi1/yi))

    def __len__(self):
        return len(self.x)

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x):
        check_type('x values', x, Iterable, Real)
        self._x = x

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, y):
        check_type('y values', y, Iterable, Real)
        self._y = y

    @property
    def breakpoints(self):
        return self._breakpoints

    @breakpoints.setter
    def breakpoints(self, breakpoints):
        check_type('breakpoints', breakpoints, Iterable, Integral)
        self._breakpoints = breakpoints

    @property
    def interpolation(self):
        return self._interpolation

    @interpolation.setter
    def interpolation(self, interpolation):
        check_type('interpolation', interpolation, Iterable, Integral)
        self._interpolation = interpolation

    @property
    def n_pairs(self):
        return len(self.x)

    @property
    def n_regions(self):
        return len(self.breakpoints)

    def integral(self):
        """Integral of the tabulated function over its tabulated range.

        Returns
        -------
        numpy.ndarray
            Array of same length as the tabulated data that represents partial
            integrals from the bottom of the range to each tabulated point.

        """

        # Create output array
        partial_sum = np.zeros(len(self.x) - 1)

        i_low = 0
        for k in range(len(self.breakpoints)):
            # Determine which x values are within this interpolation range
            i_high = self.breakpoints[k] - 1

            # Get x values and bounding (x,y) pairs
            x0 = self.x[i_low:i_high]
            x1 = self.x[i_low + 1:i_high + 1]
            y0 = self.y[i_low:i_high]
            y1 = self.y[i_low + 1:i_high + 1]

            if self.interpolation[k] == 1:
                # Histogram
                partial_sum[i_low:i_high] = y0*(x1 - x0)

            elif self.interpolation[k] == 2:
                # Linear-linear
                m = (y1 - y0)/(x1 - x0)
                partial_sum[i_low:i_high] = (y0 - m*x0)*(x1 - x0) + \
                                            m*(x1**2 - x0**2)/2

            elif self.interpolation[k] == 3:
                # Linear-log
                logx = np.log(x1/x0)
                m = (y1 - y0)/logx
                partial_sum[i_low:i_high] = y0 + m*(x1*(logx - 1) + x0)

            elif self.interpolation[k] == 4:
                # Log-linear
                m = np.log(y1/y0)/(x1 - x0)
                partial_sum[i_low:i_high] = y0/m*(np.exp(m*(x1 - x0)) - 1)

            elif self.interpolation[k] == 5:
                # Log-log
                m = np.log(y1/y0)/np.log(x1/x0)
                partial_sum[i_low:i_high] = y0/((m + 1)*x0**m)*(
                    x1**(m + 1) - x0**(m + 1))

            i_low = i_high

        return np.concatenate(([0.], np.cumsum(partial_sum)))

    def to_hdf5(self, group, name='xy'):
        """Write tabulated function to an HDF5 group

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to write to
        name : str
            Name of the dataset to create

        """
        dataset = group.create_dataset(name, data=np.vstack(
            [self.x, self.y]))
        dataset.attrs['type'] = np.string_(type(self).__name__)
        dataset.attrs['breakpoints'] = self.breakpoints
        dataset.attrs['interpolation'] = self.interpolation

    @classmethod
    def from_hdf5(cls, dataset):
        """Generate tabulated function from an HDF5 dataset

        Parameters
        ----------
        dataset : h5py.Dataset
            Dataset to read from

        Returns
        -------
        openmc.data.Tabulated1D
            Function read from dataset

        """
        if dataset.attrs['type'].decode() != cls.__name__:
            raise ValueError("Expected an HDF5 attribute 'type' equal to '"
                             + cls.__name__ + "'")

        x = dataset[0, :]
        y = dataset[1, :]
        breakpoints = dataset.attrs['breakpoints']
        interpolation = dataset.attrs['interpolation']
        return cls(x, y, breakpoints, interpolation)

    @classmethod
    def from_ace(cls, ace, idx=0, convert_units=True):
        """Create a Tabulated1D object from an ACE table.

        Parameters
        ----------
        ace : openmc.data.ace.Table
            An ACE table
        idx : int
            Offset to read from in XSS array (default of zero)
        convert_units : bool
            If the abscissa represents energy, indicate whether to convert MeV
            to eV.

        Returns
        -------
        openmc.data.Tabulated1D
            Tabulated data object

        """

        # Get number of regions and pairs
        n_regions = int(ace.xss[idx])
        n_pairs = int(ace.xss[idx + 1 + 2*n_regions])

        # Get interpolation information
        idx += 1
        if n_regions > 0:
            breakpoints = ace.xss[idx:idx + n_regions].astype(int)
            interpolation = ace.xss[idx + n_regions:idx + 2*n_regions].astype(int)
        else:
            # 0 regions implies linear-linear interpolation by default
            breakpoints = np.array([n_pairs])
            interpolation = np.array([2])

        # Get (x,y) pairs
        idx += 2*n_regions + 1
        x = ace.xss[idx:idx + n_pairs].copy()
        y = ace.xss[idx + n_pairs:idx + 2*n_pairs].copy()

        if convert_units:
            x *= EV_PER_MEV

        return Tabulated1D(x, y, breakpoints, interpolation)


class Polynomial(np.polynomial.Polynomial, Function1D):
    """A power series class.

    Parameters
    ----------
    coef : Iterable of float
        Polynomial coefficients in order of increasing degree

    """
    def to_hdf5(self, group, name='xy'):
        """Write polynomial function to an HDF5 group

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to write to
        name : str
            Name of the dataset to create

        """
        dataset = group.create_dataset(name, data=self.coef)
        dataset.attrs['type'] = np.string_(type(self).__name__)

    @classmethod
    def from_hdf5(cls, dataset):
        """Generate function from an HDF5 dataset

        Parameters
        ----------
        dataset : h5py.Dataset
            Dataset to read from

        Returns
        -------
        openmc.data.Function1D
            Function read from dataset

        """
        if dataset.attrs['type'].decode() != cls.__name__:
            raise ValueError("Expected an HDF5 attribute 'type' equal to '"
                             + cls.__name__ + "'")
        return cls(dataset[()])


class Combination(EqualityMixin):
    """Combination of multiple functions with a user-defined operator

    This class allows you to create a callable object which represents the
    combination of other callable objects by way of a series of user-defined
    operators connecting each of the callable objects.

    Parameters
    ----------
    functions : Iterable of Callable
        Functions to combine according to operations
    operations : Iterable of numpy.ufunc
        Operations to perform between functions; note that the standard order
        of operations will not be followed, but can be simulated by
        combinations of Combination objects. The operations parameter must have
        a length one less than the number of functions.


    Attributes
    ----------
    functions : Iterable of Callable
        Functions to combine according to operations
    operations : Iterable of numpy.ufunc
        Operations to perform between functions; note that the standard order
        of operations will not be followed, but can be simulated by
        combinations of Combination objects. The operations parameter must have
        a length one less than the number of functions.

    """

    def __init__(self, functions, operations):
        self.functions = functions
        self.operations = operations

    def __call__(self, x):
        ans = self.functions[0](x)
        for i, operation in enumerate(self.operations):
            ans = operation(ans, self.functions[i + 1](x))
        return ans

    @property
    def functions(self):
        return self._functions

    @functions.setter
    def functions(self, functions):
        check_type('functions', functions, Iterable, Callable)
        self._functions = functions

    @property
    def operations(self):
        return self._operations

    @operations.setter
    def operations(self, operations):
        check_type('operations', operations, Iterable, np.ufunc)
        length = len(self.functions) - 1
        check_length('operations', operations, length, length_max=length)
        self._operations = operations


class Sum(Function1D):
    """Sum of multiple functions.

    This class allows you to create a callable object which represents the sum
    of other callable objects. This is used for redundant reactions whereby the
    cross section is defined as the sum of other cross sections.

    Parameters
    ----------
    functions : Iterable of Callable
        Functions which are to be added together

    Attributes
    ----------
    functions : Iterable of Callable
        Functions which are to be added together

    """

    def __init__(self, functions):
        self.functions = list(functions)

    def __call__(self, x):
        return sum(f(x) for f in self.functions)

    @property
    def functions(self):
        return self._functions

    @functions.setter
    def functions(self, functions):
        check_type('functions', functions, Iterable, Callable)
        self._functions = functions

    def to_hdf5(self, group, name='xy'):
        """Write sum of functions to an HDF5 group

        .. versionadded:: 0.13.1

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to write to
        name : str
            Name of the dataset to create

        """
        sum_group = group.create_group(name)
        sum_group.attrs['type'] = np.string_(type(self).__name__)
        sum_group.attrs['n'] = len(self.functions)
        for i, f in enumerate(self.functions):
            f.to_hdf5(sum_group, f'func_{i+1}')

    @classmethod
    def from_hdf5(cls, group):
        """Generate sum of functions from an HDF5 group

        .. versionadded:: 0.13.1

        Parameters
        ----------
        group : h5py.Group
            Group to read from

        Returns
        -------
        openmc.data.Sum
            Functions read from the group

        """
        n = group.attrs['n']
        functions = [
            Function1D.from_hdf5(group[f'func_{i+1}'])
            for i in range(n)
        ]
        return cls(functions)


class Regions1D(EqualityMixin):
    r"""Piecewise composition of multiple functions.

    This class allows you to create a callable object which is composed
    of multiple other callable objects, each applying to a specific interval

    Parameters
    ----------
    functions : Iterable of Callable
        Functions which are to be combined in a piecewise fashion
    breakpoints : Iterable of float
        The values of the dependent variable that define the domain of
        each function. The `i`\ th and `(i+1)`\ th values are the limits of the
        domain of the `i`\ th function. Values must be monotonically increasing.

    Attributes
    ----------
    functions : Iterable of Callable
        Functions which are to be combined in a piecewise fashion
    breakpoints : Iterable of float
        The breakpoints between each function

    """

    def __init__(self, functions, breakpoints):
        self.functions = functions
        self.breakpoints = breakpoints

    def __call__(self, x):
        i = np.searchsorted(self.breakpoints, x)
        if isinstance(x, Iterable):
            ans = np.empty_like(x)
            for j in range(len(i)):
                ans[j] = self.functions[i[j]](x[j])
            return ans
        else:
            return self.functions[i](x)

    @property
    def functions(self):
        return self._functions

    @functions.setter
    def functions(self, functions):
        check_type('functions', functions, Iterable, Callable)
        self._functions = functions

    @property
    def breakpoints(self):
        return self._breakpoints

    @breakpoints.setter
    def breakpoints(self, breakpoints):
        check_iterable_type('breakpoints', breakpoints, Real)
        self._breakpoints = breakpoints


class ResonancesWithBackground(EqualityMixin):
    """Cross section in resolved resonance region.

    Parameters
    ----------
    resonances : openmc.data.Resonances
        Resolved resonance parameter data
    background : Callable
        Background cross section as a function of energy
    mt : int
        MT value of the reaction

    Attributes
    ----------
    resonances : openmc.data.Resonances
        Resolved resonance parameter data
    background : Callable
        Background cross section as a function of energy
    mt : int
        MT value of the reaction

    """


    def __init__(self, resonances, background, mt):
        self.resonances = resonances
        self.background = background
        self.mt = mt

    def __call__(self, x):
        # Get background cross section
        xs = self.background(x)

        for r in self.resonances:
            if not isinstance(r, openmc.data.resonance._RESOLVED):
                continue

            if isinstance(x, Iterable):
                # Determine which energies are within resolved resonance range
                within = (r.energy_min <= x) & (x <= r.energy_max)

                # Get resonance cross sections and add to background
                resonant_xs = r.reconstruct(x[within])
                xs[within] += resonant_xs[self.mt]
            else:
                if r.energy_min <= x <= r.energy_max:
                    resonant_xs = r.reconstruct(x)
                    xs += resonant_xs[self.mt]

        return xs

    @property
    def background(self):
        return self._background

    @background.setter
    def background(self, background):
        check_type('background cross section', background, Callable)
        self._background = background

    @property
    def mt(self):
        return self._mt

    @mt.setter
    def mt(self, mt):
        check_type('MT value', mt, Integral)
        self._mt = mt

    @property
    def resonances(self):
        return self._resonances

    @resonances.setter
    def resonances(self, resonances):
        check_type('resolved resonance parameters', resonances,
                      openmc.data.Resonances)
        self._resonances = resonances


_NAMES = (
    'fragments', 'prompt_neutrons', 'delayed_neutrons',
    'prompt_photons', 'delayed_photons', 'betas',
    'neutrinos', 'recoverable', 'total'
)


class FissionEnergyRelease(EqualityMixin):
    """Energy relased by fission reactions.

    Energy is carried away from fission reactions by many different particles.
    The attributes of this class specify how much energy is released in the form
    of fission fragments, neutrons, photons, etc.  Each component is also (in
    general) a function of the incident neutron energy.

    Following a fission reaction, most of the energy release is carried by the
    daughter nuclei fragments.  These fragments accelerate apart from the
    Coulomb force on the time scale of ~10^-20 s [1].  Those fragments emit
    prompt neutrons between ~10^-18 and ~10^-13 s after scission (although some
    prompt neutrons may come directly from the scission point) [1].  Prompt
    photons follow with a time scale of ~10^-14 to ~10^-7 s [1].  The fission
    products then emit delayed neutrons with half lives between 0.1 and 100 s.
    The remaining fission energy comes from beta decays of the fission products
    which release beta particles, photons, and neutrinos (that escape the
    reactor and do not produce usable heat).

    Use the class methods to instantiate this class from an HDF5 or ENDF
    dataset.  The :meth:`FissionEnergyRelease.from_hdf5` method builds this
    class from the usual OpenMC HDF5 data files.
    :meth:`FissionEnergyRelease.from_endf` uses ENDF-formatted data.

    References
    ----------
    [1] D. G. Madland, "Total prompt energy release in the neutron-induced
    fission of ^235U, ^238U, and ^239Pu", Nuclear Physics A 772:113--137 (2006).
    <http://dx.doi.org/10.1016/j.nuclphysa.2006.03.013>

    Attributes
    ----------
    fragments : Callable
        Function that accepts incident neutron energy value(s) and returns the
        kinetic energy of the fission daughter nuclides (after prompt neutron
        emission).
    prompt_neutrons : Callable
        Function of energy that returns the kinetic energy of prompt fission
        neutrons.
    delayed_neutrons : Callable
        Function of energy that returns the kinetic energy of delayed neutrons
        emitted from fission products.
    prompt_photons : Callable
        Function of energy that returns the kinetic energy of prompt fission
        photons.
    delayed_photons : Callable
        Function of energy that returns the kinetic energy of delayed photons.
    betas : Callable
        Function of energy that returns the kinetic energy of delayed beta
        particles.
    neutrinos : Callable
        Function of energy that returns the kinetic energy of neutrinos.
    recoverable : Callable
        Function of energy that returns the kinetic energy of all products that
        can be absorbed in the reactor (all of the energy except for the
        neutrinos).
    total : Callable
        Function of energy that returns the kinetic energy of all products.
    q_prompt : Callable
        Function of energy that returns the prompt fission Q-value (fragments +
        prompt neutrons + prompt photons - incident neutron energy).
    q_recoverable : Callable
        Function of energy that returns the recoverable fission Q-value
        (total release - neutrinos - incident neutron energy).  This value is
        sometimes referred to as the pseudo-Q-value.
    q_total : Callable
        Function of energy that returns the total fission Q-value (total release
        - incident neutron energy).

    """
    def __init__(self, fragments, prompt_neutrons, delayed_neutrons,
                 prompt_photons, delayed_photons, betas, neutrinos):
        self.fragments = fragments
        self.prompt_neutrons = prompt_neutrons
        self.delayed_neutrons = delayed_neutrons
        self.prompt_photons = prompt_photons
        self.delayed_photons = delayed_photons
        self.betas = betas
        self.neutrinos = neutrinos

    @property
    def fragments(self):
        return self._fragments

    @fragments.setter
    def fragments(self, energy_release):
        check_type('fragments', energy_release, Callable)
        self._fragments = energy_release

    @property
    def prompt_neutrons(self):
        return self._prompt_neutrons

    @prompt_neutrons.setter
    def prompt_neutrons(self, energy_release):
        check_type('prompt_neutrons', energy_release, Callable)
        self._prompt_neutrons = energy_release

    @property
    def delayed_neutrons(self):
        return self._delayed_neutrons

    @delayed_neutrons.setter
    def delayed_neutrons(self, energy_release):
        check_type('delayed_neutrons', energy_release, Callable)
        self._delayed_neutrons = energy_release

    @property
    def prompt_photons(self):
        return self._prompt_photons

    @prompt_photons.setter
    def prompt_photons(self, energy_release):
        check_type('prompt_photons', energy_release, Callable)
        self._prompt_photons = energy_release

    @property
    def delayed_photons(self):
        return self._delayed_photons

    @delayed_photons.setter
    def delayed_photons(self, energy_release):
        check_type('delayed_photons', energy_release, Callable)
        self._delayed_photons = energy_release

    @property
    def betas(self):
        return self._betas

    @betas.setter
    def betas(self, energy_release):
        check_type('betas', energy_release, Callable)
        self._betas = energy_release

    @property
    def neutrinos(self):
        return self._neutrinos

    @neutrinos.setter
    def neutrinos(self, energy_release):
        check_type('neutrinos', energy_release, Callable)
        self._neutrinos = energy_release

    @property
    def recoverable(self):
        components = ['fragments', 'prompt_neutrons', 'delayed_neutrons',
                      'prompt_photons', 'delayed_photons', 'betas']
        return sum_functions(getattr(self, c) for c in components)

    @property
    def total(self):
        components = ['fragments', 'prompt_neutrons', 'delayed_neutrons',
                      'prompt_photons', 'delayed_photons', 'betas',
                      'neutrinos']
        return sum_functions(getattr(self, c) for c in components)

    @property
    def q_prompt(self):
        # Use a polynomial to subtract incident energy.
        funcs = [self.fragments, self.prompt_neutrons, self.prompt_photons,
                 Polynomial((0.0, -1.0))]
        return sum_functions(funcs)

    @property
    def q_recoverable(self):
        # Use a polynomial to subtract incident energy.
        return sum_functions([self.recoverable, Polynomial((0.0, -1.0))])

    @property
    def q_total(self):
        # Use a polynomial to subtract incident energy.
        return sum_functions([self.total, Polynomial((0.0, -1.0))])

    @classmethod
    def from_endf(cls, ev, incident_neutron):
        """Generate fission energy release data from an ENDF file.

        Parameters
        ----------
        ev : openmc.data.endf.Evaluation
            ENDF evaluation
        incident_neutron : openmc.data.IncidentNeutron
            Corresponding incident neutron dataset

        Returns
        -------
        openmc.data.FissionEnergyRelease
            Fission energy release data

        """
        check_type('evaluation', ev, Evaluation)

        # Check to make sure this ENDF file matches the expected isomer.
        if ev.target['atomic_number'] != incident_neutron.atomic_number:
            raise ValueError('The atomic number of the ENDF evaluation does '
                             'not match the given IncidentNeutron.')
        if ev.target['mass_number'] != incident_neutron.mass_number:
            raise ValueError('The atomic mass of the ENDF evaluation does '
                             'not match the given IncidentNeutron.')
        if ev.target['isomeric_state'] != incident_neutron.metastable:
            raise ValueError('The metastable state of the ENDF evaluation '
                             'does not match the given IncidentNeutron.')
        if not ev.target['fissionable']:
            raise ValueError('The ENDF evaluation is not fissionable.')

        if (1, 458) not in ev.section:
            raise ValueError('ENDF evaluation does not have MF=1, MT=458.')

        file_obj = StringIO(ev.section[1, 458])

        # Read first record and check whether any components appear as
        # tabulated functions
        items = get_cont_record(file_obj)
        lfc = items[3]
        nfc = items[5]

        # Parse the ENDF LIST into an array.
        items, data = get_list_record(file_obj)
        npoly = items[3]

        # Associate each set of values and uncertainties with its label.
        functions = {}
        for i, name in enumerate(_NAMES):
            coeffs = data[2*i::18]

            # Ignore recoverable and total since we recalculate those directly
            if name in ('recoverable', 'total'):
                continue

            # In ENDF/B-VII.1, data for 2nd-order coefficients were mistakenly
            # not converted from MeV to eV.  Check for this error and fix it if
            # present.
            if npoly == 2:  # Only check 2nd-order data.
                # If a 5 MeV neutron causes a change of more than 100 MeV, we
                # know something is wrong.
                second_order = coeffs[2]
                if abs(second_order) * (5e6)**2 > 1e8:
                    # If we found the error, reduce 2nd-order coeff by 10**6.
                    coeffs[2] /= EV_PER_MEV

            # If multiple coefficients were given, we can create the polynomial
            # and move on to the next component
            if npoly > 0:
                functions[name] = Polynomial(coeffs)
                continue

            # If a single coefficient was given, we need to use the Sher-Beck
            # formula for energy dependence
            zeroth_order = coeffs[0]
            if name in ('delayed_photons', 'betas'):
                func = Polynomial((zeroth_order, -0.075))
            elif name == 'neutrinos':
                func = Polynomial((zeroth_order, -0.105))
            elif name == 'prompt_neutrons':
                # Prompt neutrons require nu-data.  It is not clear from
                # ENDF-102 whether prompt or total nu value should be used, but
                # the delayed neutron fraction is so small that the difference
                # is negligible. MT=18 (n, fission) might not be available so
                # try MT=19 (n, f) as well.
                if 18 in incident_neutron and not incident_neutron[18].redundant:
                    nu = [p.yield_ for p in incident_neutron[18].products
                          if p.particle == 'neutron'
                          and p.emission_mode in ('prompt', 'total')]
                elif 19 in incident_neutron:
                    nu = [p.yield_ for p in incident_neutron[19].products
                          if p.particle == 'neutron'
                          and p.emission_mode in ('prompt', 'total')]
                else:
                    raise ValueError('IncidentNeutron data has no fission '
                                     'reaction.')
                if len(nu) == 0:
                    raise ValueError(
                        'Nu data is needed to compute fission energy '
                        'release with the Sher-Beck format.'
                    )
                if len(nu) > 1:
                    raise ValueError('Ambiguous prompt/total nu value.')

                nu = nu[0]
                if isinstance(nu, Tabulated1D):
                    # Evaluate Sher-Beck polynomial form at each tabulated value
                    func = deepcopy(nu)
                    func.y = (zeroth_order + 1.307*nu.x - 8.07e6*(nu.y - nu.y[0]))
                elif isinstance(nu, Polynomial):
                    # Combine polynomials
                    if len(nu) == 1:
                        func = Polynomial([zeroth_order, 1.307])
                    else:
                        func = Polynomial(
                            [zeroth_order, 1.307 - 8.07e6*nu.coef[1]]
                            + [-8.07e6*c for c in nu.coef[2:]])
            else:
                func = Polynomial(coeffs)

            functions[name] = func

        # Check for tabulated data
        if lfc == 1:
            for _ in range(nfc):
                # Get tabulated function
                items, eifc = get_tab1_record(file_obj)

                # Determine which component it is
                ifc = items[3]
                name = _NAMES[ifc - 1]

                # Replace value in dictionary
                functions[name] = eifc

        # Build the object
        return cls(**functions)

    @classmethod
    def from_hdf5(cls, group):
        """Generate fission energy release data from an HDF5 group.

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to read from

        Returns
        -------
        openmc.data.FissionEnergyRelease
            Fission energy release data

        """

        fragments = Function1D.from_hdf5(group['fragments'])
        prompt_neutrons = Function1D.from_hdf5(group['prompt_neutrons'])
        delayed_neutrons = Function1D.from_hdf5(group['delayed_neutrons'])
        prompt_photons = Function1D.from_hdf5(group['prompt_photons'])
        delayed_photons = Function1D.from_hdf5(group['delayed_photons'])
        betas = Function1D.from_hdf5(group['betas'])
        neutrinos = Function1D.from_hdf5(group['neutrinos'])

        return cls(fragments, prompt_neutrons, delayed_neutrons, prompt_photons,
                   delayed_photons, betas, neutrinos)

    def to_hdf5(self, group):
        """Write energy release data to an HDF5 group

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to write to

        """

        self.fragments.to_hdf5(group, 'fragments')
        self.prompt_neutrons.to_hdf5(group, 'prompt_neutrons')
        self.delayed_neutrons.to_hdf5(group, 'delayed_neutrons')
        self.prompt_photons.to_hdf5(group, 'prompt_photons')
        self.delayed_photons.to_hdf5(group, 'delayed_photons')
        self.betas.to_hdf5(group, 'betas')
        self.neutrinos.to_hdf5(group, 'neutrinos')
        self.q_prompt.to_hdf5(group, 'q_prompt')
        self.q_recoverable.to_hdf5(group, 'q_recoverable')


class EnergyDistribution(EqualityMixin, ABC):
    """Abstract superclass for all energy distributions."""
    def __init__(self):
        pass

    @abstractmethod
    def to_hdf5(self, group):
        pass

    @staticmethod
    def from_hdf5(group):
        """Generate energy distribution from HDF5 data

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to read from

        Returns
        -------
        openmc.data.EnergyDistribution
            Energy distribution

        """
        energy_type = group.attrs['type'].decode()
        if energy_type == 'maxwell':
            return MaxwellEnergy.from_hdf5(group)
        elif energy_type == 'evaporation':
            return Evaporation.from_hdf5(group)
        elif energy_type == 'watt':
            return WattEnergy.from_hdf5(group)
        elif energy_type == 'madland-nix':
            return MadlandNix.from_hdf5(group)
        elif energy_type == 'discrete_photon':
            return DiscretePhoton.from_hdf5(group)
        elif energy_type == 'level':
            return LevelInelastic.from_hdf5(group)
        elif energy_type == 'continuous':
            return ContinuousTabular.from_hdf5(group)
        else:
            raise ValueError("Unknown energy distribution type: {}"
                             .format(energy_type))

    @staticmethod
    def from_endf(file_obj, params):
        """Generate energy distribution from an ENDF evaluation

        Parameters
        ----------
        file_obj : file-like object
            ENDF file positioned at the start of a section for an energy
            distribution.
        params : list
            List of parameters at the start of the energy distribution that
            includes the LF value indicating what type of energy distribution is
            present.

        Returns
        -------
        openmc.data.EnergyDistribution
            A sub-class of :class:`openmc.data.EnergyDistribution`

        """
        lf = params[3]
        if lf == 1:
            return ArbitraryTabulated.from_endf(file_obj, params)
        elif lf == 5:
            return GeneralEvaporation.from_endf(file_obj, params)
        elif lf == 7:
            return MaxwellEnergy.from_endf(file_obj, params)
        elif lf == 9:
            return Evaporation.from_endf(file_obj, params)
        elif lf == 11:
            return WattEnergy.from_endf(file_obj, params)
        elif lf == 12:
            return MadlandNix.from_endf(file_obj, params)


class ArbitraryTabulated(EnergyDistribution):
    r"""Arbitrary tabulated function given in ENDF MF=5, LF=1 represented as

    .. math::
         f(E \rightarrow E') = g(E \rightarrow E')

    Parameters
    ----------
    energy : numpy.ndarray
        Array of incident neutron energies
    pdf : list of openmc.data.Tabulated1D
        Tabulated outgoing energy distribution probability density functions

    Attributes
    ----------
    energy : numpy.ndarray
        Array of incident neutron energies
    pdf : list of openmc.data.Tabulated1D
        Tabulated outgoing energy distribution probability density functions

    """

    def __init__(self, energy, pdf):
        super().__init__()
        self.energy = energy
        self.pdf = pdf

    def to_hdf5(self, group):
        raise NotImplementedError

    @classmethod
    def from_endf(cls, file_obj, params):
        """Generate arbitrary tabulated distribution from an ENDF evaluation

        Parameters
        ----------
        file_obj : file-like object
            ENDF file positioned at the start of a section for an energy
            distribution.
        params : list
            List of parameters at the start of the energy distribution that
            includes the LF value indicating what type of energy distribution is
            present.

        Returns
        -------
        openmc.data.ArbitraryTabulated
            Arbitrary tabulated distribution

        """
        params, tab2 = get_tab2_record(file_obj)
        n_energies = params[5]

        energy = np.zeros(n_energies)
        pdf = []
        for j in range(n_energies):
            params, func = get_tab1_record(file_obj)
            energy[j] = params[1]
            pdf.append(func)
        return cls(energy, pdf)


class GeneralEvaporation(EnergyDistribution):
    r"""General evaporation spectrum given in ENDF MF=5, LF=5 represented as

    .. math::
        f(E \rightarrow E') = g(E'/\theta(E))

    Parameters
    ----------
    theta : openmc.data.Tabulated1D
        Tabulated function of incident neutron energy :math:`E`
    g : openmc.data.Tabulated1D
        Tabulated function of :math:`x = E'/\theta(E)`
    u : float
        Constant introduced to define the proper upper limit for the final
        particle energy such that :math:`0 \le E' \le E - U`

    Attributes
    ----------
    theta : openmc.data.Tabulated1D
        Tabulated function of incident neutron energy :math:`E`
    g : openmc.data.Tabulated1D
        Tabulated function of :math:`x = E'/\theta(E)`
    u : float
        Constant introduced to define the proper upper limit for the final
        particle energy such that :math:`0 \le E' \le E - U`

    """

    def __init__(self, theta, g, u):
        super().__init__()
        self.theta = theta
        self.g = g
        self.u = u

    def to_hdf5(self, group):
        raise NotImplementedError

    @classmethod
    def from_ace(cls, ace, idx=0):
        raise NotImplementedError

    @classmethod
    def from_endf(cls, file_obj, params):
        """Generate general evaporation spectrum from an ENDF evaluation

        Parameters
        ----------
        file_obj : file-like object
            ENDF file positioned at the start of a section for an energy
            distribution.
        params : list
            List of parameters at the start of the energy distribution that
            includes the LF value indicating what type of energy distribution is
            present.

        Returns
        -------
        openmc.data.GeneralEvaporation
            General evaporation spectrum

        """
        u = params[0]
        params, theta = get_tab1_record(file_obj)
        params, g = get_tab1_record(file_obj)
        return cls(theta, g, u)


class MaxwellEnergy(EnergyDistribution):
    r"""Simple Maxwellian fission spectrum represented as

    .. math::
        f(E \rightarrow E') = \frac{\sqrt{E'}}{I} e^{-E'/\theta(E)}

    Parameters
    ----------
    theta : openmc.data.Tabulated1D
        Tabulated function of incident neutron energy
    u : float
        Constant introduced to define the proper upper limit for the final
        particle energy such that :math:`0 \le E' \le E - U`

    Attributes
    ----------
    theta : openmc.data.Tabulated1D
        Tabulated function of incident neutron energy
    u : float
        Constant introduced to define the proper upper limit for the final
        particle energy such that :math:`0 \le E' \le E - U`

    """

    def __init__(self, theta, u):
        super().__init__()
        self.theta = theta
        self.u = u

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, theta):
        check_type('Maxwell theta', theta, Tabulated1D)
        self._theta = theta

    @property
    def u(self):
        return self._u

    @u.setter
    def u(self, u):
        check_type('Maxwell restriction energy', u, Real)
        self._u = u

    def to_hdf5(self, group):
        """Write distribution to an HDF5 group

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to write to

        """

        group.attrs['type'] = np.string_('maxwell')
        group.attrs['u'] = self.u
        self.theta.to_hdf5(group, 'theta')

    @classmethod
    def from_hdf5(cls, group):
        """Generate Maxwell distribution from HDF5 data

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to read from

        Returns
        -------
        openmc.data.MaxwellEnergy
            Maxwell distribution

        """
        theta = Tabulated1D.from_hdf5(group['theta'])
        u = group.attrs['u']
        return cls(theta, u)

    @classmethod
    def from_ace(cls, ace, idx=0):
        """Create a Maxwell distribution from an ACE table

        Parameters
        ----------
        ace : openmc.data.ace.Table
            An ACE table
        idx : int
            Offset to read from in XSS array (default of zero)

        Returns
        -------
        openmc.data.MaxwellEnergy
            Maxwell distribution

        """
        # Read nuclear temperature -- since units are MeV, convert to eV
        theta = Tabulated1D.from_ace(ace, idx)
        theta.y *= EV_PER_MEV

        # Restriction energy
        nr = int(ace.xss[idx])
        ne = int(ace.xss[idx + 1 + 2*nr])
        u = ace.xss[idx + 2 + 2*nr + 2*ne]*EV_PER_MEV

        return cls(theta, u)

    @classmethod
    def from_endf(cls, file_obj, params):
        """Generate Maxwell distribution from an ENDF evaluation

        Parameters
        ----------
        file_obj : file-like object
            ENDF file positioned at the start of a section for an energy
            distribution.
        params : list
            List of parameters at the start of the energy distribution that
            includes the LF value indicating what type of energy distribution is
            present.

        Returns
        -------
        openmc.data.MaxwellEnergy
            Maxwell distribution

        """
        u = params[0]
        params, theta = get_tab1_record(file_obj)
        return cls(theta, u)


class Evaporation(EnergyDistribution):
    r"""Evaporation spectrum represented as

    .. math::
        f(E \rightarrow E') = \frac{E'}{I} e^{-E'/\theta(E)}

    Parameters
    ----------
    theta : openmc.data.Tabulated1D
        Tabulated function of incident neutron energy
    u : float
        Constant introduced to define the proper upper limit for the final
        particle energy such that :math:`0 \le E' \le E - U`

    Attributes
    ----------
    theta : openmc.data.Tabulated1D
        Tabulated function of incident neutron energy
    u : float
        Constant introduced to define the proper upper limit for the final
        particle energy such that :math:`0 \le E' \le E - U`

    """

    def __init__(self, theta, u):
        super().__init__()
        self.theta = theta
        self.u = u

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, theta):
        check_type('Evaporation theta', theta, Tabulated1D)
        self._theta = theta

    @property
    def u(self):
        return self._u

    @u.setter
    def u(self, u):
        check_type('Evaporation restriction energy', u, Real)
        self._u = u

    def to_hdf5(self, group):
        """Write distribution to an HDF5 group

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to write to

        """

        group.attrs['type'] = np.string_('evaporation')
        group.attrs['u'] = self.u
        self.theta.to_hdf5(group, 'theta')

    @classmethod
    def from_hdf5(cls, group):
        """Generate evaporation spectrum from HDF5 data

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to read from

        Returns
        -------
        openmc.data.Evaporation
            Evaporation spectrum

        """
        theta = Tabulated1D.from_hdf5(group['theta'])
        u = group.attrs['u']
        return cls(theta, u)

    @classmethod
    def from_ace(cls, ace, idx=0):
        """Create an evaporation spectrum from an ACE table

        Parameters
        ----------
        ace : openmc.data.ace.Table
            An ACE table
        idx : int
            Offset to read from in XSS array (default of zero)

        Returns
        -------
        openmc.data.Evaporation
            Evaporation spectrum

        """
        # Read nuclear temperature -- since units are MeV, convert to eV
        theta = Tabulated1D.from_ace(ace, idx)
        theta.y *= EV_PER_MEV

        # Restriction energy
        nr = int(ace.xss[idx])
        ne = int(ace.xss[idx + 1 + 2*nr])
        u = ace.xss[idx + 2 + 2*nr + 2*ne]*EV_PER_MEV

        return cls(theta, u)

    @classmethod
    def from_endf(cls, file_obj, params):
        """Generate evaporation spectrum from an ENDF evaluation

        Parameters
        ----------
        file_obj : file-like object
            ENDF file positioned at the start of a section for an energy
            distribution.
        params : list
            List of parameters at the start of the energy distribution that
            includes the LF value indicating what type of energy distribution is
            present.

        Returns
        -------
        openmc.data.Evaporation
            Evaporation spectrum

        """
        u = params[0]
        params, theta = get_tab1_record(file_obj)
        return cls(theta, u)


class WattEnergy(EnergyDistribution):
    r"""Energy-dependent Watt spectrum represented as

    .. math::
        f(E \rightarrow E') = \frac{e^{-E'/a}}{I} \sinh \left ( \sqrt{bE'}
        \right )

    Parameters
    ----------
    a, b : openmc.data.Tabulated1D
        Energy-dependent parameters tabulated as function of incident neutron
        energy
    u : float
        Constant introduced to define the proper upper limit for the final
        particle energy such that :math:`0 \le E' \le E - U`

    Attributes
    ----------
    a, b : openmc.data.Tabulated1D
        Energy-dependent parameters tabulated as function of incident neutron
        energy
    u : float
        Constant introduced to define the proper upper limit for the final
        particle energy such that :math:`0 \le E' \le E - U`

    """

    def __init__(self, a, b, u):
        super().__init__()
        self.a = a
        self.b = b
        self.u = u

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, a):
        check_type('Watt a', a, Tabulated1D)
        self._a = a

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, b):
        check_type('Watt b', b, Tabulated1D)
        self._b = b

    @property
    def u(self):
        return self._u

    @u.setter
    def u(self, u):
        check_type('Watt restriction energy', u, Real)
        self._u = u

    def to_hdf5(self, group):
        """Write distribution to an HDF5 group

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to write to

        """

        group.attrs['type'] = np.string_('watt')
        group.attrs['u'] = self.u
        self.a.to_hdf5(group, 'a')
        self.b.to_hdf5(group, 'b')

    @classmethod
    def from_hdf5(cls, group):
        """Generate Watt fission spectrum from HDF5 data

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to read from

        Returns
        -------
        openmc.data.WattEnergy
            Watt fission spectrum

        """
        a = Tabulated1D.from_hdf5(group['a'])
        b = Tabulated1D.from_hdf5(group['b'])
        u = group.attrs['u']
        return cls(a, b, u)

    @classmethod
    def from_ace(cls, ace, idx):
        """Create a Watt fission spectrum from an ACE table

        Parameters
        ----------
        ace : openmc.data.ace.Table
            An ACE table
        idx : int
            Offset to read from in XSS array (default of zero)

        Returns
        -------
        openmc.data.WattEnergy
            Watt fission spectrum

        """
        # Energy-dependent a parameter -- units are MeV, convert to eV
        a = Tabulated1D.from_ace(ace, idx)
        a.y *= EV_PER_MEV

        # Advance index
        nr = int(ace.xss[idx])
        ne = int(ace.xss[idx + 1 + 2*nr])
        idx += 2 + 2*nr + 2*ne

        # Energy-dependent b parameter -- units are MeV^-1
        b = Tabulated1D.from_ace(ace, idx)
        b.y /= EV_PER_MEV

        # Advance index
        nr = int(ace.xss[idx])
        ne = int(ace.xss[idx + 1 + 2*nr])
        idx += 2 + 2*nr + 2*ne

        # Restriction energy
        u = ace.xss[idx]*EV_PER_MEV

        return cls(a, b, u)

    @classmethod
    def from_endf(cls, file_obj, params):
        """Generate Watt fission spectrum from an ENDF evaluation

        Parameters
        ----------
        file_obj : file-like object
            ENDF file positioned at the start of a section for an energy
            distribution.
        params : list
            List of parameters at the start of the energy distribution that
            includes the LF value indicating what type of energy distribution is
            present.

        Returns
        -------
        openmc.data.WattEnergy
            Watt fission spectrum

        """
        u = params[0]
        params, a = get_tab1_record(file_obj)
        params, b = get_tab1_record(file_obj)
        return cls(a, b, u)


class MadlandNix(EnergyDistribution):
    r"""Energy-dependent fission neutron spectrum (Madland and Nix) given in
    ENDF MF=5, LF=12 represented as

    .. math::
        f(E \rightarrow E') = \frac{1}{2} [ g(E', E_F(L)) + g(E', E_F(H))]

    where

    .. math::
        g(E',E_F) = \frac{1}{3\sqrt{E_F T_M}} \left [ u_2^{3/2} E_1 (u_2) -
        u_1^{3/2} E_1 (u_1) + \gamma \left ( \frac{3}{2}, u_2 \right ) - \gamma
        \left ( \frac{3}{2}, u_1 \right ) \right ] \\ u_1 = \left ( \sqrt{E'} -
        \sqrt{E_F} \right )^2 / T_M \\ u_2 = \left ( \sqrt{E'} + \sqrt{E_F}
        \right )^2 / T_M.

    Parameters
    ----------
    efl, efh : float
        Constants which represent the average kinetic energy per nucleon of the
        fission fragment (efl = light, efh = heavy)
    tm : openmc.data.Tabulated1D
        Parameter tabulated as a function of incident neutron energy

    Attributes
    ----------
    efl, efh : float
        Constants which represent the average kinetic energy per nucleon of the
        fission fragment (efl = light, efh = heavy)
    tm : openmc.data.Tabulated1D
        Parameter tabulated as a function of incident neutron energy

    """

    def __init__(self, efl, efh, tm):
        super().__init__()
        self.efl = efl
        self.efh = efh
        self.tm = tm

    @property
    def efl(self):
        return self._efl

    @efl.setter
    def efl(self, efl):
        name = 'Madland-Nix light fragment energy'
        check_type(name, efl, Real)
        check_greater_than(name, efl, 0.)
        self._efl = efl

    @property
    def efh(self):
        return self._efh

    @efh.setter
    def efh(self, efh):
        name = 'Madland-Nix heavy fragment energy'
        check_type(name, efh, Real)
        check_greater_than(name, efh, 0.)
        self._efh = efh

    @property
    def tm(self):
        return self._tm

    @tm.setter
    def tm(self, tm):
        check_type('Madland-Nix maximum temperature', tm, Tabulated1D)
        self._tm = tm

    def to_hdf5(self, group):
        """Write distribution to an HDF5 group

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to write to

        """

        group.attrs['type'] = np.string_('madland-nix')
        group.attrs['efl'] = self.efl
        group.attrs['efh'] = self.efh
        self.tm.to_hdf5(group)

    @classmethod
    def from_hdf5(cls, group):
        """Generate Madland-Nix fission spectrum from HDF5 data

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to read from

        Returns
        -------
        openmc.data.MadlandNix
            Madland-Nix fission spectrum

        """
        efl = group.attrs['efl']
        efh = group.attrs['efh']
        tm = Tabulated1D.from_hdf5(group['tm'])
        return cls(efl, efh, tm)

    @classmethod
    def from_endf(cls, file_obj, params):
        """Generate Madland-Nix fission spectrum from an ENDF evaluation

        Parameters
        ----------
        file_obj : file-like object
            ENDF file positioned at the start of a section for an energy
            distribution.
        params : list
            List of parameters at the start of the energy distribution that
            includes the LF value indicating what type of energy distribution is
            present.

        Returns
        -------
        openmc.data.MadlandNix
            Madland-Nix fission spectrum

        """
        params, tm = get_tab1_record(file_obj)
        efl, efh = params[0:2]
        return cls(efl, efh, tm)


class DiscretePhoton(EnergyDistribution):
    """Discrete photon energy distribution

    Parameters
    ----------
    primary_flag : int
        Indicator of whether the photon is a primary or non-primary photon.
    energy : float
        Photon energy (if lp==0 or lp==1) or binding energy (if lp==2).
    atomic_weight_ratio : float
        Atomic weight ratio of the target nuclide responsible for the emitted
        particle

    Attributes
    ----------
    primary_flag : int
        Indicator of whether the photon is a primary or non-primary photon.
    energy : float
        Photon energy (if lp==0 or lp==1) or binding energy (if lp==2).
    atomic_weight_ratio : float
        Atomic weight ratio of the target nuclide responsible for the emitted
        particle

    """

    def __init__(self, primary_flag, energy, atomic_weight_ratio):
        super().__init__()
        self.primary_flag = primary_flag
        self.energy = energy
        self.atomic_weight_ratio = atomic_weight_ratio

    @property
    def primary_flag(self):
        return self._primary_flag

    @primary_flag.setter
    def primary_flag(self, primary_flag):
        check_type('discrete photon primary_flag', primary_flag, Integral)
        self._primary_flag = primary_flag

    @property
    def energy(self):
        return self._energy

    @energy.setter
    def energy(self, energy):
        check_type('discrete photon energy', energy, Real)
        self._energy = energy

    @property
    def atomic_weight_ratio(self):
        return self._atomic_weight_ratio

    @atomic_weight_ratio.setter
    def atomic_weight_ratio(self, atomic_weight_ratio):
        check_type('atomic weight ratio', atomic_weight_ratio, Real)
        self._atomic_weight_ratio = atomic_weight_ratio

    def to_hdf5(self, group):
        """Write distribution to an HDF5 group

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to write to

        """

        group.attrs['type'] = np.string_('discrete_photon')
        group.attrs['primary_flag'] = self.primary_flag
        group.attrs['energy'] = self.energy
        group.attrs['atomic_weight_ratio'] = self.atomic_weight_ratio

    @classmethod
    def from_hdf5(cls, group):
        """Generate discrete photon energy distribution from HDF5 data

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to read from

        Returns
        -------
        openmc.data.DiscretePhoton
            Discrete photon energy distribution

        """
        primary_flag = group.attrs['primary_flag']
        energy = group.attrs['energy']
        awr = group.attrs['atomic_weight_ratio']
        return cls(primary_flag, energy, awr)

    @classmethod
    def from_ace(cls, ace, idx):
        """Generate discrete photon energy distribution from an ACE table

        Parameters
        ----------
        ace : openmc.data.ace.Table
            An ACE table
        idx : int
            Offset to read from in XSS array (default of zero)

        Returns
        -------
        openmc.data.DiscretePhoton
            Discrete photon energy distribution

        """
        primary_flag = int(ace.xss[idx])
        energy = ace.xss[idx + 1]*EV_PER_MEV
        return cls(primary_flag, energy, ace.atomic_weight_ratio)


class LevelInelastic(EnergyDistribution):
    r"""Level inelastic scattering

    Parameters
    ----------
    threshold : float
        Energy threshold in the laboratory system, :math:`(A + 1)/A * |Q|`
    mass_ratio : float
        :math:`(A/(A + 1))^2`

    Attributes
    ----------
    threshold : float
        Energy threshold in the laboratory system, :math:`(A + 1)/A * |Q|`
    mass_ratio : float
        :math:`(A/(A + 1))^2`

    """

    def __init__(self, threshold, mass_ratio):
        super().__init__()
        self.threshold = threshold
        self.mass_ratio = mass_ratio

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, threshold):
        check_type('level inelastic threhsold', threshold, Real)
        self._threshold = threshold

    @property
    def mass_ratio(self):
        return self._mass_ratio

    @mass_ratio.setter
    def mass_ratio(self, mass_ratio):
        check_type('level inelastic mass ratio', mass_ratio, Real)
        self._mass_ratio = mass_ratio

    def to_hdf5(self, group):
        """Write distribution to an HDF5 group

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to write to

        """

        group.attrs['type'] = np.string_('level')
        group.attrs['threshold'] = self.threshold
        group.attrs['mass_ratio'] = self.mass_ratio

    @classmethod
    def from_hdf5(cls, group):
        """Generate level inelastic distribution from HDF5 data

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to read from

        Returns
        -------
        openmc.data.LevelInelastic
            Level inelastic scattering distribution

        """
        threshold = group.attrs['threshold']
        mass_ratio = group.attrs['mass_ratio']
        return cls(threshold, mass_ratio)

    @classmethod
    def from_ace(cls, ace, idx):
        """Generate level inelastic distribution from an ACE table

        Parameters
        ----------
        ace : openmc.data.ace.Table
            An ACE table
        idx : int
            Offset to read from in XSS array (default of zero)

        Returns
        -------
        openmc.data.LevelInelastic
            Level inelastic scattering distribution

        """
        threshold = ace.xss[idx]*EV_PER_MEV
        mass_ratio = ace.xss[idx + 1]
        return cls(threshold, mass_ratio)


class ContinuousTabular(EnergyDistribution):
    """Continuous tabular distribution

    Parameters
    ----------
    breakpoints : Iterable of int
        Breakpoints defining interpolation regions
    interpolation : Iterable of int
        Interpolation codes
    energy : Iterable of float
        Incoming energies at which distributions exist
    energy_out : Iterable of openmc.stats.Univariate
        Distribution of outgoing energies corresponding to each incoming energy

    Attributes
    ----------
    breakpoints : Iterable of int
        Breakpoints defining interpolation regions
    interpolation : Iterable of int
        Interpolation codes
    energy : Iterable of float
        Incoming energies at which distributions exist
    energy_out : Iterable of openmc.stats.Univariate
        Distribution of outgoing energies corresponding to each incoming energy

    """

    def __init__(self, breakpoints, interpolation, energy, energy_out):
        super().__init__()
        self.breakpoints = breakpoints
        self.interpolation = interpolation
        self.energy = energy
        self.energy_out = energy_out

    @property
    def breakpoints(self):
        return self._breakpoints

    @breakpoints.setter
    def breakpoints(self, breakpoints):
        check_type('continuous tabular breakpoints', breakpoints,
                      Iterable, Integral)
        self._breakpoints = breakpoints

    @property
    def interpolation(self):
        return self._interpolation

    @interpolation.setter
    def interpolation(self, interpolation):
        check_type('continuous tabular interpolation', interpolation,
                      Iterable, Integral)
        self._interpolation = interpolation

    @property
    def energy(self):
        return self._energy

    @energy.setter
    def energy(self, energy):
        check_type('continuous tabular incoming energy', energy,
                      Iterable, Real)
        self._energy = energy

    @property
    def energy_out(self):
        return self._energy_out

    @energy_out.setter
    def energy_out(self, energy_out):
        check_type('continuous tabular outgoing energy', energy_out,
                      Iterable, Univariate)
        self._energy_out = energy_out

    def to_hdf5(self, group):
        """Write distribution to an HDF5 group

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to write to

        """

        group.attrs['type'] = np.string_('continuous')

        dset = group.create_dataset('energy', data=self.energy)
        dset.attrs['interpolation'] = np.vstack((self.breakpoints,
                                                 self.interpolation))

        # Determine total number of (E,p) pairs and create array
        n_pairs = sum(len(d) for d in self.energy_out)
        pairs = np.empty((3, n_pairs))

        # Create array for offsets
        offsets = np.empty(len(self.energy_out), dtype=int)
        interpolation = np.empty(len(self.energy_out), dtype=int)
        n_discrete_lines = np.empty(len(self.energy_out), dtype=int)
        j = 0

        # Populate offsets and pairs array
        for i, eout in enumerate(self.energy_out):
            n = len(eout)
            offsets[i] = j

            if isinstance(eout, Mixture):
                discrete, continuous = eout.distribution
                n_discrete_lines[i] = m = len(discrete)
                interpolation[i] = 1 if continuous.interpolation == 'histogram' else 2
                pairs[0, j:j+m] = discrete.x
                pairs[1, j:j+m] = discrete.p
                pairs[2, j:j+m] = discrete.c
                pairs[0, j+m:j+n] = continuous.x
                pairs[1, j+m:j+n] = continuous.p
                pairs[2, j+m:j+n] = continuous.c
            else:
                if isinstance(eout, Tabular):
                    n_discrete_lines[i] = 0
                    interpolation[i] = 1 if eout.interpolation == 'histogram' else 2
                elif isinstance(eout, Discrete):
                    n_discrete_lines[i] = n
                    interpolation[i] = 1
                pairs[0, j:j+n] = eout.x
                pairs[1, j:j+n] = eout.p
                pairs[2, j:j+n] = eout.c
            j += n

        # Create dataset for distributions
        dset = group.create_dataset('distribution', data=pairs)

        # Write interpolation as attribute
        dset.attrs['offsets'] = offsets
        dset.attrs['interpolation'] = interpolation
        dset.attrs['n_discrete_lines'] = n_discrete_lines

    @classmethod
    def from_hdf5(cls, group):
        """Generate continuous tabular distribution from HDF5 data

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to read from

        Returns
        -------
        openmc.data.ContinuousTabular
            Continuous tabular energy distribution

        """
        interp_data = group['energy'].attrs['interpolation']
        energy_breakpoints = interp_data[0, :]
        energy_interpolation = interp_data[1, :]
        energy = group['energy'][()]

        data = group['distribution']
        offsets = data.attrs['offsets']
        interpolation = data.attrs['interpolation']
        n_discrete_lines = data.attrs['n_discrete_lines']

        energy_out = []
        n_energy = len(energy)
        for i in range(n_energy):
            # Determine length of outgoing energy distribution and number of
            # discrete lines
            j = offsets[i]
            if i < n_energy - 1:
                n = offsets[i+1] - j
            else:
                n = data.shape[1] - j
            m = n_discrete_lines[i]

            # Create discrete distribution if lines are present
            if m > 0:
                eout_discrete = Discrete(data[0, j:j+m], data[1, j:j+m])
                eout_discrete.c = data[2, j:j+m]
                p_discrete = eout_discrete.c[-1]

            # Create continuous distribution
            if m < n:
                interp = INTERPOLATION_SCHEME[interpolation[i]]
                eout_continuous = Tabular(data[0, j+m:j+n], data[1, j+m:j+n], interp)
                eout_continuous.c = data[2, j+m:j+n]

            # If both continuous and discrete are present, create a mixture
            # distribution
            if m == 0:
                eout_i = eout_continuous
            elif m == n:
                eout_i = eout_discrete
            else:
                eout_i = Mixture([p_discrete, 1. - p_discrete],
                                 [eout_discrete, eout_continuous])
            energy_out.append(eout_i)

        return cls(energy_breakpoints, energy_interpolation,
                   energy, energy_out)

    @classmethod
    def from_ace(cls, ace, idx, ldis):
        """Generate continuous tabular energy distribution from ACE data

        Parameters
        ----------
        ace : openmc.data.ace.Table
            ACE table to read from
        idx : int
            Index in XSS array of the start of the energy distribution data
            (LDIS + LOCC - 1)
        ldis : int
            Index in XSS array of the start of the energy distribution block
            (e.g. JXS[11])

        Returns
        -------
        openmc.data.ContinuousTabular
            Continuous tabular energy distribution

        """
        # Read number of interpolation regions and incoming energies
        n_regions = int(ace.xss[idx])
        n_energy_in = int(ace.xss[idx + 1 + 2*n_regions])

        # Get interpolation information
        idx += 1
        if n_regions > 0:
            breakpoints = ace.xss[idx:idx + n_regions].astype(int)
            interpolation = ace.xss[idx + n_regions:idx + 2*n_regions].astype(int)
        else:
            breakpoints = np.array([n_energy_in])
            interpolation = np.array([2])

        # Incoming energies at which distributions exist
        idx += 2*n_regions + 1
        energy = ace.xss[idx:idx + n_energy_in]*EV_PER_MEV

        # Location of distributions
        idx += n_energy_in
        loc_dist = ace.xss[idx:idx + n_energy_in].astype(int)

        # Initialize variables
        energy_out = []

        # Read each outgoing energy distribution
        for i in range(n_energy_in):
            idx = ldis + loc_dist[i] - 1

            # intt = interpolation scheme (1=hist, 2=lin-lin)
            INTTp = int(ace.xss[idx])
            intt = INTTp % 10
            n_discrete_lines = (INTTp - intt)//10
            if intt not in (1, 2):
                warn("Interpolation scheme for continuous tabular distribution "
                     "is not histogram or linear-linear.")
                intt = 2

            n_energy_out = int(ace.xss[idx + 1])
            data = ace.xss[idx + 2:idx + 2 + 3*n_energy_out].copy()
            data.shape = (3, n_energy_out)
            data[0,:] *= EV_PER_MEV

            # Create continuous distribution
            eout_continuous = Tabular(data[0][n_discrete_lines:],
                                      data[1][n_discrete_lines:]/EV_PER_MEV,
                                      INTERPOLATION_SCHEME[intt])
            eout_continuous.c = data[2][n_discrete_lines:]

            # If discrete lines are present, create a mixture distribution
            if n_discrete_lines > 0:
                eout_discrete = Discrete(data[0][:n_discrete_lines],
                                         data[1][:n_discrete_lines])
                eout_discrete.c = data[2][:n_discrete_lines]
                if n_discrete_lines == n_energy_out:
                    eout_i = eout_discrete
                else:
                    p_discrete = min(sum(eout_discrete.p), 1.0)
                    eout_i = Mixture([p_discrete, 1. - p_discrete],
                                     [eout_discrete, eout_continuous])
            else:
                eout_i = eout_continuous

            energy_out.append(eout_i)

        return cls(breakpoints, interpolation, energy, energy_out)


_LIBRARY = {0: 'ENDF/B', 1: 'ENDF/A', 2: 'JEFF', 3: 'EFF',
            4: 'ENDF/B High Energy', 5: 'CENDL', 6: 'JENDL',
            17: 'TENDL', 18: 'ROSFOND', 21: 'SG-21', 31: 'INDL/V',
            32: 'INDL/A', 33: 'FENDL', 34: 'IRDF', 35: 'BROND',
            36: 'INGDB-90', 37: 'FENDL/A', 41: 'BROND'}

_SUBLIBRARY = {
    0: 'Photo-nuclear data',
    1: 'Photo-induced fission product yields',
    3: 'Photo-atomic data',
    4: 'Radioactive decay data',
    5: 'Spontaneous fission product yields',
    6: 'Atomic relaxation data',
    10: 'Incident-neutron data',
    11: 'Neutron-induced fission product yields',
    12: 'Thermal neutron scattering data',
    19: 'Neutron standards',
    113: 'Electro-atomic data',
    10010: 'Incident-proton data',
    10011: 'Proton-induced fission product yields',
    10020: 'Incident-deuteron data',
    10030: 'Incident-triton data',
    20030: 'Incident-helion (3He) data',
    20040: 'Incident-alpha data'
}

SUM_RULES = {1: [2, 3],
             3: [4, 5, 11, 16, 17, 22, 23, 24, 25, 27, 28, 29, 30, 32, 33, 34, 35,
                 36, 37, 41, 42, 44, 45, 152, 153, 154, 156, 157, 158, 159, 160,
                 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172,
                 173, 174, 175, 176, 177, 178, 179, 180, 181, 183, 184, 185,
                 186, 187, 188, 189, 190, 194, 195, 196, 198, 199, 200],
             4: list(range(50, 92)),
             16: list(range(875, 892)),
             18: [19, 20, 21, 38],
             27: [18, 101],
             101: [102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113, 114,
                   115, 116, 117, 155, 182, 191, 192, 193, 197],
             103: list(range(600, 650)),
             104: list(range(650, 700)),
             105: list(range(700, 750)),
             106: list(range(750, 800)),
             107: list(range(800, 850))}

ENDF_FLOAT_RE = re.compile(r'([\s\-\+]?\d*\.\d+)([\+\-]) ?(\d+)')


def py_float_endf(s):
    """Convert string of floating point number in ENDF to float.

    The ENDF-6 format uses an 'e-less' floating point number format,
    e.g. -1.23481+10. Trying to convert using the float built-in won't work
    because of the lack of an 'e'. This function allows such strings to be
    converted while still allowing numbers that are not in exponential notation
    to be converted as well.

    Parameters
    ----------
    s : str
        Floating-point number from an ENDF file

    Returns
    -------
    float
        The number

    """
    return float(ENDF_FLOAT_RE.sub(r'\1e\2\3', s))


if not _CYTHON:
    float_endf = py_float_endf


def int_endf(s):
    """Convert string of integer number in ENDF to int.

    The ENDF-6 format technically allows integers to be represented by a field
    of all blanks. This function acts like int(s) except when s is a string of
    all whitespace, in which case zero is returned.

    Parameters
    ----------
    s : str
        Integer or spaces

    Returns
    -------
    integer
        The number or 0
    """
    return 0 if s.isspace() else int(s)


def get_text_record(file_obj):
    """Return data from a TEXT record in an ENDF-6 file.

    Parameters
    ----------
    file_obj : file-like object
        ENDF-6 file to read from

    Returns
    -------
    str
        Text within the TEXT record

    """
    return file_obj.readline()[:66]


def get_cont_record(file_obj, skip_c=False):
    """Return data from a CONT record in an ENDF-6 file.

    Parameters
    ----------
    file_obj : file-like object
        ENDF-6 file to read from
    skip_c : bool
        Determine whether to skip the first two quantities (C1, C2) of the CONT
        record.

    Returns
    -------
    tuple
        The six items within the CONT record

    """
    line = file_obj.readline()
    if skip_c:
        C1 = None
        C2 = None
    else:
        C1 = float_endf(line[:11])
        C2 = float_endf(line[11:22])
    L1 = int_endf(line[22:33])
    L2 = int_endf(line[33:44])
    N1 = int_endf(line[44:55])
    N2 = int_endf(line[55:66])
    return (C1, C2, L1, L2, N1, N2)


def get_head_record(file_obj):
    """Return data from a HEAD record in an ENDF-6 file.

    Parameters
    ----------
    file_obj : file-like object
        ENDF-6 file to read from

    Returns
    -------
    tuple
        The six items within the HEAD record

    """
    line = file_obj.readline()
    ZA = int(float_endf(line[:11]))
    AWR = float_endf(line[11:22])
    L1 = int_endf(line[22:33])
    L2 = int_endf(line[33:44])
    N1 = int_endf(line[44:55])
    N2 = int_endf(line[55:66])
    return (ZA, AWR, L1, L2, N1, N2)


def get_list_record(file_obj):
    """Return data from a LIST record in an ENDF-6 file.

    Parameters
    ----------
    file_obj : file-like object
        ENDF-6 file to read from

    Returns
    -------
    list
        The six items within the header
    list
        The values within the list

    """
    # determine how many items are in list
    items = get_cont_record(file_obj)
    NPL = items[4]

    # read items
    b = []
    for i in range((NPL - 1)//6 + 1):
        line = file_obj.readline()
        n = min(6, NPL - 6*i)
        for j in range(n):
            b.append(float_endf(line[11*j:11*(j + 1)]))

    return (items, b)


def get_tab1_record(file_obj):
    """Return data from a TAB1 record in an ENDF-6 file.

    Parameters
    ----------
    file_obj : file-like object
        ENDF-6 file to read from

    Returns
    -------
    list
        The six items within the header
    openmc.data.Tabulated1D
        The tabulated function

    """
    # Determine how many interpolation regions and total points there are
    line = file_obj.readline()
    C1 = float_endf(line[:11])
    C2 = float_endf(line[11:22])
    L1 = int_endf(line[22:33])
    L2 = int_endf(line[33:44])
    n_regions = int_endf(line[44:55])
    n_pairs = int_endf(line[55:66])
    params = [C1, C2, L1, L2]

    # Read the interpolation region data, namely NBT and INT
    breakpoints = np.zeros(n_regions, dtype=int)
    interpolation = np.zeros(n_regions, dtype=int)
    m = 0
    for i in range((n_regions - 1)//3 + 1):
        line = file_obj.readline()
        to_read = min(3, n_regions - m)
        for j in range(to_read):
            breakpoints[m] = int_endf(line[0:11])
            interpolation[m] = int_endf(line[11:22])
            line = line[22:]
            m += 1

    # Read tabulated pairs x(n) and y(n)
    x = np.zeros(n_pairs)
    y = np.zeros(n_pairs)
    m = 0
    for i in range((n_pairs - 1)//3 + 1):
        line = file_obj.readline()
        to_read = min(3, n_pairs - m)
        for j in range(to_read):
            x[m] = float_endf(line[:11])
            y[m] = float_endf(line[11:22])
            line = line[22:]
            m += 1

    return params, Tabulated1D(x, y, breakpoints, interpolation)


def get_tab2_record(file_obj):
    # Determine how many interpolation regions and total points there are
    params = get_cont_record(file_obj)
    n_regions = params[4]

    # Read the interpolation region data, namely NBT and INT
    breakpoints = np.zeros(n_regions, dtype=int)
    interpolation = np.zeros(n_regions, dtype=int)
    m = 0
    for i in range((n_regions - 1)//3 + 1):
        line = file_obj.readline()
        to_read = min(3, n_regions - m)
        for j in range(to_read):
            breakpoints[m] = int(line[0:11])
            interpolation[m] = int(line[11:22])
            line = line[22:]
            m += 1

    return params, Tabulated2D(breakpoints, interpolation)


def get_intg_record(file_obj):
    """
    Return data from an INTG record in an ENDF-6 file. Used to store the
    covariance matrix in a compact format.

    Parameters
    ----------
    file_obj : file-like object
        ENDF-6 file to read from

    Returns
    -------
    numpy.ndarray
        The correlation matrix described in the INTG record
    """
    # determine how many items are in list and NDIGIT
    items = get_cont_record(file_obj)
    ndigit = items[2]
    npar = items[3]    # Number of parameters
    nlines = items[4]  # Lines to read
    NROW_RULES = {2: 18, 3: 12, 4: 11, 5: 9, 6: 8}
    nrow = NROW_RULES[ndigit]

    # read lines and build correlation matrix
    corr = np.identity(npar)
    for i in range(nlines):
        line = file_obj.readline()
        ii = int_endf(line[:5]) - 1  # -1 to account for 0 indexing
        jj = int_endf(line[5:10]) - 1
        factor = 10**ndigit
        for j in range(nrow):
            if jj+j >= ii:
                break
            element = int_endf(line[11+(ndigit+1)*j:11+(ndigit+1)*(j+1)])
            if element > 0:
                corr[ii, jj] = (element+0.5)/factor
            elif element < 0:
                corr[ii, jj] = (element-0.5)/factor

    # Symmetrize the correlation matrix
    corr = corr + corr.T - np.diag(corr.diagonal())
    return corr


def get_evaluations(filename):
    """Return a list of all evaluations within an ENDF file.

    Parameters
    ----------
    filename : str
        Path to ENDF-6 formatted file

    Returns
    -------
    list
        A list of :class:`openmc.data.endf.Evaluation` instances.

    """
    evaluations = []
    with open(str(filename), 'r') as fh:
        while True:
            pos = fh.tell()
            line = fh.readline()
            if line[66:70] == '  -1':
                break
            fh.seek(pos)
            evaluations.append(Evaluation(fh))
    return evaluations


class Evaluation:
    """ENDF material evaluation with multiple files/sections

    Parameters
    ----------
    filename_or_obj : str or file-like
        Path to ENDF file to read or an open file positioned at the start of an
        ENDF material

    Attributes
    ----------
    info : dict
        Miscellaneous information about the evaluation.
    target : dict
        Information about the target material, such as its mass, isomeric state,
        whether it's stable, and whether it's fissionable.
    projectile : dict
        Information about the projectile such as its mass.
    reaction_list : list of 4-tuples
        List of sections in the evaluation. The entries of the tuples are the
        file (MF), section (MT), number of records (NC), and modification
        indicator (MOD).

    """
    def __init__(self, filename_or_obj):
        if isinstance(filename_or_obj, (str, PurePath)):
            fh = open(str(filename_or_obj), 'r')
            need_to_close = True
        else:
            fh = filename_or_obj
            need_to_close = False
        self.section = {}
        self.info = {}
        self.target = {}
        self.projectile = {}
        self.reaction_list = []

        # Skip TPID record. Evaluators sometimes put in TPID records that are
        # ill-formated because they lack MF/MT values or put them in the wrong
        # columns.
        if fh.tell() == 0:
            fh.readline()
        MF = 0

        # Determine MAT number for this evaluation
        while MF == 0:
            position = fh.tell()
            line = fh.readline()
            MF = int(line[70:72])
        self.material = int(line[66:70])
        fh.seek(position)

        while True:
            # Find next section
            while True:
                position = fh.tell()
                line = fh.readline()
                MAT = int(line[66:70])
                MF = int(line[70:72])
                MT = int(line[72:75])
                if MT > 0 or MAT == 0:
                    fh.seek(position)
                    break

            # If end of material reached, exit loop
            if MAT == 0:
                fh.readline()
                break

            section_data = ''
            while True:
                line = fh.readline()
                if line[72:75] == '  0':
                    break
                else:
                    section_data += line
            self.section[MF, MT] = section_data

        if need_to_close:
            fh.close()

        self._read_header()

    def __repr__(self):
        name = self.target['zsymam'].replace(' ', '')
        return '<{} for {} {}>'.format(self.info['sublibrary'], name,
                                       self.info['library'])

    def _read_header(self):
        file_obj = io.StringIO(self.section[1, 451])

        # Information about target/projectile
        items = get_head_record(file_obj)
        Z, A = divmod(items[0], 1000)
        self.target['atomic_number'] = Z
        self.target['mass_number'] = A
        self.target['mass'] = items[1]
        self._LRP = items[2]
        self.target['fissionable'] = (items[3] == 1)
        try:
            library = _LIBRARY[items[4]]
        except KeyError:
            library = 'Unknown'
        self.info['modification'] = items[5]

        # Control record 1
        items = get_cont_record(file_obj)
        self.target['excitation_energy'] = items[0]
        self.target['stable'] = (int(items[1]) == 0)
        self.target['state'] = items[2]
        self.target['isomeric_state'] = m = items[3]
        self.info['format'] = items[5]
        assert self.info['format'] == 6

        # Set correct excited state for Am242_m1, which is wrong in ENDF/B-VII.1
        if Z == 95 and A == 242 and m == 1:
            self.target['state'] = 2

        # Control record 2
        items = get_cont_record(file_obj)
        self.projectile['mass'] = items[0]
        self.info['energy_max'] = items[1]
        library_release = items[2]
        self.info['sublibrary'] = _SUBLIBRARY[items[4]]
        library_version = items[5]
        self.info['library'] = (library, library_version, library_release)

        # Control record 3
        items = get_cont_record(file_obj)
        self.target['temperature'] = items[0]
        self.info['derived'] = (items[2] > 0)
        NWD = items[4]
        NXC = items[5]

        # Text records
        text = [get_text_record(file_obj) for i in range(NWD)]
        if len(text) >= 5:
            self.target['zsymam'] = text[0][0:11]
            self.info['laboratory'] = text[0][11:22]
            self.info['date'] = text[0][22:32]
            self.info['author'] = text[0][32:66]
            self.info['reference'] = text[1][1:22]
            self.info['date_distribution'] = text[1][22:32]
            self.info['date_release'] = text[1][33:43]
            self.info['date_entry'] = text[1][55:63]
            self.info['identifier'] = text[2:5]
            self.info['description'] = text[5:]
        else:
            self.target['zsymam'] = 'Unknown'

        # File numbers, reaction designations, and number of records
        for i in range(NXC):
            _, _, mf, mt, nc, mod = get_cont_record(file_obj, skip_c=True)
            self.reaction_list.append((mf, mt, nc, mod))

    @property
    def gnds_name(self):
        return gnds_name(self.target['atomic_number'],
                         self.target['mass_number'],
                         self.target['isomeric_state'])


class Tabulated2D:
    """Metadata for a two-dimensional function.

    This is a dummy class that is not really used other than to store the
    interpolation information for a two-dimensional function. Once we refactor
    to adopt GNDS-like data containers, this will probably be removed or
    extended.

    Parameters
    ----------
    breakpoints : Iterable of int
        Breakpoints for interpolation regions
    interpolation : Iterable of int
        Interpolation scheme identification number, e.g., 3 means y is linear in
        ln(x).

    """
    def __init__(self, breakpoints, interpolation):
        self.breakpoints = breakpoints
        self.interpolation = interpolation


# Isotopic abundances from Meija J, Coplen T B, et al, "Isotopic compositions
# of the elements 2013 (IUPAC Technical Report)", Pure. Appl. Chem. 88 (3),
# pp. 293-306 (2013). The "representative isotopic abundance" values from
# column 9 are used except where an interval is given, in which case the
# "best measurement" is used.
# Note that the abundances are given as atomic fractions!
NATURAL_ABUNDANCE = {
    'H1': 0.99984426, 'H2': 0.00015574, 'He3': 0.000002,
    'He4': 0.999998, 'Li6': 0.07589, 'Li7': 0.92411,
    'Be9': 1.0, 'B10': 0.1982, 'B11': 0.8018,
    'C12': 0.988922, 'C13': 0.011078, 'N14': 0.996337,
    'N15': 0.003663, 'O16': 0.9976206, 'O17': 0.000379,
    'O18': 0.0020004, 'F19': 1.0, 'Ne20': 0.9048,
    'Ne21': 0.0027, 'Ne22': 0.0925, 'Na23': 1.0,
    'Mg24': 0.78951, 'Mg25': 0.1002, 'Mg26': 0.11029,
    'Al27': 1.0, 'Si28': 0.9222968, 'Si29': 0.0468316,
    'Si30': 0.0308716, 'P31': 1.0, 'S32': 0.9504074,
    'S33': 0.0074869, 'S34': 0.0419599, 'S36': 0.0001458,
    'Cl35': 0.757647, 'Cl37': 0.242353, 'Ar36': 0.003336,
    'Ar38': 0.000629, 'Ar40': 0.996035, 'K39': 0.932581,
    'K40': 0.000117, 'K41': 0.067302, 'Ca40': 0.96941,
    'Ca42': 0.00647, 'Ca43': 0.00135, 'Ca44': 0.02086,
    'Ca46': 0.00004, 'Ca48': 0.00187, 'Sc45': 1.0,
    'Ti46': 0.0825, 'Ti47': 0.0744, 'Ti48': 0.7372,
    'Ti49': 0.0541, 'Ti50': 0.0518, 'V50': 0.0025,
    'V51': 0.9975, 'Cr50': 0.04345, 'Cr52': 0.83789,
    'Cr53': 0.09501, 'Cr54': 0.02365, 'Mn55': 1.0,
    'Fe54': 0.05845, 'Fe56': 0.91754, 'Fe57': 0.02119,
    'Fe58': 0.00282, 'Co59': 1.0, 'Ni58': 0.680769,
    'Ni60': 0.262231, 'Ni61': 0.011399, 'Ni62': 0.036345,
    'Ni64': 0.009256, 'Cu63': 0.6915, 'Cu65': 0.3085,
    'Zn64': 0.4917, 'Zn66': 0.2773, 'Zn67': 0.0404,
    'Zn68': 0.1845, 'Zn70': 0.0061, 'Ga69': 0.60108,
    'Ga71': 0.39892, 'Ge70': 0.2052, 'Ge72': 0.2745,
    'Ge73': 0.0776, 'Ge74': 0.3652, 'Ge76': 0.0775,
    'As75': 1.0, 'Se74': 0.0086, 'Se76': 0.0923,
    'Se77': 0.076, 'Se78': 0.2369, 'Se80': 0.498,
    'Se82': 0.0882, 'Br79': 0.50686, 'Br81': 0.49314,
    'Kr78': 0.00355, 'Kr80': 0.02286, 'Kr82': 0.11593,
    'Kr83': 0.115, 'Kr84': 0.56987, 'Kr86': 0.17279,
    'Rb85': 0.7217, 'Rb87': 0.2783, 'Sr84': 0.0056,
    'Sr86': 0.0986, 'Sr87': 0.07, 'Sr88': 0.8258,
    'Y89': 1.0, 'Zr90': 0.5145, 'Zr91': 0.1122,
    'Zr92': 0.1715, 'Zr94': 0.1738, 'Zr96': 0.028,
    'Nb93': 1.0, 'Mo92': 0.14649, 'Mo94': 0.09187,
    'Mo95': 0.15873, 'Mo96': 0.16673, 'Mo97': 0.09582,
    'Mo98': 0.24292, 'Mo100': 0.09744, 'Ru96': 0.0554,
    'Ru98': 0.0187, 'Ru99': 0.1276, 'Ru100': 0.126,
    'Ru101': 0.1706, 'Ru102': 0.3155, 'Ru104': 0.1862,
    'Rh103': 1.0, 'Pd102': 0.0102, 'Pd104': 0.1114,
    'Pd105': 0.2233, 'Pd106': 0.2733, 'Pd108': 0.2646,
    'Pd110': 0.1172, 'Ag107': 0.51839, 'Ag109': 0.48161,
    'Cd106': 0.01245, 'Cd108': 0.00888, 'Cd110': 0.1247,
    'Cd111': 0.12795, 'Cd112': 0.24109, 'Cd113': 0.12227,
    'Cd114': 0.28754, 'Cd116': 0.07512, 'In113': 0.04281,
    'In115': 0.95719, 'Sn112': 0.0097, 'Sn114': 0.0066,
    'Sn115': 0.0034, 'Sn116': 0.1454, 'Sn117': 0.0768,
    'Sn118': 0.2422, 'Sn119': 0.0859, 'Sn120': 0.3258,
    'Sn122': 0.0463, 'Sn124': 0.0579, 'Sb121': 0.5721,
    'Sb123': 0.4279, 'Te120': 0.0009, 'Te122': 0.0255,
    'Te123': 0.0089, 'Te124': 0.0474, 'Te125': 0.0707,
    'Te126': 0.1884, 'Te128': 0.3174, 'Te130': 0.3408,
    'I127': 1.0, 'Xe124': 0.00095, 'Xe126': 0.00089,
    'Xe128': 0.0191, 'Xe129': 0.26401, 'Xe130': 0.04071,
    'Xe131': 0.21232, 'Xe132': 0.26909, 'Xe134': 0.10436,
    'Xe136': 0.08857, 'Cs133': 1.0, 'Ba130': 0.0011,
    'Ba132': 0.001, 'Ba134': 0.0242, 'Ba135': 0.0659,
    'Ba136': 0.0785, 'Ba137': 0.1123, 'Ba138': 0.717,
    'La138': 0.0008881, 'La139': 0.9991119, 'Ce136': 0.00186,
    'Ce138': 0.00251, 'Ce140': 0.88449, 'Ce142': 0.11114,
    'Pr141': 1.0, 'Nd142': 0.27153, 'Nd143': 0.12173,
    'Nd144': 0.23798, 'Nd145': 0.08293, 'Nd146': 0.17189,
    'Nd148': 0.05756, 'Nd150': 0.05638, 'Sm144': 0.0308,
    'Sm147': 0.15, 'Sm148': 0.1125, 'Sm149': 0.1382,
    'Sm150': 0.0737, 'Sm152': 0.2674, 'Sm154': 0.2274,
    'Eu151': 0.4781, 'Eu153': 0.5219, 'Gd152': 0.002,
    'Gd154': 0.0218, 'Gd155': 0.148, 'Gd156': 0.2047,
    'Gd157': 0.1565, 'Gd158': 0.2484, 'Gd160': 0.2186,
    'Tb159': 1.0, 'Dy156': 0.00056, 'Dy158': 0.00095,
    'Dy160': 0.02329, 'Dy161': 0.18889, 'Dy162': 0.25475,
    'Dy163': 0.24896, 'Dy164': 0.2826, 'Ho165': 1.0,
    'Er162': 0.00139, 'Er164': 0.01601, 'Er166': 0.33503,
    'Er167': 0.22869, 'Er168': 0.26978, 'Er170': 0.1491,
    'Tm169': 1.0, 'Yb168': 0.00123, 'Yb170': 0.02982,
    'Yb171': 0.14086, 'Yb172': 0.21686, 'Yb173': 0.16103,
    'Yb174': 0.32025, 'Yb176': 0.12995, 'Lu175': 0.97401,
    'Lu176': 0.02599, 'Hf174': 0.0016, 'Hf176': 0.0526,
    'Hf177': 0.186, 'Hf178': 0.2728, 'Hf179': 0.1362,
    'Hf180': 0.3508, 'Ta180': 0.0001201, 'Ta181': 0.9998799,
    'W180': 0.0012, 'W182': 0.265, 'W183': 0.1431,
    'W184': 0.3064, 'W186': 0.2843, 'Re185': 0.374,
    'Re187': 0.626, 'Os184': 0.0002, 'Os186': 0.0159,
    'Os187': 0.0196, 'Os188': 0.1324, 'Os189': 0.1615,
    'Os190': 0.2626, 'Os192': 0.4078, 'Ir191': 0.373,
    'Ir193': 0.627, 'Pt190': 0.00012, 'Pt192': 0.00782,
    'Pt194': 0.32864, 'Pt195': 0.33775, 'Pt196': 0.25211,
    'Pt198': 0.07356, 'Au197': 1.0, 'Hg196': 0.0015,
    'Hg198': 0.1004, 'Hg199': 0.1694, 'Hg200': 0.2314,
    'Hg201': 0.1317, 'Hg202': 0.2974, 'Hg204': 0.0682,
    'Tl203': 0.29524, 'Tl205': 0.70476, 'Pb204': 0.014,
    'Pb206': 0.241, 'Pb207': 0.221, 'Pb208': 0.524,
    'Bi209': 1.0, 'Th230': 0.0002, 'Th232': 0.9998,
    'Pa231': 1.0, 'U234': 0.000054, 'U235': 0.007204,
    'U238': 0.992742
}

# Dictionary to give element symbols from IUPAC names
# (and some common mispellings)
ELEMENT_SYMBOL = {'neutron': 'n', 'hydrogen': 'H', 'helium': 'He',
                 'lithium': 'Li', 'beryllium': 'Be', 'boron': 'B',
                 'carbon': 'C', 'nitrogen': 'N', 'oxygen': 'O', 'fluorine': 'F',
                 'neon': 'Ne', 'sodium': 'Na', 'magnesium': 'Mg',
                 'aluminium': 'Al', 'aluminum': 'Al', 'silicon': 'Si',
                 'phosphorus': 'P', 'sulfur': 'S', 'sulphur': 'S',
                 'chlorine': 'Cl', 'argon': 'Ar', 'potassium': 'K',
                 'calcium': 'Ca', 'scandium': 'Sc', 'titanium': 'Ti',
                 'vanadium': 'V', 'chromium': 'Cr', 'manganese': 'Mn',
                 'iron': 'Fe', 'cobalt': 'Co', 'nickel': 'Ni', 'copper': 'Cu',
                 'zinc': 'Zn', 'gallium': 'Ga', 'germanium': 'Ge',
                 'arsenic': 'As', 'selenium': 'Se', 'bromine': 'Br',
                 'krypton': 'Kr', 'rubidium': 'Rb', 'strontium': 'Sr',
                 'yttrium': 'Y', 'zirconium': 'Zr', 'niobium': 'Nb',
                 'molybdenum': 'Mo', 'technetium': 'Tc', 'ruthenium': 'Ru',
                 'rhodium': 'Rh', 'palladium': 'Pd', 'silver': 'Ag',
                 'cadmium': 'Cd', 'indium': 'In', 'tin': 'Sn', 'antimony': 'Sb',
                 'tellurium': 'Te', 'iodine': 'I', 'xenon': 'Xe',
                 'caesium': 'Cs', 'cesium': 'Cs', 'barium': 'Ba',
                 'lanthanum': 'La', 'cerium': 'Ce', 'praseodymium': 'Pr',
                 'neodymium': 'Nd', 'promethium': 'Pm', 'samarium': 'Sm',
                 'europium': 'Eu', 'gadolinium': 'Gd', 'terbium': 'Tb',
                 'dysprosium': 'Dy', 'holmium': 'Ho', 'erbium': 'Er',
                 'thulium': 'Tm', 'ytterbium': 'Yb', 'lutetium': 'Lu',
                 'hafnium': 'Hf', 'tantalum': 'Ta', 'tungsten': 'W',
                 'wolfram': 'W', 'rhenium': 'Re', 'osmium': 'Os',
                 'iridium': 'Ir', 'platinum': 'Pt', 'gold': 'Au',
                 'mercury': 'Hg', 'thallium': 'Tl', 'lead': 'Pb',
                 'bismuth': 'Bi', 'polonium': 'Po', 'astatine': 'At',
                 'radon': 'Rn', 'francium': 'Fr', 'radium': 'Ra',
                 'actinium': 'Ac', 'thorium': 'Th', 'protactinium': 'Pa',
                 'uranium': 'U', 'neptunium': 'Np', 'plutonium': 'Pu',
                 'americium': 'Am', 'curium': 'Cm', 'berkelium': 'Bk',
                 'californium': 'Cf', 'einsteinium': 'Es', 'fermium': 'Fm',
                 'mendelevium': 'Md', 'nobelium': 'No', 'lawrencium': 'Lr',
                 'rutherfordium': 'Rf', 'dubnium': 'Db', 'seaborgium': 'Sg',
                 'bohrium': 'Bh', 'hassium': 'Hs', 'meitnerium': 'Mt',
                 'darmstadtium': 'Ds', 'roentgenium': 'Rg', 'copernicium': 'Cn',
                 'nihonium': 'Nh', 'flerovium': 'Fl', 'moscovium': 'Mc',
                 'livermorium': 'Lv', 'tennessine': 'Ts', 'oganesson': 'Og'}

ATOMIC_SYMBOL = {0: 'n', 1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C',
                 7: 'N', 8: 'O', 9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al',
                 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K',
                 20: 'Ca', 21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn',
                 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn', 31: 'Ga',
                 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr', 37: 'Rb',
                 38: 'Sr', 39: 'Y', 40: 'Zr', 41: 'Nb', 42: 'Mo', 43: 'Tc',
                 44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In',
                 50: 'Sn', 51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs',
                 56: 'Ba', 57: 'La', 58: 'Ce', 59: 'Pr', 60: 'Nd', 61: 'Pm',
                 62: 'Sm', 63: 'Eu', 64: 'Gd', 65: 'Tb', 66: 'Dy', 67: 'Ho',
                 68: 'Er', 69: 'Tm', 70: 'Yb', 71: 'Lu', 72: 'Hf', 73: 'Ta',
                 74: 'W', 75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au',
                 80: 'Hg', 81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At',
                 86: 'Rn', 87: 'Fr', 88: 'Ra', 89: 'Ac', 90: 'Th', 91: 'Pa',
                 92: 'U', 93: 'Np', 94: 'Pu', 95: 'Am', 96: 'Cm', 97: 'Bk',
                 98: 'Cf', 99: 'Es', 100: 'Fm', 101: 'Md', 102: 'No',
                 103: 'Lr', 104: 'Rf', 105: 'Db', 106: 'Sg', 107: 'Bh',
                 108: 'Hs', 109: 'Mt', 110: 'Ds', 111: 'Rg', 112: 'Cn',
                 113: 'Nh', 114: 'Fl', 115: 'Mc', 116: 'Lv', 117: 'Ts',
                 118: 'Og'}
ATOMIC_NUMBER = {value: key for key, value in ATOMIC_SYMBOL.items()}

# Values here are from the Committee on Data for Science and Technology
# (CODATA) 2018 recommendation (https://physics.nist.gov/cuu/Constants/).

# The value of the Boltzman constant in units of eV / K
K_BOLTZMANN = 8.617333262e-5

# Unit conversions
EV_PER_MEV = 1.0e6
JOULE_PER_EV = 1.602176634e-19

# Avogadro's constant
AVOGADRO = 6.02214076e23

# Neutron mass in units of amu
NEUTRON_MASS = 1.00866491595

# Used in atomic_mass function as a cache
_ATOMIC_MASS: Dict[str, float] = {}

# Regex for GNDS nuclide names (used in zam function)
_GNDS_NAME_RE = re.compile(r'([A-Zn][a-z]*)(\d+)((?:_[em]\d+)?)')

# Used in half_life function as a cache
_HALF_LIFE: Dict[str, float] = {}
_LOG_TWO = log(2.0)

def atomic_mass(isotope):
    """Return atomic mass of isotope in atomic mass units.

    Atomic mass data comes from the `Atomic Mass Evaluation 2020
    <https://doi.org/10.1088/1674-1137/abddaf>`_.

    Parameters
    ----------
    isotope : str
        Name of isotope, e.g., 'Pu239'

    Returns
    -------
    float
        Atomic mass of isotope in [amu]

    """
    if not _ATOMIC_MASS:

        # Load data from AME2020 file
        mass_file = os.path.join(os.path.dirname(__file__), 'mass_1.mas20.txt')
        with open(mass_file, 'r') as ame:
            # Read lines in file starting at line 37
            for line in itertools.islice(ame, 36, None):
                name = f'{line[20:22].strip()}{int(line[16:19])}'
                mass = float(line[106:109]) + 1e-6*float(
                    line[110:116] + '.' + line[117:123])
                _ATOMIC_MASS[name.lower()] = mass

        # For isotopes found in some libraries that represent all natural
        # isotopes of their element (e.g. C0), calculate the atomic mass as
        # the sum of the atomic mass times the natural abundance of the isotopes
        # that make up the element.
        for element in ['C', 'Zn', 'Pt', 'Os', 'Tl']:
            isotope_zero = element.lower() + '0'
            _ATOMIC_MASS[isotope_zero] = 0.
            for iso, abundance in isotopes(element):
                _ATOMIC_MASS[isotope_zero] += abundance * _ATOMIC_MASS[iso.lower()]

    # Get rid of metastable information
    if '_' in isotope:
        isotope = isotope[:isotope.find('_')]

    return _ATOMIC_MASS[isotope.lower()]


def atomic_weight(element):
    """Return atomic weight of an element in atomic mass units.

    Computes an average of the atomic mass of each of element's naturally
    occurring isotopes weighted by their relative abundance.

    Parameters
    ----------
    element : str
        Element symbol (e.g., 'H') or name (e.g., 'helium')

    Returns
    -------
    float
        Atomic weight of element in [amu]

    """
    weight = 0.
    for nuclide, abundance in isotopes(element):
        weight += atomic_mass(nuclide) * abundance
    if weight > 0.:
        return weight
    else:
        raise ValueError(f"No naturally-occurring isotopes for element '{element}'.")


def half_life(isotope):
    """Return half-life of isotope in seconds or None if isotope is stable

    Half-life values are from the `ENDF/B-VIII.0 decay sublibrary
    <https://www.nndc.bnl.gov/endf-b8.0/download.html>`_.

    .. versionadded:: 0.13.1

    Parameters
    ----------
    isotope : str
        Name of isotope, e.g., 'Pu239'

    Returns
    -------
    float
        Half-life of isotope in [s]

    """
    global _HALF_LIFE
    if not _HALF_LIFE:
        # Load ENDF/B-VIII.0 data from JSON file
        half_life_path = Path(__file__).with_name('half_life.json')
        _HALF_LIFE = json.loads(half_life_path.read_text())

    return _HALF_LIFE.get(isotope.lower())


def decay_constant(isotope):
    """Return decay constant of isotope in [s^-1]

    Decay constants are based on half-life values from the
    :func:`~openmc.data.half_life` function. When the isotope is stable, a decay
    constant of zero is returned.

    .. versionadded:: 0.13.1

    Parameters
    ----------
    isotope : str
        Name of isotope, e.g., 'Pu239'

    Returns
    -------
    float
        Decay constant of isotope in [s^-1]

    See also
    --------
    openmc.data.half_life

    """
    t = half_life(isotope)
    return _LOG_TWO / t if t else 0.0


def water_density(temperature, pressure=0.1013):
    """Return the density of liquid water at a given temperature and pressure.

    The density is calculated from a polynomial fit using equations and values
    from the 2012 version of the IAPWS-IF97 formulation.  Only the equations
    for region 1 are implemented here.  Region 1 is limited to liquid water
    below 100 [MPa] with a temperature above 273.15 [K], below 623.15 [K], and
    below saturation.

    Reference: International Association for the Properties of Water and Steam,
    "Revised Release on the IAPWS Industrial Formulation 1997 for the
    Thermodynamic Properties of Water and Steam", IAPWS R7-97(2012).

    Parameters
    ----------
    temperature : float
        Water temperature in units of [K]
    pressure : float
        Water pressure in units of [MPa]

    Returns
    -------
    float
        Water density in units of [g/cm^3]

    """

    # Make sure the temperature and pressure are inside the min/max region 1
    # bounds.  (Relax the 273.15 bound to 273 in case a user wants 0 deg C data
    # but they only use 3 digits for their conversion to K.)
    if pressure > 100.0:
        warn("Results are not valid for pressures above 100 MPa.")
    elif pressure < 0.0:
        raise ValueError("Pressure must be positive.")
    if temperature < 273:
        warn("Results are not valid for temperatures below 273.15 K.")
    elif temperature > 623.15:
        warn("Results are not valid for temperatures above 623.15 K.")
    elif temperature <= 0.0:
        raise ValueError('Temperature must be positive.')

    # IAPWS region 4 parameters
    n4 = [0.11670521452767e4, -0.72421316703206e6, -0.17073846940092e2,
          0.12020824702470e5, -0.32325550322333e7, 0.14915108613530e2,
          -0.48232657361591e4, 0.40511340542057e6, -0.23855557567849,
          0.65017534844798e3]

    # Compute the saturation temperature at the given pressure.
    beta = pressure**(0.25)
    E = beta**2 + n4[2] * beta + n4[5]
    F = n4[0] * beta**2 + n4[3] * beta + n4[6]
    G = n4[1] * beta**2 + n4[4] * beta + n4[7]
    D = 2.0 * G / (-F - sqrt(F**2 - 4 * E * G))
    T_sat = 0.5 * (n4[9] + D
                   - sqrt((n4[9] + D)**2  - 4.0 * (n4[8] + n4[9] * D)))

    # Make sure we aren't above saturation.  (Relax this bound by .2 degrees
    # for deg C to K conversions.)
    if temperature > T_sat + 0.2:
        warn("Results are not valid for temperatures above saturation "
             "(above the boiling point).")

    # IAPWS region 1 parameters
    R_GAS_CONSTANT = 0.461526  # kJ / kg / K
    ref_p = 16.53  # MPa
    ref_T = 1386  # K
    n1f = [0.14632971213167, -0.84548187169114, -0.37563603672040e1,
           0.33855169168385e1, -0.95791963387872, 0.15772038513228,
           -0.16616417199501e-1, 0.81214629983568e-3, 0.28319080123804e-3,
           -0.60706301565874e-3, -0.18990068218419e-1, -0.32529748770505e-1,
           -0.21841717175414e-1, -0.52838357969930e-4, -0.47184321073267e-3,
           -0.30001780793026e-3, 0.47661393906987e-4, -0.44141845330846e-5,
           -0.72694996297594e-15, -0.31679644845054e-4, -0.28270797985312e-5,
           -0.85205128120103e-9, -0.22425281908000e-5, -0.65171222895601e-6,
           -0.14341729937924e-12, -0.40516996860117e-6, -0.12734301741641e-8,
           -0.17424871230634e-9, -0.68762131295531e-18, 0.14478307828521e-19,
           0.26335781662795e-22, -0.11947622640071e-22, 0.18228094581404e-23,
           -0.93537087292458e-25]
    I1f = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 4,
           4, 4, 5, 8, 8, 21, 23, 29, 30, 31, 32]
    J1f = [-2, -1, 0, 1, 2, 3, 4, 5, -9, -7, -1, 0, 1, 3, -3, 0, 1, 3, 17, -4,
           0, 6, -5, -2, 10, -8, -11, -6, -29, -31, -38, -39, -40, -41]

    # Nondimensionalize the pressure and temperature.
    pi = pressure / ref_p
    tau = ref_T / temperature

    # Compute the derivative of gamma (dimensionless Gibbs free energy) with
    # respect to pi.
    gamma1_pi = 0.0
    for n, I, J in zip(n1f, I1f, J1f):
        gamma1_pi -= n * I * (7.1 - pi)**(I - 1) * (tau - 1.222)**J

    # Compute the leading coefficient.  This sets the units at
    #   1 [MPa] * [kg K / kJ] * [1 / K]
    # = 1e6 [N / m^2] * 1e-3 [kg K / N / m] * [1 / K]
    # = 1e3 [kg / m^3]
    # = 1 [g / cm^3]
    coeff = pressure / R_GAS_CONSTANT / temperature

    # Compute and return the density.
    return coeff / pi / gamma1_pi


def gnds_name(Z, A, m=0):
    """Return nuclide name using GNDS convention

    .. versionchanged:: 0.14.0
        Function name changed from ``gnd_name`` to ``gnds_name``

    Parameters
    ----------
    Z : int
        Atomic number
    A : int
        Mass number
    m : int, optional
        Metastable state

    Returns
    -------
    str
        Nuclide name in GNDS convention, e.g., 'Am242_m1'

    """
    if m > 0:
        return f'{ATOMIC_SYMBOL[Z]}{A}_m{m}'
    return f'{ATOMIC_SYMBOL[Z]}{A}'


def isotopes(element):
    """Return naturally occurring isotopes and their abundances

    .. versionadded:: 0.12.1

    Parameters
    ----------
    element : str
        Element symbol (e.g., 'H') or name (e.g., 'helium')

    Returns
    -------
    list
        A list of tuples of (isotope, abundance)

    Raises
    ------
    ValueError
        If the element name is not recognized

    """
    # Convert name to symbol if needed
    if len(element) > 2:
        symbol = ELEMENT_SYMBOL.get(element.lower())
        if symbol is None:
            raise ValueError(f'Element name "{element}" not recognised')
        element = symbol

    # Get the nuclides present in nature
    result = []
    for kv in NATURAL_ABUNDANCE.items():
        if re.match(r'{}\d+'.format(element), kv[0]):
            result.append(kv)

    return result


def zam(name):
    """Return tuple of (atomic number, mass number, metastable state)

    Parameters
    ----------
    name : str
        Name of nuclide using GNDS convention, e.g., 'Am242_m1'

    Returns
    -------
    3-tuple of int
        Atomic number, mass number, and metastable state

    """
    try:
        symbol, A, state = _GNDS_NAME_RE.match(name).groups()
    except AttributeError:
        raise ValueError(f"'{name}' does not appear to be a nuclide name in "
                         "GNDS format")

    if symbol not in ATOMIC_NUMBER:
        raise ValueError(f"'{symbol}' is not a recognized element symbol")

    metastable = int(state[2:]) if state else 0
    return (ATOMIC_NUMBER[symbol], int(A), metastable)


def check_type(name, value, expected_type, expected_iter_type=None, *, none_ok=False):
    """Ensure that an object is of an expected type. Optionally, if the object is
    iterable, check that each element is of a particular type.

    Parameters
    ----------
    name : str
        Description of value being checked
    value : object
        Object to check type of
    expected_type : type or Iterable of type
        type to check object against
    expected_iter_type : type or Iterable of type or None, optional
        Expected type of each element in value, assuming it is iterable. If
        None, no check will be performed.
    none_ok : bool, optional
        Whether None is allowed as a value

    """
    if none_ok and value is None:
        return

    if not isinstance(value, expected_type):
        if isinstance(expected_type, Iterable):
            msg = 'Unable to set "{}" to "{}" which is not one of the ' \
                  'following types: "{}"'.format(name, value, ', '.join(
                      [t.__name__ for t in expected_type]))
        else:
            msg = (f'Unable to set "{name}" to "{value}" which is not of type "'
                   f'{expected_type.__name__}"')
        raise TypeError(msg)

    if expected_iter_type:
        if isinstance(value, np.ndarray):
            if not issubclass(value.dtype.type, expected_iter_type):
                msg = (f'Unable to set "{name}" to "{value}" since each item '
                       f'must be of type "{expected_iter_type.__name__}"')
                raise TypeError(msg)
            else:
                return

        for item in value:
            if not isinstance(item, expected_iter_type):
                if isinstance(expected_iter_type, Iterable):
                    msg = 'Unable to set "{}" to "{}" since each item must be ' \
                          'one of the following types: "{}"'.format(
                              name, value, ', '.join([t.__name__ for t in
                                                      expected_iter_type]))
                else:
                    msg = (f'Unable to set "{name}" to "{value}" since each '
                           f'item must be of type "{expected_iter_type.__name__}"')
                raise TypeError(msg)


def check_iterable_type(name, value, expected_type, min_depth=1, max_depth=1):
    """Ensure that an object is an iterable containing an expected type.

    Parameters
    ----------
    name : str
        Description of value being checked
    value : Iterable
        Iterable, possibly of other iterables, that should ultimately contain
        the expected type
    expected_type : type
        type that the iterable should contain
    min_depth : int
        The minimum number of layers of nested iterables there should be before
        reaching the ultimately contained items
    max_depth : int
        The maximum number of layers of nested iterables there should be before
        reaching the ultimately contained items
    """
    # Initialize the tree at the very first item.
    tree = [value]
    index = [0]

    # Traverse the tree.
    while index[0] != len(tree[0]):
        # If we are done with this level of the tree, go to the next branch on
        # the level above this one.
        if index[-1] == len(tree[-1]):
            del index[-1]
            del tree[-1]
            index[-1] += 1
            continue

        # Get a string representation of the current index in case we raise an
        # exception.
        form = '[' + '{:d}, ' * (len(index)-1) + '{:d}]'
        ind_str = form.format(*index)

        # What is the current item we are looking at?
        current_item = tree[-1][index[-1]]

        # If this item is of the expected type, then we've reached the bottom
        # level of this branch.
        if isinstance(current_item, expected_type):
            # Is this deep enough?
            if len(tree) < min_depth:
                msg = (f'Error setting "{name}": The item at {ind_str} does not '
                       f'meet the minimum depth of {min_depth}')
                raise TypeError(msg)

            # This item is okay.  Move on to the next item.
            index[-1] += 1

        # If this item is not of the expected type, then it's either an error or
        # on a deeper level of the tree.
        else:
            if isinstance(current_item, Iterable):
                # The tree goes deeper here, let's explore it.
                tree.append(current_item)
                index.append(0)

                # But first, have we exceeded the max depth?
                if len(tree) > max_depth:
                    msg = (f'Error setting {name}: Found an iterable at '
                           f'{ind_str}, items in that iterable exceed the '
                           f'maximum depth of {max_depth}')
                    raise TypeError(msg)

            else:
                # This item is completely unexpected.
                msg = (f'Error setting {name}: Items must be of type '
                       f'"{expected_type.__name__}", but item at {ind_str} is '
                       f'of type "{type(current_item).__name__}"')
                raise TypeError(msg)


def check_length(name, value, length_min, length_max=None):
    """Ensure that a sized object has length within a given range.

    Parameters
    ----------
    name : str
        Description of value being checked
    value : collections.Sized
        Object to check length of
    length_min : int
        Minimum length of object
    length_max : int or None, optional
        Maximum length of object. If None, it is assumed object must be of
        length length_min.

    """

    if length_max is None:
        if len(value) < length_min:
            msg = (f'Unable to set "{name}" to "{value}" since it must be at '
                   f'least of length "{length_min}"')
            raise ValueError(msg)
    elif not length_min <= len(value) <= length_max:
        if length_min == length_max:
            msg = (f'Unable to set "{name}" to "{value}" since it must be of '
                  f'length "{length_min}"')
        else:
            msg = (f'Unable to set "{name}" to "{value}" since it must have '
                   f'length between "{length_min}" and "{length_max}"')
        raise ValueError(msg)


def check_value(name, value, accepted_values):
    """Ensure that an object's value is contained in a set of acceptable values.

    Parameters
    ----------
    name : str
        Description of value being checked
    value : collections.Iterable
        Object to check
    accepted_values : collections.Container
        Container of acceptable values

    """

    if value not in accepted_values:
        msg = (f'Unable to set "{name}" to "{value}" since it is not in '
               f'"{accepted_values}"')
        raise ValueError(msg)


def check_less_than(name, value, maximum, equality=False):
    """Ensure that an object's value is less than a given value.

    Parameters
    ----------
    name : str
        Description of the value being checked
    value : object
        Object to check
    maximum : object
        Maximum value to check against
    equality : bool, optional
        Whether equality is allowed. Defaults to False.

    """

    if equality:
        if value > maximum:
            msg = (f'Unable to set "{name}" to "{value}" since it is greater '
                   f'than "{maximum}"')
            raise ValueError(msg)
    else:
        if value >= maximum:
            msg = (f'Unable to set "{name}" to "{value}" since it is greater '
                   f'than or equal to "{maximum}"')
            raise ValueError(msg)


def check_greater_than(name, value, minimum, equality=False):
    """Ensure that an object's value is greater than a given value.

    Parameters
    ----------
    name : str
        Description of the value being checked
    value : object
        Object to check
    minimum : object
        Minimum value to check against
    equality : bool, optional
        Whether equality is allowed. Defaults to False.

    """

    if equality:
        if value < minimum:
            msg = (f'Unable to set "{name}" to "{value}" since it is less than '
                   f'"{minimum}"')
            raise ValueError(msg)
    else:
        if value <= minimum:
            msg = (f'Unable to set "{name}" to "{value}" since it is less than '
                   f'or equal to "{minimum}"')
            raise ValueError(msg)


def check_filetype_version(obj, expected_type, expected_version):
    """Check filetype and version of an HDF5 file.

    Parameters
    ----------
    obj : h5py.File
        HDF5 file to check
    expected_type : str
        Expected file type, e.g. 'statepoint'
    expected_version : int
        Expected major version number.

    """
    try:
        this_filetype = obj.attrs['filetype'].decode()
        this_version = obj.attrs['version']

        # Check filetype
        if this_filetype != expected_type:
            raise IOError(f'{obj.filename} is not a {expected_type} file.')

        # Check version
        if this_version[0] != expected_version:
            raise IOError('{} file has a version of {} which is not '
                          'consistent with the version expected by OpenMC, {}'
                          .format(this_filetype,
                                  '.'.join(str(v) for v in this_version),
                                  expected_version))
    except AttributeError:
        raise IOError(f'Could not read {obj.filename} file. This most likely '
                      'means the file was produced by a different version of '
                      'OpenMC than the one you are using.')


class CheckedList(list):
    """A list for which each element is type-checked as it's added

    Parameters
    ----------
    expected_type : type or Iterable of type
        Type(s) which each element should be
    name : str
        Name of data being checked
    items : Iterable, optional
        Items to initialize the list with

    """

    def __init__(self, expected_type, name, items=None):
        super().__init__()
        self.expected_type = expected_type
        self.name = name
        if items is not None:
            for item in items:
                self.append(item)

    def __add__(self, other):
        new_instance = copy.copy(self)
        new_instance += other
        return new_instance

    def __radd__(self, other):
        return self + other

    def __iadd__(self, other):
        check_type('CheckedList add operand', other, Iterable,
                   self.expected_type)
        for item in other:
            self.append(item)
        return self

    def append(self, item):
        """Append item to list

        Parameters
        ----------
        item : object
            Item to append

        """
        check_type(self.name, item, self.expected_type)
        super().append(item)

    def insert(self, index, item):
        """Insert item before index

        Parameters
        ----------
        index : int
            Index in list
        item : object
            Item to insert

        """
        check_type(self.name, item, self.expected_type)
        super().insert(index, item)


class AngleDistribution(EqualityMixin):
    """Angle distribution as a function of incoming energy

    Parameters
    ----------
    energy : Iterable of float
        Incoming energies in eV at which distributions exist
    mu : Iterable of openmc.stats.Univariate
        Distribution of scattering cosines corresponding to each incoming energy

    Attributes
    ----------
    energy : Iterable of float
        Incoming energies in eV at which distributions exist
    mu : Iterable of openmc.stats.Univariate
        Distribution of scattering cosines corresponding to each incoming energy

    """

    def __init__(self, energy, mu):
        super().__init__()
        self.energy = energy
        self.mu = mu

    @property
    def energy(self):
        return self._energy

    @energy.setter
    def energy(self, energy):
        check_type('angle distribution incoming energy', energy,
                      Iterable, Real)
        self._energy = energy

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, mu):
        check_type('angle distribution scattering cosines', mu,
                      Iterable, Univariate)
        self._mu = mu

    def to_hdf5(self, group):
        """Write angle distribution to an HDF5 group

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to write to

        """

        dset = group.create_dataset('energy', data=self.energy)

        # Make sure all data is tabular
        mu_tabular = [mu_i if isinstance(mu_i, Tabular) else
                      mu_i.to_tabular() for mu_i in self.mu]

        # Determine total number of (mu,p) pairs and create array
        n_pairs = sum([len(mu_i.x) for mu_i in mu_tabular])
        pairs = np.empty((3, n_pairs))

        # Create array for offsets
        offsets = np.empty(len(mu_tabular), dtype=int)
        interpolation = np.empty(len(mu_tabular), dtype=int)
        j = 0

        # Populate offsets and pairs array
        for i, mu_i in enumerate(mu_tabular):
            n = len(mu_i.x)
            offsets[i] = j
            interpolation[i] = 1 if mu_i.interpolation == 'histogram' else 2
            pairs[0, j:j+n] = mu_i.x
            pairs[1, j:j+n] = mu_i.p
            pairs[2, j:j+n] = mu_i.c
            j += n

        # Create dataset for distributions
        dset = group.create_dataset('mu', data=pairs)

        # Write interpolation as attribute
        dset.attrs['offsets'] = offsets
        dset.attrs['interpolation'] = interpolation

    @classmethod
    def from_hdf5(cls, group):
        """Generate angular distribution from HDF5 data

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to read from

        Returns
        -------
        openmc.data.AngleDistribution
            Angular distribution

        """
        energy = group['energy'][()]
        data = group['mu']
        offsets = data.attrs['offsets']
        interpolation = data.attrs['interpolation']

        mu = []
        n_energy = len(energy)
        for i in range(n_energy):
            # Determine length of outgoing energy distribution and number of
            # discrete lines
            j = offsets[i]
            if i < n_energy - 1:
                n = offsets[i+1] - j
            else:
                n = data.shape[1] - j

            interp = INTERPOLATION_SCHEME[interpolation[i]]
            mu_i = Tabular(data[0, j:j+n], data[1, j:j+n], interp)
            mu_i.c = data[2, j:j+n]

            mu.append(mu_i)

        return cls(energy, mu)

    @classmethod
    def from_ace(cls, ace, location_dist, location_start):
        """Generate an angular distribution from ACE data

        Parameters
        ----------
        ace : openmc.data.ace.Table
            ACE table to read from
        location_dist : int
            Index in the XSS array corresponding to the start of a block,
            e.g. JXS(9).
        location_start : int
            Index in the XSS array corresponding to the start of an angle
            distribution array

        Returns
        -------
        openmc.data.AngleDistribution
            Angular distribution

        """
        # Set starting index for angle distribution
        idx = location_dist + location_start - 1

        # Number of energies at which angular distributions are tabulated
        n_energies = int(ace.xss[idx])
        idx += 1

        # Incoming energy grid
        energy = ace.xss[idx:idx + n_energies]*EV_PER_MEV
        idx += n_energies

        # Read locations for angular distributions
        lc = ace.xss[idx:idx + n_energies].astype(int)
        idx += n_energies

        mu = []
        for i in range(n_energies):
            if lc[i] > 0:
                # Equiprobable 32 bin distribution
                n_bins = 32
                idx = location_dist + abs(lc[i]) - 1
                cos = ace.xss[idx:idx + n_bins + 1]
                pdf = np.zeros(n_bins + 1)
                pdf[:n_bins] = 1.0/(n_bins*np.diff(cos))
                cdf = np.linspace(0.0, 1.0, n_bins + 1)

                mu_i = Tabular(cos, pdf, 'histogram', ignore_negative=True)
                mu_i.c = cdf
            elif lc[i] < 0:
                # Tabular angular distribution
                idx = location_dist + abs(lc[i]) - 1
                intt = int(ace.xss[idx])
                n_points = int(ace.xss[idx + 1])
                # Data is given as rows of (values, PDF, CDF)
                data = ace.xss[idx + 2:idx + 2 + 3*n_points]
                data.shape = (3, n_points)

                mu_i = Tabular(data[0], data[1], INTERPOLATION_SCHEME[intt])
                mu_i.c = data[2]
            else:
                # Isotropic angular distribution
                mu_i = Uniform(-1., 1.)

            mu.append(mu_i)

        return cls(energy, mu)

    @classmethod
    def from_endf(cls, ev, mt):
        """Generate an angular distribution from an ENDF evaluation

        Parameters
        ----------
        ev : openmc.data.endf.Evaluation
            ENDF evaluation
        mt : int
            The MT value of the reaction to get angular distributions for

        Returns
        -------
        openmc.data.AngleDistribution
            Angular distribution

        """
        file_obj = StringIO(ev.section[4, mt])

        # Read HEAD record
        items = get_head_record(file_obj)
        lvt = items[2]
        ltt = items[3]

        # Read CONT record
        items = get_cont_record(file_obj)
        li = items[2]
        nk = items[4]

        # Check for obsolete energy transformation matrix. If present, just skip
        # it and keep reading
        if lvt > 0:
            warn('Obsolete energy transformation matrix in MF=4 angular '
                 'distribution.')
            for _ in range((nk + 5)//6):
                file_obj.readline()

        if ltt == 0 and li == 1:
            # Purely isotropic
            energy = np.array([0., ev.info['energy_max']])
            mu = [Uniform(-1., 1.), Uniform(-1., 1.)]

        elif ltt == 1 and li == 0:
            # Legendre polynomial coefficients
            params, tab2 = get_tab2_record(file_obj)
            n_energy = params[5]

            energy = np.zeros(n_energy)
            mu = []
            for i in range(n_energy):
                items, al = get_list_record(file_obj)
                energy[i] = items[1]
                coefficients = np.asarray([1.0] + al)
                mu.append(Legendre(coefficients))

        elif ltt == 2 and li == 0:
            # Tabulated probability distribution
            params, tab2 = get_tab2_record(file_obj)
            n_energy = params[5]

            energy = np.zeros(n_energy)
            mu = []
            for i in range(n_energy):
                params, f = get_tab1_record(file_obj)
                energy[i] = params[1]
                if f.n_regions > 1:
                    raise NotImplementedError('Angular distribution with multiple '
                                              'interpolation regions not supported.')
                mu.append(Tabular(f.x, f.y, INTERPOLATION_SCHEME[f.interpolation[0]]))

        elif ltt == 3 and li == 0:
            # Legendre for low energies / tabulated for high energies
            params, tab2 = get_tab2_record(file_obj)
            n_energy_legendre = params[5]

            energy_legendre = np.zeros(n_energy_legendre)
            mu = []
            for i in range(n_energy_legendre):
                items, al = get_list_record(file_obj)
                energy_legendre[i] = items[1]
                coefficients = np.asarray([1.0] + al)
                mu.append(Legendre(coefficients))

            params, tab2 = get_tab2_record(file_obj)
            n_energy_tabulated = params[5]

            energy_tabulated = np.zeros(n_energy_tabulated)
            for i in range(n_energy_tabulated):
                params, f = get_tab1_record(file_obj)
                energy_tabulated[i] = params[1]
                if f.n_regions > 1:
                    raise NotImplementedError('Angular distribution with multiple '
                                              'interpolation regions not supported.')
                mu.append(Tabular(f.x, f.y, INTERPOLATION_SCHEME[f.interpolation[0]]))

            energy = np.concatenate((energy_legendre, energy_tabulated))

        return AngleDistribution(energy, mu)


class CorrelatedAngleEnergy(AngleEnergy):
    """Correlated angle-energy distribution

    Parameters
    ----------
    breakpoints : Iterable of int
        Breakpoints defining interpolation regions
    interpolation : Iterable of int
        Interpolation codes
    energy : Iterable of float
        Incoming energies at which distributions exist
    energy_out : Iterable of openmc.stats.Univariate
        Distribution of outgoing energies corresponding to each incoming energy
    mu : Iterable of Iterable of openmc.stats.Univariate
        Distribution of scattering cosine for each incoming/outgoing energy

    Attributes
    ----------
    breakpoints : Iterable of int
        Breakpoints defining interpolation regions
    interpolation : Iterable of int
        Interpolation codes
    energy : Iterable of float
        Incoming energies at which distributions exist
    energy_out : Iterable of openmc.stats.Univariate
        Distribution of outgoing energies corresponding to each incoming energy
    mu : Iterable of Iterable of openmc.stats.Univariate
        Distribution of scattering cosine for each incoming/outgoing energy

    """

    _name = 'correlated'

    def __init__(self, breakpoints, interpolation, energy, energy_out, mu):
        super().__init__()
        self.breakpoints = breakpoints
        self.interpolation = interpolation
        self.energy = energy
        self.energy_out = energy_out
        self.mu = mu

    @property
    def breakpoints(self):
        return self._breakpoints

    @breakpoints.setter
    def breakpoints(self, breakpoints):
        check_type('correlated angle-energy breakpoints', breakpoints,
                      Iterable, Integral)
        self._breakpoints = breakpoints

    @property
    def interpolation(self):
        return self._interpolation

    @interpolation.setter
    def interpolation(self, interpolation):
        check_type('correlated angle-energy interpolation', interpolation,
                      Iterable, Integral)
        self._interpolation = interpolation

    @property
    def energy(self):
        return self._energy

    @energy.setter
    def energy(self, energy):
        check_type('correlated angle-energy incoming energy', energy,
                      Iterable, Real)
        self._energy = energy

    @property
    def energy_out(self):
        return self._energy_out

    @energy_out.setter
    def energy_out(self, energy_out):
        check_type('correlated angle-energy outgoing energy', energy_out,
                      Iterable, Univariate)
        self._energy_out = energy_out

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, mu):
        check_iterable_type('correlated angle-energy outgoing cosine',
                               mu, Univariate, 2, 2)
        self._mu = mu

    def to_hdf5(self, group):
        """Write distribution to an HDF5 group

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to write to

        """
        group.attrs['type'] = np.string_(self._name)

        dset = group.create_dataset('energy', data=self.energy)
        dset.attrs['interpolation'] = np.vstack((self.breakpoints,
                                                 self.interpolation))

        # Determine total number of (E,p) pairs and create array
        n_tuple = sum(len(d) for d in self.energy_out)
        eout = np.empty((5, n_tuple))

        # Make sure all mu data is tabular
        mu_tabular = []
        for i, mu_i in enumerate(self.mu):
            mu_tabular.append([mu_ij if isinstance(mu_ij, (Tabular, Discrete)) else
                               mu_ij.to_tabular() for mu_ij in mu_i])

        # Determine total number of (mu,p) points and create array
        n_tuple = sum(sum(len(mu_ij.x) for mu_ij in mu_i)
                      for mu_i in mu_tabular)
        mu = np.empty((3, n_tuple))

        # Create array for offsets
        offsets = np.empty(len(self.energy_out), dtype=int)
        interpolation = np.empty(len(self.energy_out), dtype=int)
        n_discrete_lines = np.empty(len(self.energy_out), dtype=int)
        offset_e = 0
        offset_mu = 0

        # Populate offsets and eout array
        for i, d in enumerate(self.energy_out):
            n = len(d)
            offsets[i] = offset_e

            if isinstance(d, Mixture):
                discrete, continuous = d.distribution
                n_discrete_lines[i] = m = len(discrete)
                interpolation[i] = 1 if continuous.interpolation == 'histogram' else 2
                eout[0, offset_e:offset_e+m] = discrete.x
                eout[1, offset_e:offset_e+m] = discrete.p
                eout[2, offset_e:offset_e+m] = discrete.c
                eout[0, offset_e+m:offset_e+n] = continuous.x
                eout[1, offset_e+m:offset_e+n] = continuous.p
                eout[2, offset_e+m:offset_e+n] = continuous.c
            else:
                if isinstance(d, Tabular):
                    n_discrete_lines[i] = 0
                    interpolation[i] = 1 if d.interpolation == 'histogram' else 2
                elif isinstance(d, Discrete):
                    n_discrete_lines[i] = n
                    interpolation[i] = 1
                else:
                    raise ValueError(
                        'Invalid univariate energy distribution as part of '
                        'correlated angle-energy: {}'.format(d))
                eout[0, offset_e:offset_e+n] = d.x
                eout[1, offset_e:offset_e+n] = d.p
                eout[2, offset_e:offset_e+n] = d.c

            for j, mu_ij in enumerate(mu_tabular[i]):
                if isinstance(mu_ij, Discrete):
                    eout[3, offset_e+j] = 0
                else:
                    eout[3, offset_e+j] = 1 if mu_ij.interpolation == 'histogram' else 2
                eout[4, offset_e+j] = offset_mu

                n_mu = len(mu_ij)
                mu[0, offset_mu:offset_mu+n_mu] = mu_ij.x
                mu[1, offset_mu:offset_mu+n_mu] = mu_ij.p
                mu[2, offset_mu:offset_mu+n_mu] = mu_ij.c

                offset_mu += n_mu

            offset_e += n

        # Create dataset for outgoing energy distributions
        dset = group.create_dataset('energy_out', data=eout)

        # Write interpolation on outgoing energy as attribute
        dset.attrs['offsets'] = offsets
        dset.attrs['interpolation'] = interpolation
        dset.attrs['n_discrete_lines'] = n_discrete_lines

        # Create dataset for outgoing angle distributions
        group.create_dataset('mu', data=mu)

    @classmethod
    def from_hdf5(cls, group):
        """Generate correlated angle-energy distribution from HDF5 data

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to read from

        Returns
        -------
        openmc.data.CorrelatedAngleEnergy
            Correlated angle-energy distribution

        """
        interp_data = group['energy'].attrs['interpolation']
        energy_breakpoints = interp_data[0, :]
        energy_interpolation = interp_data[1, :]
        energy = group['energy'][()]

        offsets = group['energy_out'].attrs['offsets']
        interpolation = group['energy_out'].attrs['interpolation']
        n_discrete_lines = group['energy_out'].attrs['n_discrete_lines']
        dset_eout = group['energy_out'][()]
        energy_out = []

        dset_mu = group['mu'][()]
        mu = []

        n_energy = len(energy)
        for i in range(n_energy):
            # Determine length of outgoing energy distribution and number of
            # discrete lines
            offset_e = offsets[i]
            if i < n_energy - 1:
                n = offsets[i+1] - offset_e
            else:
                n = dset_eout.shape[1] - offset_e
            m = n_discrete_lines[i]

            # Create discrete distribution if lines are present
            if m > 0:
                x = dset_eout[0, offset_e:offset_e+m]
                p = dset_eout[1, offset_e:offset_e+m]
                eout_discrete = Discrete(x, p)
                eout_discrete.c = dset_eout[2, offset_e:offset_e+m]
                p_discrete = eout_discrete.c[-1]

            # Create continuous distribution
            if m < n:
                interp = INTERPOLATION_SCHEME[interpolation[i]]

                x = dset_eout[0, offset_e+m:offset_e+n]
                p = dset_eout[1, offset_e+m:offset_e+n]
                eout_continuous = Tabular(x, p, interp, ignore_negative=True)
                eout_continuous.c = dset_eout[2, offset_e+m:offset_e+n]

            # If both continuous and discrete are present, create a mixture
            # distribution
            if m == 0:
                eout_i = eout_continuous
            elif m == n:
                eout_i = eout_discrete
            else:
                eout_i = Mixture([p_discrete, 1. - p_discrete],
                                 [eout_discrete, eout_continuous])

            # Read angular distributions
            mu_i = []
            for j in range(n):
                # Determine interpolation scheme
                interp_code = int(dset_eout[3, offsets[i] + j])

                # Determine offset and length
                offset_mu = int(dset_eout[4, offsets[i] + j])
                if offsets[i] + j < dset_eout.shape[1] - 1:
                    n_mu = int(dset_eout[4, offsets[i] + j + 1]) - offset_mu
                else:
                    n_mu = dset_mu.shape[1] - offset_mu

                # Get data
                x = dset_mu[0, offset_mu:offset_mu+n_mu]
                p = dset_mu[1, offset_mu:offset_mu+n_mu]
                c = dset_mu[2, offset_mu:offset_mu+n_mu]

                if interp_code == 0:
                    mu_ij = Discrete(x, p)
                else:
                    mu_ij = Tabular(x, p, INTERPOLATION_SCHEME[interp_code],
                                    ignore_negative=True)
                mu_ij.c = c
                mu_i.append(mu_ij)

                offset_mu += n_mu

            energy_out.append(eout_i)
            mu.append(mu_i)

        return cls(energy_breakpoints, energy_interpolation,
                   energy, energy_out, mu)

    @classmethod
    def from_ace(cls, ace, idx, ldis):
        """Generate correlated angle-energy distribution from ACE data

        Parameters
        ----------
        ace : openmc.data.ace.Table
            ACE table to read from
        idx : int
            Index in XSS array of the start of the energy distribution data
            (LDIS + LOCC - 1)
        ldis : int
            Index in XSS array of the start of the energy distribution block
            (e.g. JXS[11])

        Returns
        -------
        openmc.data.CorrelatedAngleEnergy
            Correlated angle-energy distribution

        """
        # Read number of interpolation regions and incoming energies
        n_regions = int(ace.xss[idx])
        n_energy_in = int(ace.xss[idx + 1 + 2*n_regions])

        # Get interpolation information
        idx += 1
        if n_regions > 0:
            breakpoints = ace.xss[idx:idx + n_regions].astype(int)
            interpolation = ace.xss[idx + n_regions:idx + 2*n_regions].astype(int)
        else:
            breakpoints = np.array([n_energy_in])
            interpolation = np.array([2])

        # Incoming energies at which distributions exist
        idx += 2*n_regions + 1
        energy = ace.xss[idx:idx + n_energy_in]*EV_PER_MEV

        # Location of distributions
        idx += n_energy_in
        loc_dist = ace.xss[idx:idx + n_energy_in].astype(int)

        # Initialize list of distributions
        energy_out = []
        mu = []

        # Read each outgoing energy distribution
        for i in range(n_energy_in):
            idx = ldis + loc_dist[i] - 1

            # intt = interpolation scheme (1=hist, 2=lin-lin). When discrete
            # lines are present, the value given is 10*n_discrete_lines + intt
            n_discrete_lines, intt = divmod(int(ace.xss[idx]), 10)
            if intt not in (1, 2):
                warn("Interpolation scheme for continuous tabular distribution "
                     "is not histogram or linear-linear.")
                intt = 2

            # Secondary energy distribution
            n_energy_out = int(ace.xss[idx + 1])
            data = ace.xss[idx + 2:idx + 2 + 4*n_energy_out].copy()
            data.shape = (4, n_energy_out)
            data[0,:] *= EV_PER_MEV

            # Create continuous distribution
            eout_continuous = Tabular(data[0][n_discrete_lines:],
                                      data[1][n_discrete_lines:]/EV_PER_MEV,
                                      INTERPOLATION_SCHEME[intt],
                                      ignore_negative=True)
            eout_continuous.c = data[2][n_discrete_lines:]
            if np.any(data[1][n_discrete_lines:] < 0.0):
                warn("Correlated angle-energy distribution has negative "
                     "probabilities.")

            # If discrete lines are present, create a mixture distribution
            if n_discrete_lines > 0:
                eout_discrete = Discrete(data[0][:n_discrete_lines],
                                         data[1][:n_discrete_lines])
                eout_discrete.c = data[2][:n_discrete_lines]
                if n_discrete_lines == n_energy_out:
                    eout_i = eout_discrete
                else:
                    p_discrete = min(sum(eout_discrete.p), 1.0)
                    eout_i = Mixture([p_discrete, 1. - p_discrete],
                                     [eout_discrete, eout_continuous])
            else:
                eout_i = eout_continuous

            energy_out.append(eout_i)

            lc = data[3].astype(int)

            # Secondary angular distributions
            mu_i = []
            for j in range(n_energy_out):
                if lc[j] > 0:
                    idx = ldis + abs(lc[j]) - 1

                    intt = int(ace.xss[idx])
                    n_cosine = int(ace.xss[idx + 1])
                    data = ace.xss[idx + 2:idx + 2 + 3*n_cosine]
                    data.shape = (3, n_cosine)

                    mu_ij = Tabular(data[0], data[1], INTERPOLATION_SCHEME[intt])
                    mu_ij.c = data[2]
                else:
                    # Isotropic distribution
                    mu_ij = Uniform(-1., 1.)

                mu_i.append(mu_ij)

            # Add cosine distributions for this incoming energy to list
            mu.append(mu_i)

        return cls(breakpoints, interpolation, energy, energy_out, mu)

    @classmethod
    def from_endf(cls, file_obj):
        """Generate correlated angle-energy distribution from an ENDF evaluation

        Parameters
        ----------
        file_obj : file-like object
            ENDF file positioned at the start of a section for a correlated
            angle-energy distribution

        Returns
        -------
        openmc.data.CorrelatedAngleEnergy
            Correlated angle-energy distribution

        """
        params, tab2 = get_tab2_record(file_obj)
        lep = params[3]
        ne = params[5]
        energy = np.zeros(ne)
        n_discrete_energies = np.zeros(ne, dtype=int)
        energy_out = []
        mu = []
        for i in range(ne):
            items, values = get_list_record(file_obj)
            energy[i] = items[1]
            n_discrete_energies[i] = items[2]
            # TODO: separate out discrete lines
            n_angle = items[3]
            n_energy_out = items[5]
            values = np.asarray(values)
            values.shape = (n_energy_out, n_angle + 2)

            # Outgoing energy distribution at the i-th incoming energy
            eout_i = values[:,0]
            eout_p_i = values[:,1]
            energy_out_i = Tabular(eout_i, eout_p_i, INTERPOLATION_SCHEME[lep],
                                   ignore_negative=True)
            energy_out.append(energy_out_i)

            # Legendre coefficients used for angular distributions
            mu_i = []
            for j in range(n_energy_out):
                mu_i.append(Legendre(values[j,1:]))
            mu.append(mu_i)

        return cls(tab2.breakpoints, tab2.interpolation, energy,
                   energy_out, mu)

def get_metadata(zaid, metastable_scheme='nndc'):
    """Return basic identifying data for a nuclide with a given ZAID.

    Parameters
    ----------
    zaid : int
        ZAID (1000*Z + A) obtained from a library
    metastable_scheme : {'nndc', 'mcnp'}
        Determine how ZAID identifiers are to be interpreted in the case of
        a metastable nuclide. Because the normal ZAID (=1000*Z + A) does not
        encode metastable information, different conventions are used among
        different libraries. In MCNP libraries, the convention is to add 400
        for a metastable nuclide except for Am242m, for which 95242 is
        metastable and 95642 (or 1095242 in newer libraries) is the ground
        state. For NNDC libraries, ZAID is given as 1000*Z + A + 100*m.

    Returns
    -------
    name : str
        Name of the table
    element : str
        The atomic symbol of the isotope in the table; e.g., Zr.
    Z : int
        Number of protons in the nucleus
    mass_number : int
        Number of nucleons in the nucleus
    metastable : int
        Metastable state of the nucleus. A value of zero indicates ground state.

    """

    check_type('zaid', zaid, int)
    check_value('metastable_scheme', metastable_scheme, ['nndc', 'mcnp'])

    Z = zaid // 1000
    mass_number = zaid % 1000

    if metastable_scheme == 'mcnp':
        if zaid > 1000000:
            # New SZA format
            Z = Z % 1000
            if zaid == 1095242:
                metastable = 0
            else:
                metastable = zaid // 1000000
        else:
            if zaid == 95242:
                metastable = 1
            elif zaid == 95642:
                metastable = 0
            else:
                metastable = 1 if mass_number > 300 else 0
    elif metastable_scheme == 'nndc':
        metastable = 1 if mass_number > 300 else 0

    while mass_number > 3 * Z:
        mass_number -= 100

    # Determine name
    element = ATOMIC_SYMBOL[Z]
    name = gnds_name(Z, mass_number, metastable)

    return (name, element, Z, mass_number, metastable)


def ascii_to_binary(ascii_file, binary_file):
    """Convert an ACE file in ASCII format (type 1) to binary format (type 2).

    Parameters
    ----------
    ascii_file : str
        Filename of ASCII ACE file
    binary_file : str
        Filename of binary ACE file to be written

    """

    # Read data from ASCII file
    with open(str(ascii_file), 'r') as ascii_file:
        lines = ascii_file.readlines()

    # Set default record length
    record_length = 4096

    # Open binary file
    with open(str(binary_file), 'wb') as binary_file:
        idx = 0
        while idx < len(lines):
            # check if it's a > 2.0.0 version header
            if lines[idx].split()[0][1] == '.':
                if lines[idx + 1].split()[3] == '3':
                    idx = idx + 3
                else:
                    raise NotImplementedError('Only backwards compatible ACE'
                                              'headers currently supported')
            # Read/write header block
            hz = lines[idx][:10].encode()
            aw0 = float(lines[idx][10:22])
            tz = float(lines[idx][22:34])
            hd = lines[idx][35:45].encode()
            hk = lines[idx + 1][:70].encode()
            hm = lines[idx + 1][70:80].encode()
            binary_file.write(struct.pack(str('=10sdd10s70s10s'),
                              hz, aw0, tz, hd, hk, hm))

            # Read/write IZ/AW pairs
            data = ' '.join(lines[idx + 2:idx + 6]).split()
            iz = np.array(data[::2], dtype=int)
            aw = np.array(data[1::2], dtype=float)
            izaw = [item for sublist in zip(iz, aw) for item in sublist]
            binary_file.write(struct.pack(str('=' + 16*'id'), *izaw))

            # Read/write NXS and JXS arrays. Null bytes are added at the end so
            # that XSS will start at the second record
            nxs = [int(x) for x in ' '.join(lines[idx + 6:idx + 8]).split()]
            jxs = [int(x) for x in ' '.join(lines[idx + 8:idx + 12]).split()]
            binary_file.write(struct.pack(str('=16i32i{}x'.format(record_length - 500)),
                                          *(nxs + jxs)))

            # Read/write XSS array. Null bytes are added to form a complete record
            # at the end of the file
            n_lines = (nxs[0] + 3)//4
            start = idx + _ACE_HEADER_SIZE
            xss = np.fromstring(' '.join(lines[start:start + n_lines]), sep=' ')
            extra_bytes = record_length - ((len(xss)*8 - 1) % record_length + 1)
            binary_file.write(struct.pack(str('={}d{}x'.format(
                nxs[0], extra_bytes)), *xss))

            # Advance to next table in file
            idx += _ACE_HEADER_SIZE + n_lines


def get_table(filename, name=None):
    """Read a single table from an ACE file

    Parameters
    ----------
    filename : str
        Path of the ACE library to load table from
    name : str, optional
        Name of table to load, e.g. '92235.71c'

    Returns
    -------
    openmc.data.ace.Table
        ACE table with specified name. If no name is specified, the first table
        in the file is returned.

    """

    if name is None:
        return Library(filename).tables[0]
    else:
        lib = Library(filename, name)
        if lib.tables:
            return lib.tables[0]
        else:
            raise ValueError('Could not find ACE table with name: {}'
                             .format(name))


# The beginning of an ASCII ACE file consists of 12 lines that include the name,
# atomic weight ratio, iz/aw pairs, and the NXS and JXS arrays
_ACE_HEADER_SIZE = 12


class Library(EqualityMixin):
    """A Library objects represents an ACE-formatted file which may contain
    multiple tables with data.

    Parameters
    ----------
    filename : str
        Path of the ACE library file to load.
    table_names : None, str, or iterable, optional
        Tables from the file to read in.  If None, reads in all of the
        tables. If str, reads in only the single table of a matching name.
    verbose : bool, optional
        Determines whether output is printed to the stdout when reading a
        Library

    Attributes
    ----------
    tables : list
        List of :class:`Table` instances

    """

    def __init__(self, filename, table_names=None, verbose=False):
        if isinstance(table_names, str):
            table_names = [table_names]
        if table_names is not None:
            table_names = set(table_names)

        self.tables = []

        # Determine whether file is ASCII or binary
        filename = str(filename)
        try:
            fh = open(filename, 'rb')
            # Grab 10 lines of the library
            sb = b''.join([fh.readline() for i in range(10)])

            # Try to decode it with ascii
            sb.decode('ascii')

            # No exception so proceed with ASCII - reopen in non-binary
            fh.close()
            with open(filename, 'r') as fh:
                self._read_ascii(fh, table_names, verbose)
        except UnicodeDecodeError:
            fh.close()
            with open(filename, 'rb') as fh:
                self._read_binary(fh, table_names, verbose)

    def _read_binary(self, ace_file, table_names, verbose=False,
                     recl_length=4096, entries=512):
        """Read a binary (Type 2) ACE table.

        Parameters
        ----------
        ace_file : file
            Open ACE file
        table_names : None, str, or iterable
            Tables from the file to read in.  If None, reads in all of the
            tables. If str, reads in only the single table of a matching name.
        verbose : str, optional
            Whether to display what tables are being read. Defaults to False.
        recl_length : int, optional
            Fortran record length in binary file. Default value is 4096 bytes.
        entries : int, optional
            Number of entries per record. The default is 512 corresponding to a
            record length of 4096 bytes with double precision data.

        """

        while True:
            start_position = ace_file.tell()

            # Check for end-of-file
            if len(ace_file.read(1)) == 0:
                return
            ace_file.seek(start_position)

            # Read name, atomic mass ratio, temperature, date, comment, and
            # material
            name, atomic_weight_ratio, temperature, date, comment, mat = \
                struct.unpack(str('=10sdd10s70s10s'), ace_file.read(116))
            name = name.decode().strip()

            # Read ZAID/awr combinations
            data = struct.unpack(str('=' + 16*'id'), ace_file.read(192))
            pairs = list(zip(data[::2], data[1::2]))

            # Read NXS
            nxs = list(struct.unpack(str('=16i'), ace_file.read(64)))

            # Determine length of XSS and number of records
            length = nxs[0]
            n_records = (length + entries - 1)//entries

            # verify that we are supposed to read this table in
            if (table_names is not None) and (name not in table_names):
                ace_file.seek(start_position + recl_length*(n_records + 1))
                continue

            if verbose:
                kelvin = round(temperature * EV_PER_MEV / K_BOLTZMANN)
                print("Loading nuclide {} at {} K".format(name, kelvin))

            # Read JXS
            jxs = list(struct.unpack(str('=32i'), ace_file.read(128)))

            # Read XSS
            ace_file.seek(start_position + recl_length)
            xss = list(struct.unpack(str('={}d'.format(length)),
                                     ace_file.read(length*8)))

            # Insert zeros at beginning of NXS, JXS, and XSS arrays so that the
            # indexing will be the same as Fortran. This makes it easier to
            # follow the ACE format specification.
            nxs.insert(0, 0)
            nxs = np.array(nxs, dtype=int)

            jxs.insert(0, 0)
            jxs = np.array(jxs, dtype=int)

            xss.insert(0, 0.0)
            xss = np.array(xss)

            # Create ACE table with data read in
            table = Table(name, atomic_weight_ratio, temperature, pairs,
                          nxs, jxs, xss)
            self.tables.append(table)

            # Advance to next record
            ace_file.seek(start_position + recl_length*(n_records + 1))

    def _read_ascii(self, ace_file, table_names, verbose=False):
        """Read an ASCII (Type 1) ACE table.

        Parameters
        ----------
        ace_file : file
            Open ACE file
        table_names : None, str, or iterable
            Tables from the file to read in.  If None, reads in all of the
            tables. If str, reads in only the single table of a matching name.
        verbose : str, optional
            Whether to display what tables are being read. Defaults to False.

        """

        tables_seen = set()

        lines = [ace_file.readline() for i in range(_ACE_HEADER_SIZE + 1)]

        while len(lines) != 0 and lines[0].strip() != '':
            # Read name of table, atomic mass ratio, and temperature. If first
            # line is empty, we are at end of file

            # check if it's a 2.0 style header
            if lines[0].split()[0][1] == '.':
                words = lines[0].split()
                name = words[1]
                words = lines[1].split()
                atomic_weight_ratio = float(words[0])
                temperature = float(words[1])
                commentlines = int(words[3])
                for _ in range(commentlines):
                    lines.pop(0)
                    lines.append(ace_file.readline())
            else:
                words = lines[0].split()
                name = words[0]
                atomic_weight_ratio = float(words[1])
                temperature = float(words[2])

            datastr = ' '.join(lines[2:6]).split()
            pairs = list(zip(map(int, datastr[::2]),
                             map(float, datastr[1::2])))

            datastr = '0 ' + ' '.join(lines[6:8])
            nxs = np.fromstring(datastr, sep=' ', dtype=int)

            # Detemrine number of lines in the XSS array; each line consists of
            # four values
            n_lines = (nxs[1] + 3)//4

            # Ensure that we have more tables to read in
            if (table_names is not None) and (table_names <= tables_seen):
                break
            tables_seen.add(name)

            # verify that we are supposed to read this table in
            if (table_names is not None) and (name not in table_names):
                for _ in range(n_lines - 1):
                    ace_file.readline()
                lines = [ace_file.readline() for i in range(_ACE_HEADER_SIZE + 1)]
                continue

            # Read lines corresponding to this table
            lines += [ace_file.readline() for i in range(n_lines - 1)]

            if verbose:
                kelvin = round(temperature * EV_PER_MEV / K_BOLTZMANN)
                print("Loading nuclide {} at {} K".format(name, kelvin))

            # Insert zeros at beginning of NXS, JXS, and XSS arrays so that the
            # indexing will be the same as Fortran. This makes it easier to
            # follow the ACE format specification.
            datastr = '0 ' + ' '.join(lines[8:_ACE_HEADER_SIZE])
            jxs = np.fromstring(datastr, dtype=int, sep=' ')

            datastr = '0.0 ' + ''.join(lines[_ACE_HEADER_SIZE:_ACE_HEADER_SIZE + n_lines])
            xss = np.fromstring(datastr, sep=' ')

            # When NJOY writes an ACE file, any values less than 1e-100 actually
            # get written without the 'e'. Thus, what we do here is check
            # whether the xss array is of the right size (if a number like
            # 1.0-120 is encountered, np.fromstring won't capture any numbers
            # after it). If it's too short, then we apply the ENDF float regular
            # expression. We don't do this by default because it's expensive!
            if xss.size != nxs[1] + 1:
                datastr = ENDF_FLOAT_RE.sub(r'\1e\2\3', datastr)
                xss = np.fromstring(datastr, sep=' ')
                assert xss.size == nxs[1] + 1

            table = Table(name, atomic_weight_ratio, temperature, pairs,
                          nxs, jxs, xss)
            self.tables.append(table)

            # Read all data blocks
            lines = [ace_file.readline() for i in range(_ACE_HEADER_SIZE + 1)]


class TableType(enum.Enum):
    """Type of ACE data table."""
    NEUTRON_CONTINUOUS = 'c'
    NEUTRON_DISCRETE = 'd'
    THERMAL_SCATTERING = 't'
    DOSIMETRY = 'y'
    PHOTOATOMIC = 'p'
    PHOTONUCLEAR = 'u'
    PROTON = 'h'
    DEUTERON = 'o'
    TRITON = 'r'
    HELIUM3 = 's'
    ALPHA = 'a'

    @classmethod
    def from_suffix(cls, suffix):
        """Determine ACE table type from a suffix.

        Parameters
        ----------
        suffix : str
            Single letter ACE table designator, e.g., 'c'

        Returns
        -------
        TableType
            ACE table type

        """
        for member in cls:
            if suffix.endswith(member.value):
                return member
        raise ValueError("Suffix '{}' has no corresponding ACE table type."
                         .format(suffix))


def get_libraries_from_xsdir(path):
    """Determine paths to ACE files from an MCNP xsdir file.

    Parameters
    ----------
    path : str or path-like
        Path to xsdir file

    Returns
    -------
    list
        List of paths to ACE libraries
    """
    xsdir = Path(path)

    # Find 'directory' section
    with open(path, 'r') as fh:
        lines = fh.readlines()
    for index, line in enumerate(lines):
        if line.strip().lower() == 'directory':
            break
    else:
        raise RuntimeError("Could not find 'directory' section in MCNP xsdir file")

    # Handle continuation lines indicated by '+' at end of line
    lines = lines[index + 1:]
    continue_lines = [i for i, line in enumerate(lines)
                      if line.strip().endswith('+')]
    for i in reversed(continue_lines):
        lines[i] = lines[i].strip()[:-1] + lines.pop(i + 1)

    # Create list of ACE libraries -- we use an ordered dictionary while
    # building to get O(1) membership checks while retaining insertion order
    libraries = {}
    for line in lines:
        words = line.split()
        if len(words) < 3:
            continue

        lib = (xsdir.parent / words[2]).resolve()
        if lib not in libraries:
            # Value in dictionary is not used, so we just assign None. Below a
            # list is created from the keys alone
            libraries[lib] = None

    return list(libraries.keys())


def get_libraries_from_xsdata(path):
    """Determine paths to ACE files from a Serpent xsdata file.

    Parameters
    ----------
    path : str or path-like
        Path to xsdata file

    Returns
    -------
    list
        List of paths to ACE libraries
    """
    xsdata = Path(path)
    with open(xsdata, 'r') as xsdata_file:
        # As in get_libraries_from_xsdir, we use a dict for O(1) membership
        # check while retaining insertion order
        libraries = {}
        for line in xsdata_file:
            words = line.split()
            if len(words) >= 9:
                lib = (xsdata.parent / words[8]).resolve()
                if lib not in libraries:
                    libraries[lib] = None
    return list(libraries.keys())
