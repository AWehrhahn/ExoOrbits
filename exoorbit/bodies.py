from numpy import pi
from scipy.constants import G
import astropy.units as u
from astropy.time import Time

from .util import resets_cache, time_input


class Body:
    def __init__(self, mass, radius, name="", **kwargs):
        self._orbit = None
        self.mass = mass
        self.radius = radius
        self.name = name
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):
        return self.name

    @property
    def mass(self):
        return self._mass

    @mass.setter
    @u.quantity_input(value=u.kg)
    @resets_cache
    def mass(self, value):
        self._mass = value

    @property
    def radius(self):
        return self._radius

    @radius.setter
    @u.quantity_input(value=u.km)
    @resets_cache
    def radius(self, value):
        self._radius = value

    @property
    def area(self):
        return pi * self.radius ** 2

    @property
    def circumference(self):
        return 2 * pi * self.radius

    @property
    def gravity_value(self):
        return G * self.mass

    @property
    def volume(self):
        return 4 / 3 * pi * self.radius ** 3

    @property
    def density(self):
        return self.mass / self.volume


class Star(Body):
    def __init__(self, mass, radius, effective_temperature, name="", **kwargs):
        super().__init__(mass, radius, name=name, **kwargs)
        self.effective_temperature = effective_temperature

    @property
    def effective_temperature(self):
        return self._effective_temperature

    @property
    def teff(self):
        return self._effective_temperature

    @effective_temperature.setter
    @u.quantity_input(value=u.K)
    @resets_cache
    def effective_temperature(self, value):
        self._effective_temperature = value


class Planet(Body):
    def __init__(
        self,
        mass,
        radius,
        semi_major_axis,
        period,
        eccentricity=0,
        inclination=pi / 2 * u.rad,
        argument_of_periastron=pi / 2 * u.rad,
        time_of_transit=Time(0, format="mjd"),
        name="",
        **kwargs
    ):
        super().__init__(mass, radius, name=name, **kwargs)
        self.semi_major_axis = semi_major_axis
        self.period = period
        self.eccentricity = eccentricity
        self.inclination = inclination
        self.argument_of_periastron = argument_of_periastron
        self.time_of_transit = time_of_transit

    @property
    def semi_major_axis(self):
        return self._semi_major_axis

    @property
    def a(self):
        return self._semi_major_axis

    @semi_major_axis.setter
    @u.quantity_input(value=u.km)
    @resets_cache
    def semi_major_axis(self, value):
        self._semi_major_axis = value

    @property
    def period(self):
        return self._period

    @property
    def p(self):
        return self._period

    @period.setter
    @u.quantity_input(value=u.day)
    @resets_cache
    def period(self, value):
        self._period = value

    @property
    def eccentricity(self):
        return self._eccentricity

    @property
    def ecc(self):
        return self._eccentricity

    @eccentricity.setter
    @resets_cache
    def eccentricity(self, value):
        if value > 1 or value < 0:
            raise ValueError("Eccentricity must be between 0 and 1")
        self._eccentricity = value

    @property
    def inclination(self):
        return self._inclination

    @property
    def inc(self):
        return self._inclination

    @inclination.setter
    @u.quantity_input(value=u.deg)
    @resets_cache
    def inclination(self, value):
        self._inclination = value

    @property
    def argument_of_periastron(self):
        return self._argument_of_periastron

    @property
    def w(self):
        return self._argument_of_periastron

    @argument_of_periastron.setter
    @u.quantity_input(value=u.deg)
    @resets_cache
    def argument_of_periastron(self, value):
        self._argument_of_periastron = value

    @property
    def time_of_transit(self):
        return self._time_of_transit

    @property
    def t0(self):
        return self._time_of_transit

    @time_of_transit.setter
    @time_input
    @resets_cache
    def time_of_transit(self, value):
        self._time_of_transit = value
