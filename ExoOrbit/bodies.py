from numpy import pi
from scipy.constants import G
from astropy.units import Quantity
from astropy.time import Time


class Body:
    def __init__(self, mass, radius, name="", **kwargs):
        self.mass = mass
        self.radius = radius
        self.name = name
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def mass(self):
        return self._mass

    @mass.setter
    def mass(self, value):
        if isinstance(value, Quantity):
            value = value.to("kg").value
        self._mass = value

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        if isinstance(value, Quantity):
            value = value.to("km").value
        self._radius = value

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
    pass


class Planet(Body):
    def __init__(
        self,
        mass,
        radius,
        semi_major_axis,
        period,
        eccentricity=0,
        inclination=pi / 2,
        argument_of_periastron=pi / 2,
        time_of_transit=0,
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
    def semi_major_axis(self, value):
        if isinstance(value, Quantity):
            value = value.to("km").value
        self._semi_major_axis = value

    @property
    def period(self):
        return self._period

    @property
    def p(self):
        return self._period

    @period.setter
    def period(self, value):
        if isinstance(value, Quantity):
            value = value.to("day").value
        self._period = value

    @property
    def eccentricity(self):
        return self._eccentricity

    @property
    def ecc(self):
        return self._eccentricity

    @eccentricity.setter
    def eccentricity(self, value):
        if isinstance(value, Quantity):
            value = value.to(1).value
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
    def inclination(self, value):
        if isinstance(value, Quantity):
            value = value.to("rad").value
        self._inclination = value

    @property
    def argument_of_periastron(self):
        return self._argument_of_periastron

    @property
    def w(self):
        return self._argument_of_periastron

    @argument_of_periastron.setter
    def argument_of_periastron(self, value):
        if isinstance(value, Quantity):
            value = value.to("rad").value
        self._argument_of_periastron = value

    @property
    def time_of_transit(self):
        return self._time_of_transit

    @property
    def t0(self):
        return self._time_of_transit

    @time_of_transit.setter
    def time_of_transit(self, value):
        if isinstance(value, Time):
            value = value.mjd
        self._time_of_transit = value
