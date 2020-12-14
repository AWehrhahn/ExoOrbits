import astropy.units as u
from astropy.constants import G, R
from astropy.coordinates import SkyCoord
from astropy.io.misc import yaml
from astropy.time import Time
from numpy import pi

from .util import CollectionFactory, resets_cache, time_input

# TODO: pylint shows a bunch of errors, since it can't see the members of Body
# that are added due to the decorator


@CollectionFactory
class Body:
    # fmt: off
    _fields = [
        ("mass", [], u.kg, 0 * u.kg, "Mass"),
        ("radius", [], u.km, 0 * u.km, "Radius")
    ]
    # fmt: on

    def __init__(self, **kwargs):
        self._orbit = None
        self.name = ""
        for name, _, _, default, _ in self._fields:
            setattr(self, name, default)

        for k, v in kwargs.items():
            try:
                setattr(self, k, v)
            except Exception:
                unit = getattr(self, k).unit
                value = v.value if hasattr(v, "value") else v
                print(f"WARNING: Setting units of {k} to {unit}")
                setattr(self, k, value * unit)

    def __str__(self):
        return self.name

    def to_dict(self):
        data = {}
        names = [f[0] for f in self._fields]
        for name in names:
            data[name] = getattr(self, name)
        return data

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

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

    def save(self, fname):
        data = {key: getattr(self, key) for key in self._names}
        data["name"] = self.name
        with open(fname, "w") as file:
            yaml.dump(data, stream=file)

    @classmethod
    def load(cls, fname):
        with open(fname, "r") as file:
            data = yaml.load(file)

        return cls(**data)


@CollectionFactory
class Star(Body):
    # TODO: Surface gravity could be represented more accurately as u.LogUnit("cm/s**2")
    # And Metallicity as u.Unit("dex")

    # fmt: off
    _fields = Body._fields + [
        ("effective_temperature", ["teff"], u.K, 5000 * u.K, "effective temperature"),
        ("surface_gravity", ["logg"], u.one, 4 * u.one,  "surface gravity in log(cgs) units"),
        ("metallicity", ["monh", "feh"], u.one, 0 * u.one, "metallicity in dex relative to base abundances"),
        ("turbulence_velocity", ["vturb", "vmac"], u.km/u.s, 1 * u.km / u.s, "(macro) turbulence velocity"),
        ("micro_turbulence_velocity", ["vmic"], u.km / u.s, 1 * u.km / u.s, "micro turbulence velocity"),
        ("coordinates", [], SkyCoord, SkyCoord(0, 0, unit=u.deg), "coordinates on the sky"),
        ("distance", [], u.parsec, 0 * u.parsec, "distance from the Sun"),
        ("radial_velocity", ["rv"], u.km / u.s, 0 * u.km/u.s, "radial velocity of the star"),
        ("rotational_velocity_projected", ["vsini"], u.km / u.s, 0 * u.km / u.s, "projected rotational velocity of the stellar surface"),
    ]
    # fmt: on

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


@CollectionFactory
class Planet(Body):
    # fmt: off
    _fields = Body._fields + [
        ("semi_major_axis", ["sma", "a"], u.km, 1 * u.AU, "semi major axis"),
        ("orbital_period", ["period", "p"], u.day, 365 * u.day, "orbital period"),
        ("eccentricity", ["e", "ecc"], u.one, 0 * u.one, "eccentricity"),
        ("inclination", ["inc", "i"], u.deg, 90 * u.deg, "inclination"),
        ("argument_of_periastron", ["w"], u.deg, 90 * u.deg, "argument of periastron"),
        ("time_of_transit", ["t0"], Time, Time(0, format="mjd"), "time of transit"),
        ("transit_duration", [], u.day, 0 * u.day, "duration of transit"),
    ]
    # fmt: on

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._surface_gravity = None

    @property
    def surface_gravity(self):
        if self._surface_gravity is None:
            return self.gravity_value / self.radius ** 2
        else:
            return self._surface_gravity

    @surface_gravity.setter
    @u.quantity_input(value=u.cm / u.s ** 2)
    def surface_gravity(self, value):
        self._surface_gravity = value

    @property
    def atm_molar_mass(self):
        if self.mass > 10 * u.M_earth:
            # Hydrogen (e.g. for gas giants)
            atm_molar_mass = 2.5 * (u.g / u.mol)
        else:
            # dry air (mostly nitrogen)  (e.g. for terrestial planets)
            atm_molar_mass = 29 * (u.g / u.mol)
        return atm_molar_mass

    @u.quantity_input(stellar_teff=u.K)
    def teff_from_stellar(self, stellar_teff):
        teff = ((pi * self.radius ** 2) / self.a ** 2) ** 0.25 * stellar_teff
        return teff

    @u.quantity_input(stellar_teff=u.K)
    def atm_scale_height(self, stellar_teff):
        teff = self.teff_from_stellar(stellar_teff)
        atm_scale_height = (
            R * teff * self.radius ** 2 / (G * self.mass * self.atm_molar_mass)
        )
        return atm_scale_height
