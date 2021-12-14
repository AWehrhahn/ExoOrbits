"""
This module describes the datastructures for stars and planets
It also includes functions that are not specific for the orbit, 
but rather the individual bodies

Orbit calculations are performed in the orbit module
"""
import astropy.units as u
from astropy.constants import G, R
from astropy.coordinates import SkyCoord
from astropy.io.misc import yaml
from astropy.time import Time
from astropy.units.quantity import Quantity
from numpy import pi

from .dataclasses_quantity import dataclass


@dataclass
class DataclassIO:
    def to_dict(self) -> dict:
        fields = self.__dataclass_fields__
        data = {k: getattr(self, k) for k in fields.keys()}
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "DataclassIO":
        return cls(**data)

    def save(self, filename: str):
        data = self.to_dict()
        with open(filename, "w") as f:
            yaml.dump(data, f)

    @classmethod
    def load(cls, filename: str) -> "DataclassIO":
        with open(filename, "r") as f:
            data = yaml.load(f)
        return cls.from_dict(data)


@dataclass
class Body(DataclassIO):
    #:str: name of this object
    name: str = ""
    #:Quantity(kg): total mass of the object
    mass: Quantity[u.kg] = 0 * u.kg
    #:Quantity(km): radius of the disk
    radius: Quantity[u.km] = 0 * u.km

    @property
    @u.quantity_input
    def area(self) -> Quantity[u.km ** 2]:
        """Quantity(km**2): area of the disk"""
        return pi * self.radius ** 2

    @property
    @u.quantity_input
    def circumference(self) -> Quantity[u.km]:
        """Quantity(km): circumference of the disk"""
        return 2 * pi * self.radius

    @property
    @u.quantity_input
    def gravity_value(self) -> Quantity[u.m**3/u.s**2]:
        """Quantity(): specific gravity parameter for this object"""
        return G * self.mass

    @property
    @u.quantity_input
    def volume(self) -> Quantity[u.km ** 3]:
        """Quantity(km**3): volume of this object"""
        return 4 / 3 * pi * self.radius ** 3

    @property
    @u.quantity_input
    def density(self) -> Quantity[u.kg / u.km ** 3]:
        """Quantity(kg/km**3): mass density of the object"""
        return self.mass / self.volume


@dataclass
class Star(Body):
    # TODO: Surface gravity could be represented more accurately as u.LogUnit("cm/s**2")
    # And Metallicity as u.Unit("dex")

    #:Quantity(K): effective temperature of the star
    teff: Quantity[u.K] = 5700 * u.K
    #:Quantity(log(cgs)): surface gravity of the star
    logg: Quantity[u.one] = 4.4 * u.one
    #:Quantity(1): metalicity of the star, in log units relative to the sun
    monh: Quantity[u.one] = 0 * u.one
    #:Quantity(km/s): micro turbulence parameter
    vmic: Quantity[u.km / u.s] = 1 * u.km / u.s
    #:Quantity(km/s): macro turbulence parameter
    vmac: Quantity[u.km / u.s] = 2 * u.km / u.s
    #:Quantity(km/s): projected rotational velocity of the stellar surface
    vsini: Quantity[u.km / u.s] = 2 * u.km / u.s
    #:SkyCoord: coordinates of the star in the sky
    coordinates: SkyCoord = None
    #:Quantity(pc): distance of the star from the solar system
    distance: Quantity[u.pc] = 0 * u.pc
    #:Quantity(km/s): radial velocity of the star relative to the barycentric
    radial_velocity: Quantity[u.km / u.s] = 0 * u.km / u.s


@dataclass
class Planet(Body):
    #:Quantity(AU): semi major axis of the orbit, i.e. average distance
    sma: Quantity[u.AU] = 1 * u.AU
    #:Quantity(day): orbital period, i.e. 1 year of time on the planet
    period: Quantity[u.day] = 365.25 * u.day
    #:Quantity(1): eccentricity of the orbit, 0 for circular, 1 for parabolic
    ecc: Quantity[u.one] = 0 * u.one
    #:Quantity(deg): inclination of the orbit, where 90deg is transiting
    inc: Quantity[u.deg] = 90 * u.deg
    #:Quantity(deg): argument of periastron, where 90 deg is facing the observer
    omega: Quantity[u.deg] = 90 * u.deg
    #:Time: one known time of transit
    time_of_transit: Time = Time(0, format="mjd")
    #:Quantity(day): duration of the transit, from first contact to fourth contact
    transit_duration: Quantity[u.day] = 0 * u.day

    @property
    def t0(self) -> Time:
        return self.time_of_transit

    @property
    @u.quantity_input
    def surface_gravity(self) -> Quantity[u.m / u.s ** 2]:
        """Quantity(): surface gravity in normal (non logarithmic) units"""
        return self.gravity_value / self.radius ** 2

    @property
    def planet_class(self) -> str:
        """str: archetype of the planet, e.g. gas-giant, rocky"""
        if self.mass > 10 * u.M_earth:
            return "gas-giant"
        else:
            return "rocky"

    @property
    @u.quantity_input
    def atm_molar_mass(self) -> (u.g / u.mol):
        """Quantity: average molar mass of the atmosphere"""
        if self.planet_class == "gas-giant":
            # Hydrogen (e.g. for gas giants)
            atm_molar_mass = 2.5 * (u.g / u.mol)
        elif self.planet_class == "rocky":
            # dry air (mostly nitrogen)  (e.g. for terrestial planets)
            atm_molar_mass = 29 * (u.g / u.mol)
        else:
            raise ValueError
        return atm_molar_mass

    @property
    @u.quantity_input
    def atm_surface_pressure(self) -> u.bar:
        """Quantity: atmopshere pressure at the surface"""
        if self.planet_class == "gas-giant":
            p0 = 0.01 * u.bar
        elif self.planet_class == "rocky":
            p0 = 1 * u.bar
        else:
            raise ValueError
        return p0

    @u.quantity_input
    def equilibrium_temperature(self, stellar_teff: u.K) -> u.K:
        """
        Calculate the equilibrium temperature of the planet based on the 
        stellar radiation. Assuming no reflection.

        Parameters
        ----------
        stellar_teff : u.K
            effective temperature of the star

        Returns
        -------
        equi_temp: u.K
            equilibrium temperature of the planet
        """
        teff = ((pi * self.radius ** 2) / self.a ** 2) ** 0.25 * stellar_teff
        return teff

    @u.quantity_input
    def atm_scale_height(self, surface_temp: u.K) -> u.km:
        """
        Calculate the scale height of the atmosphere, based on the surface
        temperature of the planet. The surface temperature can for example
        be calculated using the equilibrium temperature.

        Parameters
        ----------
        surface_temp : u.K
            surface temperature of the planet

        Returns
        -------
        H: u.km
            scale height of the atmosphere
        """
        teff = surface_temp
        atm_scale_height = (
            R * teff * self.radius ** 2 / (G * self.mass * self.atm_molar_mass)
        )
        return atm_scale_height


if __name__ == "__main__":
    b = Body()
    s = Star()
    p = Planet(name="WASP107b")

    p.save("test.yaml")
    p2 = Planet.load("test.yaml")
    pass
