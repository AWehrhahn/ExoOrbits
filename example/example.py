import numpy as np
import matplotlib.pyplot as plt
from astropy import constants as const
from astropy import units as q

from exoorbit import Orbit, Body, Star, Planet
from exoorbit.library import Sun, Earth, Earth_circular

if __name__ == "__main__":
    # load data from the database
    star = Star("WASP-107")
    planet = Planet("WASP-107 b")
    orbit = Orbit("WASP-107", "b")

    # Earth parameters
    orbit = Orbit(Sun, Earth)
    p = Earth.period.to_value("day")
    t0 = Earth.time_of_transit

    t = t0 + np.linspace(-p/2, p/2, 100000) * q.day

    e = orbit.eccentric_anomaly(t0)
    f = orbit.true_anomaly(t0)

    m = orbit.mean_anomaly(t)
    e = orbit.eccentric_anomaly(t)
    f = orbit.true_anomaly(t)
    d = orbit.transit_depth(t)
    p = orbit.phase_angle(t)
    r = orbit.projected_radius(t)
    rv = orbit.radial_velocity_planet(t)

    t0 = orbit.time_primary_transit()
    # t0 = orbit.time_secondary_eclipse()
    t1 = orbit.first_contact()
    t2 = orbit.second_contact()
    t3 = orbit.third_contact()
    t4 = orbit.fourth_contact()
    print(t1)
    print(t4)

    y = p.to_value("rad")
    plt.plot(t.mjd, y)

    # plt.hlines(r_s + r_p, 0, p, colors="r")
    plt.vlines((t1.mjd, t2.mjd, t3.mjd, t4.mjd), np.min(y), np.max(y), colors="r")
    plt.vlines(t0.mjd, np.min(y), np.max(y), colors="g")

    plt.xlabel("Day")
    plt.ylabel("Radial velocity [m/s]")

    plt.show()

    pass
