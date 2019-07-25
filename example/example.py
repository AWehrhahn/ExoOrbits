import numpy as np
import matplotlib.pyplot as plt
from astropy import constants as const
from astropy import units as q

from ExoOrbit import Orbit, Body
from ExoOrbit.library import Sun, Earth, Earth_circular

if __name__ == "__main__":
    # Earth parameters
    p = q.year.to("day")

    orbit = Orbit(Sun, Earth)

    t = np.linspace(-p/2, p/2, 100000)

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

    y = p
    plt.plot(t, y)

    # plt.hlines(r_s + r_p, 0, p, colors="r")
    plt.vlines((t1, t2, t3, t4), np.min(y), np.max(y), colors="r")
    plt.vlines(t0, np.min(y), np.max(y), colors="g")

    plt.xlabel("Day")
    plt.ylabel("Radial velocity [m/s]")

    plt.show()

    pass
