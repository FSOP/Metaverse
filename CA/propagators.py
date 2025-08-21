
import numpy as np
from MISC.constants import constants as const


class propagators:
    """
    This class contains methods for propagating TLE data.
    """
    def __init__(self):
        pass

    def simple_twobody(self, orbit, theta):
        """
        Propagates the TLE data for a given NORAD ID between start and end times.
        :param norad: NORAD ID of the satellite.
        :param start_time: Start time for propagation.
        :param end_time: End time for propagation.
        :return: List of propagated states.
        """
        # Implementation of TLE propagation logic goes here
        sma = orbit['sma']  # [km] Semi-major axis
        ecc = orbit['ecc']  # Eccentricity
        inc = orbit['inc']  # [rad] Inclination
        raan = orbit['raan']  # Right Ascension of Ascending Node
        argp = orbit['argp']  # Argument of Perigee
        nu = orbit['nu']      # [rad] True Anomaly

        # mean motion
        n = np.sqrt(const.MU / sma**3)  # Mean motion in rad/s
        
        E0 = 2* np.arctan(np.tan(nu /2) / np.sqrt((1 + ecc) / (1 - ecc)))  # Eccentric anomaly at epoch
        M0 = E0 - ecc * np.sin(E0)  # Mean anomaly at
        M = M0 + n * theta  # Mean anomaly at time theta

        # Solve Kepler's equation for E
        E = M  # Initial guess
        for _ in range(10):  # Iterate to solve Kepler's equation
            E_next = E + (M - (E - ecc * np.sin(E))) / (1 - ecc * np.cos(E))
            if np.abs(E_next - E) < 1e-10:  # Convergence criterion
                break
            E = E_next

        # True anomaly
        nu_new = 2 * np.arctan2(np.sqrt(1 + ecc) * np.sin(E / 2), 
                                np.sqrt(1 - ecc) * np.cos(E / 2))

        # 미완성
        # 

    def orbit_path(self, orbit, theta):
        """
        Computes the orbit path for a given TLE.
        :param orbit: Dictionary containing orbital elements.
        :param theta: [rad] True anomaly at which to compute the orbit path. 
        :return: Orbit path data.
        """
        # Implementation of orbit path computation logic goes here        
        ecc = orbit['e']  # Eccentricity
        inc = orbit['i']  # [rad] Inclination
        raan = orbit['Om']  # [rad] Right Ascension of Ascending Node
        argp = orbit['w']  # [rad] Argument of Perigee
        n = orbit['n']      # Mean motion in rad/s

        # Compute the position in the perifocal coordinate system
        n_rad = n * 2 * np.pi / 86400  # Convert mean motion to rad/s
        a = (const.MU / n_rad**2)**(1/3)  # Semi-major axis [km]

        theta = theta.ravel()
        r_pf = (a * (1 - ecc**2)) / (1 + ecc * np.cos(theta)).ravel() # Radius in perifocal coordinates
        r_pqw = np.vstack([
            r_pf * np.cos(theta),
            r_pf * np.sin(theta),
            np.zeros_like(theta)
        ])

        Rz_Om = np.array([
            [np.cos(raan), -np.sin(raan), 0],
            [np.sin(raan), np.cos(raan), 0],
            [0, 0, 1]
        ])
        Rx_i = np.array([
            [1, 0, 0],
            [0, np.cos(inc), -np.sin(inc)],
            [0, np.sin(inc), np.cos(inc)]
        ])
        Rz_w = np.array([
            [np.cos(argp), -np.sin(argp), 0],
            [np.sin(argp), np.cos(argp), 0],
            [0, 0, 1]
        ])
        R = Rz_Om @ Rx_i @ Rz_w  # Rotation matrix from perifocal to ECI coordinates
        r_eci = R @ r_pqw

        return r_eci             