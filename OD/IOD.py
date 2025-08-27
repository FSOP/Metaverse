import sys, os
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_root)

import numpy as np
import MISC.constants as const  
from OD.Gauss import Gauss
from MISC.DBmanager import DBmanager
from OD.coordinate_converter import coordinate_converter 
from datetime import datetime


def iod_main(event_id):
    """
    Initial Orbit Determination main function (Gauss-only).
    Args:
        filename: path to observation file (CSV or whitespace-delimited)
        mu: gravitational parameter (km^3/s^2)
        rsite: (3,3) array of site position vectors in inertial frame (km)
    Returns:
        GaussResult namedtuple from gauss()
    """
    dbman = DBmanager()
    cc = coordinate_converter()
    obs_data = np.array(dbman.get_obs_data(event_id))

    # Load observation data: expects columns [RA, Dec, year, month, day, hour, min, sec]
    meas = obs_data[:, 2:4]  # [azimuth, elevation] in degrees
    time = obs_data[:, 0] # datevectors
    site = [float(o) for o in obs_data[0,1]. split('_')]
    site = { 'lat': site[0], 'lon': site[1], 'alt': site[2] }

    ra, dec, LTCs, Us = cc.aer_to_radec(meas[:, 0], meas[:, 1], site, time)

    obs_ecef = cc.geodetic_to_ecef(site['lat'], site['lon'], site['alt'])

    rsite, three_meas, three_times = [], [], []
    index_meas = np.linspace(70, len(meas)-135, 3, dtype=int)
    # index_meas = [70, 130, 190]
    for i in index_meas:
        three_meas.append([ra[i], dec[i]])
        three_times.append(time[i])
        this_site = Us[i].T @ obs_ecef
        rsite.append(this_site)

    # Input validation
    if len(three_meas) != 3 or len(three_times) != 3:
        raise ValueError("File must contain exactly 3 observations for Gauss method.")

    # Call Gauss method
    result = Gauss().gauss(three_meas, three_times, np.array(rsite).T/1e3, const.MU)
    print(f"Gauss method result: {result}")
    res = {'r2_epoch': three_times[1], 'r2': result[0], 'v2': result[1]}
    print(f"Result summary: {res}")
    return res, obs_data

# Example usage:
if __name__ == "__main__":
    
    # rsite = np.array([[x1, x2, x3], [y1, y2, y3], [z1, z2, z3]])  # Replace with your site vectors
    # filename = "obs_data.txt"  # Replace with your observation file path

    result = iod_main("20250818_060440_061654")
    print("Position:", result.sat_Pos)
    print("Velocity:", result.sat_Vel)
    print("Iterations:", result.Iterations)
    print("r1:", result.r1)
    print("r3:", result.r3)