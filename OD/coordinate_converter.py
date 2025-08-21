import numpy as np
from datetime import datetime
from astropy.time import Time
from HPOP.coordinate_systems import compute_iers_matrices
from HPOP.eop import EOPManager
from HPOP.time_utils import convert_time_scales

class coordinate_converter:
    def __init__(self):
        self.eop_manager = EOPManager()
        # self.time_utils = convert_time_scales()
        pass
        
    def eci_to_ecef(self, r_eci, epoch):
        """
        Convert ECI (TEME) position to ECEF at given epoch.
        r_eci: [x, y, z] in meters
        epoch: datetime object (UTC)
        Returns: [x, y, z] in meters (ECEF)
        """
        # Compute GMST (Greenwich Mean Sidereal Time) in radians
        mjd_utc = Time(epoch, scale='utc').mjd

        x_pole, y_pole, ut1_utc, lod, dpsi, deps, dx_pole, dy_pole, tai_utc = \
            self.eop_manager.get_eop_values(mjd_utc)        
        
        mjd_ut1, mjd_tt, mjd_tdb = convert_time_scales(mjd_utc, self.eop_manager)

        E = compute_iers_matrices(mjd_tt, mjd_ut1, x_pole, y_pole)
        E_transpose = E.T

        return E @ r_eci

    def geodetic_to_ecef(self, lat_deg, lon_deg, alt_m):
        # WGS-84 parameters
        a = 6378137.0
        e2 = 6.69437999014e-3
        lat = np.radians(lat_deg)
        lon = np.radians(lon_deg)
        N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
        x = (N + alt_m) * np.cos(lat) * np.cos(lon)
        y = (N + alt_m) * np.cos(lat) * np.sin(lon)
        z = (N * (1 - e2) + alt_m) * np.sin(lat)
        return np.array([x, y, z])

    def ecef_to_enu(self, r_ecef, obs_ecef, lat_deg, lon_deg):
        # ENU rotation matrix
        lat = np.radians(lat_deg)
        lon = np.radians(lon_deg)
        R = np.array([
            [-np.sin(lon),              np.cos(lon),               0],
            [-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon),  np.cos(lat)],
            [ np.cos(lat)*np.cos(lon),  np.cos(lat)*np.sin(lon),  np.sin(lat)]
        ])
        return R @ (r_ecef - obs_ecef)

    def aer_from_enu(self, enu):
        east, north, up = enu
        rng = np.linalg.norm(enu)
        az = np.degrees(np.arctan2(east, north)) % 360
        el = np.degrees(np.arcsin(up / rng))
        return az, el, rng

    # Example usage:
    if __name__ == "__main__":
        # Satellite ECI position (meters)
        r_eci = np.array([7000e3, 0, 0])
        epoch = datetime(2025, 8, 21, 12, 0, 0)
        # Observer geodetic position
        lat_deg, lon_deg, alt_m = 37.5, 127.0, 100.0
        obs_ecef = geodetic_to_ecef(lat_deg, lon_deg, alt_m)
        # Convert satellite ECI to ECEF
        r_ecef = eci_to_ecef(r_eci, epoch)
        # Convert to ENU
        enu = ecef_to_enu(r_ecef, obs_ecef, lat_deg, lon_deg)
        # Calculate azimuth, elevation, range
        az, el, rng = aer_from_enu(enu)
        print(f"Azimuth: {az:.2f} deg, Elevation: {el:.2f} deg, Range: {rng/1000:.2f} km")