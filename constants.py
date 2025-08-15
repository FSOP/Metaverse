
class constants:
    """
    This class contains constants used throughout the metaverse project.
    """
    # Define any constants here
    EARTH_RADIUS_KM = 6371.0  # [km] Radius of the Earth in kilometers
    TLE_EPOCH_FORMAT = "%Y-%m-%dT%H:%M:%S"  # Format for TLE epoch timestamps
    MAX_TLE_AGE_DAYS = 30  # Maximum age of TLEs in days before they are considered outdated
    MU = 398600.4418  # [km^3/s^2] Standard gravitational parameter for Earth in 
    J2 = 1.08262668e-3  # [km^5/s^2] Second zonal harmonic coefficient for Earth