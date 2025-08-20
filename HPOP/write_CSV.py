import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def save_ephemeris_to_csv(times, states, filename="ephemeris.csv", sat_id="SAT1"):
    """
    Save propagated satellite ephemeris to CSV.

    Parameters
    ----------
    times : list of datetime
        List of time points (datetime objects).
    states : np.ndarray
        Array of shape (N,6) with columns [x, y, z, vx, vy, vz] (km, km/s).
    filename : str
        Output CSV filename.
    sat_id : str
        Satellite identifier.
    """

    # Norm of position vector (rho) and velocity magnitude
    rho = np.linalg.norm(states[:, 0:3], axis=1)   # km
    speed = np.linalg.norm(states[:, 3:6], axis=1) # km/s

    # Build DataFrame
    df = pd.DataFrame({
        "satellite_id": sat_id,
        "time_utc": [t.strftime("%Y-%m-%d %H:%M:%S") for t in times],
        "x_km": states[:, 0],
        "y_km": states[:, 1],
        "z_km": states[:, 2],
        "vx_kms": states[:, 3],
        "vy_kms": states[:, 4],
        "vz_kms": states[:, 5],
        "rho_km": rho,
        "speed_kms": speed,
    })

    df.to_csv(filename, index=False)
    print(f"✅ Ephemeris saved to {filename} with {len(df)} records.")


# ==== Example usage ====
if __name__ == "__main__":
    # Example data (10 points)
    start = datetime(2025, 1, 1, 0, 0, 0)
    times = [start + timedelta(seconds=10*i) for i in range(10)]

    # Random states: (x,y,z,vx,vy,vz)
    states = np.random.randn(10, 6) * 1000  # Just dummy example

    save_ephemeris_to_csv(times, states, "test_ephemeris.csv", sat_id="MY_SAT")
