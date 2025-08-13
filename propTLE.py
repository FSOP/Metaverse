import numpy as np
from sgp4.api import Satrec, jday
from datetime import datetime, timedelta, timezone

class propTLE:
    def __init__(self, tle_line1, tle_line2, start_time, end_time, step_seconds):
        self.tle_line1 = tle_line1
        self.tle_line2 = tle_line2
        self.start_time = start_time
        self.end_time = end_time
        self.step_seconds = step_seconds
        self.ephemeris = self.propagate_satellite(tle_line1, tle_line2, start_time, end_time, step_seconds)

    def propagate_satellite(self, tle_line1, tle_line2, start_time, end_time, step_seconds):
        satellite = Satrec.twoline2rv(tle_line1, tle_line2)
        num_steps = int((end_time - start_time).total_seconds() / step_seconds + 1 )
        # num_steps = int(duration_seconds // step_seconds) + 1
        ephemeris = np.zeros((num_steps, 7))  # [unix_time, x, y, z, vx, vy, vz]

        for i in range(num_steps):
            current_time = start_time + timedelta(seconds=i * step_seconds)
            unix_time = current_time.timestamp()
            jd, fr = jday(current_time.year, current_time.month, current_time.day,
                        current_time.hour, current_time.minute,
                        current_time.second + current_time.microsecond * 1e-6)
            error_code, position, velocity = satellite.sgp4(jd, fr)
            if error_code == 0:
                ephemeris[i] = [
                    unix_time,
                    position[0], position[1], position[2],
                    velocity[0], velocity[1], velocity[2]
                ]
            else:
                ephemeris[i] = [np.nan] * 7
        return ephemeris