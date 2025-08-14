from datetime import datetime, timedelta
from TLEmanager import TLEmanger
from propagators import propagators
import numpy as np

class CA_filter:
    def __init__(self):
        self.tle_manager = TLEmanger()
        self.propagator = propagators()
        pass

    def filter_outdated_tles(self, tle_data, start_time, end_time, pad_days):
        """
        Filters out TLEs that are older than 30 days from the current time.
        :param tle_data: List of tuples containing TLE data (epoch, line1, line2).
        :param current_time: Current datetime to compare against.
        :return: Filtered list of TLEs.
        """
        filtered_tles = []
        start_limit = start_time - timedelta(days=pad_days)
        end_limit = end_time + timedelta(days=pad_days)
        
        for norad, line1, line2, tle_epoch in tle_data:           
            if start_limit <= tle_epoch <= end_limit:
                filtered_tles.append((norad, line1, line2, tle_epoch))
        return filtered_tles
    
    def filter_altitude(self, tle_data, ref_line2, pad=0):
        filtered_tles = []
        ref_apogee, ref_perigee = self.tle_manager.compute_apogee_perigee(ref_line2)

        for i in range(len(tle_data)):
            norad, line1, line2, tle_epoch = tle_data[i]
            sat2_apogee, sat2_perigee = self.tle_manager.compute_apogee_perigee(line2)

            # Check if the apogee and perigee are within the specified pad
            if not((sat2_apogee < ref_perigee - pad) or (sat2_perigee > ref_apogee + pad)):
                filtered_tles.append((norad, line1, line2, tle_epoch))
        
        return filtered_tles
        
    def filter_orbitpath(self, tle_data, ref_line2, N_points=36, threshold=30):
        """
        Filters TLEs based on their orbit path.
        :param tle_data: List of tuples containing TLE data (norad, line1, line2, epoch) except reference sat.
        :param ref_line2: Reference TLE line2 for orbit path comparison.
        :param N_points: Number of points to sample along the orbit path.
        :param threshold: [km] Distance threshold for filtering.
        :return: Filtered list of TLEs.
        """
        filtered_tles = []
        N = np.linspace(0, 2 * np.pi, N_points)  # Sample points along the orbit path

        # Generate a reference orbit path using the reference satellite's TLE                        
        ref_orb = self.tle_manager.extract_elements(ref_line2)
        ref_r = self.propagator.orbit_path(ref_orb, N)  # Reference orbit path        
        
        # Iterate through the TLE except the reference TLE
        for i in range(len(tle_data)):
            norad, line1, line2, tle_epoch = tle_data[i]
            orbit = self.tle_manager.extract_elements(line2)

            # Compare the orbit with the reference orbit
            sat2_r = self.propagator.orbit_path(orbit, N)  # Assuming theta = 0 for initial comparison

            # Compare the two orbits
            distance = np.linalg.norm(ref_r[:, :, None] - sat2_r[:, None, :], axis=0)
            flag_ca = np.any(distance < threshold)         
            
            if flag_ca:
                filtered_tles.append((norad, line1, line2, tle_epoch))

        return filtered_tles

        
