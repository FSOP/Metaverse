# This file is part of the Metaverse project.
# It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution.
# No part of the Metaverse project, including this file, may be copied, modified,
# propagated, or distributed except according to the terms contained in the LICENSE file.    

# metaverse/TLEmanager.py
from MISC.DBmanager import DBmanager
from datetime import datetime, timedelta
from MISC.constants import constants as const
import math
import numpy as np

class TLEmanger:
    db_manager = None

    def __init__(self):
        self.db_manager = DBmanager()

    def tlepoch_to_datetime(self, epoch_str):
        # TLE epoch format: YYDDD.DDDDDDDD (YY = year, DDD = day of year)
        epoch = float(epoch_str)
        year = int(epoch // 1000)
        day_of_year = epoch % 1000

        # TLE years: 57-99 = 1957-1999, 00-56 = 2000-2056
        if year < 57:
            year += 2000
        else:
            year += 1900

        # Get the date for Jan 1 of the year, then add day_of_year-1 days
        dt = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
        return dt
    
    def all_tles(self):
        """
        Fetches all TLEs from the database.
        :return: List of tuples containing (norad, line1, line2).
        """
        tle_data = self.db_manager.get_all_TLEs()
        return tle_data
        # print("pause")
    
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
    

        
    # Insert TLEs from a file into the database
    def insert_tles_from_file(self, filename):
        self.db_manager.flush_TLE_data()  # Clear existing TLE data

        with open(filename, 'r') as f:
            lines = f.readlines()

        # Process every 3 lines as one TLE set
        for i in range(0, len(lines), 3):
            if i+2 >= len(lines):
                break  # incomplete TLE set
            sat_name = lines[i][1:].strip()
            line1 = lines[i+1].strip()
            line2 = lines[i+2].strip()

            # Extract NORAD number from line1 (columns 3-7)
            try:
                norad = int(line1[2:7])  # Extracts the NORAD number from line1
            except ValueError:  # Handle case where NORAD number is not an integer
                norad = 99999

            tle_epoch = line1[18:32].strip()  # Extract epoch from line1]
            epoch_datetime = self.tlepoch_to_datetime(tle_epoch)
            source = "CELESTRAK"
            self.db_manager.insert_TLE_data(source, epoch_datetime, norad, line1, line2, sat_name)

            if i % 10000 == 0:
                print(f"Inserted TLE for NORAD {norad}: {sat_name}")

    def compute_apogee_perigee(self, line2):
        """
        Computes the apogee and perigee from TLE data.
        :return: List of tuples containing (norad, apogee, perigee).
        """
        orbit = self.extract_elements(line2)
        sma = orbit['a']  # Semi-major axis [km]
        ecc = orbit['e']  # Eccentricity        
        apogee = sma * (1 + ecc) - const.EARTH_RADIUS_KM  # [km] Apogee altitude
        perigee = sma * (1 - ecc) - const.EARTH_RADIUS_KM  # [km] Perigee altitude

        return apogee, perigee

    
    def extract_elements(self, line2):
        """
        Extracts elements from TLE data.
        :param tle_data: List of tuples containing TLE data.
        :return: List of tuples containing (norad, line1, line2).
        """
        # line2 = tle_data[2]

        orbit = {}        
        orbit['i'] = np.radians(float(line2[8:16].strip()))     # inclination [rad]        
        orbit['Om'] = np.radians(float(line2[17:25].strip()))   # RAAN [rad]        
        orbit['e'] = float("0." + line2[26:33].strip())         # eccentricity        
        orbit['w'] = np.radians(float(line2[34:42].strip()))    # argument of perigee [rad]        
        orbit['n'] = float(line2[52:63].strip())                # mean motion [rev/day]
        orbit['ma'] = np.radians(float(line2[43:52].strip()))   # mean anomaly [rad]
        # orbit['n_dot'] = float(line1[33:43].strip())  # mean motion derivative [rev/day^2]
        
        n_rad = orbit['n'] * 2 * math.pi / 86400  # rad/s        
        orbit['a'] = (const.MU / n_rad**2)**(1/3)
            
        return orbit

    def extract_bstar(self,line1):
        """
        Extracts the Bstar drag term from TLE line 1.
        Bstar is in columns 54–61 (8 chars).
        Example field: '-11606-4' -> mantissa = -0.11606, exponent = -4
        """
        bstar_field = line1[53:61]  # Python uses 0-based indexing

        if len(bstar_field) != 8:
            raise ValueError("TLE line length insufficient to parse Bstar (need cols 54:61).")

        mantissa_sign = bstar_field[0] if bstar_field[0] in '+-' else '+'
        mantissa_digits = bstar_field[1:6]
        exp_sign = bstar_field[6]
        exp_digit = bstar_field[7]

        m = float(mantissa_digits) / 1e5
        if mantissa_sign == '-':
            m = -m
        e = int(exp_sign + exp_digit)

        bstar = m * 10 ** e
        return bstar


if __name__ == "__main__":
    tle_manager = TLEmanger()
    # tle_manager.all_tles()  # Fetch all TLEs from the database
    tle_path = "/home/user1229/metaverse/TLEs/20250815TLE.txt"  # Replace with your TLE file path
    tle_manager.insert_tles_from_file(tle_path)  # Replace with your TLE file path

    print("end")