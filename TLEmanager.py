# This file is part of the Metaverse project.
# It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution.
# No part of the Metaverse project, including this file, may be copied, modified,
# propagated, or distributed except according to the terms contained in the LICENSE file.    

# metaverse/TLEmanager.py
from DBmanager import DBmanager
from datetime import datetime, timedelta


class TLEmanger:
    db_manager = None

    def __init__(self):
        self.db_manager = DBmanager()


    def tle_epoch_to_datetime(self, epoch_str):
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
                norad = int(line1.split()[1][:-1])  # Extracts the NORAD number from line1
            except ValueError:  # Handle case where NORAD number is not an integer
                norad = 99999

            tle_epoch = line1.split()[3]
            epoch_datetime = self.tle_epoch_to_datetime(tle_epoch)
            source = "CELESTRAK"
            self.db_manager.insert_TLE_data(source, epoch_datetime, norad, line1, line2, sat_name)

            if i % 10000 == 0:
                print(f"Inserted TLE for NORAD {norad}: {sat_name}")


if __name__ == "__main__":
    tle_manager = TLEmanger()
    tle_manager.all_tles()  # Fetch all TLEs from the database
    # tle_path = "/home/user1229/metaverse/TLEs/20250813_TLE.txt"  # Replace with your TLE file path
    # tle_manager.insert_tles_from_file(tle_path)  # Replace with your TLE file path

    print("end")