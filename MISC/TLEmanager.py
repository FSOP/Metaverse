# This file is part of the Metaverse project.
# It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution.
# No part of the Metaverse project, including this file, may be copied, modified,
# propagated, or distributed except according to the terms contained in the LICENSE file.
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from MISC.DBmanager import DBmanager
from datetime import datetime, timedelta, timezone
import MISC.constants as const
import math
import numpy as np
import requests


class TLEmanager:
    def download_and_insert_celestrak_tle(self, url, save_path):
        """
        Downloads TLE data from Celestrak, saves to txt, and inserts into DB.
        :param url: Celestrak TLE URL
        :param save_path: Path to save the downloaded TLE txt file
        """
        print(f"Downloading TLE from {url} ...")
        response = requests.get(url)
        response.raise_for_status()
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"TLE saved to {save_path}")
        self.insert_tles_from_file(save_path)
        print("TLEs inserted into DB.")
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
        # TLE epochs are given in UTC; return timezone-aware UTC datetime
        return dt.replace(tzinfo=timezone.utc)
    
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
            # ensure tle_epoch is timezone-aware UTC for safe comparison
            try:
                if isinstance(tle_epoch, datetime) and tle_epoch.tzinfo is None:
                    tle_epoch = tle_epoch.replace(tzinfo=timezone.utc)
            except Exception:
                # if tle_epoch is not a datetime, skip that entry
                continue

            if start_limit <= tle_epoch <= end_limit:
                filtered_tles.append((norad, line1, line2, tle_epoch))
        return filtered_tles
    
        
    # Insert TLEs from a file into the database
    def insert_tles_from_file(self, filename):
        # Use bulk insert to speed up DB writes
        self.db_manager.flush_TLE_data()  # Clear existing TLE data

        with open(filename, 'r') as f:
            lines = f.readlines()

        rows = []
        # Process every 3 lines as one TLE set
        for i in range(0, len(lines), 3):
            if i + 2 >= len(lines):
                break  # incomplete TLE set
            sat_name = lines[i][1:].strip()
            line1 = lines[i + 1].strip()
            line2 = lines[i + 2].strip()

            # Extract NORAD number from line1 (columns 3-7)
            try:
                norad = int(line1[2:7])  # Extracts the NORAD number from line1
            except ValueError:  # Handle case where NORAD number is not an integer
                norad = 99999

            tle_epoch = line1[18:32].strip()  # Extract epoch from line1]
            epoch_datetime = self.tlepoch_to_datetime(tle_epoch)
            source = "CELESTRAK"
            rows.append((source, epoch_datetime, norad, line1, line2, sat_name))

            if i % 10000 == 0:
                print(f"Prepared TLE row for NORAD {norad}: {sat_name}")

        # Bulk insert for performance
        self.db_manager.insert_TLEs_bulk(rows)
        print(f"Inserted {len(rows)} TLEs into DB (bulk)")


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

    def download_tle_and_save(self, fname=None, chk_saveDb=True, source='auto'):
        """
        Download TLEs and save to file. `source` can be 'auto', 'spacetrack', or 'celestrak'.
        - 'auto': use space-track when credentials provided, otherwise celestrak
        - 'spacetrack': explicitly use space-track (login required)
        - 'celestrak': explicitly use celestrak
        """
        # Credentials and URLs
        space_user = os.getenv('SPACE_TRACK_USER')
        space_pass = os.getenv('SPACE_TRACK_PASSWORD')
        # Fallback: read from env.py if env vars not set
        if not space_user or not space_pass:
            try:
                import env as _e  # type: ignore
                space_user = space_user or getattr(_e, 'SPACE_TRACK_USER', None)
                space_pass = space_pass or getattr(_e, 'SPACE_TRACK_PASSWORD', None)
            except Exception:
                pass
        celestrak_url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"

        if fname is None:
            # use UTC timestamp for filenames
            fname = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        save_path = os.path.join(_project_root, "TLEs", f"tle_{fname}.txt")

        # Decide whether to use space-track
        if source == 'spacetrack':
            use_spacetrack = True
        elif source == 'celestrak':
            use_spacetrack = False
        else:  # auto
            use_spacetrack = bool(space_user and space_pass)

        if use_spacetrack:
            print("Using space-track.org to download TLEs")
            session = requests.Session()
            login_url = "https://www.space-track.org/ajaxauth/login"
            payload = {"identity": space_user, "password": space_pass}
            r = session.post(login_url, data=payload, timeout=30)
            if r.status_code != 200 or 'Login' in r.text:
                print("space-track login failed, falling back to Celestrak")
            else:
                # Try several likely endpoints in order
                # Use gp class with 3LE format (includes name + 2 TLE lines) for bulk TLEs
                query_candidates = [
                    "https://www.space-track.org/basicspacedata/query/class/gp/EPOCH/%3Enow-30/orderby/NORAD_CAT_ID,EPOCH/format/3le",
                    "https://www.space-track.org/basicspacedata/query/class/tle_latest/format/tle",
                    "https://www.space-track.org/basicspacedata/query/class/tle/format/tle",
                ]
                for query_url in query_candidates:
                    try:
                        r2 = session.get(query_url, timeout=120)
                        if r2.status_code == 200 and r2.text:
                            with open(save_path, "w", encoding="utf-8") as f:
                                f.write(r2.text)
                            print(f"TLE saved to {save_path} (space-track) via {query_url}")
                            if chk_saveDb:
                                self.insert_tles_from_file(save_path)
                            return save_path
                        else:
                            body = (r2.text or '')[:400]
                            print(f"space-track query {query_url} returned {r2.status_code}; body: {body!r}")
                    except Exception as e:
                        print(f"space-track query {query_url} failed: {e}")
                print("space-track queries exhausted; falling back to Celestrak")

        # Use Celestrak as fallback or explicitly requested
        print("Using Celestrak to download TLEs")
        _headers = {"User-Agent": "Mozilla/5.0 (compatible; TLEmanager/1.0)"}
        response = requests.get(celestrak_url, timeout=60, headers=_headers)
        response.raise_for_status()
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(response.text)

        if chk_saveDb:
            self.insert_tles_from_file(save_path)

        print(f"TLE downloaded and saved to {save_path}")
        return save_path



# module intended for import; example usage removed