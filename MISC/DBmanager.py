import mariadb
import requests
import csv

class DBmanager:
    user       = "mverse"
    password   = "dlstn"
    host       = "localhost"
    port       = 3306
    database   = "Metaverse"

    def __init__(self):
        self.conn = mariadb.connect(
            user        =self.user,
            password    =self.password,
            host        =self.host,
            port        =self.port,
            database    =self.database
        )
        self.curr = self.conn.cursor()
    
    def get_unique_obsid(self):
        query = "SELECT DISTINCT(event_id) FROM observations"
        self.curr.execute(query)
        return self.curr.fetchall()
    
    def get_obs_data(self, event_id):
        query = "SELECT * FROM observations WHERE event_id = ?"
        self.curr.execute(query, (event_id,))
        return self.curr.fetchall()

    def insert_obs_data(self, epoch, azimuth, elevation, rng, obs_site, event_id, aux):
        query = "INSERT INTO observations (epoch, obs_site, azimuth, elevation, sat_range, event_id, aux_data) VALUES (?, ?, ?, ?, ?, ?, ?)"
        self.curr.execute(query, (epoch, obs_site, azimuth, elevation, rng, event_id, aux))
        self.conn.commit()

    def get_single_TLE(self, norad):
        query = "SELECT norad, line1, line2, creation_date FROM TLE_DATA WHERE norad = ?"
        self.curr.execute(query, (norad,))
        return self.curr.fetchall()

    def get_all_TLEs(self):
        query = "SELECT norad, line1, line2, creation_date FROM TLE_DATA order by norad"
        self.curr.execute(query)
        return self.curr.fetchall()

    def insert_TLE_data(self, str_source, dt_creation_date, int_norad, str_line1, str_line2, str_sat_name):
        query = "INSERT INTO TLE_DATA (TLE_source, creation_date, norad, line1, line2, sat_name) VALUES (?, ?, ?, ?, ?, ?)"
        self.curr.execute(query, (str_source, dt_creation_date, int_norad, str_line1, str_line2, str_sat_name))
        self.conn.commit()

    def flush_TLE_data(self):
        query = "DELETE FROM TLE_DATA"
        self.curr.execute(query)
        self.conn.commit()

    def flush_SATCAT_data(self):
        query = "DELETE FROM SATCAT"
        self.curr.execute(query)
        self.conn.commit()

    def get_tle_count(self):
        query = "SELECT COUNT(*) FROM TLE_DATA"
        self.curr.execute(query)
        return self.curr.fetchone()[0]
    
    def insert_CA(self, norad1, norad2, name1, name2, tca, closest_distance_km):
        query = "INSERT INTO CA (sat1_norad, sat1_name, sat2_norad, sat2_name, tca, miss_distance, probability) VALUES (?, ?, ?, ?, ?, ?, ?)"
        self.curr.execute(query, (norad1, name1, norad2, name2, tca, closest_distance_km, 0.0))
        self.conn.commit()
    
    def download_and_insert_satcat(self):
        url = "https://celestrak.org/pub/satcat.csv"
        response = requests.get(url)
        response.raise_for_status()
        csv_data = response.content.decode("utf-8").splitlines()
        reader = csv.DictReader(csv_data)

        if len(csv_data) < 1000:
            print(f"Failed to download SATCAT")
            return

        print("SATCAT download complete")
        self.flush_SATCAT_data()

        rows = []
        for row in reader:
            rows.append((
                row["OBJECT_NAME"],
                row["OBJECT_ID"],
                int(row["NORAD_CAT_ID"]),
                row["OBJECT_TYPE"],
                row["OPS_STATUS_CODE"],
                row["OWNER"],
                row["LAUNCH_DATE"] or None,
                row["LAUNCH_SITE"],
                row["DECAY_DATE"] or None,
                float(row["PERIOD"]) if row["PERIOD"] else None,
                float(row["INCLINATION"]) if row["INCLINATION"] else None,
                float(row["APOGEE"]) if row["APOGEE"] else None,
                float(row["PERIGEE"]) if row["PERIGEE"] else None,
                float(row["RCS"]) if row["RCS"] else None,
                row["DATA_STATUS_CODE"],
                row["ORBIT_CENTER"],
                row["ORBIT_TYPE"]
            ))

        query = """
        INSERT INTO SATCAT (
            OBJECT_NAME, OBJECT_ID, NORAD_CAT_ID, OBJECT_TYPE, OPS_STATUS_CODE, OWNER,
            LAUNCH_DATE, LAUNCH_SITE, DECAY_DATE, PERIOD, INCLINATION, APOGEE, PERIGEE,
            RCS, DATA_STATUS_CODE, ORBIT_CENTER, ORBIT_TYPE
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        self.curr.executemany(query, rows)
        self.conn.commit()
        print("SATCAT DB insert complete")
    
    def insert_SATCAT(self, satcat_row):
        query = """
        INSERT INTO SATCAT (
            OBJECT_NAME, OBJECT_ID, NORAD_CAT_ID, OBJECT_TYPE, OPS_STATUS_CODE, OWNER,
            LAUNCH_DATE, LAUNCH_SITE, DECAY_DATE, PERIOD, INCLINATION, APOGEE, PERIGEE,
            RCS, DATA_STATUS_CODE, ORBIT_CENTER, ORBIT_TYPE
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        self.curr.execute(query, (
            satcat_row["OBJECT_NAME"],
            satcat_row["OBJECT_ID"],
            int(satcat_row["NORAD_CAT_ID"]),
            satcat_row["OBJECT_TYPE"],
            satcat_row["OPS_STATUS_CODE"],
            satcat_row["OWNER"],
            satcat_row["LAUNCH_DATE"] or None,
            satcat_row["LAUNCH_SITE"],
            satcat_row["DECAY_DATE"] or None,
            float(satcat_row["PERIOD"]) if satcat_row["PERIOD"] else None,
            float(satcat_row["INCLINATION"]) if satcat_row["INCLINATION"] else None,
            float(satcat_row["APOGEE"]) if satcat_row["APOGEE"] else None,
            float(satcat_row["PERIGEE"]) if satcat_row["PERIGEE"] else None,
            float(satcat_row["RCS"]) if satcat_row["RCS"] else None,
            satcat_row["DATA_STATUS_CODE"],
            satcat_row["ORBIT_CENTER"],
            satcat_row["ORBIT_TYPE"]
        ))
        self.conn.commit()
    
    def get_SATCAT_info(self, norad_id):
        query = "SELECT OBJECT_NAME, OBJECT_ID, NORAD_CAT_ID, OBJECT_TYPE, OPS_STATUS_CODE, OWNER, LAUNCH_DATE, LAUNCH_SITE, DECAY_DATE, PERIOD, INCLINATION, APOGEE, PERIGEE, RCS, DATA_STATUS_CODE, ORBIT_CENTER, ORBIT_TYPE FROM SATCAT WHERE NORAD_CAT_ID = ?"
        self.curr.execute(query, (norad_id,))
        row = self.curr.fetchone()
        if row is None:
            return None
        columns = ["OBJECT_NAME", "OBJECT_ID", "NORAD_CAT_ID", "OBJECT_TYPE", "OPS_STATUS_CODE", "OWNER", "LAUNCH_DATE", "LAUNCH_SITE", "DECAY_DATE", "PERIOD", "INCLINATION", "APOGEE", "PERIGEE", "RCS", "DATA_STATUS_CODE", "ORBIT_CENTER", "ORBIT_TYPE"]
        return dict(zip(columns, row))

    def __exit__(self, exc_type, exc_value, traceback):
        self.curr.close()
        self.conn.close()
    
if __name__ == "__main__":
    dbm = DBmanager()
    dbm.download_and_insert_satcat()