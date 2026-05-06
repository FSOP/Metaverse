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

    def get_latest_TLE(self, norad):
        """
        Returns the latest TLE (by creation_date) for a single NORAD.
        """
        query = "SELECT norad, line1, line2, creation_date FROM TLE_DATA WHERE norad = ? ORDER BY creation_date DESC LIMIT 1"
        self.curr.execute(query, (norad,))
        row = self.curr.fetchone()
        return [row] if row else []

    def get_latest_TLE_asof(self, norad, as_of_dt):
        """Returns the latest TLE (by creation_date) for a NORAD where creation_date <= as_of_dt."""
        query = (
            "SELECT norad, line1, line2, creation_date "
            "FROM TLE_DATA "
            "WHERE norad = ? AND creation_date <= ? "
            "ORDER BY creation_date DESC LIMIT 1"
        )
        self.curr.execute(query, (norad, as_of_dt))
        row = self.curr.fetchone()
        return [row] if row else []

    def get_all_TLEs(self):
        query = "SELECT norad, line1, line2, creation_date FROM TLE_DATA order by norad"
        self.curr.execute(query)
        return self.curr.fetchall()

    def get_latest_TLEs(self):
        """
        Returns the latest TLE (by creation_date) for each NORAD.
        """
        query = """
        SELECT t1.norad, t1.line1, t1.line2, t1.creation_date
        FROM TLE_DATA t1
        JOIN (
            SELECT norad, MAX(creation_date) AS max_dt
            FROM TLE_DATA
            GROUP BY norad
        ) t2
          ON t1.norad = t2.norad
         AND t1.creation_date = t2.max_dt
        ORDER BY t1.norad
        """
        self.curr.execute(query)
        return self.curr.fetchall()

    def get_latest_TLEs_asof(self, as_of_dt):
        """Returns the latest TLE (by creation_date) for each NORAD where creation_date <= as_of_dt."""
        query = """
        SELECT t1.norad, t1.line1, t1.line2, t1.creation_date
        FROM TLE_DATA t1
        JOIN (
            SELECT norad, MAX(creation_date) AS max_dt
            FROM TLE_DATA
            WHERE creation_date <= ?
            GROUP BY norad
        ) t2
          ON t1.norad = t2.norad
         AND t1.creation_date = t2.max_dt
        ORDER BY t1.norad
        """
        self.curr.execute(query, (as_of_dt,))
        return self.curr.fetchall()

    def insert_TLE_data(self, str_source, dt_creation_date, int_norad, str_line1, str_line2, str_sat_name):
        query = "INSERT INTO TLE_DATA (TLE_source, creation_date, norad, line1, line2, sat_name) VALUES (?, ?, ?, ?, ?, ?)"
        self.curr.execute(query, (str_source, dt_creation_date, int_norad, str_line1, str_line2, str_sat_name))
        self.conn.commit()

    def insert_TLEs_bulk(self, rows):
        """
        Insert multiple TLE rows in a single transaction for speed.
        `rows` should be an iterable of tuples: (source, creation_date, norad, line1, line2, sat_name)
        Uses executemany and a single commit to avoid per-row commits.
        """
        if not rows:
            return
        query = "INSERT INTO TLE_DATA (TLE_source, creation_date, norad, line1, line2, sat_name) VALUES (%s, %s, %s, %s, %s, %s)"
        self.curr.executemany(query, rows)
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
    
    def insert_CA(self, norad1, norad2, name1, name2, tca, closest_distance_km, probability=0.0, creation_date=None):
        """
        CA(충돌 분석) 결과를 DB에 저장.
        creation_date가 None이면 현재 UTC 시각을 자동 사용.
        """
        from datetime import datetime, timezone
        if creation_date is None:
            creation_date = datetime.now(timezone.utc)
        query = "INSERT INTO CA (sat1_norad, sat1_name, sat2_norad, sat2_name, tca, miss_distance, probability, creation_date) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
        self.curr.execute(query, (norad1, name1, norad2, name2, tca, closest_distance_km, probability, creation_date))
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

    def update_last_analysis_time(self, dt):
        """
        system_status 테이블의 마지막 분석 시간(last_analysis_time)을 dt로 업데이트
        만약 row가 없으면 새로 insert, 있으면 update
        """
        self.curr.execute("SELECT id FROM system_status LIMIT 1")
        row = self.curr.fetchone()
        if row:
            self.curr.execute("UPDATE system_status SET last_analysis_time = ? WHERE id = ?", (dt, row[0]))
        else:
            # last_tle_update에 NULL이 들어가면 에러 발생하므로 dt와 동일하게 저장
            self.curr.execute("INSERT INTO system_status (last_tle_update, last_analysis_time) VALUES (?, ?)", (dt, dt))
        self.conn.commit()
    
    def update_last_tle_update(self, dt):
        """
        system_status 테이블의 마지막 TLE 업데이트 시간(last_tle_update)을 dt로 업데이트
        만약 row가 없으면 새로 insert, 있으면 update
        """
        self.curr.execute("SELECT id FROM system_status LIMIT 1")
        row = self.curr.fetchone()
        if row:
            self.curr.execute("UPDATE system_status SET last_tle_update = ? WHERE id = ?", (dt, row[0]))
        else:
            self.curr.execute("INSERT INTO system_status (last_tle_update, last_analysis_time) VALUES (?, ?)", (dt, None))
        self.conn.commit()
    
    def get_primary_norad_list(self):
            """
            primary_satellites 테이블에서 norad_cat_id 리스트 반환
            """
            query = "SELECT norad_cat_id FROM primary_satellites"
            self.curr.execute(query)
            return [row[0] for row in self.curr.fetchall()]
    
    def insert_reentry_event(self, crash_id, creation_date, start_time, norad_cat_id, object_name, object_type, rcs, orbit_crash, time_crash, lat_crash, lon_crash, alt_crash, prob_crash, inbound_info):
            """
            reentry_events 테이블에 위성 재진입(추락) 결과를 저장
            orbit_crash, inbound_info는 JSON/text로 저장
            """
            query = """
            INSERT INTO reentry_events (
                crash_id, creation_date, start_time, norad_cat_id, object_name, object_type, rcs,
                orbit_crash, time_crash, lat_crash, lon_crash, alt_crash, prob_crash, inbound_info
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            self.curr.execute(query, (
                crash_id, creation_date, start_time, norad_cat_id, object_name, object_type, rcs,
                orbit_crash, time_crash, lat_crash, lon_crash, alt_crash, prob_crash, inbound_info
            ))
            self.conn.commit()

    def __exit__(self, exc_type, exc_value, traceback):
        self.curr.close()
        self.conn.close()

# module intended for import; example usage removed