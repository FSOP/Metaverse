import mariadb

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


    def get_single_TLE(self, norad):
        query = "SELECT line1, line2 FROM TLE_DATA WHERE norad = ?"
        self.curr.execute(query, (norad,))
        return self.curr.fetchall()

    def get_all_TLEs(self):
        query = "SELECT norad, line1, line2, creation_date FROM TLE_DATA"
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

    def get_tle_count(self):
        query = "SELECT COUNT(*) FROM TLE_DATA"
        self.curr.execute(query)
        return self.curr.fetchone()[0]
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.curr.close()
        self.conn.close()