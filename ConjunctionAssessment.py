
from DBmanager import DBmanager
from TLEmanager import TLEmanger
from CA_filter import CA_filter
from datetime import datetime, timedelta

count_tle = DBmanager().get_tle_count()
tle_all = TLEmanger().all_tles()
print(f"Total TLE records in database: {count_tle}") 

now_epoch = datetime.now()             # Reference epoch for TLE data
analysis_start = now_epoch                      # Start of analysis period
analysis_end = now_epoch + timedelta(days=1)    # End of analysis period

for i in [1]:

    # filter 1) Outdated TLEs
    CA_filter().filter_outdated_tles(tle_all, analysis_start, analysis_end, pad_days=30)
    pass