
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

# filter 1) Outdated TLEs
filtered_tle = CA_filter().filter_outdated_tles(tle_all, analysis_start, analysis_end, pad_days=10)
print(f"1st Filtered TLE records: {len(filtered_tle)}")  # Number of TLEs after filtering

for i in [1]:    
    ref_line2 = filtered_tle[i][2]  # Reference TLE line2 for altitude comparison

    # filter 2) Altitude
    remain_tle = CA_filter().filter_altitude(filtered_tle[i+1:], ref_line2, pad=0)
    print(f"2nd Filtered TLE records: {len(remain_tle)}")  # Number of TLEs after altitude filtering

    # filter 3) Orbit path    
    remain_tle = CA_filter().filter_orbitpath(remain_tle, ref_line2)    
    print(f"3rd Filtered TLE records: {len(remain_tle)}")  # Number of TLEs after orbit path filtering
    
    pass


