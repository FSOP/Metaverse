
from MISC.DBmanager import DBmanager
from MISC.TLEmanager import TLEmanger
from CA.CA_filter import CA_filter
from datetime import datetime, timedelta

tle_manager = TLEmanger()
CA_filter = CA_filter()

count_tle = DBmanager().get_tle_count()
tle_all = tle_manager.all_tles()
print(f"Total TLE records in database: {count_tle}") 

now_epoch = datetime.now()             # Reference epoch for TLE data
analysis_start = now_epoch                      # Start of analysis period
analysis_end = now_epoch + timedelta(days=1)    # End of analysis period

# filter 1) Outdated TLEs
filtered_tle = tle_manager.filter_outdated_tles(tle_all, analysis_start, analysis_end, pad_days=10)
print(f"1st Filtered TLE records: {len(filtered_tle)}")  # Number of TLEs after filtering

for i in range(1000):    
    ref_line2 = filtered_tle[i][2]  # Reference TLE line2 for altitude comparison

    # filter 2) Altitude
    remain_tle = CA_filter.filter_altitude(filtered_tle[i+1:], ref_line2, pad=0)
    # print(f"2nd Filtered TLE records: {len(remain_tle)}")  # Number of TLEs after altitude filtering

    # filter 3) Orbit path    
    remain_tle = CA_filter.filter_orbitpath(remain_tle, ref_line2)    
    # print(f"3rd Filtered TLE records: {len(remain_tle)}")  # Number of TLEs after orbit path filtering
    
    # filter 4) Time
    ref_sat = filtered_tle[i]  # Reference satellite for time filtering
    remain_events = CA_filter.filter_time(ref_sat, remain_tle, analysis_days=10, time_window=300.0, d_tol_km=100.0)
    # print(f"4th Filtered TLE records: {len(remain_tle)}")  # Number of TLEs after time filtering
    pass

    # filter 5) Conjunction assessment
    ca_res = CA_filter.fine_filter_min_distance(ref_sat, remain_tle, remain_events, dt_s=1.0)
    print(f"5th Filtered TLE records: {len(ca_res)}")  # Number of TLEs after conjunction assessment

    
    for r in ca_res:
        # print(f"SAT1: {r['sat1_norad']}, SAT2:{r['sat2_norad']}, TCA: {r['closest_time']}, Miss_distance(km): {r['closest_distance_km']}")
        DBmanager().insert_CA(
            r['sat1_norad'], r['sat2_norad'], "NONAME1", "NONAME2", r['closest_time'], r['closest_distance_km']
        )
