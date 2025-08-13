from datetime import datetime, timedelta

class CA_filter:
    def __init__(self):
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
                filtered_tles.append((epoch, line1, line2))
        return filtered_tles
        