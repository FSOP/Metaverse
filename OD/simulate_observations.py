import os
import sys
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_root)

from datetime import datetime, timedelta
from OD.coordinate_converter import coordinate_converter as cc
from OD.simulator_tools import Observation
from MISC.DBmanager import DBmanager

def simulate_observations(sat, epochs, site):
    """
    Simulate azimuth, elevation, range for each epoch.
    Returns: list of dicts with {'epoch', 'azimuth', 'elevation', 'range'}
    """
    CC = cc()
    obs_tools = Observation()  # Dummy initialization
    ephemeris = obs_tools.propagate_j2(sat, epochs, step_sec=60)
    # _, res_el, res_epoch, res_state = obs_tools.compute_aer_accesses(ephemeris, site, CC)
    res_obs, res_states = obs_tools.compute_aer_accesses(ephemeris, site, CC)
    peak_epoch, peak_state = obs_tools.find_access_peaks(res_obs, res_states)

    hpop_result = []
    aux_data = []
    for i, p_epoch in enumerate(peak_epoch):    # peak는 하나의 pass를 의미함
        # Refine the access window with HPOP
        hpop_ephem = obs_tools.refine_access_with_hpop(p_epoch, peak_state[i], minutes=15, step_sec=1)
        # Compute AER for the refined ephemeris
        t_res, _ = obs_tools.compute_aer_accesses(hpop_ephem, site, CC)
        hpop_result.append(t_res)

        # eps = [entry['epoch'] for entry in t_res]
        eps = t_res[:, 0]
        event_id = f"{eps[0].strftime('%Y%m%d_%H%M%S')}_{eps[-1].strftime('%H%M%S')}"
        aux_data.append({'id': event_id, 'site': site, 'state_epoch': peak_epoch[i], 'state': peak_state[i]})
    return hpop_result, aux_data

if __name__ == "__main__":
    dbman = DBmanager()
    start_time = datetime(2025, 8, 17, 23, 20, 00)
    start_time = datetime(2025, 8, 18, 6, 0, 0)
    epochs = [start_time, start_time + timedelta(days=1/24)]
    SWITCH_WRITE_DB = 0

    sat = {
        'sma': 6878.14,
        'ecc': 0,
        'inc': 43,
        'raan': 66.2117,
        'argp': 0,
        'nu': 23.2427,
        'epoch': datetime(2025, 8, 17, 23, 28, 58)
    }

    site = {
        'lat': 37.7,
        'lon': 127.1,
        'alt': 0
    }

    obs_data, aux_data = simulate_observations(sat, epochs, site)
    

    for passes, aux in zip(obs_data, aux_data):
        for obs in passes:
            epoch, azimuth, elevation, rng = obs[0:5]           

            event_id = aux['id']
            site_str = f"{site['lat']}_{site['lon']}_{site['alt']}"
            state_epoch = aux['state_epoch'].strftime('%Y-%m-%d %H:%M:%S')
            state = ','.join(f'{x:.4f}' for x in aux['state']/1e3)  # Convert to km

            str_aux = f"{state_epoch},{state}"
            if SWITCH_WRITE_DB:
                dbman.insert_obs_data(epoch, azimuth, elevation, rng, site_str, event_id, str_aux)
            else:
                print(f"{epoch}, {azimuth:.3f}, {elevation:.3f}, {rng/1e3:.3f}, {str_aux}")
            pass
        # dbman.insert_obs_data(**aux)
