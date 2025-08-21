from OD.coordinate_converter import coordinate_converter as cc
from MISC.J2Propagator import J2_Propagator
from HPOP.easyHPOP_handle import HPOP_handle
from scipy.signal import find_peaks
from datetime import datetime, timedelta
import numpy as np


class Observation:
    def __init__(self):
        self.hpop_handle = HPOP_handle()

    def propagate_j2(self, sat, epochs, step_sec=60):
        """Propagate orbit using J2 model."""
        ephemeris, _, _ = J2_Propagator(sat, (epochs[0], epochs[-1]), step_sec)
        return ephemeris

    def compute_aer_accesses(self,ephemeris, site, CC):
        """Compute AER and filter visible accesses."""
        obs_ecef = CC.geodetic_to_ecef(site['lat'], site['lon'], site['alt'])
        result, res_ephemeris = [], []        
        # res_el, res_az, res_epoch, res_state = [], [], []
        for t in range(len(ephemeris[:,0])):
            this_res = []            
            this_epoch = ephemeris[t, 0]
            r_eci = ephemeris[t, 1:4]
            r_ecef = CC.eci_to_ecef(r_eci, this_epoch)
            enu = CC.ecef_to_enu(r_ecef, obs_ecef, site['lat'], site['lon'])
            az, el, rng = CC.aer_from_enu(enu)
            if el <= 0:
                continue
            # results.append({'epoch': this_epoch, 'azimuth': az, 'elevation': el, 'range': rng})            
            this_res = [this_epoch, az, el, rng]
            result.append(this_res)
            res_ephemeris.append(ephemeris[t, 1:7])
        return np.array(result), np.array(res_ephemeris)

    def find_access_peaks(self, res_obs, res_states):
        res_el = res_obs[:, 2]
        res_epoch = res_obs[:, 0]
        """Find local maxima in elevation (access peaks)."""
        peaks, _ = find_peaks(res_el, height=0)
        return res_epoch[peaks], res_states[peaks, :]

    def refine_access_with_hpop(self,p_epoch, p_state, minutes=15, step_sec=1):
        """Refine each access window with HPOP propagation."""
        # hpop_handle = HPOP_handle()
        start_time = p_epoch - timedelta(minutes=minutes)
        end_time = p_epoch + timedelta(minutes=minutes)
        ephemeris = self.hpop_handle.easyHpop(p_state, p_epoch, (start_time, end_time), step_size=step_sec)
        # Optionally process ephemeris for refined AER here
        return ephemeris