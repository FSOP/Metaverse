# easyHPOP_handle.py

import numpy as np
import sys, os
from HPOP.force_models import ForceModel
from HPOP.eop import EOPManager
from HPOP.astroephemeris import EphemerisManager
from HPOP.gravity_models import GravityModel # 새로 만든 클래스를 import
from HPOP.atmosphere import AtmosphereModel

# import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from HPOP.propagator import propagate_with_scipy

# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

class HPOP_handle:
    def __init__(self):
        self.ephem_file = os.path.join(PROJECT_ROOT, 'de440.bsp')
        self.gravity_file = os.path.join(PROJECT_ROOT, 'EGM2008.gfc')    
        self.CONSTANTS = {
            'GM_Earth': 3.986004418e14,     # m^3/s^2
            'GM_Sun': 1.32712440018e20,
            'GM_Moon': 4.9048695e12,
            'R_Earth': 6378137.0,            # 지구 평균 반경 [m]
            'AU': 149597870700.0,            # 천문단위 [m]
            'P_Sol': 4.56e-6,                # 1 AU에서의 태양 압력 [N/m^2]
            'omega_Earth': 7.292115e-5       # 지구 자전 각속도 [rad/s]
        }

        self.AUX_PARAMS = {
            'mass': 1000.0, 'area_drag': 10.0, 'area_solar': 10.0,
            'Cd': 2.35, 'Cr': 1.0, 'n_max': 70, 'm_max': 70,
            'sun': False, 'moon': False, 'sRad': False, 'drag': True
        }

        # Create managers/models once
        self.eop_manager = EOPManager()
        self.ephem_manager = EphemerisManager(self.ephem_file)
        self.gravity_model = GravityModel(self.gravity_file, n_max=self.AUX_PARAMS['n_max'], m_max=self.AUX_PARAMS['m_max'])
        self.atmosphere_model = AtmosphereModel()
        self.force_model = ForceModel(
            consts=self.CONSTANTS,
            aux_params=self.AUX_PARAMS,
            eop_manager=self.eop_manager,
            ephem_manager=self.ephem_manager,
            gravity_model=self.gravity_model,
            atmosphere_model=self.atmosphere_model
        )

    def easyHpop(self, state, state_epoch, analysis_period, step_size):
        # 위성 초기 상태 벡터
        r0 = np.array(state[:3])
        v0 = np.array(state[3:])
        y_initial = np.hstack((r0, v0))

        # ForceModel is reused; propagate_with_scipy will update aux_params['Mjd_UTC'] as needed
        ephemeris = propagate_with_scipy(
            state_epoch, analysis_period, step_size,
            y_initial, self.force_model, rtol=1e-12, atol=1e-12
        )

        ep = [state_epoch + timedelta(seconds=sec) for sec in ephemeris[:,0]]
        ephemeris = np.delete(ephemeris, 0, axis=1)
        ephemeris = np.hstack((np.array(ep).reshape(-1, 1), ephemeris))
        return ephemeris
        

if __name__ == "__main__":
    hpop = HPOP_handle()
