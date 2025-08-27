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

class getFM:
    aux  = {
                'mass': 1000.0, 'area_drag': 10.0, 'area_solar': 10.0,
                'Cd': 2.2, 'Cr': 1.0, 'n_max': 70, 'm_max': 70,
                'sun': False, 'moon': False, 'sRad': False, 'drag': True
            }
    def __init__(self, AUX_PARAMS=aux):
        PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
        self.ephem_file = os.path.join(PROJECT_ROOT, 'de440.bsp')
        self.gravity_file = os.path.join(PROJECT_ROOT, 'EGM2008.gfc')     
        if AUX_PARAMS is not None:
            self.AUX_PARAMS = AUX_PARAMS
        else:
            self.AUX_PARAMS = {
                'mass': 1000.0, 'area_drag': 10.0, 'area_solar': 10.0,
                'Cd': 2.2, 'Cr': 1.0, 'n_max': 70, 'm_max': 70,
                'sun': False, 'moon': False, 'sRad': False, 'drag': True
            }

        # Create managers/models once
        self.eop_manager = EOPManager()
        self.ephem_manager = EphemerisManager(self.ephem_file)
        self.gravity_model = GravityModel(self.gravity_file, n_max=self.AUX_PARAMS['n_max'], m_max=self.AUX_PARAMS['m_max'])
        self.atmosphere_model = AtmosphereModel()
        self.force_model = ForceModel(
            aux_params=self.AUX_PARAMS,
            eop_manager=self.eop_manager,
            ephem_manager=self.ephem_manager,
            gravity_model=self.gravity_model,
            atmosphere_model=self.atmosphere_model
        ) 
        self.CS = self.gravity_model.get_CS()