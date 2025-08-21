import os, sys
from datetime import datetime, timedelta
from HPOP.force_models import ForceModel
from MISC.TLEmanager import TLEmanger
from CA.CA_filter import CA_filter
from CA.RA_ephemeris import propagate_hpop_to_surface

from HPOP.force_models import ForceModel
from HPOP.propagator import propagate_with_scipy
from HPOP.eop import EOPManager
from HPOP.astroephemeris import EphemerisManager
from HPOP.gravity_models import GravityModel # 새로 만든 클래스를 import
from HPOP.atmosphere import AtmosphereModel



ALTITUDE_THRESHOLD = 170.0  # km
BSTAR_THRESHOLD = 0.05

def main():
    tle_manager = TLEmanger()
    CA_manager = CA_filter()   

    start_epoch = datetime.now()
    end_epoch = start_epoch + timedelta(days=10)

    # Step 1: Load TLEs
    all_tles = tle_manager.all_tles()

    # Step 2: Filter outdated TLEs
    tles = tle_manager.filter_outdated_tles(all_tles, start_epoch, end_epoch, pad_days=10)

    # Step 3: Filter by apogee/perigee
    tles = CA_manager.filter_perigee(tles, ALTITUDE_THRESHOLD)

    # Step 4: Filter by Bstar
    tles = CA_manager.filter_BSTAR(tles, BSTAR_THRESHOLD)  # example value

    print(f"Number of whole TLE: {len(all_tles)} Filtered TLEs: {len(tles)}")
    pass

    # Step 5: For each filtered TLE, analyze reentry
    results = []
    for tle in tles:
        sat = tle
        code, reentry_time, state_vector = CA_manager.get_state_at_altitude(sat, start_epoch, ALTITUDE_THRESHOLD, 30, 10)
        result = {
            "norad": sat[0],
            "code": code,  # 0: success, 1: error, 2: already-decayed
            "reentry_time": reentry_time,
            "state_vector": state_vector
        }
        results.append(result)
        print(f"Satellite {sat[0]}: code={code}, reentry_time={reentry_time}, state={state_vector}")


    # HPOP ForceModel 객체 생성 (초기화는 HPOP_example.py 참고)
    CONSTANTS = {
    'GM_Earth': 3.986004418e14,     # m^3/s^2
    'GM_Sun': 1.32712440018e20,
    'GM_Moon': 4.9048695e12,
    'R_Earth': 6378137.0,            # 지구 평균 반경 [m]
    'AU': 149597870700.0,            # 천문단위 [m]
    'P_Sol': 4.56e-6,                # 1 AU에서의 태양 압력 [N/m^2]
    'omega_Earth': 7.292115e-5       # 지구 자전 각속도 [rad/s]
    }

    AUX_PARAMS = {
    'mass': 1000.0, 'area_drag': 10.0, 'area_solar': 10.0,
    'Cd': 2.35, 'Cr': 1.0, 'n_max': 70, 'm_max': 70,
    'sun': False, 'moon': False, 'sRad': False, 'drag': True
    }

    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    ephem_file = os.path.join(PROJECT_ROOT, 'de440.bsp')    
    gravity_file = os.path.join(PROJECT_ROOT, 'EGM2008.gfc')


    eop_manager = EOPManager()
    ephem_manager = EphemerisManager(ephem_file)
    gravity_model = GravityModel(gravity_file, n_max=AUX_PARAMS['n_max'], m_max=AUX_PARAMS['m_max'])
    atmosphere_model = AtmosphereModel()

    force_model = ForceModel(
        consts=CONSTANTS, 
        aux_params=AUX_PARAMS, 
        eop_manager=eop_manager,
        ephem_manager=ephem_manager,
        gravity_model=gravity_model,
        atmosphere_model=atmosphere_model
    )

    result_impact = []
    for result in results:
        if result['code'] == 0 and result['state_vector'] is not None:
            print(f"Propagating satellite {result['norad']} to surface...")
            # HPOP을 이용해 위성이 지표면에 도달할 때까지 propagate
            impact_time, impact_location = propagate_hpop_to_surface(result['state_vector'], result['reentry_time'], force_model)
            result['impact_time'] = impact_time # 지표면 도달 시간
            result['impact_location'] = impact_location # 지표면 도달 위치

            result_impact.append({
                "norad": result['norad'],
                "impact_time": impact_time,
                "impact_location": impact_location
            })
            print(f"Satellite {result['norad']} impact at {impact_location} on {impact_time}")
            print(f"Number of Total events: {len(results)} / this event number: {len(result_impact)}")

    # Example: print summary
    print(f"\nReentry analysis summary:")
    for res in result_impact:
        print(res)

if __name__ == "__main__":
    main()