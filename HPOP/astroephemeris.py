# ephemeris.py (수정된 코드)

import numpy as np
from jplephem.spk import SPK

class EphemerisManager:
    def __init__(self, ephem_file_path):
        print(f"천체력 파일 로딩 중: {ephem_file_path}")
        self.kernel = SPK.open(ephem_file_path)
        print("천체력 파일 로딩 완료.")
        
        # jplephem 천체 ID
        # 0: SSB, 1: Mercury, 2: Venus, 3: EMB, 4: Mars, ..., 10: Sun, 301: Moon, 399: Earth
        self.ssb_center_bodies = {
            'sun': 10, 'mercury': 1, 'venus': 2, 'earth_bary': 3, 
            'mars': 4, 'jupiter': 5, 'saturn': 6, 'uranus': 7,
            'neptune': 8, 'pluto': 9
        }
        # self.earth_id = 399
        # self.earth_id = 3
        self.moon_id = 301
        self.emb_id = 3 # Earth-Moon Barycenter ID

    def get_positions(self, mjd_tdb):
        """
        주어진 MJD(TDB) 시점에서 주요 천체들의 위치 벡터를 계산합니다.
        모든 위치는 태양계 질량중심(SSB)을 기준으로 한 ICRF 좌표계입니다.
        """
        jd_tdb = mjd_tdb + 2400000.5
        positions_km = {}
        
        # 1. SSB를 중심으로 한 천체들의 위치 계산
        for name, body_id in self.ssb_center_bodies.items():
            positions_km[name] = self.kernel[0, body_id].compute(jd_tdb)
            
        # 2. 지구의 위치 계산 (SSB 기준)
        
        positions_km['earth'] = positions_km['earth_bary'] + self.kernel[3, 399].compute(jd_tdb)
        
        # 3. 달의 위치 계산 (SSB 기준)
        # (a) EMB에서 달까지의 벡터
        pos_moon_from_emb = self.kernel[self.ssb_center_bodies['earth_bary'], self.moon_id].compute(jd_tdb)
        # (b) SSB에서 EMB까지의 벡터 (이미 계산됨) + EMB에서 달까지의 벡터
        positions_km['moon'] = positions_km['earth_bary'] + pos_moon_from_emb
        
        return positions_km