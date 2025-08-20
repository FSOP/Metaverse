# eop.py

import numpy as np
# from astropy.utils.iers import IersA
from astropy.utils.iers import IERS_A as IersA
from astropy.time import Time

class EOPManager:
    """
    Astropy를 사용하여 IERS EOP(지구 방향 매개변수) 데이터를 관리하고 제공합니다.
    데이터 파일을 자동으로 다운로드하고 특정 시점의 보간된 값을 제공합니다.
    """
    def __init__(self):
        """
        EOPManager를 초기화하고 IERS A 데이터를 로드합니다.
        데이터가 로컬에 없으면 Astropy가 자동으로 다운로드합니다.
        """
        print("EOP 데이터를 로딩합니다 (첫 실행 시 자동으로 다운로드됩니다)...")
        self.iers_a = IersA.open()
        print("EOP 데이터 로딩 완료.")

    def get_eop_values(self, mjd_utc):
        """
        주어진 MJD(UTC) 시점에 대한 EOP 값들을 보간하여 반환합니다.
        MATLAB의 IERS(eopdata, MJD_UTC, 'l') 호출과 유사한 역할을 합니다.
        
        Args:
            mjd_utc (float): EOP 값을 조회할 시점 (MJD, UTC 스케일).
        
        Returns:
            tuple: x_pole, y_pole, UT1_UTC, LOD, dpsi, deps, dx_pole, dy_pole, TAI_UTC
        """
        # Astropy의 Time 객체를 사용하여 MJD 시점 지정
        t = Time(mjd_utc, format='mjd')

        # polar motion (x_pole, y_pole) 값 보간
        # 값의 단위는 arcseconds 입니다.
        x_pole = np.interp(mjd_utc, self.iers_a['MJD'].to_value('d'), self.iers_a['PM_x'].to_value('arcsec'))
        y_pole = np.interp(mjd_utc, self.iers_a['MJD'].to_value('d'), self.iers_a['PM_y'].to_value('arcsec'))

        # UT1-UTC 값 보간 (Astropy는 편리한 메서드를 제공합니다)
        ut1_utc = self.iers_a.ut1_utc(t).to_value('s')

        # Length of Day (LOD) 값 보간
        # 값의 단위는 milliseconds 이므로 초(seconds) 단위로 변환합니다.
        # lod = np.interp(mjd_utc, self.iers_a['MJD'].to_value('d'), self.iers_a['LOD'].to_value('arcsec')) / 1000.0
        lod = 0

        # TODO: dpsi, deps 등은 다른 IERS 데이터(Bulletin B)에서 오므로,
        # 최고 정밀도가 필요한 경우 추가 구현이 필요합니다. 여기서는 0으로 둡니다.
        dpsi = 0.0
        deps = 0.0
        
        # 극 운동의 시간 미분값 (dx_pole, dy_pole)
        # TODO: 정밀한 계산을 위해서는 EOP 데이터의 수치 미분이 필요합니다.
        # 여기서는 0으로 둡니다.
        dx_pole = 0.0
        dy_pole = 0.0

        # TAI-UTC 차이 (Astropy Time 객체에서 직접 얻을 수 있습니다)
        # GPS-UTC 값 등 다른 시간 차이도 이와 유사하게 얻을 수 있습니다.
        tai_utc = t.tai.mjd - t.utc.mjd
        tai_utc *= 86400.0 # MJD 차이를 초(seconds) 단위로 변환

        return x_pole, y_pole, ut1_utc, lod, dpsi, deps, dx_pole, dy_pole, tai_utc