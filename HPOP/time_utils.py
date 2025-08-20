# time_utils.py

from astropy.time import Time

def convert_time_scales(mjd_utc, eop_manager):
    """
    주어진 MJD(UTC)를 다른 주요 시간 척도(UT1, TT, TDB)로 변환합니다.
    Astropy의 Time 객체를 사용하여 정밀한 변환을 수행합니다.

    Args:
        mjd_utc (float): 변환할 기준 시점 (MJD, UTC 스케일).
        eop_manager (EOPManager): EOP 값을 제공하는 객체.

    Returns:
        tuple: mjd_ut1, mjd_tt, mjd_tdb
    """
    # 1. Astropy의 Time 객체를 UTC 기준으로 생성합니다.
    time_utc = Time(mjd_utc, format='mjd', scale='utc')

    # 2. EOP 데이터를 사용하여 Astropy가 UT1을 정확하게 계산하도록 설정합니다.
    #    EOPManager에서 얻은 UT1-UTC 값을 직접 지정해 줍니다.
    x_pole, y_pole, ut1_utc, lod, _, _, _, _, _ = eop_manager.get_eop_values(mjd_utc)
    time_utc.delta_ut1_utc = ut1_utc

    # 3. Time 객체의 내장 변환 기능을 사용하여 각 시간 척도의 MJD 값을 얻습니다.
    mjd_ut1 = time_utc.ut1.mjd
    mjd_tt = time_utc.tt.mjd
    mjd_tdb = time_utc.tdb.mjd

    return mjd_ut1, mjd_tt, mjd_tdb