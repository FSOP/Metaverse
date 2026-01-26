# run_once.py
# ---------------------------------------------
# 패키지(프로젝트) 설치 후 최초 1회 실행하는 스크립트입니다.
# 목적: 이미 DB가 구축된 상태에서 TLE와 SATCAT 데이터를 자동으로 다운로드하여 DB에 저장합니다.
# 사용법: 가상환경 활성화 후, 아래 명령어로 실행
#   python run_once.py
#
# - SATCAT: 위성 카탈로그 데이터
# - TLE: Two-Line Element 궤도 데이터
#
# 이 스크립트를 한 번 실행하면 최신 SATCAT/TLE 데이터가 DB에 저장되어
# 이후 분석/전파/충돌 예측 등에 바로 활용할 수 있습니다.
# ---------------------------------------------
from MISC.TLEmanager import TLEmanager
from MISC.DBmanager import DBmanager
import pymsis
from datetime import datetime

tleman = TLEmanager()
dbman = DBmanager()

tleman.download_tle_and_save(chk_saveDb=True)
# TLE 업데이트 후 마지막 TLE 업데이트 시간 기록
dbman.update_last_tle_update(datetime.now())
dbman.download_and_insert_satcat()

pymsis.utils.download_f107_ap() # F10.7 및 Ap 지수 데이터 다운로드