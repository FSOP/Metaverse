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
from MISC import config
import pymsis
import json
import os
import tempfile
from datetime import datetime, timezone

tleman = TLEmanager()
dbman = DBmanager()

snapshot_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
tle_save_path = tleman.download_tle_and_save(chk_saveDb=True, source=config.TLE_SOURCE)
# TLE 업데이트 후 마지막 TLE 업데이트 시간 기록 (UTC)
dbman.update_last_tle_update(datetime.now(timezone.utc))
dbman.download_and_insert_satcat()

pymsis.utils.download_f107_ap() # F10.7 및 Ap 지수 데이터 다운로드

# ── SUBTASK_REPORT 출력 (job_worker.py가 TLE 업로드에 사용) ──
# TLE txt 파일을 파싱하여 JSON 임시 파일로 저장
_parsed_tles_path = None
if tle_save_path and os.path.exists(tle_save_path):
    try:
        _items = []
        with open(tle_save_path, "r", encoding="utf-8") as _f:
            _lines = _f.readlines()
        for _i in range(0, len(_lines) - 2, 3):
            _name = _lines[_i].strip().lstrip("0")
            _l1 = _lines[_i + 1].strip()
            _l2 = _lines[_i + 2].strip()
            if not (_l1.startswith("1 ") and _l2.startswith("2 ")):
                continue
            try:
                _norad = int(_l1[2:7])
            except ValueError:
                _norad = 99999
            _items.append({"norad": _norad, "name": _name, "line1": _l1, "line2": _l2})
        if _items:
            _tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, encoding="utf-8"
            )
            json.dump(_items, _tmp)
            _tmp.close()
            _parsed_tles_path = _tmp.name
    except Exception as _e:
        print(f"[run_once] TLE JSON 변환 실패: {_e}")

_tle_update_result = {"status": "done", "source": config.TLE_SOURCE}
if _parsed_tles_path:
    _tle_update_result["parsed_tles_path"] = _parsed_tles_path

_subtask_report = {
    "snapshot_id": snapshot_id,
    "subtask_results": {
        "tle_update": _tle_update_result,
    },
}
print(f"SUBTASK_REPORT:{json.dumps(_subtask_report, ensure_ascii=False)}")