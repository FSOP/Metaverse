"""
job_worker.py — 서버 작업 큐 폴링 워커

서버의 작업 큐를 주기적으로 폴링하고,
작업 타입에 맞는 기존 분석 프로그램을 실행한 뒤 결과 상태를 서버로 반환하는
로컬 오케스트레이터 워커입니다.

[작업 흐름]
  A) 범용 큐(/api/tasks/*):
      - TLE_UPDATE  -> run_once.py
      - CA_ANALYSIS -> ConjunctionAssessment.py
      - CRASH_ANALYSIS -> ReentryAnalysis.py
      - complete/fail로 상태 반영

  B) CA 전용 큐(/api/analysis/*):
      - 충돌분석 작업 -> ConjunctionAssessment.py 위임
      - result 전송으로 상태 반영

[사용법]
  python job_worker.py                  # 기본: 15초 주기 폴링, 무한 루프
  python job_worker.py --once           # 한 번만 폴링 후 종료
  python job_worker.py --interval 30    # 30초 주기 폴링
  python job_worker.py --dry            # 작업 큐만 확인하고 분석하지 않음
  python job_worker.py --debug          # 디버그 로깅 활성화 (debug.log)
  
[디버그 로깅]
  --debug 플래그로 활성화하면 debug.log 파일에 상세 로그 기록:
  - API 요청/응답 전체 데이터
  - 작업 수신/할당/전송 상세 내용
  - TCA 탐색 3단계 상세 결과
  - 충돌 확률 계산 과정
  - Time Series 생성 결과
  - 궤도 요소 추출 결과
  
[프로그래밍으로 디버그 로깅 제어]
  from job_worker import enable_debug_logging, disable_debug_logging
  
  enable_debug_logging()   # 활성화
  # ... 작업 수행 ...
  disable_debug_logging()  # 비활성화
"""

import os
import sys
import json
import time
import logging
import argparse
import subprocess
import tempfile
import signal
import socket
import traceback
import requests
from datetime import datetime, timedelta, timezone

from MISC import config

# ─── 로깅 설정 ───
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("job_worker.log"),
    ],
)
log = logging.getLogger(__name__)

# ─── 디버깅 로그 설정 ───
_debug_enabled = False
_debug_log_handler = None
_active_client = None
_inflight_work = None


def enable_debug_logging(filename: str = "debug.log"):
    """디버깅 로그를 활성화합니다.
    
    API 요청/응답, 작업 상세 내용을 별도 파일에 기록합니다.
    
    Args:
        filename: 디버그 로그 파일명 (기본: debug.log)
    """
    global _debug_enabled, _debug_log_handler
    if _debug_enabled:
        log.info("디버그 로깅이 이미 활성화되어 있습니다.")
        return
    
    # 기존 디버그 로그 파일이 있으면 백업
    if os.path.exists(filename):
        backup_name = filename.replace(".log", f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        try:
            os.rename(filename, backup_name)
            log.info(f"기존 디버그 로그를 백업했습니다: {backup_name}")
        except Exception as e:
            log.warning(f"디버그 로그 백업 실패: {e}")
    
    # 새 디버그 로거 핸들러 생성
    _debug_log_handler = logging.FileHandler(filename, encoding="utf-8")
    _debug_log_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(funcName)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    _debug_log_handler.setFormatter(formatter)
    
    # 로거에 핸들러 추가
    log.addHandler(_debug_log_handler)
    _debug_enabled = True
    log.info(f"디버그 로깅 활성화됨: {filename}")


def disable_debug_logging():
    """디버깅 로그를 비활성화합니다."""
    global _debug_enabled, _debug_log_handler
    if not _debug_enabled:
        log.info("디버그 로깅이 활성화되지 않았습니다.")
        return
    
    if _debug_log_handler:
        log.removeHandler(_debug_log_handler)
        _debug_log_handler.close()
        _debug_log_handler = None
    
    _debug_enabled = False
    log.info("디버그 로깅 비활성화됨")


def _debug_log(msg: str, level: str = "DEBUG"):
    """디버그 로깅 활성화 시에만 메시지를 기록합니다.
    
    Args:
        msg: 로그 메시지
        level: 로그 레벨 ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    """
    if not _debug_enabled:
        return
    
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }
    log.log(level_map.get(level, logging.DEBUG), msg)


def _debug_log_request(method: str, url: str, params: dict = None, json_data: dict = None):
    """API 요청을 디버그 로깅합니다."""
    if not _debug_enabled:
        return
    msg = f"[API 요청] {method} {url}"
    if params:
        msg += f"\n  Params: {json.dumps(params, ensure_ascii=False)}"
    if json_data:
        msg += f"\n  JSON: {json.dumps(json_data, ensure_ascii=False, default=str)[:500]}"
    _debug_log(msg, "DEBUG")


def _debug_log_response(url: str, status_code: int, text: str = None):
    """API 응답을 디버그 로깅합니다."""
    if not _debug_enabled:
        return
    msg = f"[API 응답] {status_code} {url}"
    if text:
        # 긴 응답은 요약
        if len(text) > 500:
            msg += f"\n  응답(일부): {text[:500]}..."
        else:
            msg += f"\n  응답: {text}"
    _debug_log(msg, "DEBUG")


def _debug_log_dict(title: str, data: dict):
    """딕셔너리 데이터를 보기 좋게 디버그 로깅합니다."""
    if not _debug_enabled:
        return
    msg = f"[{title}]\n{json.dumps(data, indent=2, ensure_ascii=False, default=str)}"
    _debug_log(msg, "DEBUG")


def _tail_text(s: str, n: int = 4000) -> str:
    """문자열 끝 n자만 반환. 긴 stdout/stderr 잘라내기용."""
    if s is None:
        return ""
    s = str(s)
    return s[-n:] if len(s) > n else s


def _set_active_client(client):
    """현재 API 클라이언트를 글로벌에 저장 (중단 시 서버 통지용)."""
    global _active_client
    _active_client = client


def _set_inflight_work(queue_name: str, item_id: int, item_type: str):
    """현재 진행 중인 작업 정보를 기록 (시그널 핸들러에서 fail 전송에 사용)."""
    global _inflight_work
    _inflight_work = {
        "queue": queue_name,
        "id": int(item_id),
        "type": str(item_type),
        "started_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _clear_inflight_work():
    """진행 중 작업 정보 초기화."""
    global _inflight_work
    _inflight_work = None


def _notify_inflight_interrupted(reason: str):
    """워커가 비정상 중단될 때 진행 중이던 작업을 서버에 fail로 통지."""
    if not _active_client or not _inflight_work:
        return

    queue_name = _inflight_work.get("queue")
    item_id = _inflight_work.get("id")
    item_type = _inflight_work.get("type")
    started_at = _inflight_work.get("started_at")

    try:
        if queue_name == "tasks":
            msg = f"worker interrupted: {reason} (type={item_type}, started_at={started_at})"
            ok = _active_client.fail_task(item_id, msg)
            if ok:
                log.warning("중단된 범용 작업 fail 전송 성공 (task_id=%s)", item_id)
            else:
                log.error("중단된 범용 작업 fail 전송 실패 (task_id=%s)", item_id)
            return

        if queue_name == "analysis":
            payload = {
                "status": "failed",
                "error_message": f"worker interrupted: {reason}",
                "result": {
                    "job_type": item_type,
                    "started_at": started_at,
                    "interrupted_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "worker_interrupted": True,
                },
            }
            ok = _active_client.submit_result(item_id, payload)
            if ok:
                log.warning("중단된 CA 전용 작업 실패 결과 전송 성공 (job_id=%s)", item_id)
            else:
                log.error("중단된 CA 전용 작업 실패 결과 전송 실패 (job_id=%s)", item_id)
    except Exception as e:
        log.error("중단 작업 통지 중 예외 발생: %s", e)


def _handle_termination_signal(signum, frame):
    """SIGTERM/SIGINT 수신 시 진행 중 작업을 fail 처리하고 KeyboardInterrupt 발생."""
    sig_name = signal.Signals(signum).name if signum else "UNKNOWN"
    log.warning("종료 시그널 수신: %s", sig_name)
    _notify_inflight_interrupted(f"signal {sig_name}")
    raise KeyboardInterrupt


def _project_root() -> str:
    """프로젝트 루트 디렉토리 경로 반환."""
    return os.path.dirname(os.path.abspath(__file__))


def _python_exec() -> str:
    """현재 venv의 Python 인터프리터 경로 반환."""
    return sys.executable if sys.executable else "python3"


def _default_worker_id() -> str:
    """워커 식별자 반환 (WORKER_ID env → hostname-job-worker 순)."""
    explicit = os.getenv("WORKER_ID")
    if explicit:
        return explicit
    return f"{socket.gethostname()}-job-worker"


def _normalize_job_type(job: dict) -> str:
    """CA 전용 큐 작업의 타입을 정규화 (별칭 → 표준명)."""
    raw = (
        job.get("job_type")
        or job.get("task_type")
        or job.get("analysis_type")
        or job.get("type")
        or ""
    )
    jt = str(raw).strip().lower().replace("-", "_").replace(" ", "_")

    aliases = {
        "ca": "ca_batch",
        "ca_analysis": "ca_batch",
        "conjunction": "ca_batch",
        "conjunction_assessment": "ca_batch",
        "collision": "collision_pair",
        "collision_analysis": "collision_pair",
        "manual_ca": "collision_pair",
        "manual_collision": "collision_pair",
        "pair_ca": "collision_pair",
        "reentry": "reentry",
        "reentry_analysis": "reentry",
        "tle": "tle_update",
        "tle_update": "tle_update",
        "tle_updater": "tle_update",
    }
    jt = aliases.get(jt, jt)

    if jt:
        return jt

    # 타입 필드가 없으면 payload로 추론
    if job.get("tle_primary_line1") and job.get("tle_secondary_line1"):
        return "collision_pair"
    if job.get("sat1_norad") and job.get("sat2_norad"):
        return "collision_pair"
    return "unknown"


def _normalize_task_type(task: dict) -> str:
    """범용 큐 task의 타입을 정규화 (별칭 → 표준명)."""
    raw = task.get("task_type") or task.get("type") or ""
    tt = str(raw).strip().upper()
    aliases = {
        "TLE": "TLE_UPDATE",
        "TLE_UPDATER": "TLE_UPDATE",
        "CA": "CA_ANALYSIS",
        "CA_BATCH": "CA_ANALYSIS",
        "CONJUNCTION": "CA_ANALYSIS",
        "CRASH": "CRASH_ANALYSIS",
        "REENTRY": "CRASH_ANALYSIS",
    }
    return aliases.get(tt, tt)


def _parse_task_payload(task: dict) -> dict:
    """task의 payload를 dict로 파싱 (str이면 JSON 디코드)."""
    payload = task.get("payload", {})
    if isinstance(payload, dict):
        return payload
    if payload is None:
        return {}
    if isinstance(payload, str):
        s = payload.strip()
        if not s:
            return {}
        try:
            parsed = json.loads(s)
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except Exception:
            return {"raw": payload}
    return {"raw": payload}


def _run_subprocess_with_heartbeat(cmd, timeout_s: int, heartbeat_cb=None, heartbeat_interval_s: int = 20):
    """서브프로세스 실행 중 주기적으로 heartbeat 콜백을 호출합니다."""
    proc = subprocess.Popen(
        cmd,
        cwd=_project_root(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    t0 = time.perf_counter()
    hb_interval = max(1, int(heartbeat_interval_s or 20))

    while True:
        elapsed = time.perf_counter() - t0
        remaining = timeout_s - elapsed
        if remaining <= 0:
            proc.kill()
            stdout, stderr = proc.communicate()
            return {
                "timed_out": True,
                "return_code": None,
                "stdout": stdout or "",
                "stderr": stderr or "",
                "elapsed": time.perf_counter() - t0,
            }

        wait_s = min(hb_interval, max(1, int(remaining)))
        try:
            stdout, stderr = proc.communicate(timeout=wait_s)
            return {
                "timed_out": False,
                "return_code": proc.returncode,
                "stdout": stdout or "",
                "stderr": stderr or "",
                "elapsed": time.perf_counter() - t0,
            }
        except subprocess.TimeoutExpired:
            if heartbeat_cb:
                try:
                    heartbeat_cb()
                except Exception as hb_err:
                    _debug_log(f"heartbeat 콜백 오류: {hb_err}", "WARNING")


def _run_script_job(script_name: str, args=None, timeout_s: int = 7200,
                    heartbeat_cb=None, heartbeat_interval_s: int = 20) -> dict:
    """Python 스크립트를 서브프로세스로 실행하고 결과를 {status, result} dict로 반환."""
    args = args or []
    cmd = [_python_exec(), script_name] + list(args)
    cmd_text = " ".join(cmd)
    _debug_log(f"스크립트 실행: {cmd_text}", "INFO")

    try:
        run = _run_subprocess_with_heartbeat(
            cmd,
            timeout_s=timeout_s,
            heartbeat_cb=heartbeat_cb,
            heartbeat_interval_s=heartbeat_interval_s,
        )
        elapsed = run["elapsed"]
        stdout_raw = run["stdout"]
        stdout_tail = _tail_text(stdout_raw)
        stderr_tail = _tail_text(run["stderr"])

        # _tail_text 로 잘리기 전에 전체 stdout 에서 SUBTASK_REPORT 를 먼저 파싱해둔다.
        # (stdout 이 수 MB 이어도 SUBTASK_REPORT 를 안정적으로 추출 가능)
        _sr_inline = _parse_subtask_report(stdout_raw)
        _sr_kv = {"subtask_report_inline": _sr_inline} if _sr_inline is not None else {}

        if run["timed_out"]:
            return {
                "status": "failed",
                "error_message": f"스크립트 타임아웃({timeout_s}s)",
                "result": {
                    "command": cmd_text,
                    "duration_sec": round(elapsed, 3),
                    "stdout_tail": stdout_tail,
                    "stderr_tail": stderr_tail,
                    **_sr_kv,
                },
            }

        if run["return_code"] == 0:
            return {
                "status": "done",
                "result": {
                    "command": cmd_text,
                    "return_code": run["return_code"],
                    "duration_sec": round(elapsed, 3),
                    "stdout_tail": stdout_tail,
                    "stderr_tail": stderr_tail,
                    **_sr_kv,
                },
            }
        return {
            "status": "failed",
            "error_message": f"스크립트 실패(return_code={run['return_code']})",
            "result": {
                "command": cmd_text,
                "return_code": run["return_code"],
                "duration_sec": round(elapsed, 3),
                "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
                **_sr_kv,
            },
        }
    except Exception as e:
        return {
            "status": "failed",
            "error_message": f"스크립트 실행 예외: {e}",
            "result": {
                "command": cmd_text,
                "duration_sec": None,
            },
        }


def _run_conjunction_single_pair(job: dict, timeout_s: int = 3600,
                                 heartbeat_cb=None, heartbeat_interval_s: int = 20) -> dict:
    """ConjunctionAssessment.py 단일쌍 모드를 실행하고 RESULT_JSON을 파싱합니다."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as tf:
            json.dump(job, tf, ensure_ascii=False, default=str)
            tmp_path = tf.name

        cmd = [
            _python_exec(),
            "ConjunctionAssessment.py",
            "--single-pair",
            "--pair-input-json",
            tmp_path,
        ]
        cmd_text = " ".join(cmd)
        _debug_log(f"단일쌍 CA 스크립트 실행: {cmd_text}", "INFO")

        run = _run_subprocess_with_heartbeat(
            cmd,
            timeout_s=timeout_s,
            heartbeat_cb=heartbeat_cb,
            heartbeat_interval_s=heartbeat_interval_s,
        )
        elapsed = run["elapsed"]

        if run["timed_out"]:
            return {
                "status": "failed",
                "error_message": f"단일쌍 분석 타임아웃({timeout_s}s)",
                "result": {
                    "command": cmd_text,
                    "duration_sec": round(elapsed, 3),
                    "stdout_tail": _tail_text(run["stdout"]),
                    "stderr_tail": _tail_text(run["stderr"]),
                },
            }

        stdout = run["stdout"]
        stderr = run["stderr"]

        result_line = None
        for line in reversed(stdout.splitlines()):
            if line.startswith("RESULT_JSON:"):
                result_line = line[len("RESULT_JSON:"):].strip()
                break

        if result_line:
            try:
                parsed = json.loads(result_line)
                if isinstance(parsed, dict) and "status" in parsed:
                    parsed.setdefault("result", {})
                    if isinstance(parsed.get("result"), dict):
                        parsed["result"].setdefault("command", cmd_text)
                        parsed["result"].setdefault("duration_sec", round(elapsed, 3))
                        parsed["result"].setdefault("stdout_tail", _tail_text(stdout))
                        parsed["result"].setdefault("stderr_tail", _tail_text(stderr))
                    return parsed
            except Exception as e:
                return {
                    "status": "failed",
                    "error_message": f"RESULT_JSON 파싱 실패: {e}",
                    "result": {
                        "command": cmd_text,
                        "return_code": run["return_code"],
                        "duration_sec": round(elapsed, 3),
                        "stdout_tail": _tail_text(stdout),
                        "stderr_tail": _tail_text(stderr),
                    },
                }

        return {
            "status": "failed",
            "error_message": f"단일쌍 결과 파싱 실패(return_code={run['return_code']})",
            "result": {
                "command": cmd_text,
                "return_code": run["return_code"],
                "duration_sec": round(elapsed, 3),
                "stdout_tail": _tail_text(stdout),
                "stderr_tail": _tail_text(stderr),
            },
        }
    except Exception as e:
        return {
            "status": "failed",
            "error_message": f"단일쌍 분석 실행 예외: {e}",
            "result": {
                "duration_sec": None,
            },
        }
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def _parse_subtask_report(stdout: str) -> dict | None:
    """run_once.py 출력에서 SUBTASK_REPORT JSON 라인을 파싱합니다."""
    if not stdout:
        return None
    for line in reversed(stdout.splitlines()):
        if line.startswith("SUBTASK_REPORT:"):
            try:
                return json.loads(line[len("SUBTASK_REPORT:"):].strip())
            except Exception:
                pass
    return None


def _handle_tle_update_job(job: dict, heartbeat_cb=None, heartbeat_interval_s: int = 20) -> dict:
    """TLE 업데이트 작업 처리 — Delta 방식 우선, 실패 시 기존 batch 방식 fallback.

    Delta 흐름:
      1. run_once.py 실행 (TLE 다운로드 + DB 저장)
      2. SUBTASK_REPORT에서 delta_info 추출
      3. 변경분만 /api/tle/delta 로 전송
    """
    _debug_log_dict("TLE 업데이트 작업", job)
    args = []
    if bool(job.get("skip_satcat", False)):
        args.append("--skip-satcat")
    if bool(job.get("skip_spaceweather", False)):
        args.append("--skip-spaceweather")
    if bool(job.get("skip_eop", False)):
        args.append("--skip-eop")

    res = _run_script_job(
        "run_once.py",
        args=args,
        timeout_s=int(job.get("timeout_s", 3600)),
        heartbeat_cb=heartbeat_cb,
        heartbeat_interval_s=heartbeat_interval_s,
    )

    # SUBTASK_REPORT 파싱 — 서브태스크별 성공/실패  추적
    # 1순위: _run_script_job 이 잘리기 전 전체 stdout 에서 미리 파싱해 둔 결과 사용
    # 2순위: fallback — stdout_tail (잘렸을 수 있음)
    report = res.get("result", {}).get("subtask_report_inline")
    if report is None:
        stdout_text = res.get("result", {}).get("stdout_tail", "")
        report = _parse_subtask_report(stdout_text)
    if report is None:
        log.warning("SUBTASK_REPORT 파싱 실패 — delta 업로드를 건너뜁니다")
    if report:
        res.setdefault("result", {})["subtask_report"] = report
        res["result"]["snapshot_id"] = report.get("snapshot_id")

        # ── Delta 업로드 시도 ──
        delta_info = report.get("subtask_results", {}).get("tle_update", {}).get("delta_info")
        if delta_info and _active_client:
            # changes 는 run_once.py 가 임시 파일에 저장했으므로 파일에서 읽는다
            changes_file = delta_info.get("changes_file")
            if changes_file and os.path.exists(changes_file):
                try:
                    with open(changes_file, "r", encoding="utf-8") as _cf:
                        changes = json.load(_cf)
                except Exception as _cfe:
                    log.error("Delta changes 파일 읽기 실패: %s", _cfe)
                    changes = []
                finally:
                    try:
                        os.remove(changes_file)
                    except Exception:
                        pass
            else:
                # 구버전 호환: SUBTASK_REPORT 에 직접 포함된 경우
                changes = delta_info.get("changes", [])
                if not changes and changes_file:
                    log.error("Delta changes 파일 없음: %s", changes_file)
            snapshot_id = delta_info.get("snapshot_id", report.get("snapshot_id"))
            payload = {
                "snapshot_id": snapshot_id,
                "source": delta_info.get("source", "space-track.org"),
                "mode": delta_info.get("mode", "unknown"),
                "downloaded_at": delta_info.get("downloaded_at",
                                                datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")),
                "total_local_count": delta_info.get("total_local_count", 0),
                "changes": changes,
            }
            _debug_log(f"Delta 업로드: {len(changes)}건 변경분 전송", "INFO")
            upload_ok, err_msg = _active_client.upload_tle_delta(payload)
            if upload_ok:
                new_cnt = sum(1 for c in changes if c.get("change_type") == "new")
                upd_cnt = sum(1 for c in changes if c.get("change_type") == "updated")
                res["result"]["tle_upload_success"] = True
                res["result"]["tle_upload_mode"] = "delta"
                res["result"]["tle_delta_stats"] = {
                    "new": new_cnt, "updated": upd_cnt, "total_changes": len(changes),
                }
                log.info("Delta TLE 업로드 완료: new=%d, updated=%d", new_cnt, upd_cnt)
            else:
                log.error("Delta TLE 업로드 실패: %s", err_msg)
                res["status"] = "failed"
                res["error_message"] = f"Delta TLE upload failed: {err_msg}"

        # ── Fallback: delta_info 없으면 기존 batch 방식 ──
        elif not delta_info:
            tle_update_info = report.get("subtask_results", {}).get("tle_update", {})
            parsed_tles_path = tle_update_info.get("parsed_tles_path")

            if parsed_tles_path and os.path.exists(parsed_tles_path):
                try:
                    with open(parsed_tles_path, "r", encoding="utf-8") as f:
                        parsed_items = json.load(f)

                    if _active_client and parsed_items:
                        batch_size = 2000
                        total_batches = (len(parsed_items) + batch_size - 1) // batch_size
                        success_count = 0
                        _debug_log(f"Fallback: 총 {len(parsed_items)}개의 TLE를 {total_batches}번 나누어 전송합니다.", "INFO")
                        last_error = ""
                        for i in range(0, len(parsed_items), batch_size):
                            chunk = parsed_items[i : i + batch_size]
                            upload_payload = {
                                "downloaded_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                                "snapshot_id": report.get("snapshot_id"),
                                "items": chunk,
                            }
                            upload_ok, err_msg = _active_client.upload_tles(upload_payload)
                            if upload_ok:
                                success_count += 1
                            else:
                                log.error("TLE batch 업로드 실패 (배치 %d/%d)", i // batch_size + 1, total_batches)
                                last_error = err_msg
                                break
                        if success_count == total_batches:
                            res["result"]["tle_upload_success"] = True
                            res["result"]["tle_upload_mode"] = "batch_fallback"
                            res["result"]["tle_upload_batches"] = f"{success_count}/{total_batches}"
                        else:
                            res["status"] = "failed"
                            res["error_message"] = f"TLE batch upload failed: {last_error}"
                            res["result"]["tle_upload_batches"] = f"{success_count}/{total_batches}"

                    os.remove(parsed_tles_path)
                except Exception as e:
                    log.error("Failed to process parsed TLEs: %s", e)
                    res["status"] = "failed"
                    res["error_message"] = f"Failed to process parsed TLEs: {e}"
                    res.setdefault("result", {})["tle_upload_error"] = str(e)

        # delta_info에서 임시 파일이 남아있으면 정리
        if delta_info:
            for _tmp_key in ("parsed_out_path", "changes_file"):
                _tmp = delta_info.get(_tmp_key)
                if _tmp and os.path.exists(_tmp):
                    try:
                        os.remove(_tmp)
                    except Exception:
                        pass

    if res.get("status") == "done":
        res["result"]["job_type"] = "tle_update"
        res["result"]["message"] = "TLE 업데이트 작업 및 업로드 완료"
    return res


def _handle_ca_batch_job(job: dict, heartbeat_cb=None, heartbeat_interval_s: int = 20) -> dict:
    """CA 분석 작업 처리 — ConjunctionAssessment.py를 서브프로세스로 실행."""
    _debug_log_dict("CA 배치 작업", job)

    # 수동/쌍대 CA 입력이면 ConjunctionAssessment 단일쌍 모드 사용
    if job.get("tle_primary_line1") and job.get("tle_secondary_line1"):
        res = _run_conjunction_single_pair(
            job,
            timeout_s=int(job.get("timeout_s", 3600)),
            heartbeat_cb=heartbeat_cb,
            heartbeat_interval_s=heartbeat_interval_s,
        )
        if res.get("status") == "done" and isinstance(res.get("result"), dict):
            res["result"]["job_type"] = "collision_pair"
            res["result"].setdefault("message", "수동 충돌분석 완료(ConjunctionAssessment --single-pair)")
        return res

    args = []
    norad = job.get("norad") or job.get("primary_norad") or job.get("sat1_norad")
    if norad:
        args += ["--norad", str(norad)]

    # 세부 파라미터는 명시적으로 전달된 경우에만 오버라이드
    if job.get("fast") is True:
        args.append("--fast")

    analysis_days = job.get("analysis_days", job.get("days"))
    try:
        if analysis_days is not None:
            args += ["--analysis-days", str(float(analysis_days))]
    except Exception:
        pass

    # 세부 파라미터 오버라이드 허용
    if job.get("min_dt") is not None:
        args += ["--min-dt", str(float(job.get("min_dt")))]
    if job.get("refine_step") is not None:
        args += ["--refine-step", str(float(job.get("refine_step")))]

    if bool(job.get("no_send", False)):
        args.append("--no-send")
    if bool(job.get("no_db", False)):
        args.append("--no-db")

    res = _run_script_job(
        "ConjunctionAssessment.py",
        args=args,
        timeout_s=int(job.get("timeout_s", 7200)),
        heartbeat_cb=heartbeat_cb,
        heartbeat_interval_s=heartbeat_interval_s,
    )
    if res.get("status") == "done":
        res["result"]["job_type"] = "ca_batch"
        res["result"]["message"] = "CA 분석 작업 완료(ConjunctionAssessment.py 실행)"
        res["result"]["ca_args"] = args
    return res


def _handle_reentry_job(job: dict, heartbeat_cb=None, heartbeat_interval_s: int = 20) -> dict:
    """재진입 분석 작업 처리 — ReentryAnalysis.py 서브프로세스 실행."""
    _debug_log_dict("재진입 분석 작업", job)
    res = _run_script_job(
        "ReentryAnalysis.py",
        args=[],
        timeout_s=int(job.get("timeout_s", 14400)),
        heartbeat_cb=heartbeat_cb,
        heartbeat_interval_s=heartbeat_interval_s,
    )
    if res.get("status") == "done":
        res["result"]["job_type"] = "reentry"
        res["result"]["message"] = "재진입 분석 작업 완료(ReentryAnalysis.py 실행)"
    return res


def _handle_launch_sim_job(job: dict, heartbeat_cb=None, heartbeat_interval_s: int = 20) -> dict:
    """발사 궤적 시뮬레이션 작업 처리.

    job payload 예시:
      {
        "site": "cape_canaveral_slc40",   # 또는 lat/lon/alt
        "lat": 28.56, "lon": -80.58, "alt": 3.0,
        "orbit_alt_km": 550,
        "orbit_apo_km": 550,              # 생략 시 = orbit_alt_km (원궤도)
        "orbit_inc_deg": 53.0,
        "orbit_type": "circular",         # circular|gto|sso|molniya|lunar|custom
        "payload_mass_kg": 5000,
        "launch_time": "2026-03-05T12:00:00Z",  # 생략 시 현재시각
      }
    """
    _debug_log_dict("발사 시뮬레이션 작업", job)

    try:
        from LaunchSim.launch_api import run_simulation_from_params, LaunchSimSender

        # output 경로 설정
        import tempfile
        output_dir = os.path.join(tempfile.gettempdir(), "launch_sim")
        os.makedirs(output_dir, exist_ok=True)
        ts_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_prefix = os.path.join(output_dir, f"launch_{ts_tag}")

        params = dict(job)
        params["output_prefix"] = output_prefix
        params.setdefault("sim_id", f"job_launch_{ts_tag}")

        # heartbeat (시뮬레이션이 오래 걸릴 수 있으므로)
        if heartbeat_cb:
            heartbeat_cb()

        result = run_simulation_from_params(params)

        # API 전송
        if result.get("status") == "done" and _active_client:
            sender = LaunchSimSender(base_url=_active_client.base_url)
            payload = result.get("result", {})

            ok = sender.send_result(payload)
            if ok:
                result["result"]["api_upload"] = "success"
                _debug_log("발사 시뮬레이션 결과 API 전송 성공", "INFO")
            else:
                result["result"]["api_upload"] = "failed"
                _debug_log("발사 시뮬레이션 결과 API 전송 실패", "WARNING")

            # ephemeris 파일 업로드
            for key in ("ephemeris_eci", "ephemeris_ecef"):
                eph_path = payload.get(key)
                if eph_path and os.path.exists(eph_path):
                    sender.upload_ephemeris(eph_path, sim_id=payload.get("sim_id"))

        if result.get("status") == "done":
            result.setdefault("result", {})["job_type"] = "launch_sim"
            result["result"]["message"] = "발사 궤적 시뮬레이션 완료"

        return result

    except Exception as e:
        _debug_log(f"발사 시뮬레이션 작업 오류: {e}", "ERROR")
        log.error("발사 시뮬레이션 작업 오류: %s", e)
        return {
            "status": "failed",
            "error_message": str(e),
            "result": {"job_type": "launch_sim"},
        }


def process_job(job: dict, heartbeat_cb=None, heartbeat_interval_s: int = 20) -> dict:
    """큐 작업 유형에 따라 적절한 분석 핸들러를 실행합니다."""
    job_type = _normalize_job_type(job)
    _debug_log(f"작업 타입 판별: raw={job.get('job_type') or job.get('task_type')} -> normalized={job_type}", "INFO")

    # 1) 명시적 작업 타입
    if job_type == "tle_update":
        return _handle_tle_update_job(job, heartbeat_cb=heartbeat_cb, heartbeat_interval_s=heartbeat_interval_s)
    if job_type == "ca_batch":
        return _handle_ca_batch_job(job, heartbeat_cb=heartbeat_cb, heartbeat_interval_s=heartbeat_interval_s)
    if job_type == "reentry":
        return _handle_reentry_job(job, heartbeat_cb=heartbeat_cb, heartbeat_interval_s=heartbeat_interval_s)
    if job_type == "collision_pair":
        # 충돌분석은 항상 ConjunctionAssessment.py로 위임
        return _handle_ca_batch_job(job, heartbeat_cb=heartbeat_cb, heartbeat_interval_s=heartbeat_interval_s)
    if job_type == "launch_sim":
        return _handle_launch_sim_job(job, heartbeat_cb=heartbeat_cb, heartbeat_interval_s=heartbeat_interval_s)

    # 2) fallback: 수동 CA payload 형태면 ConjunctionAssessment.py로 위임
    if job.get("tle_primary_line1") and job.get("tle_secondary_line1"):
        return _handle_ca_batch_job(job, heartbeat_cb=heartbeat_cb, heartbeat_interval_s=heartbeat_interval_s)
    if job.get("sat1_norad") and job.get("sat2_norad"):
        return _handle_ca_batch_job(job, heartbeat_cb=heartbeat_cb, heartbeat_interval_s=heartbeat_interval_s)

    return {
        "status": "failed",
        "error_message": f"지원하지 않는 작업 타입: {job_type}",
        "result": {
            "job_type": job_type,
            "supported_types": ["tle_update", "ca_batch", "reentry", "collision_pair", "launch_sim"],
        },
    }


def process_generic_task(task: dict, heartbeat_cb=None, heartbeat_interval_s: int = 20) -> dict:
    """범용 큐 task_type에 따라 배치 작업을 실행합니다."""
    task_type = _normalize_task_type(task)
    payload = _parse_task_payload(task)
    _debug_log(f"범용 작업 타입 판별: raw={task.get('task_type')} -> normalized={task_type}", "INFO")
    _debug_log_dict("범용 작업 payload", payload)

    if task_type == "TLE_UPDATE":
        out = _handle_tle_update_job(payload, heartbeat_cb=heartbeat_cb, heartbeat_interval_s=heartbeat_interval_s)
        out.setdefault("result", {})["task_type"] = task_type
        return out

    if task_type == "CA_ANALYSIS":
        out = _handle_ca_batch_job(payload, heartbeat_cb=heartbeat_cb, heartbeat_interval_s=heartbeat_interval_s)
        out.setdefault("result", {})["task_type"] = task_type
        return out

    if task_type == "CRASH_ANALYSIS":
        out = _handle_reentry_job(payload, heartbeat_cb=heartbeat_cb, heartbeat_interval_s=heartbeat_interval_s)
        out.setdefault("result", {})["task_type"] = task_type
        return out

    if task_type == "LAUNCH_SIM":
        out = _handle_launch_sim_job(payload, heartbeat_cb=heartbeat_cb, heartbeat_interval_s=heartbeat_interval_s)
        out.setdefault("result", {})["task_type"] = task_type
        return out

    return {
        "status": "failed",
        "error_message": f"지원하지 않는 task_type: {task_type}",
        "result": {
            "task_type": task_type,
            "supported_types": ["TLE_UPDATE", "CA_ANALYSIS", "CRASH_ANALYSIS", "LAUNCH_SIM"],
        },
    }


# ═══════════════════════════════════════════════════════════════
#  서버 API 통신 클래스
# ═══════════════════════════════════════════════════════════════

class AnalysisJobClient:
    """서버 analysis_jobs API와 통신하는 클라이언트."""

    def __init__(self, base_url: str, api_token: str = None, worker_id: str = None):
        self.base_url = base_url.rstrip("/")
        self.worker_id = worker_id or _default_worker_id()
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

        # 인증 토큰 설정 (우선순위: 인자 > CA_API_TOKEN env > EPHEMERIS_API_KEY)
        token = api_token or os.getenv("CA_API_TOKEN") or getattr(config, "EPHEMERIS_API_KEY", None)
        if token:
            self.session.headers.update({"Authorization": f"Bearer {token}"})
            _debug_log("Authorization 헤더 설정 완료", "INFO")
        else:
            log.warning("API 토큰이 설정되지 않았습니다. claim/complete/fail/result 요청에서 401이 발생할 수 있습니다.")

    def fetch_queue(self, limit: int = 10):
        """pending 상태의 작업 목록을 가져옵니다."""
        url = f"{self.base_url}/api/analysis/queue"
        try:
            _debug_log_request("GET", url, params={"limit": limit})
            resp = self.session.get(url, params={"limit": limit}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            _debug_log_response(url, resp.status_code, resp.text[:1000])
            _debug_log(f"작업 큐 수신: {len(data)}건", "INFO")
            return data
        except requests.RequestException as e:
            _debug_log(f"작업 큐 수신 실패: {e}", "ERROR")
            log.error("작업 큐 수신 실패: %s", e)
            return []

    def claim_job(self, job_id: int, worker_id: str = None):
        """작업을 할당(claim)합니다. 성공 시 True."""
        url = f"{self.base_url}/api/analysis/claim/{job_id}"
        body = {"worker_id": worker_id or self.worker_id}
        try:
            _debug_log_request("POST", url, json_data=body)
            resp = self.session.post(url, json=body, timeout=10)
            _debug_log_response(url, resp.status_code, resp.text)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == "claimed":
                    _debug_log(f"작업 할당 성공: job_id={job_id}", "INFO")
                    return True
            log.warning("작업 할당 실패 (job_id=%s): %s", job_id, resp.text)
            _debug_log(f"작업 할당 실패: job_id={job_id}, {resp.text}", "WARNING")
            return False
        except requests.RequestException as e:
            _debug_log(f"작업 할당 요청 오류: job_id={job_id}, {e}", "ERROR")
            log.error("작업 할당 요청 오류 (job_id=%s): %s", job_id, e)
            return False

    def submit_result(self, job_id: int, payload: dict, worker_id: str = None):
        """분석 결과를 서버로 전송합니다."""
        url = f"{self.base_url}/api/analysis/result/{job_id}"
        body = dict(payload or {})
        wid = worker_id or self.worker_id
        body.setdefault("worker_id", wid)
        body.setdefault("worker_name", wid)
        try:
            _debug_log_request("POST", url, json_data=body)
            resp = self.session.post(url, json=body, timeout=30)
            resp.raise_for_status()
            _debug_log_response(url, resp.status_code, resp.text[:500])
            _debug_log(f"결과 전송 성공: job_id={job_id}", "INFO")
            log.info("결과 전송 성공 (job_id=%s), 서버 응답: %s", job_id, resp.text[:200])
            return True
        except requests.RequestException as e:
            _debug_log(f"결과 전송 실패: job_id={job_id}, {e}", "ERROR")
            log.error("결과 전송 실패 (job_id=%s): %s", job_id, e)
            return False

    def download_orbit_file(self, filename: str, save_dir: str = "data/orbits"):
        """서버에서 궤도 파일을 다운로드합니다."""
        url = f"{self.base_url}/storage/orbits/{filename}"
        os.makedirs(save_dir, exist_ok=True)
        local_path = os.path.join(save_dir, filename)
        try:
            _debug_log_request("GET", url)
            resp = self.session.get(url, timeout=30, stream=True)
            resp.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            _debug_log(f"궤도 파일 다운로드 완료: {local_path}", "INFO")
            log.info("궤도 파일 다운로드 완료: %s", local_path)
            return local_path
        except requests.RequestException as e:
            _debug_log(f"궤도 파일 다운로드 실패: {filename}, {e}", "ERROR")
            log.error("궤도 파일 다운로드 실패 (%s): %s", filename, e)
            return None

    def upload_tles(self, payload: dict):
        """서버로 TLE를 업로드합니다. (/api/tle/batch) — 기존 전체 업로드 방식"""
        url = f"{self.base_url}/api/tle/batch"
        try:
            _debug_log_request("POST", url, json_data={"items_count": len(payload.get("items", []))})
            resp = self.session.post(url, json=payload, timeout=60)
            if resp.status_code not in (200, 201):
                err_msg = f"status {resp.status_code}: {resp.text[:100]}"
                log.error("TLE 업로드 실패 %s", err_msg)
                return False, err_msg
            _debug_log_response(url, resp.status_code, resp.text[:500])
            _debug_log("TLE 업로드 성공", "INFO")
            log.info("TLE 업로드 성공, 서버 응답: %s", resp.text[:200])
            return True, ""
        except requests.RequestException as e:
            _debug_log(f"TLE 업로드 예외 발생: {e}", "ERROR")
            log.error("TLE 업로드 예외 발생: %s", e)
            return False, str(e)

    def upload_tle_delta(self, payload: dict):
        """변경된 TLE만 서버로 전송합니다. (/api/tle/delta)

        Args:
            payload: {
                "snapshot_id": str,
                "source": str,           # "space-track.org" | "celestrak"
                "mode": str,             # "full" | "incremental"
                "downloaded_at": str,
                "total_local_count": int,
                "changes": [{"norad", "name", "line1", "line2", "change_type"}, ...]
            }
        Returns:
            (success: bool, error_msg: str)
        """
        url = f"{self.base_url}/api/tle/delta"
        n_changes = len(payload.get("changes", []))
        try:
            _debug_log_request("POST", url, json_data={
                "snapshot_id": payload.get("snapshot_id"),
                "source": payload.get("source"),
                "mode": payload.get("mode"),
                "changes_count": n_changes,
            })
            log.info("Delta TLE 업로드 시작: %d건 변경분", n_changes)
            resp = self.session.post(url, json=payload, timeout=120)
            if resp.status_code not in (200, 201):
                err_msg = f"status {resp.status_code}: {resp.text[:200]}"
                log.error("Delta TLE 업로드 실패: %s", err_msg)
                return False, err_msg
            _debug_log_response(url, resp.status_code, resp.text[:500])
            _debug_log(f"Delta TLE 업로드 성공 ({n_changes}건)", "INFO")
            log.info("Delta TLE 업로드 성공, 서버 응답: %s", resp.text[:300])
            return True, ""
        except requests.RequestException as e:
            _debug_log(f"Delta TLE 업로드 예외: {e}", "ERROR")
            log.error("Delta TLE 업로드 예외: %s", e)
            return False, str(e)

    def fetch_tasks_queue(self, limit: int = 10):
        """범용 큐(pending task) 목록을 조회합니다. (/api/tasks/queue)"""
        url = f"{self.base_url}/api/tasks/queue"
        try:
            _debug_log_request("GET", url, params={"limit": limit})
            resp = self.session.get(url, params={"limit": limit}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            _debug_log_response(url, resp.status_code, resp.text[:1000])
            _debug_log(f"범용 큐 수신: {len(data)}건", "INFO")
            return data
        except requests.RequestException as e:
            _debug_log(f"범용 큐 수신 실패: {e}", "ERROR")
            log.error("범용 큐 수신 실패: %s", e)
            return []

    def claim_task(self, task_id: int, worker_id: str = None):
        """범용 큐 작업을 claim 합니다. (/api/tasks/claim/<task_id>)"""
        url = f"{self.base_url}/api/tasks/claim/{task_id}"
        body = {"worker_id": worker_id or self.worker_id}
        try:
            _debug_log_request("POST", url, json_data=body)
            resp = self.session.post(url, json=body, timeout=10)
            _debug_log_response(url, resp.status_code, resp.text)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == "claimed":
                    return True
            log.warning("범용 작업 할당 실패 (task_id=%s): %s", task_id, resp.text)
            return False
        except requests.RequestException as e:
            log.error("범용 작업 할당 요청 오류 (task_id=%s): %s", task_id, e)
            return False

    def complete_task(self, task_id: int, worker_id: str = None):
        """범용 큐 작업을 complete 처리합니다. (/api/tasks/complete/<task_id>)"""
        url = f"{self.base_url}/api/tasks/complete/{task_id}"
        wid = worker_id or self.worker_id
        body = {"worker_name": wid, "worker_id": wid}
        try:
            _debug_log_request("POST", url, json_data=body)
            resp = self.session.post(url, json=body, timeout=10)
            _debug_log_response(url, resp.status_code, resp.text)
            resp.raise_for_status()
            return True
        except requests.RequestException as e:
            log.error("범용 작업 complete 실패 (task_id=%s): %s", task_id, e)
            return False

    def fail_task(self, task_id: int, error_message: str, worker_id: str = None):
        """범용 큐 작업을 fail 처리합니다. (/api/tasks/fail/<task_id>)"""
        url = f"{self.base_url}/api/tasks/fail/{task_id}"
        wid = worker_id or self.worker_id
        body = {
            "worker_name": wid,
            "worker_id": wid,
            "error_message": str(error_message),
        }
        try:
            _debug_log_request("POST", url, json_data=body)
            resp = self.session.post(url, json=body, timeout=10)
            _debug_log_response(url, resp.status_code, resp.text)
            resp.raise_for_status()
            return True
        except requests.RequestException as e:
            log.error("범용 작업 fail 실패 (task_id=%s): %s", task_id, e)
            return False

    def send_task_heartbeat(self, task_id: int, worker_id: str = None,
                            progress: int = None, stage: str = None,
                            message: str = None, eta_sec: int = None,
                            lease_sec: int = None):
        url = f"{self.base_url}/api/tasks/heartbeat/{task_id}"
        body = {
            "worker_id": worker_id or self.worker_id,
        }
        if progress is not None:
            body["progress"] = int(progress)
        if stage is not None:
            body["stage"] = str(stage)
        if message is not None:
            body["message"] = str(message)
        if eta_sec is not None:
            body["eta_sec"] = int(eta_sec)
        if lease_sec is not None:
            body["lease_sec"] = int(lease_sec)

        try:
            _debug_log_request("POST", url, json_data=body)
            resp = self.session.post(url, json=body, timeout=10)
            _debug_log_response(url, resp.status_code, resp.text)
            if resp.status_code in (200, 201):
                return True
            if resp.status_code in (404, 409):
                log.warning("task heartbeat 거부 (task_id=%s): %s", task_id, resp.text)
                return False
            resp.raise_for_status()
            return True
        except requests.RequestException as e:
            log.warning("task heartbeat 실패 (task_id=%s): %s", task_id, e)
            return False

    def send_job_heartbeat(self, job_id: int, worker_id: str = None,
                           progress: int = None, stage: str = None,
                           message: str = None, eta_sec: int = None,
                           lease_sec: int = None):
        url = f"{self.base_url}/api/analysis/heartbeat/{job_id}"
        body = {
            "worker_id": worker_id or self.worker_id,
        }
        if progress is not None:
            body["progress"] = int(progress)
        if stage is not None:
            body["stage"] = str(stage)
        if message is not None:
            body["message"] = str(message)
        if eta_sec is not None:
            body["eta_sec"] = int(eta_sec)
        if lease_sec is not None:
            body["lease_sec"] = int(lease_sec)

        try:
            _debug_log_request("POST", url, json_data=body)
            resp = self.session.post(url, json=body, timeout=10)
            _debug_log_response(url, resp.status_code, resp.text)
            if resp.status_code in (200, 201):
                return True
            if resp.status_code in (404, 409):
                log.warning("analysis heartbeat 거부 (job_id=%s): %s", job_id, resp.text)
                return False
            resp.raise_for_status()
            return True
        except requests.RequestException as e:
            log.warning("analysis heartbeat 실패 (job_id=%s): %s", job_id, e)
            return False


# (SGP4/TCA/RIC/충돌분석 유틸리티: ConjunctionAssessment.py로 이관 완료, 삭제됨)


def print_job_detail(job):
    """작업 데이터를 보기 좋게 출력합니다."""
    print("\n" + "─" * 60)
    print(f"  작업 ID:        {job.get('id')}")
    print(f"  상태:           {job.get('status')}")
    print(f"  작업 타입:      {_normalize_job_type(job)}")
    print(f"  생성일:         {job.get('created_at')}")
    print(f"  Primary 위성:   {job.get('sat1_name') or '(미설정)'} (NORAD {job.get('sat1_norad')})")
    print(f"  Secondary 위성: {job.get('sat2_name') or '(미설정)'} (NORAD {job.get('sat2_norad')})")
    print(f"  TCA 힌트:       {job.get('tca') or '(없음)'}")
    print(f"  궤도 소스 Pri:  {job.get('orbit_source_primary')}")
    print(f"  궤도 소스 Sec:  {job.get('orbit_source_secondary')}")
    if job.get("tle_primary_line1"):
        print(f"  TLE Pri L1:     {job.get('tle_primary_line1')}")
        print(f"  TLE Pri L2:     {job.get('tle_primary_line2')}")
    if job.get("tle_secondary_line1"):
        print(f"  TLE Sec L1:     {job.get('tle_secondary_line1')}")
        print(f"  TLE Sec L2:     {job.get('tle_secondary_line2')}")
    if job.get("orbit_file_primary"):
        print(f"  궤도 파일 Pri:  {job.get('orbit_file_primary')}")
    if job.get("orbit_file_secondary"):
        print(f"  궤도 파일 Sec:  {job.get('orbit_file_secondary')}")
    print("─" * 60)


def print_result_preview(payload):
    """분석 결과를 미리보기로 출력합니다."""
    if payload.get("status") == "failed":
        print(f"\n  ✗ 분석 실패: {payload.get('error_message')}")
        if payload.get("result"):
            print(f"  · 상세: {payload.get('result')}")
        return

    r = payload.get("result", {})
    # 범용 워커 결과 출력 (script 실행형 작업)
    if "command" in r and "return_code" in r:
        print("\n  ┌─── 작업 결과 요약 ───")
        print(f"  │ Job Type:            {r.get('job_type')}")
        print(f"  │ Command:             {r.get('command')}")
        print(f"  │ Return Code:         {r.get('return_code')}")
        print(f"  │ Duration:            {r.get('duration_sec')} sec")
        print("  └──────────────────────")
        return

    print("\n  ┌─── 분석 결과 요약 ───")
    print(f"  │ TCA:                 {r.get('tca')}")
    print(f"  │ Miss Distance:       {r.get('miss_distance_km')} km")
    cp = r.get('collision_probability')
    cp_str = f"{cp:.6e}" if isinstance(cp, (int, float)) else str(cp)
    print(f"  │ Collision Prob:      {cp_str}")
    print(f"  │ Relative Velocity:   {r.get('relative_velocity_km_s')} km/s")
    ric = r.get("ric_at_tca", {})
    print(f"  │ RIC at TCA:          R={ric.get('R')} I={ric.get('I')} C={ric.get('C')} km")
    ts = r.get("time_series", [])
    print(f"  │ Time Series:         {len(ts)} points")
    if ts:
        print(f"  │   첫 번째:           {ts[0]}")
        print(f"  │   마지막:            {ts[-1]}")
    po = r.get("primary_orbit", {})
    print(f"  │ Primary Orbit:       a={po.get('a_km')}km e={po.get('e')} i={po.get('i_deg')}°")
    so = r.get("secondary_orbit", {})
    print(f"  │ Secondary Orbit:     a={so.get('a_km')}km e={so.get('e')} i={so.get('i_deg')}°")
    print("  └──────────────────────")


# ═══════════════════════════════════════════════════════════════
#  메인 루프
# ═══════════════════════════════════════════════════════════════

def main():
    """메인 진입점 — CLI 인자 파싱 후 폴링 루프 실행."""
    parser = argparse.ArgumentParser(description="서버 작업 큐 폴링 워커")
    parser.add_argument("--once", action="store_true",
                        help="한 번만 폴링 후 종료")
    parser.add_argument("--interval", type=int, default=15,
                        help="폴링 주기 (초, 기본 15)")
    parser.add_argument("--dry", action="store_true",
                        help="작업 큐만 확인하고 분석하지 않음 (수신 데이터 확인용)")
    parser.add_argument("--debug", action="store_true",
                        help="디버그 로깅 활성화 (debug.log에 상세 기록)")
    parser.add_argument("--token", type=str, default=None,
                        help="API Bearer 토큰 (미지정 시 CA_API_TOKEN env 또는 EPHEMERIS_API_KEY 사용)")
    parser.add_argument("--worker-id", type=str, default=None,
                        help="워커 식별자 (미지정 시 HOSTNAME-job-worker 또는 WORKER_ID env 사용)")
    parser.add_argument("--heartbeat-interval", type=int, default=20,
                        help="작업 실행 중 heartbeat 전송 주기(초, 기본 20)")
    parser.add_argument("--lease-sec", type=int, default=90,
                        help="heartbeat 시 서버에 요청할 lease 연장 시간(초, 기본 90)")
    args = parser.parse_args()

    # 디버그 로깅 활성화 여부
    if args.debug:
        enable_debug_logging("debug.log")

    base_url = config.API_BASE_URL
    worker_id = args.worker_id or _default_worker_id()
    client = AnalysisJobClient(base_url, api_token=args.token, worker_id=worker_id)
    _set_active_client(client)

    signal.signal(signal.SIGTERM, _handle_termination_signal)
    signal.signal(signal.SIGINT, _handle_termination_signal)

    # 현재 토큰 설정 상태 출력 (보안상 값은 마스킹)
    token_in_use = args.token or os.getenv("CA_API_TOKEN") or getattr(config, "EPHEMERIS_API_KEY", None)
    token_status = "설정됨" if token_in_use else "없음"

    print("=" * 60)
    print("  작업 큐 폴링 워커")
    print(f"  서버: {base_url}")
    print(f"  폴링 주기: {args.interval}초")
    print(f"  모드: {'DRY RUN (큐 확인만)' if args.dry else '분석 실행'}")
    print(f"  디버그 로그: {'활성화 (debug.log)' if args.debug else '비활성화'}")
    print(f"  API 토큰: {token_status}")
    print(f"  WORKER_ID: {worker_id}")
    print(f"  Heartbeat: {args.heartbeat_interval}초 / lease {args.lease_sec}초")
    print("=" * 60)

    while True:
        try:
            # 1) 큐 A(범용) + 큐 B(CA 전용) 조회
            tasks = client.fetch_tasks_queue(limit=5)
            jobs = client.fetch_queue(limit=5)

            if not tasks and not jobs:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 대기 중인 작업 없음")
                _debug_log("대기 중인 작업 없음(범용/CA 전용 모두 비어있음)", "DEBUG")
                if args.once:
                    break
                time.sleep(args.interval)
                continue

            # dry 모드이면 여기서 종료
            if args.dry:
                print("\n[DRY RUN] 분석을 수행하지 않고 종료합니다.")
                print("\n[범용 큐 응답 원본 JSON]")
                print(json.dumps(tasks, indent=2, ensure_ascii=False, default=str))
                print("\n[CA 전용 큐 응답 원본 JSON]")
                print(json.dumps(jobs, indent=2, ensure_ascii=False, default=str))
                _debug_log_dict("범용 큐 응답 (DRY RUN)", {"tasks": tasks})
                _debug_log_dict("CA 전용 큐 응답 (DRY RUN)", {"jobs": jobs})
                break

            # 2) 큐 A: 범용 작업 처리 (/api/tasks/*)
            if tasks:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 범용 큐 수신: {len(tasks)}건")

            for task in tasks:
                task_id = task["id"]
                task_type = _normalize_task_type(task)
                _debug_log(f"범용 작업 처리 시작: task_id={task_id}, task_type={task_type}", "INFO")

                if not client.claim_task(task_id, worker_id=worker_id):
                    log.warning("범용 작업 할당 실패, 다음 폴링에서 재시도. task_id=%s", task_id)
                    continue

                print(f"\n✓ 범용 작업 할당 성공 (task_id={task_id}, type={task_type})")
                _set_inflight_work("tasks", task_id, task_type)
                try:
                    client.send_task_heartbeat(
                        task_id,
                        worker_id=worker_id,
                        progress=1,
                        stage="started",
                        message=f"{task_type} started",
                        lease_sec=args.lease_sec,
                    )

                    def _task_hb():
                        client.send_task_heartbeat(
                            task_id,
                            worker_id=worker_id,
                            progress=50,
                            stage="running",
                            message=f"{task_type} running",
                            lease_sec=args.lease_sec,
                        )

                    t_start = time.perf_counter()
                    result_payload = process_generic_task(
                        task,
                        heartbeat_cb=_task_hb,
                        heartbeat_interval_s=args.heartbeat_interval,
                    )
                    t_elapsed = time.perf_counter() - t_start
                    print(f"작업 소요시간: {t_elapsed:.1f}초")
                    print_result_preview(result_payload)

                    if result_payload.get("status") == "done":
                        ok = client.complete_task(task_id, worker_id=worker_id)
                        if ok:
                            print(f"✓ 범용 작업 complete 성공 (task_id={task_id})")
                        else:
                            print(f"✗ 범용 작업 complete 실패 (task_id={task_id})")
                    else:
                        err = result_payload.get("error_message", "unknown error")
                        ok = client.fail_task(task_id, err, worker_id=worker_id)
                        if ok:
                            print(f"✓ 범용 작업 fail 전송 성공 (task_id={task_id})")
                        else:
                            print(f"✗ 범용 작업 fail 전송 실패 (task_id={task_id})")
                except Exception as e:
                    log.error("범용 작업 처리 중 예외 (task_id=%s): %s", task_id, e, exc_info=True)
                    _debug_log(f"범용 작업 예외: task_id={task_id}\n{traceback.format_exc()}", "ERROR")
                    try:
                        client.fail_task(task_id, f"worker exception: {e}", worker_id=worker_id)
                    except Exception:
                        log.error("범용 작업 fail 전송도 실패 (task_id=%s)", task_id)
                finally:
                    _clear_inflight_work()

            # 3) 큐 B: CA 전용 작업 처리 (/api/analysis/*)
            if jobs:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] CA 전용 큐 수신: {len(jobs)}건")
                for j in jobs:
                    print_job_detail(j)

            for job in jobs:
                job_id = job["id"]
                job_type = _normalize_job_type(job)
                _debug_log(f"CA 전용 작업 처리 시작: job_id={job_id}, job_type={job_type}", "INFO")

                if not client.claim_job(job_id, worker_id=worker_id):
                    log.warning("CA 전용 작업 할당 실패, 다음 폴링에서 재시도. job_id=%s", job_id)
                    continue

                print(f"\n✓ CA 전용 작업 할당 성공 (job_id={job_id}, type={job_type})")
                _set_inflight_work("analysis", job_id, job_type)
                try:
                    client.send_job_heartbeat(
                        job_id,
                        worker_id=worker_id,
                        progress=1,
                        stage="started",
                        message=f"{job_type} started",
                        lease_sec=args.lease_sec,
                    )

                    def _job_hb():
                        client.send_job_heartbeat(
                            job_id,
                            worker_id=worker_id,
                            progress=50,
                            stage="running",
                            message=f"{job_type} running",
                            lease_sec=args.lease_sec,
                        )

                    t_start = time.perf_counter()
                    result_payload = process_job(
                        job,
                        heartbeat_cb=_job_hb,
                        heartbeat_interval_s=args.heartbeat_interval,
                    )
                    t_elapsed = time.perf_counter() - t_start
                    print(f"작업 소요시간: {t_elapsed:.1f}초")
                    print_result_preview(result_payload)

                    preview = dict(result_payload)
                    if preview.get("result") and isinstance(preview["result"], dict) and preview["result"].get("time_series"):
                        ts = preview["result"]["time_series"]
                        preview_result = dict(preview["result"])
                        preview_result["time_series"] = f"[... {len(ts)}개 항목 ...]"
                        preview["result"] = preview_result
                    _debug_log_dict("CA 전용 결과 payload (요약)", preview)

                    ok = client.submit_result(job_id, result_payload, worker_id=worker_id)
                    if ok:
                        print(f"✓ CA 전용 결과 전송 성공 (job_id={job_id})")
                    else:
                        print(f"✗ CA 전용 결과 전송 실패 (job_id={job_id})")
                except Exception as e:
                    log.error("CA 전용 작업 처리 중 예외 (job_id=%s): %s", job_id, e, exc_info=True)
                    _debug_log(f"CA 전용 작업 예외: job_id={job_id}\n{traceback.format_exc()}", "ERROR")
                    try:
                        client.submit_result(job_id, {
                            "status": "failed",
                            "error_message": f"worker exception: {e}",
                        }, worker_id=worker_id)
                    except Exception:
                        log.error("CA 전용 작업 fail 전송도 실패 (job_id=%s)", job_id)
                finally:
                    _clear_inflight_work()

        except KeyboardInterrupt:
            print("\n워커 중지됨 (Ctrl+C)")
            _debug_log("워커 중지 (Ctrl+C)", "INFO")
            break
        except Exception as e:
            log.error("메인 루프 오류: %s", e, exc_info=True)
            _debug_log(f"메인 루프 오류: {e}\n{traceback.format_exc()}", "ERROR")

        if args.once:
            break

        time.sleep(args.interval)

    print("\n워커 종료.")
    _debug_log("워커 종료", "INFO")
    if args.debug:
        disable_debug_logging()


def _install_crash_handler():
    """프로세스 비정상 종료 시에도 로그가 남도록 sys.excepthook을 설정합니다."""
    _original_hook = sys.excepthook

    def _crash_hook(exc_type, exc_value, exc_tb):
        if exc_type is KeyboardInterrupt:
            _original_hook(exc_type, exc_value, exc_tb)
            return
        tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        log.critical("치명적 오류로 워커 종료:\n%s", tb_str)
        _original_hook(exc_type, exc_value, exc_tb)

    sys.excepthook = _crash_hook


if __name__ == "__main__":
    _install_crash_handler()
    main()
