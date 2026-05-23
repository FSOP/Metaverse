"""
ca_api.py — CA 분석 결과 API 전송 모듈

[역할]
  충돌 분석 결과 이벤트를 REST API 서버로 전송한다.
  단건 전송(send_event)과 배치 전송(send_batch) 모두 지원.

[전송 포맷]
  POST /api/ca_events (단건) 또는 POST /api/ca_events/batch (배치)
  서버가 WORKER_ALLOWED_IPS를 설정한 경우 IP만으로 인증 (토큰 불필요).
  미설정 환경(개발/테스트)에서는 CA_API_TOKEN Bearer 토큰으로 fallback.

[페이로드 필드]
  sat1_norad, sat2_norad, sat1_name, sat2_name,
  tca, creation_date, miss_distance, probability

[재시도]
  최대 3회 재시도, 지수 백오프 (1s, 2s, 4s)
"""

import os
import time
import logging
from datetime import datetime, timezone
import requests

try:
    from MISC import config as _config
except ImportError:
    try:
        import sys, os as _os
        sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
        from MISC import config as _config
    except ImportError:
        _config = None  # type: ignore


logging.getLogger(__name__)


class CAEventSender:
    def __init__(self, base_url: str, batch_size: int = 0, max_retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size if batch_size > 1 else 0
        self.max_retries = max_retries
        self.session = requests.Session()

    def _get_headers(self):
        headers = {"Content-Type": "application/json"}
        # Bearer 토큰은 서버가 WORKER_ALLOWED_IPS 미설정(개발/테스트) 환경에서만 필요.
        # 운영 환경에서는 IP 화이트리스트로 인증하므로 토큰 없이도 동작.
        token = (getattr(_config, 'CA_API_TOKEN', None)
                 or os.getenv("CA_API_TOKEN")
                 or getattr(_config, 'EPHEMERIS_API_KEY', None))
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _format_time(self, dt):
        if dt is None:
            return None
        if isinstance(dt, str):
            s = dt
            try:
                # parse ISO-like strings and return UTC time formatted as 'YYYY-MM-DD HH:MM:SS'
                if s.endswith('Z'):
                    t = datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")
                    t = t.replace(tzinfo=timezone.utc)
                    return t.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                t = datetime.fromisoformat(s)
                if t.tzinfo is None:
                    # assume naive times are UTC
                    t = t.replace(tzinfo=timezone.utc)
                else:
                    t = t.astimezone(timezone.utc)
                return t.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                return s
        if isinstance(dt, datetime):
            # return UTC formatted as 'YYYY-MM-DD HH:MM:SS'
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            t = dt.astimezone(timezone.utc)
            return t.strftime("%Y-%m-%d %H:%M:%S")

    def _build_api_event(self, colli):
        try:
            sat1 = colli.get("SAT_1", {}) if isinstance(colli, dict) else {}
            sat2 = colli.get("SAT_2", {}) if isinstance(colli, dict) else {}
            payload = {
                "sat1_norad": int(sat1.get("SAT_ID") or colli.get("sat1_norad") or 0),
                "sat2_norad": int(sat2.get("SAT_ID") or colli.get("sat2_norad") or 0),
                "tca": self._format_time(colli.get("TCA") or colli.get("tca")),
                "creation_date": self._format_time(colli.get("Creation_date") or colli.get("creation_date")),
                "miss_distance": float(colli.get("MIN_RNG") or colli.get("miss_distance") or 0.0),
                "probability": float(colli.get("probability") or colli.get("probability") or 0.0),
            }
            sat1_name = sat1.get("SAT_Name") or colli.get("sat1_name")
            sat2_name = sat2.get("SAT_Name") or colli.get("sat2_name")
            if sat1_name:
                payload["sat1_name"] = sat1_name
            if sat2_name:
                payload["sat2_name"] = sat2_name
            return payload
        except Exception as e:
            logging.error("Failed to build API event payload: %s", e)
            return {}

    def _post_with_retry(self, endpoint: str, payload):
        headers = self._get_headers()
        url = f"{self.base_url}{endpoint}"
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.post(url, json=payload, headers=headers, timeout=10)
                if response.status_code in (200, 201):
                    return True
                logging.error(
                    "API request failed (status=%s): %s",
                    response.status_code,
                    response.text,
                )
            except requests.RequestException as exc:
                logging.error("API request exception: %s", exc)

            if attempt < self.max_retries:
                sleep_s = 2 ** (attempt - 1)
                time.sleep(sleep_s)
        return False

    def send_event(self, event):
        payload = self._build_api_event(event)
        if not payload:
            logging.error("Empty payload, skipping send for event: %s", getattr(event, "CDM_ID", "<no-id>"))
            return False
        ok = self._post_with_retry("/api/ca_events", payload)
        if not ok:
            cdm = event.get("CDM_ID") if isinstance(event, dict) else None
            tca = self._format_time(event.get("TCA") if isinstance(event, dict) else None)
            logging.error("Failed to send event CDM_ID=%s TCA=%s", cdm, tca)
        return ok

    def send_batch(self, events):
        items = []
        ids = []
        for e in events:
            p = self._build_api_event(e)
            if p:
                items.append(p)
            if isinstance(e, dict):
                ids.append({"cdm": e.get("CDM_ID"), "tca": self._format_time(e.get("TCA"))})

        if not items:
            logging.error("No valid items to send in batch. ids=%s", ids)
            return False

        payload = {"items": items}
        ok = self._post_with_retry("/api/ca_events/batch", payload)
        if not ok:
            logging.error("Batch send failed. ids=%s", ids)
        return ok
