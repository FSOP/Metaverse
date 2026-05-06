"""Reentry analysis result uploader (ephemeris upload + summary).

Implements upload_sampled_ephemeris and build_orbit_summary directly.
"""
from typing import List, Optional, Dict, Any, Tuple
import os
import gzip
import json as _json
import time
import math
import requests
from datetime import datetime, timezone
import shutil

try:
    import msgpack
    _HAS_MSGPACK = True
except Exception:
    msgpack = None  # type: ignore
    _HAS_MSGPACK = False

from MISC.config import API_BASE_URL, EPHEMERIS_UPLOAD_PATH, EPHEMERIS_API_KEY
from MISC.stk_ephemeris import write_stk_ephemeris

EARTH_MEAN_RADIUS_M = 6_371_000.0

def _ensure_iso(dt: Any) -> str:
    if isinstance(dt, str):
        return dt
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return str(dt)

def serialize_ephemeris(rows: List[List[Any]], prefer: str = "msgpack", compress: bool = True) -> Tuple[bytes, str, Optional[str]]:
    import numpy as np
    def to_primitive(x):
        # Recursively convert numpy arrays/scalars and dicts to Python primitives
        if isinstance(x, np.ndarray):
            return [to_primitive(i) for i in x.tolist()]
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, dict):
            return {k: to_primitive(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [to_primitive(i) for i in x]
        if isinstance(x, datetime):
            return _ensure_iso(x)
        return x

    serializable = []
    # Debug: print type of first few rows
    for i, r in enumerate(rows[:3]):
        print(f"[serialize_ephemeris] Row {i} type: {type(r)}")
    for r in rows:
        if r is None or (hasattr(r, '__len__') and len(r) == 0):
            continue
        # numpy 배열이면 리스트로 변환
        if isinstance(r, np.ndarray):
            r = r.tolist()
        elif not isinstance(r, list):
            try:
                r = list(r)
            except Exception:
                r = [r]
        # 시간은 ISO 문자열로 변환해서 직렬화 가능한 형태로 만듦
        t = _ensure_iso(r[0])
        serializable.append([t] + [to_primitive(x) for x in r[1:]])

    use_msgpack = prefer == "msgpack" and _HAS_MSGPACK
    if use_msgpack:
        packed = msgpack.packb(serializable, use_bin_type=True)
        content_type = "application/x-msgpack"
        payload = packed
    else:
        payload = _json.dumps(serializable, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        content_type = "application/json"

    content_encoding = None
    if compress:
        payload = gzip.compress(payload)
        content_encoding = "gzip"

    return payload, content_type, content_encoding

def build_orbit_summary(rows: List[List[Any]], sample_rate_minutes: Optional[float] = None) -> Dict[str, Any]:
    if not rows:
        return {}
    times = [_ensure_iso(r[0]) for r in rows]
    first = rows[0][0]
    last = rows[-1][0]
    if sample_rate_minutes is None and len(rows) >= 2:
        dt = (rows[1][0] - rows[0][0]).total_seconds() / 60.0
        sample_rate_minutes = round(dt, 3)

    alts = []
    for r in rows:
        try:
            x, y, z = float(r[1]), float(r[2]), float(r[3])
            norm = math.sqrt(x * x + y * y + z * z)
            alts.append(norm - EARTH_MEAN_RADIUS_M)
        except Exception:
            continue

    summary: Dict[str, Any] = {
        "start_time": _ensure_iso(first),
        "end_time": _ensure_iso(last),
        "sample_rate_minutes": sample_rate_minutes,
        "points_count": len(rows),
    }
    if alts:
        summary.update({"min_alt_m": min(alts), "max_alt_m": max(alts)})

    return summary

def upload_ephemeris(endpoint: str, payload: bytes, content_type: str, content_encoding: Optional[str] = None, headers: Optional[Dict[str, str]] = None, timeout: int = 30, max_retries: int = 3, params: Optional[Dict[str, str]] = None, use_multipart: bool = False, form_fields: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    hdrs = headers.copy() if headers else {}
    if not use_multipart:
        hdrs.setdefault("Content-Type", content_type)
        if content_encoding:
            hdrs.setdefault("Content-Encoding", content_encoding)

    attempt = 0
    backoff = 1.0
    last_exc = None
    while attempt < max_retries:
        try:
            if use_multipart:
                files = {}
                if content_encoding:
                    files['file'] = ('ephemeris.bin', payload, content_type, {'Content-Encoding': content_encoding})
                else:
                    files['file'] = ('ephemeris.bin', payload, content_type)
                resp = requests.post(endpoint, files=files, data=form_fields or {}, headers=hdrs, params=params, timeout=timeout)
            else:
                resp = requests.post(endpoint, data=payload, headers=hdrs, timeout=timeout)
            resp.raise_for_status()
            try:
                return resp.json()
            except Exception:
                return {"status_code": resp.status_code, "text": resp.text}
        except Exception as e:
            last_exc = e
            attempt += 1
            time.sleep(backoff)
            backoff *= 2.0

    raise last_exc

def upload_sampled_ephemeris(rows: List[List[Any]], prefer: str = "msgpack", compress: bool = True, timeout: int = 30, max_retries: int = 3, crash_id: Optional[str] = None, use_multipart: bool = True, metadata: Optional[Dict[str, Any]] = None, debug_metadata: Optional[Dict[str, Any]] = None) -> str:
    payload, content_type, content_encoding = serialize_ephemeris(rows, prefer=prefer, compress=compress)
    endpoint = f"{API_BASE_URL.rstrip('/')}" + EPHEMERIS_UPLOAD_PATH
    headers = None
    if EPHEMERIS_API_KEY:
        headers = {"Authorization": f"Bearer {EPHEMERIS_API_KEY}"}

    params = None
    if crash_id:
        params = {"crash_id": crash_id, "ephemeris_format": "bin"}

    form_fields = None
    if metadata is not None and use_multipart:
        def _fmt_sql_dt(s):
            if s is None:
                return None
            if isinstance(s, str):
                for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S"):
                    try:
                        from datetime import datetime
                        dt = datetime.strptime(s, fmt)
                        return dt.strftime("%Y-%m-%d %H:%M:%S")
                    except Exception:
                        continue
            return s

        try:
            md = dict(metadata)
            if 'creation_date' in md:
                md['creation_date'] = _fmt_sql_dt(md.get('creation_date'))
            if 'time_crash' in md:
                md['time_crash'] = _fmt_sql_dt(md.get('time_crash'))
            form_fields = {"metadata": _json.dumps(md, ensure_ascii=False)}
        except Exception:
            form_fields = {"metadata": str(metadata)}

    # 직접 아카이브 디렉터리에 바로 기록합니다 (tmp_uploads 사용 안함)
    try:
        cid = crash_id or (metadata.get('crash_id') if metadata else 'unknown')
        safe_cid = str(cid).replace('/', '_')

        archive_root = os.path.join(os.path.dirname(__file__), '..', 'data', 'archives', 'ephemeris')
        archive_root = os.path.abspath(archive_root)
        os.makedirs(archive_root, exist_ok=True)
        archive_dir = os.path.join(archive_root, safe_cid)
        os.makedirs(archive_dir, exist_ok=True)

        ts = time.strftime('%Y%m%d_%H%M%S')

        # payload (바이너리) 저장
        try:
            payload_fname = os.path.join(archive_dir, f"{safe_cid}_payload_{ts}.bin")
            with open(payload_fname, 'wb') as f:
                f.write(payload)
        except Exception:
            pass

        # metadata 저장
        try:
            meta_fname = os.path.join(archive_dir, f"{safe_cid}_metadata_{ts}.json")
            with open(meta_fname, 'w', encoding='utf-8') as f:
                f.write(_json.dumps(metadata or {}, ensure_ascii=False, indent=2))
        except Exception:
            pass

        # debug metadata 저장
        try:
            if debug_metadata is not None:
                debug_fname = os.path.join(archive_dir, f"{safe_cid}_debug_{ts}.json")
                with open(debug_fname, 'w', encoding='utf-8') as f:
                    f.write(_json.dumps(debug_metadata, ensure_ascii=False, indent=2, default=str))
        except Exception:
            pass

        # payload 디코드(필요시) 및 샘플 저장
        try:
            raw = payload
            if content_encoding == 'gzip':
                raw = gzip.decompress(payload)
            decoded = None
            if content_type == 'application/x-msgpack':
                try:
                    decoded = msgpack.unpackb(raw, raw=False)
                except Exception:
                    decoded = None
            elif content_type == 'application/json':
                try:
                    decoded = _json.loads(raw.decode('utf-8'))
                except Exception:
                    decoded = None
            if decoded is not None:
                try:
                    sample = decoded[:20] if isinstance(decoded, list) else decoded
                    dec_fname = os.path.join(archive_dir, f"{safe_cid}_decoded_{ts}.json")
                    with open(dec_fname, 'w', encoding='utf-8') as f:
                        f.write(_json.dumps(sample, ensure_ascii=False, indent=2, default=str))
                except Exception:
                    pass
        except Exception:
            pass

        # TLE 파일 저장 (디버그 메타데이터에 TLE 라인이 하나라도 있으면 저장)
        try:
            tle_lines = []
            if debug_metadata is not None:
                l1 = debug_metadata.get('tle_line1')
                l2 = debug_metadata.get('tle_line2')
                if l1:
                    tle_lines.append(l1.strip())
                if l2:
                    tle_lines.append(l2.strip())
            # fallback: 메타데이터에 TLE가 들어있을 수도 있으므로 확인
            if not tle_lines and metadata is not None:
                l1 = metadata.get('tle_line1') if isinstance(metadata, dict) else None
                l2 = metadata.get('tle_line2') if isinstance(metadata, dict) else None
                if l1:
                    tle_lines.append(l1.strip())
                if l2:
                    tle_lines.append(l2.strip())

            if tle_lines:
                tle_path = os.path.join(archive_dir, f"{safe_cid}_tle.txt")
                with open(tle_path, 'w', encoding='utf-8') as tf:
                    for ln in tle_lines:
                        tf.write(ln + "\n")
                try:
                    print(f"[reentry_uploader] Wrote TLE file: {tle_path}")
                except Exception:
                    pass
        except Exception:
            pass

        # STK ephemeris 파일 저장
        try:
            stk_path = os.path.join(archive_dir, f"{safe_cid}_ephemeris_{ts}.e")
            write_stk_ephemeris(rows, stk_path)
        except Exception:
            pass
    except Exception:
        pass

    resp = upload_ephemeris(endpoint, payload, content_type, content_encoding, headers=headers, timeout=timeout, max_retries=max_retries, params=params, use_multipart=use_multipart, form_fields=form_fields)
    if isinstance(resp, dict):
        return resp.get('url') or resp.get('id') or str(resp)
    return str(resp)
