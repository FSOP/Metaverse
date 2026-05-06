"""STK 형식의 ephemeris 파일(.e)을 쓰는 유틸리티 모듈입니다.

입력 형식:
    rows: 각 행이 [datetime, x, y, z, vx, vy, vz] 형태의 리스트
    위치 단위는 미터(m), 속도 단위는 미터/초(m/s)로 가정합니다.

이 모듈은 STK이 읽을 수 있는 Ephemeris 블록을 생성하며,
시간 오프셋(ScenarioEpoch으로부터의 초)을 기록합니다.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, List


def _format_stk_epoch(dt: datetime) -> str:
    """STK의 ScenarioEpoch 문자열 형식으로 변환합니다.

    예: "1 Jan 2026 12:34:56.789000"
    월은 영어 단축형(%b)을 사용하고, 소수 초까지 포함합니다.
    """
    month = dt.strftime("%b")
    return f"{dt.day} {month} {dt.year} {dt.strftime('%H:%M:%S.%f')}"


def write_stk_ephemeris(rows: List[List[Any]], file_path: str) -> None:
    """주어진 ephemeris 행들을 STK .e 파일로 기록합니다.

    처리 절차:
    1) 첫 번째 행의 시간값을 ScenarioEpoch으로 사용합니다.
    2) 헤더(포인트 수, 보간법 등)를 기록합니다.
    3) 각 행을 ScenarioEpoch으로부터의 초(seconds)로 환산해
       EphemerisTimePosVel 블록에 기록합니다.

    유의사항:
    - 시간 값이 문자열인 경우 ISO UTC 형식("%Y-%m-%dT%H:%M:%SZ")을
      우선 파싱하려 시도합니다. 실패하면 ScenarioEpoch을 사용합니다.
    - timezone 정보가 없는 datetime은 UTC로 간주합니다.
    """
    if not rows:
        return

    # ScenarioEpoch으로 첫 번째 행의 시간을 사용
    epoch_dt = rows[0][0]
    if isinstance(epoch_dt, str):
        try:
            epoch_dt = datetime.strptime(epoch_dt, "%Y-%m-%dT%H:%M:%SZ")
            epoch_dt = epoch_dt.replace(tzinfo=timezone.utc)
        except Exception:
            epoch_dt = datetime.now(timezone.utc)
    elif isinstance(epoch_dt, datetime) and epoch_dt.tzinfo is None:
        epoch_dt = epoch_dt.replace(tzinfo=timezone.utc)

    epoch_str = _format_stk_epoch(epoch_dt)
    num_points = len(rows)

    # 숫자 출력 포맷: 충분한 유효자릿수로 지수형 표기
    def fmt(x: float) -> str:
        return f"{x:.16e}"

    with open(file_path, "w", encoding="utf-8") as f:
        # STK 버전 + Ephemeris 블록 시작
        f.write("stk.v.12.0\n\n")
        f.write("# WrittenBy    STK_v12.0.1\n\n")
        f.write("BEGIN Ephemeris\n\n")

        # 메타데이터: 포인트 개수, 시나리오 epoch, 보간방법 등
        f.write(f"    NumberOfEphemerisPoints\t\t {num_points}\n\n")
        f.write(f"    ScenarioEpoch\t\t {epoch_str}\n\n")
        f.write("    InterpolationMethod\t\t Lagrange\n\n")
        f.write("    InterpolationSamplesM1\t\t 5\n\n")
        f.write("    CentralBody\t\t Earth\n\n")
        f.write("    CoordinateSystem\t\t ICRF\n\n")
        f.write("    EphemerisTimePosVel\t\t\n\n")

        # 실제 데이터: 각 행은 (시나리오 epoch으로부터의 초, x y z vx vy vz)
        for r in rows:
            t = r[0]
            if isinstance(t, str):
                try:
                    t = datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                except Exception:
                    t = epoch_dt
            if isinstance(t, datetime) and t.tzinfo is None:
                t = t.replace(tzinfo=timezone.utc)

            dt_seconds = (t - epoch_dt).total_seconds() if isinstance(t, datetime) else 0.0
            x, y, z = float(r[1]), float(r[2]), float(r[3])
            vx, vy, vz = float(r[4]), float(r[5]), float(r[6])
            f.write(
                f" {fmt(dt_seconds)}  {fmt(x)} {fmt(y)} {fmt(z)} {fmt(vx)} {fmt(vy)} {fmt(vz)}\n"
            )

        # Ephemeris 블록 종료
        f.write("\nEND Ephemeris\n")
