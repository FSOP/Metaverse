"""
LaunchSim — 위성 발사 궤적 시뮬레이션 패키지

3-DOF (point-mass) 발사 궤적을 시뮬레이션합니다.
Falcon-9 Block 5를 기본 차량 모델로 제공하며,
STK ephemeris (.e) 형식으로 결과를 저장합니다.

[주요 구성]
  vehicle.py       — 발사체 모델 (Stage, Vehicle, falcon9_block5)
  environment.py   — 좌표변환, 대기모델, 중력모델
  launch_sites.py  — 알려진 발사장 데이터베이스
  trajectory.py    — 3-DOF 발사 궤적 전파기 (LaunchTrajectory)
  run_launch_sim.py — CLI 실행기 & 예제

[사용 예시]
  from LaunchSim.trajectory import LaunchTrajectory
  from LaunchSim.vehicle import falcon9_block5
  from LaunchSim.launch_sites import get_site

  vehicle = falcon9_block5(payload_mass_kg=5000)
  site = get_site("cape_canaveral_slc40")
  sim = LaunchTrajectory(vehicle, site, launch_time, target_alt_km=550, target_inc_deg=53.0)
  result = sim.run()
  result.save_stk_ephemeris("output.e")
"""

from LaunchSim.vehicle import Stage, Vehicle, falcon9_block5
from LaunchSim.launch_sites import get_site, make_custom_site, LAUNCH_SITES
from LaunchSim.trajectory import LaunchTrajectory

__all__ = [
    "Stage", "Vehicle", "falcon9_block5",
    "get_site", "make_custom_site", "LAUNCH_SITES",
    "LaunchTrajectory",
]
