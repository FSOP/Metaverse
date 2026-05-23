# 성능 최적화 기록 (CA 분석 엔진)

> 작성일: 2026-05-23  
> 대상 파일: `CA/CA_filter.py`  
> 환경: Python 3.x, numpy 2.2, sgp4 라이브러리 (C++ 백엔드), CPU 2코어

---

## 배경

충돌 분석(Conjunction Assessment) 파이프라인의 처리 속도가 느리다는 문제가 있었다.
코드를 분석한 결과, **계산 자체가 느린 것이 아니라 빠른 C 함수를 Python 루프로 1개씩
호출하는 패턴**이 병목임을 확인하였다.

numpy, sgp4(SatrecArray), scipy.cKDTree 등 핵심 연산은 이미 C/C++ 백엔드로
실행되므로 언어 변경보다 **호출 패턴 변경**이 더 효과적이다.

---

## 병목 분석

| 단계 | 함수 | 문제 패턴 | 기존 비용 |
|------|------|-----------|-----------|
| 1단계 | `filter_altitude` | 위성마다 `compute_apogee_perigee` 개별 호출 (Python 루프 N회) | O(N) Python 함수 호출 |
| 3단계 | `filter_time` (시간 배열) | `jday()` 를 Python 루프로 n_coarse(≈5,000)회 호출 | 루프 5,000회 |
| 3단계 | `filter_time` (위성 전파) | 위성 1개씩 `SatrecArray([sat_o])` 생성 후 전파 (N회 반복) | SatrecArray 생성 N회, sgp4 N회 |
| 4단계 | `fine_filter_min_distance` | Python 루프로 1초마다 개별 `satrec.sgp4()` 호출 (후보당 ≈600회) | sgp4 개별 호출 600×M회 |

---

## 최적화 내용 (2026-05-23 적용)

### 병목 1+4: `filter_altitude` numpy 벡터화

**변경 전**
```python
for i in range(len(tle_data)):
    sat2_apogee, sat2_perigee = self.tle_manager.compute_apogee_perigee(line2)
    if not ((sat2_apogee < ref_perigee - pad_km) or ...):
        filtered_tles.append(...)
```

**변경 후**
```python
# TLE line2 슬라이싱 → numpy 배열 → 벡터 연산 1회
sat_apogee, sat_perigee = self._batch_apogee_perigee(line2_list)
keep = ~((sat_apogee < ref_perigee - pad_km) | (sat_perigee > ref_apogee + pad_km))
return [tle_data[i] for i in np.where(keep)[0]]
```

- 신규 정적 메서드: `_batch_apogee_perigee(line2_list)`
- 공식 동일 (`a*(1±e) - R_earth`), 정확도 변화 없음
- 예상 효과: **2~5배 가속**

---

### 병목 2: `filter_time` — jday 배열 numpy 벡터화

**변경 전**
```python
for i in range(n_coarse):           # ≈ 5,040회 (7일 / 120s)
    t_i = start + timedelta(seconds=i * base_coarse_dt)
    jd_i, fr_i = jday(t_i.year, ...)
    jds_coarse[i] = jd_i
```

**변경 후**
```python
jd0, fr0 = jday(start.year, ...)
jd_total = (jd0 + fr0) + np.arange(n_steps) * (dt_s / 86400.0)
jds = np.floor(jd_total)
frs = jd_total - jds
```

- 신규 정적 메서드: `_make_jd_arrays(t0, n_steps, dt_s)` — `filter_time`, `fine_filter_min_distance`, refinement 구간 모두 공유
- `jday()` 호출 1회로 감소 (기존 n_steps회)
- 예상 효과: **시간 배열 생성 10~50배 가속**

---

### 병목 3: `filter_time` — 후보 위성 전체를 단일 SatrecArray로 배치 전파

**변경 전** (위성 N개를 N번 따로 전파)
```python
for other_sat in tle_data:
    o_arr = SatrecArray([sat_o])          # 1개짜리 배열 생성
    e_o, r_o, v_o = o_arr.sgp4(jds, frs) # N번 호출
```

**변경 후** (전체 N개를 1번에 전파)
```python
batch_arr = SatrecArray([sr for _, sr, _ in cand_list])
e_batch, r_batch, _ = batch_arr.sgp4(jds_coarse, frs_coarse)
# r_batch[i] = i번째 위성의 전체 궤적 (n_coarse, 3)
```

- `SatrecArray` 생성 1회, `sgp4` 호출 1회로 감소
- refinement 구간도 `SatrecArray([sat_ref, sat_o])` 2개짜리로 단일 호출
- 예상 효과: **filter_time 단독 5~15배 가속** (후보 위성 수에 비례)

---

### 병목 4: `fine_filter_min_distance` — Python 루프 SGP4 → SatrecArray 배치

**변경 전** (후보당 ≈600회 개별 SGP4 호출)
```python
for t in times:
    jd, fr = jday(...)
    e1, r1, v1 = sat_ref.sgp4(jd, fr)  # 개별 호출
    e2, r2, v2 = sat_o.sgp4(jd, fr)
```

**변경 후** (전체 시간 배열을 1번에 전파)
```python
pair_arr = SatrecArray([sat_ref, sat_o])
e_pair, r_pair, v_pair = pair_arr.sgp4(jds, frs)   # 1회 호출
dists = np.linalg.norm(r_pair[0] - r_pair[1], axis=1)
min_idx = int(np.argmin(dists))
```

- secondary refinement (0.1s)도 동일 방식 적용
- 정확도 변화 없음 (동일 SGP4 엔진, 동일 시간 그리드)
- 예상 효과: **fine_filter 단독 3~8배 가속**

---

## 종합 예상 효과

| 최적화 항목 | 적용 전 | 적용 후 | 정확도 영향 |
|------------|---------|---------|-----------|
| filter_altitude 벡터화 | Python 루프 N회 | numpy 1회 | 없음 |
| jday 배열 numpy화 | Python 루프 5,000회 | numpy 1회 | 없음 |
| filter_time 배치 SGP4 | SatrecArray N회 | SatrecArray 1회 | 없음 |
| fine_filter 배치 SGP4 | 개별 600×M회 | SatrecArray M회 | 없음 |
| **전체** | **기준** | **3~10배 단축 예상** | **없음** |

> 참고: CPU 2코어 환경 기준. 위성 수(N)와 후보 수(M)가 클수록 효과 증가.

---

## 변경되지 않은 항목 (정확도 유지)

- SGP4 전파 엔진 자체 (sgp4 라이브러리 동일)
- 시간 그리드 해상도: coarse 120s, refine 1~30s, fine 1s + 0.1s
- 최소 접근 거리 임계값: 5km (SOCRATES 기준)
- Alfano 2D 최대 충돌확률 계산 알고리즘
- KDTree 기반 궤도 경로 필터 (`filter_orbitpath`)
- 고도 필터 판정 조건 (동일 공식)

---

## 향후 개선 여지

| 항목 | 예상 효과 | 난이도 |
|------|---------|--------|
| `filter_orbitpath` segment 거리 계산 numpy화 | 중간 | 중간 |
| Numba JIT (`@jit`) — 남은 Python 루프 가속 | 중간 | 낮음 |
| CPU 코어 추가 (하드웨어) | 코어 수 비례 | 하드웨어 |
