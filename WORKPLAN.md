# 작업 계획서 — 분석 컴퓨터 이전 후 복구 작업

> **작성일**: 2026-05-07  
> **담당**: 분석 컴퓨터 이전(구 `/home/user1229/` → 현재 Ubuntu 서버) 후 작업 큐 정상화  
> **하네스 프로그래밍 원칙 적용**: 작업 전/후 commit & push, 계획 문서 선행 작성, 검증 절차 명시

---

## 배경 및 현황

`https://lab.sejong-stm.cloud/` 서버와 연동하는 분석 컴퓨터를 새 머신으로 이전한 후 아래 증상이 발생:

| 증상 | 영향 |
|------|------|
| CA 분석 결과가 웹에 표시 안 됨 | `CA_API_TOKEN` 환경변수 미설정 |
| TLE 업데이트 후 `SUBTASK_REPORT 파싱 실패` 경고 | `run_once.py`가 JSON 출력 안 함 |
| TLE 파일 저장 경로 오류 | `/home/user1229/` 하드코딩 → 수정 완료(TLEmanager.py) |
| `job_worker.py`가 git 미추적 | untracked 상태 |

---

## 근본 원인 분석

### 문제 1: CA_API_TOKEN 누락
- **위치**: `MISC/ca_api.py`, `LaunchSim/launch_api.py`, `job_worker.py`
- **원인**: 이전 머신에서 OS 환경변수(`export CA_API_TOKEN=...`)로 설정되어 있었으나, 새 머신에서 재설정 안 됨
- **결과**: `ca_api.py`의 `os.getenv("CA_API_TOKEN")`이 `None`을 반환 → 모든 CA 이벤트 전송 실패

### 문제 2: SUBTASK_REPORT 파싱 실패
- **위치**: `run_once.py` ↔ `job_worker.py`
- **원인**: `job_worker.py`가 `run_once.py` stdout에서 `SUBTASK_REPORT:{...}` JSON 라인을 기대하지만, 현재 `run_once.py`는 이 형식으로 출력하지 않음
- **결과**: TLE delta 업로드를 건너뜀 → TLE가 서버에 반영 안 됨

### 문제 3: TLE 저장 경로 하드코딩 (수정 완료)
- **위치**: `MISC/TLEmanager.py`
- **원인**: `save_path = f"/home/user1229/metaverse/TLEs/tle_{fname}.txt"` 하드코딩
- **수정**: `os.path.dirname(os.path.abspath(__file__))` 기반 상대경로로 변경 (이미 적용됨)

### 문제 4: job_worker.py git 미추적
- `job_worker.py`가 untracked 상태로 버전관리 밖에 있음

---

## 작업 계획

### 작업 0: Before 커밋 (현재 상태 스냅샷)
현재 변경된 파일들을 커밋하여 작업 전 기준점 확보.

**대상 파일**:
- `MISC/TLEmanager.py` (modified - 경로 수정)
- `job_worker.py` (새로 추가)
- `WORKPLAN.md` (이 파일)

```bash
cd ~/metaverse
git add MISC/TLEmanager.py job_worker.py WORKPLAN.md
git commit -m "before: 분석 컴퓨터 이전 후 복구 작업 전 스냅샷"
git push origin main
```

---

### 작업 1: CA_API_TOKEN을 env.py 및 config.py에 통합

**목표**: `env.py`에 `CA_API_TOKEN`을 추가하고, `ca_api.py`가 `MISC/config._env_get()`을 통해 읽도록 수정

**변경 파일**:
- `env.py` — `CA_API_TOKEN = "실제토큰값"` 추가
- `MISC/config.py` — `CA_API_TOKEN = _env_get("CA_API_TOKEN")` 추가
- `MISC/ca_api.py` — `os.getenv()` 대신 `config.CA_API_TOKEN` 사용

**검증**:
```bash
# 1. Python 셸에서 토큰 로딩 확인
cd ~/metaverse && source venv/bin/activate
python -c "from MISC import config; print('Token:', bool(config.CA_API_TOKEN))"

# 2. ca_api.py 단독 테스트 (dry-run)
python -c "
from MISC.ca_api import CAEventSender
from MISC import config
sender = CAEventSender(config.API_BASE_URL)
headers = sender._get_headers()
print('Headers set:', headers is not None)
"
```
**기대 결과**: `Token: True`, `Headers set: True`

---

### 작업 2: run_once.py에 SUBTASK_REPORT JSON 출력 추가

**목표**: TLE 업데이트 완료 후 `job_worker.py`가 파싱할 수 있는 JSON을 stdout에 출력

**출력 형식** (job_worker.py의 `_parse_subtask_report` 참고):
```
SUBTASK_REPORT:{"snapshot_id": "...", "subtask_results": {"tle_update": {...}}}
```

**변경 파일**:
- `run_once.py` — 실행 결과를 `SUBTASK_REPORT:` 접두사와 함께 JSON으로 출력

**검증**:
```bash
# 1. 직접 실행하여 SUBTASK_REPORT 라인 확인
cd ~/metaverse && source venv/bin/activate
python run_once.py 2>&1 | grep "^SUBTASK_REPORT:"

# 2. job_worker.py 파싱 함수 단위 테스트
python -c "
import sys; sys.path.insert(0, '.')
from job_worker import _parse_subtask_report
# 샘플 stdout 생성
sample = 'SUBTASK_REPORT:{\"snapshot_id\": \"test\", \"subtask_results\": {}}'
result = _parse_subtask_report(sample)
print('Parsed:', result is not None, result)
"
```
**기대 결과**: `SUBTASK_REPORT:` 라인 출력됨, 파싱 결과 `snapshot_id` 포함

---

### 작업 3: After 커밋 & 푸시

```bash
cd ~/metaverse
git add MISC/config.py MISC/ca_api.py run_once.py WORKPLAN.md
git commit -m "fix: CA_API_TOKEN config 통합 및 run_once.py SUBTASK_REPORT 출력 추가"
git push origin main
```

---

### 작업 4: job_worker 재시작 및 최종 검증

**현재 실행 중인 worker PID 확인 후 재시작**:
```bash
# 현재 worker 확인
ps aux | grep job_worker | grep -v grep

# 정상 여부 확인 (재시작 없이 로그 모니터링)
tail -f ~/metaverse/job_worker.log
```

**통합 검증 시나리오**:
1. 웹 관리자 페이지에서 TLE_UPDATE 작업 수동 생성
2. job_worker.log에서 `SUBTASK_REPORT 파싱 실패` 경고 **미발생** 확인
3. job_worker.log에서 `Delta TLE 업로드 완료` 로그 확인
4. 웹에서 최근 TLE 업데이트 시간 갱신 확인
5. CA 분석 작업 수동 생성
6. `ca_api_errors.log`에 `CA_API_TOKEN is missing` 오류 **미발생** 확인
7. 웹에서 CA 이벤트 목록에 결과 표시 확인

---

## 파일별 변경 요약

| 파일 | 변경 유형 | 내용 |
|------|-----------|------|
| `env.py` | 수정 (git 제외) | `CA_API_TOKEN` 추가 |
| `MISC/config.py` | 수정 | `CA_API_TOKEN` 항목 추가 |
| `MISC/ca_api.py` | 수정 | `os.getenv()` → `config.CA_API_TOKEN` |
| `MISC/TLEmanager.py` | 수정 완료 | 경로 하드코딩 제거, env.py fallback 추가 |
| `run_once.py` | 수정 | SUBTASK_REPORT JSON 출력 추가 |
| `job_worker.py` | 추가 | git 추적 시작 |
| `WORKPLAN.md` | 신규 | 이 문서 |

---

## 롤백 방법

작업 전 커밋으로 돌아가는 방법:
```bash
git log --oneline -5           # 커밋 해시 확인
git checkout <before-commit-hash> -- <파일명>   # 특정 파일 복원
# 또는 전체 롤백:
git reset --hard <before-commit-hash>
```

> **주의**: `env.py`는 git에서 제외되므로 별도 백업 필요
> `cp env.py env.py.bak`

---

## 작업 이력

| 날짜 | 내용 | 커밋 |
|------|------|------|
| 2026-05-07 | 작업 계획서 작성, 문제 분석 | `before: 분석 컴퓨터 이전 후 복구 작업 전 스냅샷` |
| 2026-05-07 | CA_API_TOKEN config.py 통합 (env.py fallback), ca_api.py 수정 | `fix:` |
| 2026-05-07 | TLEmanager.download_tle_and_save() → save_path 반환하도록 수정 | `fix:` |
| 2026-05-07 | run_once.py → SUBTASK_REPORT JSON 출력 추가, TLE JSON 임시파일 생성 | `fix:` |

### 검증 결과

| 검증 항목 | 결과 |
|-----------|------|
| `config.CA_API_TOKEN` 로딩 | ✅ True |
| `ca_api._get_headers()` 반환 | ✅ Authorization 헤더 포함 |
| `_parse_subtask_report()` 파싱 | ✅ snapshot_id 추출 성공 |
| TLE 파일 → JSON 파싱 31,177개 | ✅ 정상 |
