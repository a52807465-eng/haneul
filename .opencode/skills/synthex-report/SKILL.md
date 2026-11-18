---
name: synthex-report
description: SynthEx를 Solar Pro 4로 실행하고 실제 실행 기록에서 DOCX/PDF 보고서를 재생성한다. 물질 입력부터 실패/성공 판정까지 재현 가능하게 수행해야 할 때 사용한다.
---

# Skill: synthex-report

이 스킬은 **실제 저장된 기록만** 근거로 SynthEx + Solar Pro 4 실행 보고서를 만든다. 생성되지 않은 경로·조건·수율·실험값을 만들지 않는다.

## 0. 절대 규칙

1. API 키는 환경변수로만 읽고 어느 파일에도 쓰지 않는다.
2. 실제 성공한 단계만 `completed`로 기록한다. 실패하면 즉시 중단하고 실패 단계를 남긴다.
3. `strategies.json`의 문장은 제안일 뿐 검증된 경로가 아니다.
4. 합성 조건과 수율은 검증된 경로 기록에 있을 때만 기재하고, 없으면 `미확보`로 쓴다.
5. 합성 구조 이미지는 실제 확보된 경로가 있을 때만 포함한다. 없으면 이미지를 넣지 않는다.
6. 보고서 개선은 이 `SKILL.md` 또는 `scripts/generate_final_report.py`를 수정한 뒤 재생성한다. 생성된 본문을 직접 고쳐 꾸미지 않는다.
7. DOCX/PDF는 최대 20쪽이어야 한다. 생성기가 초과 시 오류를 내면 원인을 줄이고 다시 생성한다.
8. `examples/aspirin-failed-run/`는 실제 실패 예제이며 성공 사례가 아니다. 새 실행과 혼동하지 않는다.

## 1. 사전 확인

저장소 루트에서 다음을 확인한다.

```bash
python --version
python -m pip install -e ".[solar,report]"
python scripts/synthex_solar_pro4_run_staged.py --help
python scripts/generate_final_report.py --help
```

필요 파일:

- `scripts/synthex_adapters/solar.py`
- `scripts/synthex_solar_pro4_run_staged.py`
- `scripts/generate_final_report.py`
- `src/synthex/`, `configs/`, `schemas/`

## 2. 입력 수집

사용자에게 다음을 받는다.

- **SMILES** (필수): 이름만 주어지면 SMILES를 추정하지 말고 요청한다.
- `target-id` (선택): 없으면 안전한 id를 만든다.
- 제약 조건(선택): 짧은 경로, 구입 가능한 출발물질 등.

아스피린 예: `CC(=O)Oc1ccccc1C(=O)O`

## 3. 키 설정

```bash
# macOS / Linux
export SOLAR_API_KEY='<YOUR_UPSTAGE_API_KEY>'
# Windows PowerShell
$env:SOLAR_API_KEY = '<YOUR_UPSTAGE_API_KEY>'
```

다른 허용 변수: `OPENCODE_SOLAR_KEY`, `UPSTAGE_API_KEY`, `OPENCODE_API_KEY`.

## 4. SynthEx 실행

```bash
python scripts/synthex_solar_pro4_run_staged.py \
  --smiles "<SMILES>" \
  --target-id "<TARGET_ID>" \
  --constraint "<CONSTRAINT>" \
  --output-base runs
```

실행 후 `runs/<target-id>/`에 다음이 있어야 한다.

- `manifest.json`
- `execution.log`
- `strategies.json`
- `run-report.json` (경로 단계까지 진입한 경우)

실패해도 위 기록은 남는다. 실패 시 재시도 전에 오류를 먼저 읽는다.

## 5. 근거 취합 및 실패 판정

1. `manifest.json`의 `status`, `current_stage`, `completed_stages`, `failure`를 읽는다.
2. `execution.log`에서 단계 시작/성공/실패 이벤트를 확인한다.
3. `strategies.json`에서 실제 생성 문장만 읽는다.
4. `run-report.json`에서 LLM 호출, 토큰, 단계 시간을 읽는다.
5. `status=failed`이면 성공으로 쓰지 않는다. `routes` 실패 시 downstream 단계는 `not_started`다.
6. 후속 실험이 별도 디렉터리에 있으면 별도 기록으로만 언급하고 본 실행 manifest를 대체하지 않는다.

## 6. 보고서 생성

```bash
python scripts/generate_final_report.py \
  --run-dir runs/<target-id> \
  --output-dir reports/<target-id> \
  --title "<REPORT_TITLE>"
```

생성기 출력:

- `report.md`
- `report.docx`
- `report.pdf`
- `sources.json`

생성기는 `manifest.json`을 기준으로 상태를 쓰며, 조건·수율은 기록에 없으면 `미확보`로 표시한다. PDF가 20쪽을 넘으면 생성이 실패하므로 내용을 줄인 뒤 다시 실행한다.

## 7. 재현 예제(오프라인)

```bash
python scripts/generate_final_report.py \
  --run-dir examples/aspirin-failed-run \
  --output-dir reports/example-aspirin-failed \
  --title "아스피린 SynthEx 실행 보고서 (실패 예제)"
```

이 예제는 실제 2026-11-18 실행의 개인정보 제거 사본이며 `failed`다. 최종 경로/조건/수율은 미확보다.

## 8. 최종 확인

- [ ] `report.docx`, `report.pdf`, `report.md`, `sources.json` 존재
- [ ] PDF 페이지 수 ≤ 20
- [ ] 보고서에 API 키/개인 경로 없음
- [ ] 실패 실행이 성공으로 표현되지 않음
- [ ] 조건·수율은 실제 기록 또는 `미확보`
- [ ] 근거 목록에 실제 파일 경로와 SHA-256 포함
