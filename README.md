# SynthEx + Solar Pro 4 — 재현 가능한 실행 및 보고서 저장소

> 이 저장소는 **실패 실행을 정직하게 기록하고 재생성하는 것**을 목적으로 합니다.
> 포함된 아스피린 예제는 `failed` 실행이며 합성 경로 탐색 성공 사례가 아닙니다.

## 1. 프로젝트 목적

- SynthEx(Submission 기준) 파이프라인을 Solar Pro 4(Upstage, OpenAI 호환 API)에 연결합니다.
- 물질 SMILES를 입력받아 전략 생성 → 경로 생성 순서로 실행합니다.
- 성공/실패, 오류, 사용량, 단계 시간을 체크포인트와 실행 기록으로 남깁니다.
- `synthex-report` 스킬로 같은 기록에서 Markdown/DOCX/PDF 보고서를 **재생성**합니다.
- 실제 확보되지 않은 경로·조건·수율은 만들지 않고 `미확보`로 표시합니다.

## 2. 저장소 구조

```text
.
├── .opencode/skills/synthex-report/
│   └── SKILL.md                 # 스킬 정의(입력→실행→근거→보고서)
├── scripts/
│   ├── synthex_adapters/solar.py         # Solar Pro 4 LLMClient 어댑터
│   ├── synthex_solar_pro4_run_staged.py  # 전략→경로 단계별 실행기
│   ├── probe_solar_pro4.py               # API 계약 확인(선택)
│   └── generate_final_report.py          # MD/DOCX/PDF 생성기(20쪽 제한)
├── examples/aspirin-failed-run/          # 실제 실패 실행의 비공개정보 제거 사본
├── src/, tests/, configs/, docs/, schemas/
├── .env.example
├── .gitignore
├── pyproject.toml
└── README.md
```

실행 결과 기본 위치는 `runs/<target-id>/`, 보고서 기본 위치는 `reports/<target-id>/`입니다. 둘 다 커밋되지 않습니다.

## 3. 설치

Python 3.11–3.14를 권장합니다.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

python -m pip install -e ".[solar,report]"
python -m pip install -e ".[dev]"   # 테스트용(선택)
```

## 4. Solar Pro 4 API URL 및 모델 설정

| 항목 | 값 |
|---|---|
| 엔드포인트 | `https://api.upstage.ai/v1/chat/completions` |
| OpenAI 호환 base URL | `https://api.upstage.ai/v1` |
| 요청 모델 | `solar-pro4` |
| 실제 응답 모델 예시 | `solar-pro4-260806` (기록에 남은 값) |
| 스크립트 어댑터 | `scripts/synthex_adapters/solar.py` |

SynthEx 설정 리터럴 `llm.provider`는 `"gemini"`만 허용하므로 내부 설정에는 `gemini`가 기록될 수 있습니다. 실제 네트워크 호출은 Solar 어댑터가 수행하며, 실행 보고서에는 실제 provider/model/응답 모델을 별도로 기록합니다.

모델 변경은 실행 스크립트의 `--model` 값 또는 어댑터 기본값을 수정하세요. (현재 실행기 기본 모델은 `solar-pro4`입니다.)

## 5. API 키 환경변수

키는 **환경변수로만** 주입합니다. 파일에 쓰거나 커밋하지 마세요.

우선순위: `SOLAR_API_KEY` → `OPENCODE_SOLAR_KEY` → `UPSTAGE_API_KEY` → `OPENCODE_API_KEY`

```bash
# macOS / Linux
export SOLAR_API_KEY='<YOUR_UPSTAGE_API_KEY>'

# Windows PowerShell
$env:SOLAR_API_KEY = '<YOUR_UPSTAGE_API_KEY>'
```

`.env`를 사용하려면 `.env.example`을 복사해 값을 채우세요. `.env`는 `.gitignore`로 제외됩니다.

## 6. 물질명 및 SMILES 입력

- **SMILES가 필요합니다.** 이름만으로는 SMILES를 추정하지 않습니다.
- `--smiles`에 SMILES, `--target-id`에 안전한 식별자, `--constraint`에 탐색 지시를 줍니다.

예시(아스피린):

```bash
--smiles "CC(=O)Oc1ccccc1C(=O)O" --target-id aspirin --constraint "가능한 한 짧고 상업적으로 구입 가능한 출발 물질을 사용하는 경로를 탐색하라."
```

## 7. SynthEx 실행

저장소 루트에서 실행합니다.

```bash
python scripts/synthex_solar_pro4_run_staged.py \
  --smiles "CC(=O)Oc1ccccc1C(=O)O" \
  --target-id aspirin \
  --constraint "가능한 한 짧고 상업적으로 구입 가능한 출발 물질을 사용하는 경로를 탐색하라." \
  --output-base runs
```

주요 옵션: `--strategy-count`, `--route-step-limit`, `--route-proposal-attempts`, `--route-temperature`, `--thinking-level`, `--seed-strategy`.

단계는 `strategies` → `routes` 순서입니다. 실패하면 즉시 중단하고 실패 단계와 오류를 기록합니다. 성공하지 않은 단계는 `not_started`로 남습니다.

## 8. `synthex-report` 스킬 호출

스킬 정의는 `.opencode/skills/synthex-report/SKILL.md`에 있습니다.

- 프로젝트 스킬로 사용: 저장소 루트에서 OpenCode를 실행하면 프로젝트 스킬로 인식됩니다.
- 전역 설치: `.opencode/skills/synthex-report` 디렉터리를 `~/.config/opencode/skills/synthex-report`로 복사합니다.

요청 예시:

```text
synthex-report 스킬을 사용해서 runs/aspirin/ 기록으로 보고서를 재생성해 줘.
```

스킬은 항상 `SKILL.md`의 절차를 따릅니다. 보고서 본문을 손으로 고치지 말고, 개선 사항은 `SKILL.md` 또는 생성기에 반영한 뒤 다시 생성합니다.

## 9. DOCX/PDF 생성 방법

```bash
python scripts/generate_final_report.py \
  --run-dir runs/aspirin \
  --output-dir reports/aspirin \
  --title "아스피린 SynthEx 실행 보고서"
```

생성 파일:

- `report.md` — Markdown 원본
- `report.docx` — 제출용 DOCX
- `report.pdf` — 제출용 PDF (최대 20쪽, 초과 시 오류)
- `sources.json` — 근거 파일 경로와 SHA-256

예제 재현(네트워크 호출 없음):

```bash
python scripts/generate_final_report.py \
  --run-dir examples/aspirin-failed-run \
  --output-dir reports/example-aspirin-failed \
  --title "아스피린 SynthEx 실행 보고서 (실패 예제)"
```

## 10. 성공 및 실패 결과 확인 방법

`runs/<target-id>/manifest.json`이 기준입니다.

| 필드 | 의미 |
|---|---|
| `status` | `completed` 또는 `failed` |
| `current_stage` | 마지막으로 도달한 단계 |
| `completed_stages` | 실제 완료된 단계 목록 |
| `failure.category` / `failure.message` | 실패 원류 |
| `llm_usage` | 호출 수·토큰 사용량 |

판정 규칙:

- **성공**: 경로 체크포인트가 생성되고 `ArtifactStore.validate()`를 통과한 경우에만 해당합니다.
- **실패**: strategy 생성 성공만으로는 성공이 아닙니다. `routes` 실패 시 전체 실행은 `failed`입니다.
- 예제 아스피린 실행은 `status=failed`, `current_stage=routes`입니다. 최종 합성 경로/조건/수율은 **미확보**입니다.

## 11. 개인정보·비밀정보 정책

- API 키, 토큰, 비밀번호는 커밋하지 않습니다.
- 개인 홈 경로(`C:\Users\...`)는 예제에서도 제거했습니다.
- `runs/`, `reports/`, `.env`, 루트 `*.log`, 임시 파일은 `.gitignore`로 제외합니다.

## 12. 라이선스

Upstream SynthEx 라이선스와 서드파티 고지를 따릅니다. (`LICENSE`, `THIRD_PARTY_NOTICES.md` 참조)
