# 실제 실패 실행 예제 — 아스피린

이 디렉터리는 2026-11-18에 수행된 **실제 실패 실행**의 기록에서 개인 경로만 제거한 사본입니다.

- 상태: `failed` / `current_stage: routes`
- 완료된 단계: `strategies` (전략 3개 생성)
- 실패 단계: `routes` (`ValueError`)
- 최종 합성 경로: **미확보**
- 합성 조건·수율: **미확보**
- 합성 구조 이미지: **미포함** (확보된 경로 없음)

이 예제는 성공 사례가 아닙니다. 새 실행 결과는 이 디렉터리를 덮어쓰지 말고 `runs/<target-id>/` 및 `reports/<target-id>/`에 저장하세요.

파일:
- `manifest.json` — 최종 상태·실패 메시지·사용량
- `execution.log` — 단계 이벤트
- `strategies.json` — 실제 생성된 전략 3개
- `run-report.json` — 단계별 결과 요약
- `seed-manifest.json` — 후속 시드 실험의 원자 맵 참조 오류 기록
