# Image-Model Curriculum

목표: Diffusion 기반 이미지 생성 개념 → 3D 표현 확장 → RIFT_Engine 오토리깅(`AutoRigger.js`) 연결까지.

학습 방식: 개념 설명 + 힌트 주석이 달린 `...` 스켈레톤 코드를 채워넣고, `__main__`에서 직접 결과를 검증. 반복되는 패턴은 힌트 없이 스스로 구현.

---

## Phase 1. 2D Diffusion 완성

- [x] 1. NoiseScheduler + Sinusoidal Time Embedding — `scratch/diffusion.py`
- [ ] 2. ResBlock (time embedding 주입) — `scratch/unet.py` ← **현재 위치, forward() 채우기부터**
- [ ] 3. Self-Attention block (bottleneck용) — multihead.py의 attention을 feature map에 적용
- [ ] 4. Down/Up sample block + skip connection (인코더-디코더 구조)
- [ ] 5. 전체 UNet 조립 (down → bottleneck(attn) → up, skip 연결)
- [ ] 6. Training loop (ε 예측, MSE loss) — `train.py` 패턴 재사용
- [ ] 7. DDPM Sampling (역과정, x_t → x_0)
- [ ] 8. 실제 학습 + 샘플 이미지 생성 확인 (MNIST 등)

## Phase 2. 조건부 생성 (Text/Class Conditioning)

RIFT 프롬프트 패널이 텍스트로 캐릭터를 생성해야 하니 필요한 개념.

- [ ] 9. Class-conditional embedding (label을 time embedding처럼 주입)
- [ ] 10. Classifier-Free Guidance (CFG)
- [ ] 11. 텍스트 임베딩 연동 — PsyGPT 토크나이저/트랜스포머를 텍스트 인코더로 재활용

## Phase 3. 2D → 3D 표현으로 확장

로컬에서 3D diffusion을 완전히 밑바닥부터 데이터 없이 재현하는 건 비현실적이므로,
개념 이해 + RIFT가 이미 쓰는 외부 API(Tripo3D/Hunyuan3D) 이해를 결합.

- [ ] 12. 3D 표현 방식 개념 (voxel / point cloud / mesh / SDF) — 개념+간단 시각화 위주
- [ ] 13. Point cloud diffusion 미니 구현 — Phase1 로직을 3D 좌표에 적용 (UNet 대신 PointNet류 MLP)
- [ ] 14. `RIFT_Engine/src/agent/generators/MeshGenerator.js`의 실제 외부 API 호출 코드 리딩 →
      그 API가 내부적으로 쓰는 diffusion 기법과 연결지어 이해

## Phase 4. 오토리깅 개념 → `AutoRigger.js` 연결

- [ ] 15. 스켈레톤/본 계층구조, Humanoid bone mapping 개념
- [ ] 16. Skinning weight 개념 (Linear Blend Skinning, geodesic distance 기반) —
      미니 구현: 막대인간 2본짜리로 weight 계산
- [ ] 17. RigNet류 학습 기반 오토리깅 vs 현재 `RIFT_Engine/src/agent/rigging/AutoRigger.js`가
      실제 쓰는 방식 코드 대조
- [ ] 18. 배운 개념으로 `AutoRigger.js` 개선/디버깅 포인트 찾기 — 실전 연결

## Phase 5. 파이프라인 통합

- [ ] 19. 이미지/3D 생성 → 리깅 → `ChampRegistry` AST 주입까지 전체 흐름을 다시 그려보고,
      직접 만든/이해한 모델을 끼울 지점 설계

---

## 진행 로그

- 2026-08-27: 커리큘럼 최초 작성. Phase 1의 1번(diffusion.py) 완료, 2번(unet.py ResBlock)
  forward() 채우다 중단된 상태에서 이어가기로 함.
