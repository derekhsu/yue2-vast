# SPEC — YuE2 on vast.ai

版本：v1（draft，待 review）
日期：2026-09-15

## Problem Statement

需要在雲端 GPU 上自架 YuE2（`m-a-p/YuE2-3B`）音樂生成服務，提供可程式化呼叫的 API。本機（Apple M4）無 CUDA，無法跑官方 pipeline（要求 NVIDIA GPU + BF16）。YuE2 權重為 CC BY-NC 4.0 非商用授權，用途限 POC／評測／個人創作。比照 minimax-music-gen 的 proven pattern：vast.ai 按需租用、zero-image-build、隨用隨停。

## Goals

1. 在 vast.ai 單卡 instance 上跑起 YuE2 inference，提供 `POST /v1/audio/speech`（沿用 minimax-music-gen 的 contract 形狀，client 可複用）。
2. 暴露 YuE2 的白盒能力：ABC 樂譜輸入（`abc`）、三種 CoT 模式（`full`/`melody`/`off`）、`plan()` 分段 API。
3. 部署可重現：所有版本 pin 死（repo commit、HF 權重 revision、wheel 依賴），新 instance 從零到可用不需人工介入。
4. 成本可控：閒置 stop 保留 disk，重開免重抓權重。
5. 安全：對外只暴露一個帶 auth 的入口（Caddy/Portal），inference port 只綁 127.0.0.1。

## Non-Goals

- **不做 Web UI**：YuE2 無現成 UI（HF 討論串在敲碗 ComfyUI 但尚無）；POC 用 curl/smoke test 驗收。
- **不做 vLLM server 模式**：官方 vLLM 路線目標是 H800 並發 32（373 songs/hr），POC 單卡 FIFO 已夠。
- **不做 SheetSage2 cover pipeline**：轉譜是獨立環境（MERT2 encoder），client 端自行準備 ABC；server 只接受現成 `abc` 輸入。
- **不做 GGUF/audio.cpp 路線**：自訂 API，非官方 contract。
- **不自建 Docker image**：`vastai/pytorch` + `PROVISIONING_SCRIPT` 已足夠。
- **不做 SSE streaming**：YuE2 的 `on_token` callback 存在但 POC 不暴露；阻塞回應即可。
- **不做商用**：CC BY-NC 4.0 權重，授權紅線不碰。

## User Stories

- As a **呼叫方（程式/agent）**，我想 POST 歌詞+風格到 `/v1/audio/speech`，拿回 FLAC/WAV，以便整合進 pipeline。
- As a **呼叫方**，我想傳 `abc` 樂譜 + `cot=melody` 做 cover/重編曲，以便精確控制旋律。
- As a **呼叫方**，我想先 `POST /v1/audio/plan` 拿 ABC 譜、人工改完再送回生成，以便做 agentic editing。
- As a **維運者（自己）**，我想 instance 重開後服務自動恢復（權重已在 disk 上），以便隨用隨停控制成本。
- As a **維運者**，我想用固定 seed 重現同一首歌，以便除錯與比較 prompt 效果。

## Requirements

### P0 — Must Have

| # | 需求 | 驗收條件 |
|---|------|----------|
| R1 | vast.ai instance 開機後全自動完成部署：建 venv → 裝依賴 → 下載權重 → 啟動服務 | 新 instance 從建立到 `/health` 回 200 無需人工介入；`PROVISIONING_SCRIPT` 指向 repo 內 `onstart.sh` |
| R2 | inference server 提供生成 contract | `POST /v1/audio/speech` 接受 `input`（歌詞）/`instructions`（style）/`seed`/`cot`/`abc`/`cfg_scale`/`id`/`response_format`，回 FLAC 或 WAV；`GET /v1/models`、`GET /health` 可用 |
| R3 | plan API | `POST /v1/audio/plan` 回 `{abc, truncated, timing}`，不產音訊 |
| R4 | 對外單一入口 + auth | inference（7862）只綁 127.0.0.1；對外經 Caddy :8787（Portal auth + TLS）；inference 層另驗 `Authorization: Bearer $MUSIC_API_KEY`（defense in depth），錯誤 key 回 401 |
| R5 | 版本鎖定 | YuE repo commit、YuE2-3B / YuE2-Vae HF revision、依賴版本全部 pin 死，文件記錄 |
| R6 | 權重與產出持久化 | `HF_HOME=/workspace/hf`、artifacts 存 `/workspace/outputs/<id>/`；stop → start 後不需重抓權重 |
| R7 | smoke test | `smoke_test.sh`：`/health` 200 → 無 key 回 401 → 短歌詞生成 → FLAC magic bytes + 非零大小驗證 |
| R8 | Python 3.12 環境 | venv 用 `uv venv --python 3.12` 建在 `/workspace`（持久），不依賴 image 內建 Python 版本 |

### P1 — Nice to Have

| # | 需求 | 驗收條件 |
|---|------|----------|
| R9 | artifacts 全留檔 | 每次生成 `save_artifacts()`：score.abc、semantic.npy、latent.npy、request/config/result.json |
| R10 | 低 VRAM 選項 | env `MEMORY_BUDGET_GIB` 可調（預設 24；≤12 時 pipeline 自動切小 VAE core frames） |

### P2 — Future

- Web UI（等上游 ComfyUI/社群方案成熟，或自架極簡 Gradio）
- vLLM server 模式（高並發需求出現時）
- SheetSage2 轉譜 sidecar（cover 工作流全自動化）
- yue2-music agent skill 整合（官方 repo 內建 `skills/yue2-music/`）
- 自建 all-in-one Docker image（頻繁開新 instance 時）

## 技術決策（已定）

| 項目 | 決定 | 理由 |
|------|------|------|
| 後端 | 官方 `yue2-infer`（`pip install .` from pinned commit）+ 自寫薄 FastAPI wrapper | 官方 pipeline 已封裝好（`YuE2Pipeline.from_pretrained` → `pipe()` → `SongResult`）；無未合併 PR 風險（對比 Music 3 的 diffusers 坑） |
| 部署 | `vastai/pytorch:cuda-12.8.1-auto` + `PROVISIONING_SCRIPT` → `onstart.sh` | 沿用 M2 驗證過的模式；明確 tag 不用 `@vastai-automatic-tag`（M2 坑 2） |
| Python | `uv venv --python 3.12` 於 `/workspace/venv312` | README 要求 3.12；image 內建版本不保證；venv 持久化在 disk 上 |
| GPU | 1× RTX 3090（24GB，~$0.13–0.2/h） | 官方實測峰值 11.2 GiB；3090 綽綽有餘且便宜；趕時間才上 4090 |
| Disk | ≥40GB | 權重 ~8GB + venv（torch ~5GB）+ HF cache + outputs |
| RAM | ≥24GB host | 官方要求 |
| 輸出格式 | FLAC 預設（PCM_24，上游原生），WAV 可選（PCM_16） | 上游 `save()` 只支援 .flac/.wav；FLAC 檔案小適合傳輸 |

## API Contract

```
POST /v1/audio/speech          Authorization: Bearer $MUSIC_API_KEY
  input            歌詞；[Verse]/[Chorus] 等 tag 獨立一行
  instructions     style prompt（genre/BPM/key/vocal/arrangement）
  seed             固定 → 可重現；省略 = 隨機，實際值從 X-Seed header 取回
  cot              full（預設，旋律+和弦規劃）| melody（cover 推薦）| off（直生）
  abc              自帶 ABC 樂譜（需 cot=full|melody）
  cfg_scale        文本引導強度（預設 full/melody=1.0、off=1.01）
  id               filename-safe 識別子（artifacts 目錄名；預設 song-<seed>）
  response_format  flac（預設）| wav
→ audio/flac 或 audio/wav；headers: X-Seed, X-Request-Id, X-Audio-Seconds, X-Truncated

POST /v1/audio/plan            同 auth；body 同 speech（回 plan 不產音訊）
→ {"id", "abc", "truncated", "timing"}

GET /health                    無 auth；ready 前回 503
GET /v1/models                 同 auth
```

## 檔案結構

```
yue2-vast/
├── SPEC.md                  ← 本檔（what）
├── PLAN.md                  ← 部署細節與調查記錄（how）
├── inference/
│   ├── server.py            ← FastAPI wrapper：FIFO queue + Bearer auth + artifacts
│   └── requirements.txt     ← server 端 deps（yue2-infer 本體由 onstart 從 git 裝）
├── scripts/
│   ├── onstart.sh           ← vast.ai provisioning：venv→deps→權重→supervisor→portal
│   └── smoke_test.sh        ← R7 驗收腳本
├── .env.example             ← MUSIC_API_KEY, HF_TOKEN
└── .gitignore               ← .env, outputs/, hf-cache/, *.flac, *.wav
```

## Success Metrics

| 指標 | 目標 |
|------|------|
| 冷啟動（新 instance → `/health` 200） | < 15 min（權重僅 ~8GB） |
| 熱啟動（stop → start → ready） | < 3 min |
| 3.6 分鐘歌生成時間 | < 90s（3090 推估；4090 官方 71s） |
| smoke test | `ALL CHECKS PASSED` |
| 單次生成成本 | < $0.01/首（3090 @ ~$0.15/h） |

## 風險

| 風險 | 緩解 |
|------|------|
| CC BY-NC 4.0 權重：不可商用 | 紅線寫死；只供 POC/個人評測；對外提供服務即違約 |
| `yue2-infer` 依賴 `torch==2.10.0` 硬 pin | 獨立 venv 安裝，不碰 image 內建 torch；uv 解析 |
| 歌曲長度不可控（由歌詞段落數決定） | smoke test 用短歌詞（1 verse + 1 chorus）；文件註明無 duration 參數 |
| 同步 endpoint 無法中途取消 | POC 接受；client timeout 設 > 預期生成時間 |
| vast.ai 閒置計費 | 不用時 stop（保留 disk）而非 destroy |
| deverified 機器 CDI 坑（M2 教訓） | 只租 verified；3090 價位 verified 供給充足 |

## Open Questions

| 問題 | 待誰回答 | 阻塞？ |
|------|----------|--------|
| 對外暴露：Portal/Caddy auth 是否夠用 | M2 部署後實測 | 否 |
| 3090 實際生成速度（官方只有 4090/H800 數據） | M2 實測 | 否 |
| `yue2` CLI 的 `generate`/`batch` 子命令是否值得包進 server | 使用後評估 | 否 |

## 里程碑

| # | 內容 | 驗收 |
|---|------|------|
| M1 | 檔案就緒：server.py + requirements + onstart.sh + smoke_test.sh | 本地 code review 通過 |
| M2 | vast.ai 首次部署：template + instance + provisioning 全自動 | smoke test `ALL CHECKS PASSED` |
| M3 | 固化：stop/start 驗證持久化、成本記錄、文件補實測數據 | 熱啟動 < 3min；PLAN.md 補實測速度/成本 |
