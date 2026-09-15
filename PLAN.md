# YuE2 on vast.ai — 部署規劃

## 目標

在 vast.ai 單卡 GPU instance 上部署 YuE2（`m-a-p/YuE2-3B`），對外提供：

- `POST /v1/audio/speech` API（沿用 minimax-music-gen contract 形狀）
- `POST /v1/audio/plan`（白盒 ABC 樂譜規劃，不產音訊）

## 決策記錄

| 項目 | 決定 | 理由 |
|---|---|---|
| 後端 | 官方 `yue2-infer` + 自寫薄 FastAPI wrapper | 官方 pipeline 完整封裝；無 diffusers 未合併 PR 風險 |
| Python | `uv venv --python 3.12` 於 `/workspace/venv312` | README 要求 3.12；venv 持久化在 instance disk |
| GPU | RTX 3090 24GB（~$0.13–0.2/h） | 官方實測峰值 11.2 GiB（4090）；3090 便宜且 verified 供給充足 |
| UI | 不做 | 無現成方案；POC 用 curl/smoke test |
| 量化 | 不用 | 原生 BF16 峰值僅 11GB，無量化需求；fp8 選項留待需要時 |

## 參考項目（調查結果）

| 項目 | 用途 |
|---|---|
| [multimodal-art-projection/YuE](https://github.com/multimodal-art-projection/YuE) | 官方 repo；`pip install .` 安裝 `yue2-infer`；release tag `yue2-v0.1.6` |
| [m-a-p/YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B) | 主權重 3.63B（~8GB）；rev `29b3558dd46954a0cd9021dc76d5c91864a0f1c7` |
| [m-a-p/YuE2-Vae](https://huggingface.co/m-a-p/YuE2-Vae) | 預設 VAE decoder（聽感好）；rev `9a94e1d0ea9f8087e98f77fa88df4a4068104d2a` |
| [m-a-p/YuE2-Vae-legacy](https://huggingface.co/m-a-p/YuE2-Vae-legacy) | benchmark 復現用（本部署不用） |
| [map-yue2.github.io](https://map-yue2.github.io/) | Demo + agentic editing 展示 |
| minimax-music-gen/PLAN.md | 同 pattern 前作；M2 坑位清單直接沿用 |

## 架構

```
internet
   │  vast.ai Caddy：TLS + auth（OPEN_BUTTON_TOKEN / WEB_PASSWORD）
   │  PORTAL_CONFIG: localhost:8787:17862:/:YuE2 API
   ▼
:8787 (external) → Caddy → :17862 yue2-inference (FastAPI, 127.0.0.1)
   │  Authorization: Bearer MUSIC_API_KEY（defense in depth）
   ▼
YuE2Pipeline.from_pretrained("m-a-p/YuE2-3B", vae="m-a-p/YuE2-Vae",
                             revision=<pin>, vae_revision=<pin>)
   │
   ▼
GPU 0  BF16, 峰值 ~11.2 GiB（官方 4090 實測）
```

- inference 只綁 127.0.0.1；對外唯一入口是 Caddy 代理的 8787。
- 服務由 Supervisor 管理（autorestart），log 進 vast.ai logging。
- artifacts 存 `/workspace/outputs/<id>/`（score.abc、semantic.npy、latent.npy、audio.flac、request/config/result.json）。

## 硬體需求（vast.ai 篩選條件）

| 項目 | 需求 |
|---|---|
| GPU | 1× ≥16GB VRAM + BF16（RTX 3090/4090/A5000/L40S；官方下限 24GB 是保守值） |
| Disk | ≥40GB（權重 ~8GB + venv ~5GB + HF cache + outputs） |
| RAM | ≥24GB host |
| Image | `vastai/pytorch:cuda-12.8.1-auto`（明確 tag） |
| 網路 | 首次要抓 ~8GB 權重 + ~5GB pip wheels |

## Image 策略

**主路線（採用）**：`vastai/pytorch:cuda-12.8.1-auto` + `PROVISIONING_SCRIPT` 指向 repo 的 `onstart.sh` raw URL。開機時建 venv + pip install（全 wheel 無編譯）+ `hf download` + 啟動。disk 跨 stop/start 保留 → 安裝成本只付一次。

**備選**：自建 all-in-one Dockerfile 推 GHCR。只在需要頻繁開新 instance 時才值得。

## 版本鎖定（reproducible）

| 元件 | Pin |
|---|---|
| YuE repo | commit `9c6c4b349be978b06a9d0d958471a07a6cdeff4d`（tag `yue2-v0.1.6`） |
| YuE2-3B | rev `29b3558dd46954a0cd9021dc76d5c91864a0f1c7` |
| YuE2-Vae | rev `9a94e1d0ea9f8087e98f77fa88df4a4068104d2a` |
| yue2-infer deps | pyproject 自帶硬 pin（torch==2.10.0、transformers==4.57.6、hf-hub==0.36.2 等） |
| server deps | `inference/requirements.txt` pinned |

## vast.ai 部署流程

1. Repo push GitHub → template 設 `PROVISIONING_SCRIPT` 指向 `onstart.sh` 的 raw URL
2. 建立 instance：`vastai/pytorch:cuda-12.8.1-auto`、disk 40、`--ssh --direct`、env `MUSIC_API_KEY` / `HF_TOKEN`（選用）/ `HF_HOME=/workspace/hf`
   - ⚠️ `--ssh` 模式不跑完整 boot → `--onstart-cmd 'exec /opt/instance-tools/bin/boot_default.sh'`（M2 坑 1）
   - ⚠️ `PORTAL_CONFIG` 的 `|` 會被 `--env` 截斷 → onstart.sh 寫 `/etc/environment`（M2 坑 3）
3. 選 offer：`gpu_name=RTX_3090 verified=true rentable=true disk_space>=40`，排序 `dph_total`
4. 首次啟動：venv → `pip install .`（YuE repo）→ `hf download` 兩個 pinned revision → load pipeline → `/health` 200
5. 驗證：`smoke_test.sh` 產短歌，檢查 FLAC magic bytes
6. 對外：Caddy :8787 + Portal auth，**不裸開 port**

## API contract

```
POST /v1/audio/speech
  input            歌詞（[Verse]/[Chorus] tag 獨立一行）
  instructions     style prompt
  seed             固定 → 可重現；省略 = 隨機，X-Seed header 取回
  cot              full | melody | off（預設 full）
  abc              自帶 ABC 樂譜（需 cot=full|melody）
  cfg_scale        0–20（預設 full/melody=1.0、off=1.01）
  id               filename-safe（artifacts 目錄名）
  response_format  flac（預設）| wav
→ audio/flac；headers: X-Seed, X-Request-Id, X-Audio-Seconds, X-Truncated

POST /v1/audio/plan → {"id", "abc", "truncated", "timing"}
```

## 風險與注意事項

- **CC BY-NC 4.0**：權重非商用。對外提供服務即違約——只供 POC/個人評測。
- **torch==2.10.0 硬 pin**：yue2-infer pyproject 鎖死；獨立 venv 安裝避免與 image torch 衝突。
- **歌長不可控**：由歌詞段落數決定，無 duration 參數。smoke test 用短歌詞。
- **同步無取消**：client timeout 設 > 預期生成時間（3090 估 ~2-3min/首）。
- **成本**：3090 ~$0.15/h 閒置也計費；不用時 stop（保留 disk）而非 destroy。
- **M2 沿用坑位**：`--ssh` 不跑完整 boot、`@vastai-automatic-tag` 解析錯誤、`PORTAL_CONFIG` 截斷、`create instance` success:false 孤兒檢查。

## 里程碑

1. **M1 檔案就緒**：server.py + requirements + onstart.sh + smoke_test.sh，code review 通過
2. **M2 vast.ai 單次部署**：手動開 instance + onstart script，smoke test 通過
3. **M3 固化**：stop/start 驗證持久化、成本記錄、文件補實測數據

## M2 部署記錄（2026-09-15，instance 51131340，TW RTX 3090 $0.161/h）

### 已完成
- Instance 建立 + provisioning 全自動跑通（~10 min：venv312 → pip install → 7.3GB 權重 → supervisor）
- `yue2-inference` RUNNING，`/health` 回 `{"status":"ready","model":"m-a-p/YuE2-3B","rev":"29b3558…"}`
- Caddy RUNNING，`/etc/portal.yaml` 正確生成（8787→7862 YuE2 API）
- smoke test **ALL CHECKS PASSED**：health 200 / 無 key 401 / models 列表 / 短歌生成 8.8MB FLAC（seed=42）
- 外部存取驗證：`http://<ip>:<mapped-8787>/health?token=$OPEN_BUTTON_TOKEN` → 200；`/v1/audio/speech?token=…` + `Authorization: Bearer $MUSIC_API_KEY` → 422（cot 驗證生效）

### 發現
- **雙層 auth**：Caddy 要 `?token=$OPEN_BUTTON_TOKEN`（query param 或 Bearer），inference 再驗 `MUSIC_API_KEY`。兩個都要。
- Caddy 初次 FATAL 是預期行為：onstart.sh 寫 `/etc/environment` 後 restart 即恢復（M2 坑 3 的解法有效）。
- 3090 生成速度：短歌（1 verse + 1 chorus）約 2 min 內完成（含 plan + semantic + NAR + VAE）。
- 權重實際 7.3GB（HF cache），比預估 8GB 略小。

### 成本記錄
- 本次部署 ~15 min ≈ $0.04；instance 已 **stop**（disk 保留，重開免重抓）
