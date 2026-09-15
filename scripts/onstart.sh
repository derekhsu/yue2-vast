#!/bin/bash
# vast.ai PROVISIONING_SCRIPT — runs once after Supervisor starts (marker: /.provisioning_complete).
# Idempotent by design; safe to re-run manually.
#
# Required env (set in the vast.ai template):
#   MUSIC_API_KEY   — bearer key for the inference server
# Optional env:
#   HF_TOKEN          — only if HF repos ever require auth (currently public)
#   DEPLOY_REPO       — git URL of THIS repo (default below). Must be reachable from the instance.
#   DEPLOY_REF        — branch/tag/commit to check out (default: main)
#   MEMORY_BUDGET_GIB — VRAM budget for YuE2Pipeline (default 24; <=12 picks smaller VAE chunks)
set -euo pipefail

: "${MUSIC_API_KEY:?MUSIC_API_KEY must be set in the vast.ai template env}"

DEPLOY_REPO="${DEPLOY_REPO:-https://github.com/derekhsu/yue2-vast.git}"
DEPLOY_REF="${DEPLOY_REF:-main}"
YUE_REPO="https://github.com/multimodal-art-projection/YuE.git"
YUE_REF="9c6c4b349be978b06a9d0d958471a07a6cdeff4d"   # tag yue2-v0.1.6
MODEL_REPO="m-a-p/YuE2-3B"
MODEL_REV="29b3558dd46954a0cd9021dc76d5c91864a0f1c7"
VAE_REPO="m-a-p/YuE2-Vae"
VAE_REV="9a94e1d0ea9f8087e98f77fa88df4a4068104d2a"
VENV=/workspace/venv312

export HF_HOME="${HF_HOME:-/workspace/hf}"

log() { echo "[onstart] $*"; }

UV="$(command -v uv || echo /venv/main/bin/uv)"

# --- 1. Fetch this repo (inference server + scripts) --------------------------
if [ ! -d /workspace/deploy/.git ]; then
    log "cloning deploy repo $DEPLOY_REPO@$DEPLOY_REF"
    git clone "$DEPLOY_REPO" /workspace/deploy
    git -C /workspace/deploy checkout "$DEPLOY_REF"
else
    log "deploy repo present, fetching $DEPLOY_REF"
    git -C /workspace/deploy fetch origin
    git -C /workspace/deploy checkout "$DEPLOY_REF"
fi

# --- 2. Python 3.12 venv + deps ------------------------------------------------
# yue2-infer hard-pins torch==2.10.0 etc.; install into a dedicated venv so the
# image's own torch is untouched. Venv lives on /workspace → survives stop/start.
if [ ! -x "$VENV/bin/python" ]; then
    log "creating python 3.12 venv at $VENV"
    "$UV" venv --python 3.12 "$VENV"
fi
. "$VENV/bin/activate"

if [ ! -d /workspace/YuE/.git ]; then
    log "cloning YuE repo @ $YUE_REF"
    git clone "$YUE_REPO" /workspace/YuE
fi
git -C /workspace/YuE checkout "$YUE_REF"

if [ ! -f /workspace/.deps_done ]; then
    log "installing yue2-infer + server deps"
    "$UV" pip install --no-cache-dir /workspace/YuE
    "$UV" pip install --no-cache-dir -r /workspace/deploy/inference/requirements.txt
    python -c "import torch, yue2; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
    touch /workspace/.deps_done
fi

# --- 3. Model weights (~8GB total) ---------------------------------------------
if [ ! -f /workspace/.weights_done ]; then
    log "downloading $MODEL_REPO@$MODEL_REV"
    hf download "$MODEL_REPO" --revision "$MODEL_REV"
    log "downloading $VAE_REPO@$VAE_REV"
    hf download "$VAE_REPO" --revision "$VAE_REV"
    touch /workspace/.weights_done
fi

# --- 4. Supervisor app ----------------------------------------------------------
cat > /opt/supervisor-scripts/yue2-inference.sh << EOF
#!/bin/bash
. $VENV/bin/activate
export HF_HOME="$HF_HOME"
export MUSIC_API_KEY="$MUSIC_API_KEY"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec python /workspace/deploy/inference/server.py \
    --host 127.0.0.1 --port 7862 \
    --api-key "\$MUSIC_API_KEY" \
    --memory-budget-gib "${MEMORY_BUDGET_GIB:-24}" \
    --artifacts-dir /workspace/outputs
EOF
chmod +x /opt/supervisor-scripts/yue2-inference.sh

if ! grep -q 'yue2-inference' /etc/supervisor/conf.d/*.conf 2>/dev/null; then
cat > /etc/supervisor/conf.d/yue2-inference.conf << 'EOF'
[program:yue2-inference]
command=/opt/supervisor-scripts/yue2-inference.sh
autostart=true
autorestart=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0
EOF
fi

supervisorctl reread && supervisorctl update
log "done — inference on 127.0.0.1:7862 (external via Caddy :8787)"

# --- 5. Portal config -----------------------------------------------------------
# PORTAL_CONFIG contains '|' which vast.ai's --env parsing truncates, so we set
# it here instead of relying on the template env. caddy_config_manager.py
# regenerates /etc/portal.yaml from PORTAL_CONFIG when the file is absent.
PORTAL_LINE='PORTAL_CONFIG="localhost:1111:11111:/:Instance Portal|localhost:8787:7862:/:YuE2 API"'
grep -q '^PORTAL_CONFIG=' /etc/environment || echo "$PORTAL_LINE" >> /etc/environment
if ! grep -q 'YuE2 API' /etc/portal.yaml 2>/dev/null; then
    rm -f /etc/portal.yaml
    supervisorctl restart caddy || true
fi
