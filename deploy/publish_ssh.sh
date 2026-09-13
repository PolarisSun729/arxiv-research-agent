#!/usr/bin/env bash
# GitHub runner 只负责传包和触发普通用户部署；凭据写入临时私有目录，不进入命令输出。
set -euo pipefail
umask 077

: "${DEPLOY_HOST:?请配置 DEPLOY_HOST}"
: "${DEPLOY_USER:?请配置 DEPLOY_USER}"
: "${DEPLOY_SSH_KEY:?请配置 DEPLOY_SSH_KEY}"
: "${DEPLOY_KNOWN_HOSTS:?请配置经过核验的 DEPLOY_KNOWN_HOSTS}"
: "${GITHUB_SHA:?缺少发布提交}"
: "${GITHUB_RUN_ID:?缺少 Actions run ID}"
: "${GITHUB_RUN_ATTEMPT:?缺少 Actions attempt}"
DEPLOY_PORT=${DEPLOY_PORT:-22}
DEPLOY_ROOT=${DEPLOY_ROOT:-/opt/arxiv-research-agent}
ARTIFACT_DIR=${ARTIFACT_DIR:-temp/ci-release}

# 参数会进入远端 shell，提前限定字符集；不把 GitHub 上的自由文本拼成 shell 指令。
[[ "$DEPLOY_HOST" =~ ^[a-zA-Z0-9][a-zA-Z0-9.-]*$ ]]
[[ "$DEPLOY_USER" =~ ^[a-z_][a-z0-9_-]*$ ]]
[[ "$DEPLOY_PORT" =~ ^[0-9]+$ ]] && ((DEPLOY_PORT >= 1 && DEPLOY_PORT <= 65535))
[[ "$DEPLOY_ROOT" =~ ^/[a-zA-Z0-9_/-]+$ && "$DEPLOY_ROOT" != / && "$DEPLOY_ROOT" != */../* ]]
[[ "$GITHUB_SHA" =~ ^[0-9a-f]{40}$ && "$GITHUB_RUN_ID" =~ ^[0-9]+$ && "$GITHUB_RUN_ATTEMPT" =~ ^[0-9]+$ ]]

release_id="$GITHUB_SHA-$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT"
incoming="$DEPLOY_ROOT/incoming/$release_id"
private_dir=$(mktemp -d)
trap 'rm -rf -- "$private_dir"' EXIT
printf '%s\n' "$DEPLOY_SSH_KEY" > "$private_dir/key"
printf '%s\n' "$DEPLOY_KNOWN_HOSTS" > "$private_dir/known_hosts"
chmod 600 "$private_dir/key" "$private_dir/known_hosts"
options=(-i "$private_dir/key" -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o "UserKnownHostsFile=$private_dir/known_hosts" -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=6)
target="$DEPLOY_USER@$DEPLOY_HOST"

# 主机指纹由操作员预先核验，不在 CI 中临时信任 ssh-keyscan 返回的任何主机。
ssh -p "$DEPLOY_PORT" "${options[@]}" "$target" "mkdir -p '$incoming'"
scp -P "$DEPLOY_PORT" "${options[@]}" "$ARTIFACT_DIR/release.tar.gz" deploy/apply_release.py deploy/release_common.py "$target:$incoming/"
digest=$(tr -d '\r\n' < "$ARTIFACT_DIR/release.sha256")
[[ "$digest" =~ ^[0-9a-f]{64}$ ]]
ssh -p "$DEPLOY_PORT" "${options[@]}" "$target" \
  "python3 '$incoming/apply_release.py' --root '$DEPLOY_ROOT' --archive '$incoming/release.tar.gz' --sha256 '$digest' --commit '$GITHUB_SHA' --release-id '$release_id'"
