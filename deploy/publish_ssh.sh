#!/usr/bin/env bash
# GitHub runner 只负责传包和触发普通用户部署；凭据写入临时私有目录，不进入命令输出。
#
# 运行位置：GitHub Actions 的 deploy 作业（.github/workflows/deploy.yml），
# 在下载同一 run 中 package_release.py 产出的 artifact 之后执行。
#
# 做三件事：
#   1. 在服务器上创建 <DEPLOY_ROOT>/incoming/<release_id>/ 暂存目录；
#   2. scp 上传 release.tar.gz、apply_release.py、release_common.py；
#   3. 远程以普通用户执行 apply_release.py 完成安装、切换和健康检查。
#
# 必需环境变量（来自 GitHub production 环境的 secrets / vars 及 Actions 内置变量）：
#   DEPLOY_HOST          服务器主机名或 IP
#   DEPLOY_USER          部署用户（如 arxiv）
#   DEPLOY_SSH_KEY       部署专用 SSH 私钥内容
#   DEPLOY_KNOWN_HOSTS   预先核验过的服务器主机指纹（known_hosts 格式）
#   GITHUB_SHA / GITHUB_RUN_ID / GITHUB_RUN_ATTEMPT  Actions 内置变量，组成唯一的 release_id
# 可选环境变量：DEPLOY_PORT（默认 22）、DEPLOY_ROOT（默认 /opt/arxiv-research-agent）、
#              ARTIFACT_DIR（默认 temp/ci-release，即 package_release.py 的输出目录）

# -e 任一命令失败即退出；-u 使用未定义变量即报错；-o pipefail 管道中任一环节失败即失败。
set -euo pipefail
# 本脚本创建的文件默认仅当前用户可读写（私钥文件尤其需要）。
umask 077

# ${VAR:?msg}：变量未设置或为空时打印 msg 并退出；":" 是空命令，只用来触发这个展开检查。
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
# 每一行都是一个 [[ ]] 测试，不匹配时返回非零，配合 set -e 直接终止脚本。
[[ "$DEPLOY_HOST" =~ ^[a-zA-Z0-9][a-zA-Z0-9.-]*$ ]]
[[ "$DEPLOY_USER" =~ ^[a-z_][a-z0-9_-]*$ ]]
[[ "$DEPLOY_PORT" =~ ^[0-9]+$ ]] && ((DEPLOY_PORT >= 1 && DEPLOY_PORT <= 65535))
# 部署根目录必须是绝对路径、不能是 /、不能含 ".." 路径段。
[[ "$DEPLOY_ROOT" =~ ^/[a-zA-Z0-9_/-]+$ && "$DEPLOY_ROOT" != / && "$DEPLOY_ROOT" != */../* ]]
[[ "$GITHUB_SHA" =~ ^[0-9a-f]{40}$ && "$GITHUB_RUN_ID" =~ ^[0-9]+$ && "$GITHUB_RUN_ATTEMPT" =~ ^[0-9]+$ ]]

# release_id 绑定提交、run ID 和重试次数，同一提交重跑 Actions 也会得到新的版本目录。
release_id="$GITHUB_SHA-$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT"
incoming="$DEPLOY_ROOT/incoming/$release_id"
# 私钥和 known_hosts 写入临时目录，脚本退出（无论成功失败）时由 trap 删除。
private_dir=$(mktemp -d)
trap 'rm -rf -- "$private_dir"' EXIT
printf '%s\n' "$DEPLOY_SSH_KEY" > "$private_dir/key"
printf '%s\n' "$DEPLOY_KNOWN_HOSTS" > "$private_dir/known_hosts"
chmod 600 "$private_dir/key" "$private_dir/known_hosts"
# SSH 公共选项：
#   BatchMode=yes             禁止任何交互式提示（如输入密码），出问题直接失败
#   IdentitiesOnly=yes        只使用 -i 指定的私钥，不尝试 ssh-agent 中的其他密钥
#   StrictHostKeyChecking=yes 主机指纹不在 known_hosts 中就拒绝连接，防中间人
#   UserKnownHostsFile=...    只信任本次写入的 known_hosts，不读 runner 上的默认文件
#   ConnectTimeout / ServerAlive*  连接超时 20 秒；每 15 秒心跳，连续 6 次无响应断开
options=(-i "$private_dir/key" -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o "UserKnownHostsFile=$private_dir/known_hosts" -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=6)
target="$DEPLOY_USER@$DEPLOY_HOST"

# 主机指纹由操作员预先核验，不在 CI 中临时信任 ssh-keyscan 返回的任何主机。
# 步骤 1：创建远端暂存目录。
ssh -p "$DEPLOY_PORT" "${options[@]}" "$target" "mkdir -p '$incoming'"
# 步骤 2：上传发布包和服务器端安装脚本（注意 scp 的端口参数是大写 -P）。
scp -P "$DEPLOY_PORT" "${options[@]}" "$ARTIFACT_DIR/release.tar.gz" deploy/apply_release.py deploy/release_common.py "$target:$incoming/"
# 读取整包哈希并去掉换行，校验格式后作为参数传给服务器端二次校验。
digest=$(tr -d '\r\n' < "$ARTIFACT_DIR/release.sha256")
[[ "$digest" =~ ^[0-9a-f]{64}$ ]]
# 步骤 3：远程执行安装。使用系统 python3 运行，apply_release.py 只依赖标准库；
# 其标准输出（部署结果 JSON）会显示在 Actions 日志中。
ssh -p "$DEPLOY_PORT" "${options[@]}" "$target" \
  "python3 '$incoming/apply_release.py' --root '$DEPLOY_ROOT' --archive '$incoming/release.tar.gz' --sha256 '$digest' --commit '$GITHUB_SHA' --release-id '$release_id'"
