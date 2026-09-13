# 跨平台部署指南

本文档说明如何在 Windows 开发环境和 Linux 生产环境之间无缝切换。

**Debian 12 全新服务器及 GitHub 自动发布请优先按 [CI/CD 部署手册](operations/cicd-deployment.md) 操作。** 新流程采用 `current/releases/shared` 目录和专用 systemd 模板；下文保留手工部署说明，不要把两种目录布局混用。

首次启动前按 [第三阶段安全部署说明](SECURITY_STAGE3_PLAN.md) 配置 JWT 签名材料、认证数据库、初始管理员和 `ALLOWED_ORIGINS`。公网继续使用持久化 Redis、可信代理、脱敏审计与 HTTPS；前端使用账号密码登录，任何密钥都不能写入 `VITE_` 变量。会话、笔记、偏好和画像按账号隔离，论文与索引共享；旧 API Key 仅作为显式兼容模式保留。

---

## 🔄 **平台自适应配置**

系统已实现 Milvus 平台自动检测，无需手动修改配置：

| 平台 | Milvus 模式 | 默认 URI | 内存占用 |
|------|------------|----------|---------|
| **Windows** | Milvus Standalone (Docker) | `http://localhost:19530` | ~2.5GB |
| **Linux** | Milvus Lite (嵌入式) | `./backend-data/milvus_lite.db` | ~100MB |

### **工作原理**

`backend/utils/config.py` 会自动检测操作系统：

```python
# 自动检测平台
IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"

# Windows: 使用完整 Milvus (Docker)
# Linux: 使用 Milvus Lite (本地文件)
def _get_default_milvus_uri() -> str:
    if IS_WINDOWS:
        return "http://localhost:19530"
    else:
        return str(BACKEND_DATA_ROOT / "milvus_lite.db")
```

---

## 💻 **Windows 开发环境配置**

### **1. 安装 Docker Desktop**

下载并安装：https://www.docker.com/products/docker-desktop/

### **2. 初始化本地安全配置**

在仓库根目录、项目 conda 环境中执行：

```powershell
conda activate new_rag
python scripts/init_security.py --profile development
```

脚本创建开发 `.env` 和 `backend/config/development.jwt-secret`，不覆盖已有文件，不打印密钥。JWT、Redis、MinIO 密码分别随机生成；Windows 使用当前账号的 NTFS ACL，Linux 文件使用 `0600`。在 `.env` 中自行填写 `ALIYUN_API_KEY`，不要把密钥发到聊天或写进前端变量。

开发配置只允许本机 Vite Origin，后端监听 `127.0.0.1`，限流使用单进程内存。生产配置独立存放，不能将开发配置当作公网配置。

### **3. 启动 Milvus Standalone（Windows）**

使用仓库 [docker-compose.yml](../docker-compose.yml)，MinIO 与 Milvus 从同一 `.env` 读取匹配的凭据，缺少值会拒绝创建容器。

```powershell
docker compose --env-file .env config --quiet
docker compose --env-file .env up -d
```

端口均绑定本机。不要输出完整 `docker compose config`，其中会展开真实密码；用 `--quiet` 做语法校验。Redis 为可选 security profile；开发默认无需启动它，生产步骤见下文。

### **4. 启动开发环境**

首次在仓库根目录创建管理员，确认认证库配置与后端一致：

```powershell
python scripts/manage_users.py create-admin --username admin --email admin@example.com
```

密码使用隐藏输入；命令默认读取根目录 `.env`，也可显式指定 `--env-file .env`。启动后在前端输入此账号密码。

```powershell
# 后端
cd backend
python main.py

# 前端（新终端）
cd new_frontend
npm run dev
```

---

## 🐧 **Linux 生产环境部署**

### **1. 系统准备**

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装依赖
sudo apt install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    tesseract-ocr \
    tesseract-ocr-chi-sim \
    tesseract-ocr-eng \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    nginx
```

### **2. 部署项目**

```bash
# 克隆项目
cd /opt
sudo git clone <your-repo-url> arxiv-research-agent
sudo chown -R $USER:$USER arxiv-research-agent
cd arxiv-research-agent

# 创建虚拟环境
python3.10 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install --upgrade pip
pip install -r requirements.txt
pip install gunicorn
```

### **3. 生成生产配置、启动 Redis、创建管理员**

在部署服务器项目根目录运行，脚本生成全新的 `.env.production` 和 `backend/config/production.jwt-secret`：

```bash
python scripts/init_security.py --profile production --domain arxiv.001769.xyz
# 在私有文件中填写 ALIYUN_API_KEY；不要将文件内容贴到终端日志或前端。
nano .env.production
docker compose --env-file .env.production --profile security config --quiet
docker compose --env-file .env.production --profile security up -d redis
docker compose --env-file .env.production --profile security exec redis redis-cli ping
python scripts/manage_users.py create-admin --username admin --email admin@001769.xyz --env-file .env.production
```

将管理员邮箱改成自己的地址。密码通过隐藏输入确认，不创建默认账号。Redis 应返回 `PONG`；容器启用密码、AOF 每次刷盘和 `noeviction`，不可用时业务入口返回 503，不会退回内存额度。

生产配置的关键行为：

| 配置 | 生成值或语义 |
| --- | --- |
| `AUTH_MODE` / 注册 | `jwt` / 默认关闭 |
| `JWT_SECRET_FILE` | `backend/config/production.jwt-secret`，独立随机签名材料 |
| `AUTH_DATABASE_PATH` | `backend/data/auth/production.sqlite3`，与开发账号库分开 |
| `ALLOWED_ORIGINS` | `https://arxiv.001769.xyz` |
| `BACKEND_HOST` | `127.0.0.1` |
| `TRUSTED_PROXY_IPS` | `127.0.0.1,::1`，仅适用于同机 Nginx |
| `RATE_LIMIT_STORAGE` | 带独立密码的本机 Redis URI，与容器密码一致 |
| `AUDIT_LOG_FILE` | `-`，交给 systemd journal 收集 |
| MinIO | 独立随机账号/密码，同步提供给 Milvus |

后端优先使用进程环境，然后是根 `.env`、`backend/.env`。生产的 `.env.production` **不会由直接执行 `python backend/main.py` 自动加载**：systemd 通过 `EnvironmentFile` 注入它，用户管理命令必须传 `--env-file .env.production`。不要同时保留会覆盖生产配置的旧 shell 环境变量。

初始化脚本拒绝覆盖已有文件，不能用它轮换现有部署。迁移既有配置时逐项补齐；已有 Redis/MinIO 凭据变化必须同步应用和存储服务。若保留开发环境的账号与兴趣历史，还需迁移匹配的认证库、业务库及相关资产，不能只创建同名账号。

Linux 默认采用 Milvus Lite，运行 Redis 不需要启动其他容器。若显式使用 Standalone，设置 `MILVUS_URI=http://127.0.0.1:19530`，并用相同 `.env.production` 启动完整 Compose 栈。

认证目录/数据库使用 `0700`/`0600`，过宽权限会阻止启动。JWT/模型密钥、账号库、日志和所有数据端口均不应对公网开放。

### **4. 构建前端**

```bash
cd new_frontend
npm ci
npm run build
cd ..
```

### **5. 小内存服务器可选 Swap**

```bash
# 创建 2GB swap
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 永久生效
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# 优化 swap 策略
sudo sysctl vm.swappiness=10
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
```

### **6. 配置 systemd 服务**

使用维护中的 [systemd 模板](../deploy/arxiv-agent.service)，修改部署账号、工作目录及虚拟环境路径。模板禁用 Uvicorn 自动解释转发头，由应用依据可信代理名单解析原始 TCP 对端。

```bash
sudo cp deploy/arxiv-agent.service /etc/systemd/system/arxiv-agent.service
sudoedit /etc/systemd/system/arxiv-agent.service
sudo systemctl daemon-reload
sudo systemctl enable arxiv-agent
sudo systemctl start arxiv-agent
sudo systemctl status arxiv-agent
```

模板读取 `.env.production`，使用单 worker 和 `UMask=0077`。同机扩容必须共享同一个认证 SQLite、JWT 签名材料和 Redis，并设置 `AUDIT_LOG_FILE=-`，由 journald 或日志平台收集审计，避免多个进程同时轮转一个文件。多主机运行需先迁移共享认证存储，不能每台机器复制一份 SQLite，也不支持用网络文件系统共享 SQLite。

认证库、WAL/SHM、签名文件及备份不能放入前端静态目录。使用 SQLite 在线备份或停服务后做一致性备份；恢复认证库时轮换签名材料并重启全部实例，避免旧会话记录被重新启用。JWT 模式进程重启后的待运行 Agent 续跑因授权上下文丢失而失败，需用户重新登录后发起新的恢复请求。

### **7. 配置 arxiv.001769.xyz 与 HTTPS**

先在 `001769.xyz` 的 DNS 中将子域名 `arxiv` 的 A 记录指向服务器公网 IPv4。有可用 IPv6 时才添加 AAAA。以下模板采用同域前端与 `/api`，不需要单独的 API 子域名。防火墙仅开放 80/443 及受限管理入口；8001、6379、19530、9000、9001、19091 保持回环监听。

首次申请证书使用 [bootstrap 配置](../deploy/nginx-bootstrap.conf)。它只开放 ACME 验证目录，其他 HTTP 请求返回 503，不提供明文登录页面或业务 API：

```bash
sudo apt install -y certbot
sudo mkdir -p /var/www/letsencrypt
sudo cp deploy/nginx-bootstrap.conf /etc/nginx/sites-available/arxiv-agent
# 首次创建此链接；若已存在，保持原链接，不要重复添加同域站点。
sudo ln -s /etc/nginx/sites-available/arxiv-agent /etc/nginx/sites-enabled/arxiv-agent
sudo nginx -t
sudo systemctl reload nginx
sudo certbot certonly --webroot -w /var/www/letsencrypt -d arxiv.001769.xyz
```

证书成功后，用完整 [HTTPS 配置](../deploy/nginx.conf) 替换**同一个**站点文件：

```bash
sudo cp deploy/nginx.conf /etc/nginx/sites-available/arxiv-agent
sudo nginx -t
sudo systemctl reload nginx
sudo certbot renew --dry-run
```

保留 HTTP 上的 ACME 目录供续期；其他 HTTP 请求以 308 跳转至固定 HTTPS 主机。配置启用 TLS 1.2/1.3、HSTS，限制请求体 1 MiB，覆盖外部 XFF，保留 SSE 流式传输。Nginx 的静态目录必须与实际 `new_frontend/dist` 一致，证书文件不存在时不能跳过 `nginx -t`。

为自动续期添加 reload hook，保证 Nginx 重新读取新证书：

```bash
sudo mkdir -p /etc/letsencrypt/renewal-hooks/deploy
printf '#!/bin/sh\nnginx -t && systemctl reload nginx\n' | sudo tee /etc/letsencrypt/renewal-hooks/deploy/reload-nginx.sh >/dev/null
sudo chmod 755 /etc/letsencrypt/renewal-hooks/deploy/reload-nginx.sh
sudo systemctl enable --now certbot.timer
```

验证：`curl -I http://arxiv.001769.xyz/` 应为 308；`curl https://arxiv.001769.xyz/health` 应为 200；无认证访问 `/api/auth/me` 应为 401。`/health` 只证明存活，登录、受保护请求和 Redis 持久性仍要分别验收。域名变更时同时更新两份 Nginx 文件、证书路径和 `ALLOWED_ORIGINS`。

## 安全回归与发布边界

在 Windows 的 `new_rag` 中运行：

```powershell
conda activate new_rag
python scripts/test_security.py
python scripts/test_security.py --full
python scripts/doctor.py basic
```

前两个入口隔离凭据、数据库、trace 和缓存；专项包括三阶段认证以及安全初始化回归。Linux/Git Bash 可以运行 `bash test_security.sh`。业务依赖就绪、Docker、真实 Redis、DNS/证书、模型服务及向量库可用性应分别确认，离线回归不能替代这些部署验收。

同步 arXiv 搜索和 PDF 下载在线程池运行，保留认证上下文；线程池及 SQLite 写入仍有容量上限。小内存实例应限制并发并监测 Docling 峰值，不能将下面的资源估算视为容量保证。

## 🔧 **手动覆盖配置**

如果需要在特定平台手动指定 Milvus 模式：

### **Windows 上使用 Milvus Lite**
```powershell
$env:MILVUS_URI = "./backend-data/milvus_lite.db"
```

### **Linux 上使用完整 Milvus**
```bash
export MILVUS_URI="http://localhost:19530"
```

---

## 📊 **资源占用对比**

### **Windows 开发环境**
- Milvus Standalone: 2-2.5GB
- Python 后端: 400-600MB
- 总计: ~2.5-3GB
- ✅ 开发机通常有 8GB+ 内存，无压力

### **Linux 生产环境（2GB 服务器）**
- Milvus Lite: 50-150MB
- Python 后端: 400-600MB
- Docling + OCR: 200-400MB
- 系统开销: 200-300MB
- 总计: ~0.85-1.45GB
- ✅ 在 2GB 限制内，配合 Swap 稳定运行

---

## ✅ **验证部署**

### **验证 Milvus 模式**

在项目根目录运行：

```bash
# 显示当前 Milvus 配置
python -c "
import sys
sys.path.insert(0, 'backend')
from utils.config import MILVUS_CONFIG
import platform
print(f'Platform: {platform.system()}')
print(f'Milvus URI: {MILVUS_CONFIG[\"uri\"]}')
"
```

预期输出：
- **Windows**: `Platform: Windows, Milvus URI: http://localhost:19530`
- **Linux**: `Platform: Linux, Milvus URI: /path/to/backend-data/milvus_lite.db`

### **健康检查**

```bash
# 后端健康检查
curl http://127.0.0.1:8001/health

# Milvus 连接检查（仅完整 Milvus）
curl http://127.0.0.1:19091/healthz
```

---

## 🐛 **常见问题**

### **Q1: Windows 上 Docker 启动失败**
- 确保 WSL2 已启用
- 确保 Docker Desktop 正在运行
- 检查端口 19530 是否被占用

### **Q2: Linux 上内存不足**
- 确认已配置 Swap
- 降低并发 worker 数量
- 考虑关闭 OCR 功能

### **Q3: Milvus Lite 数据库文件过大**
```bash
# 清理未使用的索引
cd backend
python -c "
from pymilvus import MilvusClient
client = MilvusClient(uri='./backend-data/milvus_lite.db')
# 清理逻辑
"
```

### **Q4: 跨平台数据迁移**

Milvus Standalone 和 Milvus Lite 的数据格式不同，迁移需要：

```python
# 从 Windows 导出
from pymilvus import MilvusClient
client_src = MilvusClient(uri="http://localhost:19530")
data = client_src.query(collection_name="your_collection", filter="", output_fields=["*"])

# 在 Linux 导入
client_dst = MilvusClient(uri="./backend-data/milvus_lite.db")
client_dst.insert(collection_name="your_collection", data=data)
```

---

## 📚 **参考链接**

- [Milvus Lite 文档](https://milvus.io/docs/milvus_lite.md)
- [Milvus Standalone 安装](https://milvus.io/docs/install_standalone-docker.md)
- [项目主 README](../README.md)
- [第一阶段安全部署](SECURITY_STAGE1_PLAN.md)
- [第二阶段安全部署](SECURITY_STAGE2_PLAN.md)
