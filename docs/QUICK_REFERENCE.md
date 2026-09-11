# 快速参考指南

本文档提供常用命令和故障排除的快速参考。

---

## 🚀 **快速启动**

### **Windows 开发环境**

```powershell
# 一键启动（推荐）
.\start-dev.ps1

# 或手动启动
docker-compose up -d           # 启动 Milvus
cd backend && python main.py   # 启动后端（新终端）
cd new_frontend && npm run dev # 启动前端（另一个新终端）
```

### **Linux 生产环境**

```bash
# 首次部署
bash deploy-linux.sh

# 启动服务
sudo systemctl start arxiv-agent
sudo systemctl status arxiv-agent

# 查看日志
sudo journalctl -u arxiv-agent -f
```

---

## 🔍 **配置检查**

### **验证 Milvus 配置**

```bash
python scripts/check_milvus.py
```

### **检查环境变量**

```bash
# Linux
env | grep -E "ALIYUN|MILVUS|DOCLING"

# Windows
Get-ChildItem Env: | Where-Object { $_.Name -match "ALIYUN|MILVUS|DOCLING" }
```

---

## 📊 **监控命令**

### **资源监控**

```bash
# 一键监控（Linux）
bash scripts/monitor.sh

# 实时监控
watch -n 5 bash scripts/monitor.sh

# 手动检查
free -h                          # 内存使用
df -h                            # 磁盘使用
ps aux --sort=-%mem | head -10   # 内存占用 Top 10
htop                             # 交互式监控
```

### **服务状态**

```bash
# systemd 服务
sudo systemctl status arxiv-agent
sudo systemctl restart arxiv-agent
sudo systemctl stop arxiv-agent

# 查看日志
sudo journalctl -u arxiv-agent -n 100      # 最近 100 条
sudo journalctl -u arxiv-agent --since today  # 今天的日志
sudo journalctl -u arxiv-agent -f          # 实时跟踪
```

### **Milvus 健康检查**

```bash
# Milvus Standalone（Windows）
curl http://localhost:9091/healthz

# Milvus Lite（Linux）
python -c "
from pymilvus import MilvusClient
client = MilvusClient(uri='./backend-data/milvus_lite.db')
print('Collections:', client.list_collections())
"
```

---

## 🛠️ **常用操作**

### **环境变量管理**

```bash
# 编辑配置
nano .env

# 重新加载配置
sudo systemctl restart arxiv-agent

# 临时设置（仅当前会话）
export BACKEND_LOG_LEVEL=DEBUG
```

### **数据库管理**

```bash
# SQLite 数据库位置
ls -lh backend-data/*.db

# 查看数据库大小
du -sh backend-data/

# 备份数据库
tar -czf backup-$(date +%Y%m%d).tar.gz backend-data/
```

### **清理操作**

```bash
# 清理 Python 缓存
find . -type d -name "__pycache__" -exec rm -rf {} +

# 清理临时文件
rm -rf temp/*

# 清理 Milvus Lite 数据库（谨慎！）
rm backend-data/milvus_lite.db
```

---

## 🐛 **故障排除**

### **问题 1：内存不足**

**症状**：服务频繁重启，OOM killed

**解决方案**：
```bash
# 1. 检查内存使用
free -h

# 2. 配置 Swap（如果未配置）
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 3. 降低并发数
nano .env
# 设置：
# PAPER_QA_BUILD_LLM_MAX_WORKERS=1
# PROFILE_EVIDENCE_MAX_WORKERS=1
# RETRIEVAL_ROUTE_MAX_WORKERS=1

# 4. 关闭 OCR（如果不需要）
# DOCLING_OCR_ENABLED=false

# 5. 重启服务
sudo systemctl restart arxiv-agent
```

### **问题 2：Milvus 连接失败**

**Windows（Milvus Standalone）**：
```powershell
# 检查 Docker 服务
docker ps

# 查看 Milvus 日志
docker logs milvus-standalone

# 重启 Milvus
docker-compose restart

# 完全重建
docker-compose down
docker-compose up -d
```

**Linux（Milvus Lite）**：
```bash
# 检查数据库文件权限
ls -la backend-data/milvus_lite.db

# 检查磁盘空间
df -h

# 重新创建数据库
rm backend-data/milvus_lite.db
sudo systemctl restart arxiv-agent
```

### **问题 3：API Key 无效**

```bash
# 检查环境变量
echo $ALIYUN_API_KEY

# 测试 API Key
curl -X POST "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding" \
  -H "Authorization: Bearer $ALIYUN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"text-embedding-v1","input":{"texts":["测试"]}}'

# 重新设置
nano .env
# 更新 ALIYUN_API_KEY=xxx
sudo systemctl restart arxiv-agent
```

### **问题 4：端口被占用**

```bash
# 检查端口占用
sudo netstat -tlnp | grep :8001

# 或使用 ss
sudo ss -tlnp | grep :8001

# 杀死占用进程
sudo kill -9 <PID>

# 或更改端口
nano deploy/arxiv-agent.service
# 修改 -b 127.0.0.1:8002
```

### **问题 5：前端无法访问后端**

```bash
# 检查 Nginx 配置
sudo nginx -t

# 查看 Nginx 错误日志
sudo tail -f /var/log/nginx/error.log

# 检查后端是否运行
curl http://127.0.0.1:8001/api/health

# 重启 Nginx
sudo systemctl restart nginx
```

---

## 📦 **更新部署**

### **更新代码**

```bash
# 1. 拉取最新代码
git pull origin main

# 2. 更新 Python 依赖
source .venv/bin/activate
pip install -r requirements.txt

# 3. 重新构建前端
cd new_frontend
npm ci
npm run build
cd ..

# 4. 重启服务
sudo systemctl restart arxiv-agent
```

### **回滚版本**

```bash
# 查看提交历史
git log --oneline -10

# 回滚到指定版本
git checkout <commit-hash>

# 重新部署
bash deploy-linux.sh
```

---

## 🔒 **安全检查**

### **检查开放端口**

```bash
# 查看监听端口
sudo ss -tlnp

# 检查防火墙
sudo ufw status
```

### **更新系统**

```bash
# Ubuntu/Debian
sudo apt update && sudo apt upgrade -y

# 检查安全更新
sudo unattended-upgrades --dry-run
```

---

## 📈 **性能优化**

### **调整 Worker 数量**

根据服务器配置调整并发数：

| 内存 | LLM Workers | Evidence Workers | Retrieval Workers |
|------|-------------|------------------|-------------------|
| 2GB  | 1           | 1                | 1-2               |
| 4GB  | 2           | 1-2              | 2-3               |
| 8GB+ | 4           | 2                | 4                 |

```bash
nano .env
# 修改对应值后重启
sudo systemctl restart arxiv-agent
```

### **启用缓存**

```bash
nano .env
# 添加：
PAPER_QA_BUILD_CACHE_ENABLED=true
```

---

## 🔗 **有用的链接**

- **项目主页**: [README.md](../README.md)
- **完整部署指南**: [DEPLOYMENT.md](DEPLOYMENT.md)
- **Docker Compose**: [docker-compose.yml](../docker-compose.yml)
- **Milvus 文档**: https://milvus.io/docs
- **FastAPI 文档**: https://fastapi.tiangolo.com/
- **Vue 3 文档**: https://vuejs.org/

---

## 📞 **获取帮助**

如果遇到问题：

1. 查看日志：`sudo journalctl -u arxiv-agent -n 100`
2. 检查配置：`python scripts/check_milvus.py`
3. 监控资源：`bash scripts/monitor.sh`
4. 提交 Issue：https://github.com/PolarisSun729/arxiv-research-agent/issues
