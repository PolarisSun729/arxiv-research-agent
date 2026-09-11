#!/bin/bash
# Linux 生产环境部署脚本
# 适用于 2GB 内存服务器（使用 Milvus Lite）

set -e

echo "========================================"
echo "arXiv Research Agent - Linux 部署"
echo "========================================"
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 检查是否为 root
if [ "$EUID" -eq 0 ]; then
    echo -e "${RED}❌ 请不要使用 root 用户运行此脚本${NC}"
    echo "   使用普通用户，脚本会在需要时提示输入 sudo 密码"
    exit 1
fi

# 检查系统
if [ ! -f /etc/os-release ]; then
    echo -e "${RED}❌ 无法检测操作系统${NC}"
    exit 1
fi

source /etc/os-release
echo -e "${CYAN}📌 操作系统: ${NC}$PRETTY_NAME"

# 检查内存
TOTAL_MEM=$(free -m | awk '/^Mem:/{print $2}')
echo -e "${CYAN}📌 总内存: ${NC}${TOTAL_MEM}MB"

if [ "$TOTAL_MEM" -lt 1800 ]; then
    echo -e "${RED}❌ 内存不足 2GB，建议至少 2GB 内存${NC}"
    read -p "是否继续？(y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 步骤 1: 安装系统依赖
echo ""
echo -e "${YELLOW}🔧 步骤 1/7: 安装系统依赖${NC}"
echo "需要 sudo 权限安装以下软件包："
echo "  - Python 3.10"
echo "  - Tesseract OCR"
echo "  - Poppler (PDF 工具)"
echo "  - 其他系统库"
echo ""

read -p "是否安装？(y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    sudo apt update
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
    echo -e "${GREEN}✅ 系统依赖安装完成${NC}"
else
    echo -e "${YELLOW}⏭️  跳过系统依赖安装${NC}"
fi

# 步骤 2: 创建虚拟环境
echo ""
echo -e "${YELLOW}🔧 步骤 2/7: 创建 Python 虚拟环境${NC}"

if [ ! -d ".venv" ]; then
    python3.10 -m venv .venv
    echo -e "${GREEN}✅ 虚拟环境创建完成${NC}"
else
    echo -e "${GREEN}✅ 虚拟环境已存在${NC}"
fi

# 激活虚拟环境
source .venv/bin/activate

# 步骤 3: 安装 Python 依赖
echo ""
echo -e "${YELLOW}🔧 步骤 3/7: 安装 Python 依赖${NC}"
pip install --upgrade pip
pip install -r requirements.txt
pip install gunicorn
echo -e "${GREEN}✅ Python 依赖安装完成${NC}"

# 步骤 4: 配置环境变量
echo ""
echo -e "${YELLOW}🔧 步骤 4/7: 配置生产环境变量${NC}"

if [ ! -f ".env.production" ]; then
    # 与手工部署共用初始化入口；独立生成 JWT、Redis、MinIO 凭据并设置私有权限。
    python scripts/init_security.py --profile production --domain arxiv.001769.xyz
else
    echo -e "${GREEN}✅ .env.production 已存在，保留现有凭据和配置${NC}"
fi

# 步骤 5: 配置 Swap
chmod 600 .env.production

echo ""
echo -e "${YELLOW}🔧 步骤 5/7: 配置 Swap 分区${NC}"

if [ -f "/swapfile" ]; then
    echo -e "${GREEN}✅ Swap 已配置${NC}"
else
    echo "建议为 2GB 内存服务器配置 2GB Swap"
    read -p "是否配置？(y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sudo fallocate -l 2G /swapfile
        sudo chmod 600 /swapfile
        sudo mkswap /swapfile
        sudo swapon /swapfile

        # 永久生效
        if ! grep -q "/swapfile" /etc/fstab; then
            echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
        fi

        # 优化 swap 策略
        sudo sysctl vm.swappiness=10
        if ! grep -q "vm.swappiness" /etc/sysctl.conf; then
            echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
        fi

        echo -e "${GREEN}✅ Swap 配置完成${NC}"
    else
        echo -e "${YELLOW}⏭️  跳过 Swap 配置${NC}"
    fi
fi

# 步骤 6: 构建前端
echo ""
echo -e "${YELLOW}🔧 步骤 6/7: 构建前端${NC}"

if [ -d "new_frontend" ]; then
    cd new_frontend

    if [ ! -d "node_modules" ]; then
        echo "安装前端依赖..."
        npm ci
    fi

    echo "构建生产版本..."
    npm run build

    cd ..
    echo -e "${GREEN}✅ 前端构建完成${NC}"
else
    echo -e "${YELLOW}⚠️  new_frontend 目录不存在${NC}"
fi

# 步骤 7: 验证 Milvus 配置
echo ""
echo -e "${YELLOW}🔧 步骤 7/7: 验证 Milvus 配置${NC}"
python scripts/check_milvus.py

# 完成
echo ""
echo "========================================"
echo -e "${GREEN}✅ 部署准备完成${NC}"
echo "========================================"
echo ""
echo -e "${CYAN}📝 下一步操作：${NC}"
echo ""
echo -e "${CYAN}1️⃣  编辑配置文件（必需）：${NC}"
echo "   nano .env.production"
echo "   # 填写模型密钥，按部署文档完成 DNS 与 HTTPS；签名文件已独立生成"
echo "   docker compose --env-file .env.production --profile security up -d redis"
echo "   python scripts/manage_users.py create-admin --username admin --email admin@001769.xyz --env-file .env.production"
echo ""
echo -e "${CYAN}2️⃣  测试启动后端：${NC}"
echo "   source .venv/bin/activate"
echo "   # 生产环境通过下方 systemd 服务加载 .env.production，不读取开发 .env"
echo ""
echo -e "${CYAN}3️⃣  配置 systemd 服务（生产环境）：${NC}"
echo "   sudo cp deploy/arxiv-agent.service /etc/systemd/system/"
echo "   sudo systemctl enable arxiv-agent"
echo "   sudo systemctl start arxiv-agent"
echo ""
echo -e "${CYAN}4️⃣  配置 Nginx 与 HTTPS（公网必需）：${NC}"
echo "   参考 docs/DEPLOYMENT.md"
echo ""
echo -e "${CYAN}5️⃣  查看服务状态：${NC}"
echo "   sudo systemctl status arxiv-agent"
echo ""
echo -e "${CYAN}6️⃣  查看日志：${NC}"
echo "   sudo journalctl -u arxiv-agent -f"
echo ""
