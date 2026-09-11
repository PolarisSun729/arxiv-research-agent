#!/bin/bash
# 资源监控脚本 - 用于 2GB 服务器性能监控

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

echo "========================================"
echo "arXiv Research Agent - 资源监控"
echo "========================================"
echo ""

# 内存使用
echo -e "${CYAN}📊 内存使用情况：${NC}"
echo ""
free -h

TOTAL_MEM=$(free -m | awk '/^Mem:/{print $2}')
USED_MEM=$(free -m | awk '/^Mem:/{print $3}')
MEM_PERCENT=$(awk "BEGIN {printf \"%.1f\", ($USED_MEM/$TOTAL_MEM)*100}")

echo ""
if (( $(echo "$MEM_PERCENT > 85" | bc -l) )); then
    echo -e "${RED}⚠️  内存使用率: ${MEM_PERCENT}% (高负载)${NC}"
elif (( $(echo "$MEM_PERCENT > 70" | bc -l) )); then
    echo -e "${YELLOW}⚠️  内存使用率: ${MEM_PERCENT}% (中等负载)${NC}"
else
    echo -e "${GREEN}✅ 内存使用率: ${MEM_PERCENT}% (正常)${NC}"
fi

# Swap 使用
echo ""
echo -e "${CYAN}💾 Swap 使用情况：${NC}"
echo ""
SWAP_TOTAL=$(free -m | awk '/^Swap:/{print $2}')
SWAP_USED=$(free -m | awk '/^Swap:/{print $3}')

if [ "$SWAP_TOTAL" -eq 0 ]; then
    echo -e "${YELLOW}⚠️  Swap 未配置${NC}"
    echo "   建议: sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile"
else
    SWAP_PERCENT=$(awk "BEGIN {printf \"%.1f\", ($SWAP_USED/$SWAP_TOTAL)*100}")
    echo "   总计: ${SWAP_TOTAL}MB"
    echo "   已用: ${SWAP_USED}MB"
    echo "   使用率: ${SWAP_PERCENT}%"

    if (( $(echo "$SWAP_PERCENT > 50" | bc -l) )); then
        echo -e "${RED}   ⚠️  Swap 使用率较高，可能存在内存压力${NC}"
    fi
fi

# CPU 使用
echo ""
echo -e "${CYAN}⚡ CPU 使用情况：${NC}"
echo ""
CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')
echo "   使用率: ${CPU_USAGE}%"

# 磁盘使用
echo ""
echo -e "${CYAN}💿 磁盘使用情况：${NC}"
echo ""
df -h | grep -E '^/dev/' | awk '{print "   "$1": "$3"/"$2" ("$5")"}'

# 进程监控
echo ""
echo -e "${CYAN}🔍 arXiv Agent 进程：${NC}"
echo ""

if pgrep -f "gunicorn.*main:app" > /dev/null; then
    PID=$(pgrep -f "gunicorn.*main:app" | head -1)
    PROCESS_MEM=$(ps -p $PID -o rss= | awk '{printf "%.1fMB", $1/1024}')
    PROCESS_CPU=$(ps -p $PID -o %cpu= | awk '{printf "%.1f%%", $1}')

    echo -e "${GREEN}✅ 后端服务运行中${NC}"
    echo "   PID: $PID"
    echo "   内存: $PROCESS_MEM"
    echo "   CPU: $PROCESS_CPU"

    # 详细进程信息
    echo ""
    echo "   进程树："
    ps -f --ppid $PID | tail -n +2 | awk '{print "      "$2": "$8" (MEM: "$(ps -p $2 -o rss= | awk "{printf \"%.0fMB\", \$1/1024}")")"}'
else
    echo -e "${RED}❌ 后端服务未运行${NC}"
    echo "   启动: sudo systemctl start arxiv-agent"
fi

# Milvus Lite 数据库大小
echo ""
echo -e "${CYAN}🗄️  Milvus Lite 数据库：${NC}"
echo ""

MILVUS_DB=$(find . -name "milvus_lite.db" 2>/dev/null | head -1)
if [ -n "$MILVUS_DB" ]; then
    DB_SIZE=$(du -h "$MILVUS_DB" | cut -f1)
    echo "   位置: $MILVUS_DB"
    echo "   大小: $DB_SIZE"
else
    echo "   未找到 milvus_lite.db（可能尚未创建）"
fi

# 系统负载
echo ""
echo -e "${CYAN}📈 系统负载：${NC}"
echo ""
uptime | awk -F'load average:' '{print "   "$2}'

# 网络连接
echo ""
echo -e "${CYAN}🌐 活动连接：${NC}"
echo ""
ACTIVE_CONN=$(ss -tunap | grep -c ":8001")
echo "   端口 8001 活动连接数: $ACTIVE_CONN"

# 日志最新错误（如果是 systemd 服务）
if systemctl is-active --quiet arxiv-agent; then
    echo ""
    echo -e "${CYAN}📝 最近日志（最近 10 条）：${NC}"
    echo ""
    sudo journalctl -u arxiv-agent -n 10 --no-pager | sed 's/^/   /'
fi

echo ""
echo "========================================"
echo -e "${GREEN}监控完成${NC}"
echo "========================================"
echo ""
echo "💡 提示："
echo "   - 实时监控: watch -n 5 bash scripts/monitor.sh"
echo "   - 查看完整日志: sudo journalctl -u arxiv-agent -f"
echo "   - 性能分析: htop"
echo ""
