#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Milvus 配置验证脚本
用于检查当前平台的 Milvus 配置是否正确
"""
import sys
import platform
from pathlib import Path

# 设置 UTF-8 输出（Windows 兼容）
if platform.system() == "Windows":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 添加 backend 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

try:
    from utils.config import MILVUS_CONFIG, IS_WINDOWS, IS_LINUX
    from pymilvus import MilvusClient
except ImportError as e:
    print(f"[ERROR] 导入失败: {e}")
    print("请确保已安装依赖: pip install -r requirements.txt")
    sys.exit(1)


def check_milvus_config():
    """检查 Milvus 配置"""
    print("=" * 60)
    print("[*] Milvus 配置检查")
    print("=" * 60)

    # 平台信息
    os_name = platform.system()
    print(f"\n[Platform] 操作系统: {os_name}")
    print(f"[Platform] Python 版本: {platform.python_version()}")

    # Milvus 配置
    milvus_uri = MILVUS_CONFIG["uri"]
    print(f"\n[Milvus] URI: {milvus_uri}")

    # 判断 Milvus 模式
    is_standalone = milvus_uri.startswith("http://") or milvus_uri.startswith("https://")
    is_lite = not is_standalone

    if is_standalone:
        print(f"[Milvus] 模式: Milvus Standalone (完整版)")
        print(f"[Milvus] 预估内存: ~2-2.5GB")
        print(f"[Milvus] 适合场景: Windows 开发环境")
    else:
        print(f"[Milvus] 模式: Milvus Lite (轻量版)")
        print(f"[Milvus] 预估内存: ~50-150MB")
        print(f"[Milvus] 适合场景: Linux 生产环境 / 资源受限环境")

    # 平台匹配检查
    print(f"\n[Check] 平台匹配度检查:")
    if IS_WINDOWS and is_standalone:
        print(f"[OK] Windows + Milvus Standalone - 推荐配置")
    elif IS_LINUX and is_lite:
        print(f"[OK] Linux + Milvus Lite - 推荐配置")
    elif IS_WINDOWS and is_lite:
        print(f"[WARN] Windows + Milvus Lite - 可用但非推荐")
        print(f"       建议: 使用 Docker 运行完整 Milvus")
    elif IS_LINUX and is_standalone:
        print(f"[WARN] Linux + Milvus Standalone - 高内存占用")
        print(f"       建议: 如果内存 < 4GB，切换到 Milvus Lite")

    # 连接测试
    print(f"\n[Test] 连接测试:")
    try:
        client = MilvusClient(uri=milvus_uri)
        print(f"[OK] 连接成功")

        # 列出集合
        collections = client.list_collections()
        if collections:
            print(f"[Data] 现有集合: {', '.join(collections)}")
        else:
            print(f"[Data] 暂无集合")

    except Exception as e:
        print(f"[ERROR] 连接失败: {e}")

        if is_standalone:
            print(f"\n[Solution] 解决建议 (Milvus Standalone):")
            print(f"   1. 确保 Docker Desktop 正在运行")
            print(f"   2. 启动 Milvus: docker-compose up -d")
            print(f"   3. 检查服务状态: docker-compose ps")
            print(f"   4. 检查端口占用: netstat -ano | findstr 19530 (Windows)")
        else:
            print(f"\n[Solution] 解决建议 (Milvus Lite):")
            print(f"   1. 确保目录可写: {Path(milvus_uri).parent}")
            print(f"   2. 检查磁盘空间是否充足")
            print(f"   3. 首次运行会自动创建数据库文件")

    print("\n" + "=" * 60)


def main():
    try:
        check_milvus_config()
    except Exception as e:
        print(f"\n[ERROR] 检查失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
