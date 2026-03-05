#!/bin/bash

# WhisperLiveKit Docker 部署脚本

set -e

echo "=== WhisperLiveKit Docker 部署脚本 ==="
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查 Docker 是否安装
if ! command -v docker &> /dev/null; then
    echo -e "${RED}错误: Docker 未安装${NC}"
    echo "请先安装 Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# 检查 Docker Compose 是否安装
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}错误: Docker Compose 未安装${NC}"
    echo "请先安装 Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi

# 显示菜单
echo "请选择部署方式:"
echo "1) 开发环境 (CPU)"
echo "2) 生产环境 (GPU)"
echo "3) 仅构建镜像"
echo "4) 停止服务"
echo "5) 查看日志"
echo "6) 清理数据"
echo ""
read -p "请输入选项 (1-6): " choice

case $choice in
    1)
        echo -e "${GREEN}正在部署开发环境 (CPU)...${NC}"
        docker-compose -f docker-compose.yml up -d --build
        echo -e "${GREEN}部署完成！${NC}"
        echo "访问: http://localhost:8000"
        ;;
    2)
        echo -e "${GREEN}正在部署生产环境 (GPU)...${NC}"
        
        # 检查 NVIDIA Docker
        if ! docker info | grep -q "nvidia"; then
            echo -e "${YELLOW}警告: 未检测到 NVIDIA Docker 运行时${NC}"
            echo "如需 GPU 支持，请先安装:"
            echo "  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
            read -p "是否继续 (CPU 模式)? (y/n) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        fi
        
        docker-compose -f docker-compose.prod.yml up -d --build
        echo -e "${GREEN}部署完成！${NC}"
        echo "访问: http://localhost"
        ;;
    3)
        echo -e "${GREEN}正在构建镜像...${NC}"
        docker build -t whisperlivekit:latest .
        echo -e "${GREEN}镜像构建完成！${NC}"
        docker images | grep whisperlivekit
        ;;
    4)
        echo -e "${YELLOW}正在停止服务...${NC}"
        docker-compose -f docker-compose.yml down
        docker-compose -f docker-compose.prod.yml down
        echo -e "${GREEN}服务已停止${NC}"
        ;;
    5)
        echo -e "${GREEN}查看日志...${NC}"
        docker logs -f whisper-asr
        ;;
    6)
        echo -e "${RED}警告: 这将删除所有数据，包括模型缓存！${NC}"
        read -p "确定要继续吗? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            docker-compose -f docker-compose.yml down -v
            docker-compose -f docker-compose.prod.yml down -v
            docker volume rm whisperlivekit_model-cache 2>/dev/null || true
            echo -e "${GREEN}数据已清理${NC}"
        fi
        ;;
    *)
        echo -e "${RED}无效选项${NC}"
        exit 1
        ;;
esac
