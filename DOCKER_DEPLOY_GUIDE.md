# WhisperLiveKit Docker 部署指南

## 目录
- [快速开始](#快速开始)
- [环境要求](#环境要求)
- [部署步骤](#部署步骤)
- [常用命令](#常用命令)
- [故障排除](#故障排除)
- [高级配置](#高级配置)

---

## 快速开始

### 方式一：使用脚本（推荐）

**Linux/Mac:**
```bash
chmod +x deploy.sh
./deploy.sh
```

**Windows:**
```powershell
.\deploy.ps1
```

### 方式二：手动部署

```bash
# 1. 构建镜像
docker build -t whisperlivekit:latest .

# 2. 启动服务
docker-compose up -d

# 3. 查看日志
docker logs -f whisper-asr
```

---

## 环境要求

### 基础要求
- Docker 20.10+
- Docker Compose 2.0+
- 4GB+ 可用内存
- 10GB+ 磁盘空间

### GPU 支持（可选）
- NVIDIA GPU
- NVIDIA 驱动 525.60.13+
- NVIDIA Container Toolkit

**安装 NVIDIA Container Toolkit:**
```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

---

## 部署步骤

### 1. 准备配置文件

确保以下文件存在：
- `docker-compose.yml` - 开发环境配置
- `docker-compose.prod.yml` - 生产环境配置
- `keywords_example.txt` - 关键字文件
- `Dockerfile` - 镜像构建文件

### 2. 配置关键字文件

编辑 `keywords_example.txt`，添加你需要的关键字：

```text
# 每行一个关键字
百度
阿里巴巴
腾讯
语音识别
WhisperLiveKit
```

### 3. 选择部署模式

#### 开发模式（CPU）

适合开发和测试：

```bash
docker-compose -f docker-compose.yml up -d
```

特点：
- 使用 CPU 运行
- 启动快速
- 适合开发调试

#### 生产模式（GPU）

适合生产环境：

```bash
docker-compose -f docker-compose.prod.yml up -d
```

特点：
- 使用 GPU 加速
- 包含 Nginx 反向代理
- 资源限制和监控
- 自动重启

### 4. 验证部署

```bash
# 查看容器状态
docker ps

# 查看日志
docker logs whisper-asr

# 测试服务
curl http://localhost:8000
```

### 5. 访问服务

- **直接访问**: http://localhost:8000
- **通过 Nginx** (生产模式): http://localhost

---

## 常用命令

### 容器管理

```bash
# 查看运行中的容器
docker ps

# 查看所有容器（包括停止的）
docker ps -a

# 停止服务
docker-compose down

# 重启服务
docker-compose restart

# 进入容器内部
docker exec -it whisper-asr /bin/bash
```

### 日志管理

```bash
# 查看实时日志
docker logs -f whisper-asr

# 查看最近 100 行日志
docker logs --tail 100 whisper-asr

# 查看特定时间段的日志
docker logs --since 10m whisper-asr
```

### 镜像管理

```bash
# 查看所有镜像
docker images

# 删除镜像
docker rmi whisperlivekit:latest

# 清理未使用的镜像
docker image prune
```

### 数据卷管理

```bash
# 查看数据卷
docker volume ls

# 查看模型缓存内容
docker volume inspect whisperlivekit_model-cache

# 清理未使用的卷
docker volume prune
```

---

## 故障排除

### 问题一：端口被占用

**错误信息:**
```
Bind for 0.0.0.0:8000 failed: port is already allocated
```

**解决方案:**
```bash
# 查找占用端口的进程
sudo lsof -i :8000

# 停止占用端口的进程
sudo kill -9 <PID>

# 或修改 docker-compose.yml 使用其他端口
ports:
  - "8080:8000"  # 改为 8080 端口
```

### 问题二：GPU 不可用

**错误信息:**
```
nvidia-container-cli: initialization error
```

**解决方案:**
```bash
# 检查 NVIDIA 驱动
nvidia-smi

# 检查 NVIDIA Docker 运行时
docker info | grep nvidia

# 重启 Docker 服务
sudo systemctl restart docker

# 如果没有 GPU，修改 docker-compose.yml 移除 GPU 配置
deploy:
  resources:
    limits:
      cpus: '4'
      memory: 8G
```

### 问题三：模型下载失败

**错误信息:**
```
Connection timeout while downloading model
```

**解决方案:**
```bash
# 设置 Hugging Face 镜像
export HF_ENDPOINT=https://hf-mirror.com

# 预下载模型到本地
# 修改 docker-compose.yml 挂载本地缓存
volumes:
  - /path/to/local/hf/cache:/root/.cache/huggingface/hub
```

### 问题四：内存不足

**错误信息:**
```
RuntimeError: CUDA out of memory
```

**解决方案:**
```bash
# 修改 docker-compose.yml 减小内存限制
deploy:
  resources:
    limits:
      memory: 4G  # 减小内存限制

# 使用更小的模型
command: [
  "--model", "tiny",  # 改为 tiny 模型
  ...
]
```

### 问题五：权限错误

**错误信息:**
```
Permission denied
```

**解决方案:**
```bash
# Linux/Mac: 修改文件权限
chmod 644 keywords_example.txt
chmod 755 deploy.sh

# 或修改 docker-compose.yml 使用 root 用户
services:
  whisper-asr:
    user: root
```

---

## 高级配置

### 1. 自定义模型路径

如果你有本地模型，可以挂载到容器：

```yaml
volumes:
  - ./models:/app/models:ro
environment:
  - MODEL_PATH=/app/models/faster-whisper-small
```

### 2. 配置 SSL/HTTPS

准备 SSL 证书文件：
- `ssl/cert.pem`
- `ssl/key.pem`

然后启动生产模式：
```bash
docker-compose -f docker-compose.prod.yml up -d
```

### 3. 多实例部署

使用 Docker Swarm 或 Kubernetes 实现负载均衡：

```bash
# 初始化 Swarm
docker swarm init

# 部署服务
docker stack deploy -c docker-compose.prod.yml whisper

# 扩展实例数
docker service scale whisper_whisper-asr=3
```

### 4. 监控和告警

添加 Prometheus 监控：

```yaml
# docker-compose.monitoring.yml
version: '3.8'
services:
  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"
  
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
```

### 5. 自动更新

使用 Watchtower 自动更新镜像：

```yaml
# 添加到 docker-compose.prod.yml
watchtower:
  image: containrrr/watchtower
  volumes:
    - /var/run/docker.sock:/var/run/docker.sock
  command: --interval 3600 whisper-asr
```

---

## 性能优化

### CPU 优化

```yaml
deploy:
  resources:
    limits:
      cpus: '4'
    reservations:
      cpus: '2'
```

### GPU 优化

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          device_ids: ['0']  # 指定 GPU 设备
          capabilities: [gpu]
```

### 模型缓存优化

```yaml
volumes:
  - model-cache:/root/.cache/huggingface/hub
driver_opts:
  type: none
  o: bind
  device: /mnt/fast-disk/hf-cache  # 使用高速磁盘
```

---

## 安全建议

1. **使用非 root 用户运行容器**
2. **限制容器资源使用**
3. **定期更新基础镜像**
4. **使用防火墙限制端口访问**
5. **启用日志审计**
6. **使用 Secrets 管理敏感信息**

---

## 参考链接

- [Docker 官方文档](https://docs.docker.com/)
- [Docker Compose 文档](https://docs.docker.com/compose/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [WhisperLiveKit GitHub](https://github.com/QuentinFuxa/WhisperLiveKit)

---

## 获取帮助

遇到问题？
1. 查看日志：`docker logs whisper-asr`
2. 检查状态：`docker ps`
3. 提交 Issue：[GitHub Issues](https://github.com/QuentinFuxa/WhisperLiveKit/issues)
