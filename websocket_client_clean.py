"""
简洁版 WebSocket 客户端 - 参照前端实现，正确处理 LocalAgreement 策略

使用方法:
1. 先启动服务器: wlk --model tiny --language zh --pcm-input
2. 运行此脚本: python websocket_client_clean.py
3. 选择麦克风，对着麦克风说话
"""

import asyncio
import websockets
import json
import pyaudio
import sys
import hashlib

WS_URL = "ws://localhost:8000/asr"
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.1
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)

# 清屏函数
def clear_screen():
    print("\033[2J\033[H", end="")


def list_microphones():
    """列出所有可用的麦克风"""
    audio = pyaudio.PyAudio()
    print("\n可用的麦克风设备：")
    print("-" * 50)
    
    default_input = audio.get_default_input_device_info()
    default_index = default_input['index']
    
    microphones = []
    for i in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:  # 只显示输入设备
            is_default = " [默认]" if i == default_index else ""
            print(f"  {len(microphones)}: {info['name']}{is_default}")
            microphones.append(i)
    
    print("-" * 50)
    audio.terminate()
    return microphones, default_index


def select_microphone(microphones, default_index):
    """让用户选择麦克风"""
    while True:
        try:
            choice = input(f"\n请选择麦克风 (0-{len(microphones)-1}, 回车使用默认): ").strip()
            if choice == "":
                print(f"使用默认麦克风")
                return default_index
            idx = int(choice)
            if 0 <= idx < len(microphones):
                return microphones[idx]
            else:
                print("无效的选择，请重新输入")
        except ValueError:
            print("请输入数字")


async def send_audio(websocket, device_index):
    """从麦克风读取音频并发送到服务器"""
    audio = pyaudio.PyAudio()
    
    # 获取设备信息
    device_info = audio.get_device_info_by_index(device_index)
    print(f"\n使用麦克风: {device_info['name']}\n")
    
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=CHUNK_SIZE
    )

    print("🎤 开始录音，请说话...（按 Ctrl+C 停止）\n")

    try:
        while True:
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            await websocket.send(data)
            await asyncio.sleep(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()


def compute_signature(lines, buffer_transcription, buffer_diarization, status):
    """
    计算数据签名，用于去重（参照前端实现）
    """
    # 提取关键字段生成签名
    lines_key = [
        {
            'speaker': line.get('speaker'),
            'text': line.get('text'),
            'start': line.get('start'),
            'end': line.get('end')
        }
        for line in (lines or [])
    ]
    
    signature_data = {
        'lines': lines_key,
        'buffer_transcription': buffer_transcription or "",
        'buffer_diarization': buffer_diarization or "",
        'status': status
    }
    
    # 使用 JSON 字符串的哈希作为签名
    signature_str = json.dumps(signature_data, sort_keys=True)
    return hashlib.md5(signature_str.encode()).hexdigest()


async def receive_results(websocket):
    """
    接收服务器的转录结果 - 参照前端实现
    
    前端关键逻辑：
    1. lines 数组包含所有已完成的转录行（累计的）
    2. buffer_transcription 是当前正在识别的临时文本
    3. 使用 signature 去重，只有数据变化时才更新显示
    """
    
    # 状态跟踪（参照前端 lastSignature）
    last_signature = None
    
    # 已输出的行记录（用于识别新行）
    displayed_lines_count = 0
    
    print("[接收] 等待转录结果...\n")
    
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                
                # 处理不同类型的消息
                msg_type = data.get("type", "")
                
                # 1. 配置消息
                if msg_type == "config":
                    continue
                    
                # 2. 结束消息
                elif msg_type == "ready_to_stop":
                    break
                
                # 3. LocalAgreement 策略的转录结果 (FrontData 格式)
                # 参照前端：提取字段
                lines = data.get("lines", []) or []
                buffer_transcription = data.get("buffer_transcription", "") or ""
                buffer_diarization = data.get("buffer_diarization", "") or ""
                status = data.get("status", "active_transcription")
                
                # 参照前端：计算签名去重
                signature = compute_signature(lines, buffer_transcription, buffer_diarization, status)
                
                if signature == last_signature:
                    # 数据没有变化，跳过（参照前端逻辑）
                    continue
                
                last_signature = signature
                
                # 处理已完成的行（参照前端：只输出新增的行）
                # lines 是累计数组，只输出新增加的行
                new_lines_output = []
                if len(lines) > displayed_lines_count:
                    # 有新的已完成行
                    for i in range(displayed_lines_count, len(lines)):
                        line = lines[i]
                        line_text = line.get("text", "").strip()
                        
                        if line_text:
                            new_lines_output.append(line_text)
                    
                    displayed_lines_count = len(lines)
                
                # 如果有新完成的行，先清除临时显示，输出行，再恢复临时显示
                if new_lines_output:
                    # 清除当前行
                    print("\r" + " " * 80 + "\r", end="")
                    # 输出新完成的行
                    for text in new_lines_output:
                        print(f"✓ {text}")
                    # 如果有缓冲区内容，在新行显示
                    if buffer_transcription:
                        print(f"⟳ {buffer_transcription}", end="", flush=True)
                elif buffer_transcription:
                    # 只有缓冲区更新，覆盖当前行
                    print(f"\r⟳ {buffer_transcription}", end="", flush=True)
                else:
                    # 清除临时显示
                    print("\r" + " " * 80 + "\r", end="")

            except json.JSONDecodeError:
                pass

    except websockets.exceptions.ConnectionClosed:
        pass
    
    # 结束时换行
    print("\n")


async def main():
    """主函数"""
    clear_screen()
    print("=" * 50)
    print("      实时语音转录客户端")
    print("=" * 50)
    
    # 列出并选择麦克风
    microphones, default_index = list_microphones()
    if not microphones:
        print("❌ 没有找到可用的麦克风")
        return
    
    device_index = select_microphone(microphones, default_index)
    
    try:
        async with websockets.connect(WS_URL) as websocket:
            # 等待配置消息
            config_msg = await websocket.recv()
            try:
                config_data = json.loads(config_msg)
                print(f"⚙️  服务器配置: useAudioWorklet={config_data.get('useAudioWorklet', False)}")
            except:
                print(f"⚙️  服务器返回: {config_msg}")
            
            # 同时运行发送和接收
            await asyncio.gather(
                send_audio(websocket, device_index),
                receive_results(websocket)
            )

    except (websockets.exceptions.ConnectionClosed, websockets.exceptions.InvalidHandshake, OSError):
        print(f"\n❌ 无法连接到服务器")
        print(f"   请确认服务器已启动: wlk --model tiny --language zh --pcm-input")
    except Exception as e:
        print(f"\n❌ 错误: {e}")


if __name__ == "__main__":
    try:
        import websockets
        import pyaudio
    except ImportError:
        print("请先安装依赖: pip install websockets pyaudio")
        sys.exit(1)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 再见！")
