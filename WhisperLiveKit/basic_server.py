import asyncio
import logging
import pathlib
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from whisperlivekit import (AudioProcessor, TranscriptionEngine,
                            get_inline_ui_html, parse_args)
from whisperlivekit.hotword_service import HotwordService
import whisperlivekit.web as webpkg

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger().setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

args = parse_args()
transcription_engine = None
hotword_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):    
    global transcription_engine, hotword_service
    
    # 初始化转录引擎
    transcription_engine = TranscriptionEngine(
        **vars(args),
    )
    
    # 初始化唤醒词检测服务
    if args.hotword_model_dir:
        # 确定关键词文件路径
        keywords_file = args.hotword_keywords_file
        if not keywords_file:
            keywords_file = f"{args.hotword_model_dir}/keywords.txt"
        
        hotword_service = HotwordService(
            model_dir=args.hotword_model_dir,
            keywords_file=keywords_file,
            threshold=args.hotword_threshold,
            sample_rate=args.hotword_sample_rate,
            num_threads=args.hotword_threads
        )
        logger.info(f"唤醒词检测服务初始化完成，模型目录: {args.hotword_model_dir}，阈值: {args.hotword_threshold}")
    else:
        hotword_service = None
        logger.info("未配置唤醒词模型目录，唤醒词检测服务未初始化")
    
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件目录
web_dir = pathlib.Path(webpkg.__file__).parent
app.mount("/web", StaticFiles(directory=str(web_dir)), name="web")

@app.get("/")
async def get():
    return HTMLResponse(get_inline_ui_html())


async def handle_websocket_results(websocket, results_generator):
    """Consumes results from the audio processor and sends them via WebSocket."""
    try:
        async for response in results_generator:
            await websocket.send_json(response.to_dict())
        # when the results_generator finishes it means all audio has been processed
        logger.info("Results generator finished. Sending 'ready_to_stop' to client.")
        await websocket.send_json({"type": "ready_to_stop"})
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected while handling results (client likely closed connection).")
    except Exception as e:
        logger.exception(f"Error in WebSocket results handler: {e}")


@app.websocket("/asr")
async def websocket_endpoint(websocket: WebSocket):
    global transcription_engine
    audio_processor = AudioProcessor(
        transcription_engine=transcription_engine,
    )
    await websocket.accept()
    logger.info("WebSocket connection opened.")

    try:
        await websocket.send_json({"type": "config", "useAudioWorklet": bool(args.pcm_input)})
    except Exception as e:
        logger.warning(f"Failed to send config to client: {e}")
            
    results_generator = await audio_processor.create_tasks()
    websocket_task = asyncio.create_task(handle_websocket_results(websocket, results_generator))

    try:
        while True:
            message = await websocket.receive_bytes()
            await audio_processor.process_audio(message)
    except KeyError as e:
        if 'bytes' in str(e):
            logger.warning(f"Client has closed the connection.")
        else:
            logger.error(f"Unexpected KeyError in websocket_endpoint: {e}", exc_info=True)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected by client during message receiving loop.")
    except Exception as e:
        logger.error(f"Unexpected error in websocket_endpoint main loop: {e}", exc_info=True)
    finally:
        logger.info("Cleaning up WebSocket endpoint...")
        if not websocket_task.done():
            websocket_task.cancel()
        try:
            await websocket_task
        except asyncio.CancelledError:
            logger.info("WebSocket results handler task was cancelled.")
        except Exception as e:
            logger.warning(f"Exception while awaiting websocket_task completion: {e}")
            
        await audio_processor.cleanup()
        logger.info("WebSocket endpoint cleaned up successfully.")


@app.websocket("/hotword")
async def hotword_websocket_endpoint(websocket: WebSocket):
    """唤醒词检测WebSocket端点"""
    global hotword_service
    
    # 检查唤醒词服务是否可用
    if not hotword_service:
        logger.error("唤醒词检测服务不可用，拒绝连接")
        await websocket.close(code=1008, reason="Hotword service not available")
        return
    
    await websocket.accept()
    
    # 生成连接ID
    import uuid
    connection_id = str(uuid.uuid4())[:8]
    logger.info(f"唤醒词检测WebSocket连接打开，ID: {connection_id}")
    
    # 为连接创建KWS流
    if not hotword_service.create_stream(connection_id):
        logger.error(f"无法为连接 {connection_id} 创建KWS流")
        await websocket.close(code=1011, reason="Failed to create KWS stream")
        return
    
    try:
        # 发送初始配置
        await websocket.send_json({
            "type": "config",
            "connection_id": connection_id,
            "sample_rate": hotword_service.sample_rate,
            "threshold": hotword_service.threshold,
            "useAudioWorklet": bool(args.pcm_input)
        })
        
        # 主循环：接收音频数据
        while True:
            message = await websocket.receive_bytes()
            # 处理音频数据，检测唤醒词
            wakeword = hotword_service.process_audio(connection_id, message)
            
            if wakeword:
                # 发送唤醒词检测通知
                await websocket.send_json({
                    "type": "wakeword_detected",
                    "wakeword": wakeword,
                    "timestamp": time.time()
                })
                logger.info(f"向连接 {connection_id} 发送唤醒词检测通知: {wakeword}")
                
    except WebSocketDisconnect:
        logger.info(f"唤醒词检测WebSocket连接断开，ID: {connection_id}")
    except KeyError as e:
        if 'bytes' in str(e):
            logger.warning(f"客户端 {connection_id} 已关闭连接")
        else:
            logger.error(f"连接 {connection_id} 的KeyError: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"唤醒词检测WebSocket端点异常 (连接: {connection_id}): {e}", exc_info=True)
    finally:
        # 清理资源
        hotword_service.delete_stream(connection_id)
        logger.info(f"唤醒词检测连接 {connection_id} 清理完成")


def main():
    """Entry point for the CLI command."""
    import uvicorn
    
    uvicorn_kwargs = {
        "app": "whisperlivekit.basic_server:app",
        "host":args.host, 
        "port":args.port, 
        "reload": False,
        "log_level": "info",
        "lifespan": "on",
    }
    
    ssl_kwargs = {}
    if args.ssl_certfile or args.ssl_keyfile:
        if not (args.ssl_certfile and args.ssl_keyfile):
            raise ValueError("Both --ssl-certfile and --ssl-keyfile must be specified together.")
        ssl_kwargs = {
            "ssl_certfile": args.ssl_certfile,
            "ssl_keyfile": args.ssl_keyfile
        }

    if ssl_kwargs:
        uvicorn_kwargs = {**uvicorn_kwargs, **ssl_kwargs}
    if args.forwarded_allow_ips:
        uvicorn_kwargs = { **uvicorn_kwargs, "forwarded_allow_ips" : args.forwarded_allow_ips }

    uvicorn.run(**uvicorn_kwargs)

if __name__ == "__main__":
    main()
