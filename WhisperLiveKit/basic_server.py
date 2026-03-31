import asyncio
import logging
import pathlib
import time
from contextlib import asynccontextmanager
from datetime import timedelta

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, status, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordRequestForm

from whisperlivekit import (AudioProcessor, TranscriptionEngine,
                            get_inline_ui_html, parse_args)
from whisperlivekit.hotword_service import HotwordService
from whisperlivekit.auth import user_manager, create_access_token, get_current_active_user, ACCESS_TOKEN_EXPIRE_MINUTES
from whisperlivekit.keywords_manager import KeywordsManager
from whisperlivekit.wakewords_manager import WakewordsManager
from whisperlivekit.config_manager import config_manager
from whisperlivekit.database import get_all_meetings, get_meeting_by_id, create_meeting, delete_meeting, update_meeting_title
import whisperlivekit.web as webpkg
import os
import shutil
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger().setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 解析命令行参数
args = parse_args()

# 从配置文件加载配置并更新args对象
config = config_manager.get_config()
for key, value in config.items():
    if hasattr(args, key):
        setattr(args, key, value)
    elif key == "backend_policy":
        # backend_policy参数在parse_args中可能是其他名称
        if hasattr(args, "backend_policy"):
            setattr(args, "backend_policy", value)

transcription_engine = None
hotword_service = None
keywords_manager = None
wakewords_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):    
    global transcription_engine, hotword_service, keywords_manager, wakewords_manager
    
    # 初始化转录引擎
    transcription_engine = TranscriptionEngine(
        **vars(args),
    )
    
    # 初始化关键词管理器
    keywords_file = args.keywords_file if hasattr(args, 'keywords_file') else 'keywords.txt'
    keywords_manager = KeywordsManager(keywords_file)
    
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
        
        # 初始化唤醒词管理器
        wakewords_manager = WakewordsManager(args.hotword_model_dir)
        
        logger.info(f"唤醒词检测服务初始化完成，模型目录: {args.hotword_model_dir}，阈值: {args.hotword_threshold}")
    else:
        hotword_service = None
        wakewords_manager = None
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
app.mount("/web", StaticFiles(directory=str(web_dir), html=True), name="web")

# 挂载管理系统静态文件目录
admin_dir = web_dir / "admin"
if admin_dir.exists():
    app.mount("/admin", StaticFiles(directory=str(admin_dir), html=True), name="admin")

# 挂载会议系统静态文件目录
meeting_dir = web_dir / "meeting"
if meeting_dir.exists():
    app.mount("/meeting", StaticFiles(directory=str(meeting_dir), html=True), name="meeting")

@app.get("/meeting")
async def meeting_home_redirect():
    # 访问 /meeting 时进行重定向到 /meeting/meeting_home.html，以确保前端相对路径能正确解析
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/meeting/meeting_home.html")

@app.get("/")
async def get():
    return HTMLResponse(get_inline_ui_html())

# 在下载路由中添加异常处理
@app.get("/web/download/sdk.zip")
async def download_sdk():
    try:
        sdk_path = web_dir / "download" / "sdk.zip"
        if not sdk_path.exists():
            raise HTTPException(status_code=404, detail="SDK文件不存在")
        
        return FileResponse(
            path=str(sdk_path),
            filename="WhisperLiveKit-Web-SDK.zip",
            media_type="application/zip",
            headers={
                "Content-Disposition": "attachment; filename=WhisperLiveKit-Web-SDK.zip",
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    except Exception as e:
        logger.warning(f"下载过程中发生异常: {e}")
        raise
        
# 挂载会议录音文件夹
records_dir = pathlib.Path(__file__).parent / "data" / "records"
records_dir.mkdir(parents=True, exist_ok=True)
app.mount("/data/records", StaticFiles(directory=str(records_dir)), name="records")

# 会议记录相关接口
@app.get("/api/meetings")
async def api_get_meetings():
    meetings = get_all_meetings()
    return JSONResponse(content={"meetings": meetings})

@app.get("/api/meetings/{meeting_id}")
async def api_get_meeting_detail(meeting_id: str):
    meeting = get_meeting_by_id(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
    return JSONResponse(content=meeting)

@app.post("/api/meetings")
async def api_create_meeting(
    title: str = Form(...),
    start_time: str = Form(...),
    duration: int = Form(...),
    transcription_data: str = Form(...),
    audio_file: UploadFile = File(...)
):
    # 保存音频文件
    filename = audio_file.filename
    safe_filename = f"{int(time.time())}_{filename}"
    file_path = records_dir / safe_filename
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(audio_file.file, buffer)
        
    audio_url = f"/data/records/{safe_filename}"
    
    # 保存到数据库
    try:
        # Validate JSON
        json.loads(transcription_data)
        meeting_id = create_meeting(
            title=title,
            start_time=start_time,
            duration=duration,
            audio_path=audio_url,
            transcription_data=transcription_data
        )
        return JSONResponse(content={"id": meeting_id, "message": "Meeting created successfully"})
    except Exception as e:
        logger.error(f"Error creating meeting: {e}")
        # 如果数据库保存失败，尝试删除刚上传的文件
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail="Failed to create meeting record")

@app.delete("/api/meetings/{meeting_id}")
async def api_delete_meeting(meeting_id: str):
    meeting = get_meeting_by_id(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
        
    # 删除音频文件
    if meeting.get("audio_path"):
        filename = meeting["audio_path"].split("/")[-1]
        file_path = records_dir / filename
        if file_path.exists():
            file_path.unlink()
            
    # 删除数据库记录
    deleted = delete_meeting(meeting_id)
    if not deleted:
        raise HTTPException(status_code=500, detail="Failed to delete meeting record")
        
    return JSONResponse(content={"message": "Meeting deleted successfully"})

@app.patch("/api/meetings/{meeting_id}")
async def api_update_meeting(meeting_id: str, title: str = Form(...)):
    meeting = get_meeting_by_id(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
    
    updated = update_meeting_title(meeting_id, title)
    if not updated:
        raise HTTPException(status_code=500, detail="Failed to update meeting title")
    
    return JSONResponse(content={"message": "Meeting title updated successfully"})

# 认证相关接口
@app.post("/api/auth/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = user_manager.authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/api/auth/logout")
async def logout(current_user: dict = Depends(get_current_active_user)):
    # 前端处理退出，后端无需特殊处理
    return {"message": "Logout successful"}

@app.get("/api/auth/me")
async def get_me(current_user: dict = Depends(get_current_active_user)):
    return {"username": current_user["username"]}

# 专业术语管理接口
@app.get("/api/keywords")
async def get_keywords(current_user: dict = Depends(get_current_active_user)):
    if not keywords_manager:
        raise HTTPException(status_code=404, detail="Keywords manager not initialized")
    return keywords_manager.get_keywords()

@app.post("/api/keywords")
async def add_keyword(keyword: str = Form(...), current_user: dict = Depends(get_current_active_user)):
    if not keywords_manager:
        raise HTTPException(status_code=404, detail="Keywords manager not initialized")
    success = keywords_manager.add_keyword(keyword)
    if success:
        return {"message": "Keyword added successfully"}
    else:
        raise HTTPException(status_code=400, detail="Failed to add keyword")

@app.put("/api/keywords/{old_keyword}")
async def update_keyword(old_keyword: str, new_keyword: str = Form(...), current_user: dict = Depends(get_current_active_user)):
    if not keywords_manager:
        raise HTTPException(status_code=404, detail="Keywords manager not initialized")
    success = keywords_manager.update_keyword(old_keyword, new_keyword)
    if success:
        return {"message": "Keyword updated successfully"}
    else:
        raise HTTPException(status_code=400, detail="Failed to update keyword")

@app.delete("/api/keywords/{keyword}")
async def delete_keyword(keyword: str, current_user: dict = Depends(get_current_active_user)):
    if not keywords_manager:
        raise HTTPException(status_code=404, detail="Keywords manager not initialized")
    success = keywords_manager.delete_keyword(keyword)
    if success:
        return {"message": "Keyword deleted successfully"}
    else:
        raise HTTPException(status_code=400, detail="Failed to delete keyword")

# 唤醒词管理接口
@app.get("/api/wakewords")
async def get_wakewords(current_user: dict = Depends(get_current_active_user)):
    if not wakewords_manager:
        raise HTTPException(status_code=404, detail="Wakewords manager not initialized")
    return wakewords_manager.get_wakewords()

@app.post("/api/wakewords")
async def add_wakeword(word: str = Form(...), boost: float = Form(None), threshold: float = Form(None), current_user: dict = Depends(get_current_active_user)):
    if not wakewords_manager:
        raise HTTPException(status_code=404, detail="Wakewords manager not initialized")
    success = wakewords_manager.add_wakeword(word, boost, threshold)
    if success:
        return {"message": "Wakeword added successfully"}
    else:
        raise HTTPException(status_code=400, detail="Failed to add wakeword")

@app.put("/api/wakewords/{old_word}")
async def update_wakeword(old_word: str, new_word: str = Form(...), boost: float = Form(None), threshold: float = Form(None), current_user: dict = Depends(get_current_active_user)):
    if not wakewords_manager:
        raise HTTPException(status_code=404, detail="Wakewords manager not initialized")
    success = wakewords_manager.update_wakeword(old_word, new_word, boost, threshold)
    if success:
        return {"message": "Wakeword updated successfully"}
    else:
        raise HTTPException(status_code=400, detail="Failed to update wakeword")

@app.delete("/api/wakewords/{word}")
async def delete_wakeword(word: str, current_user: dict = Depends(get_current_active_user)):
    if not wakewords_manager:
        raise HTTPException(status_code=404, detail="Wakewords manager not initialized")
    success = wakewords_manager.delete_wakeword(word)
    if success:
        return {"message": "Wakeword deleted successfully"}
    else:
        raise HTTPException(status_code=400, detail="Failed to delete wakeword")

@app.post("/api/wakewords/convert")
async def convert_wakewords(current_user: dict = Depends(get_current_active_user)):
    if not wakewords_manager:
        raise HTTPException(status_code=404, detail="Wakewords manager not initialized")
    success = wakewords_manager.convert_wakewords()
    if success:
        return {"message": "Wakewords converted successfully"}
    else:
        raise HTTPException(status_code=400, detail="Failed to convert wakewords")

# 配置管理接口
@app.get("/api/config")
async def get_config(current_user: dict = Depends(get_current_active_user)):
    """获取当前配置"""
    return config_manager.get_config()

@app.put("/api/config")
async def update_config(config_data: dict, current_user: dict = Depends(get_current_active_user)):
    """更新配置"""
    success = config_manager.update_config(config_data)
    if success:
        return {"message": "配置更新成功", "restart_required": True}
    else:
        raise HTTPException(status_code=400, detail="配置更新失败")

@app.post("/api/config/reset")
async def reset_config(current_user: dict = Depends(get_current_active_user)):
    """重置配置为默认值"""
    success = config_manager.reset_config()
    if success:
        return {"message": "配置重置成功", "restart_required": True}
    else:
        raise HTTPException(status_code=400, detail="配置重置失败")

@app.get("/api/config/defaults")
async def get_default_config(current_user: dict = Depends(get_current_active_user)):
    """获取默认配置"""
    return config_manager.get_default_config_dict()

@app.post("/api/config/restart")
async def restart_service(current_user: dict = Depends(get_current_active_user)):
    """重启服务（通知管理员需要手动重启）"""
    logger.info("收到服务重启请求，请手动重启Docker容器使配置生效")
    return {"message": "配置已保存，请手动重启Docker容器使配置生效", "manual_restart_required": True}


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
