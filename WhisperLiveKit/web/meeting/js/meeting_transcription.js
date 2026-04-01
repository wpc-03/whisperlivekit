document.addEventListener('DOMContentLoaded', async function() {
    const btnBack = document.getElementById('btn-back');
    const btnPause = document.getElementById('btn-pause');
    const btnStop = document.getElementById('btn-stop');
    const recordingTime = document.querySelector('.recording-time');
    const recordingTitle = document.querySelector('.recording-title');
    const statusBadge = document.getElementById('transcription-status-badge');
    const durationText = document.getElementById('duration-text');
    const transcriptionList = document.getElementById('transcription-list');
    const transcriptionContent = document.getElementById('transcription-content');

    let isRecording = false; // Is active and capturing
    let isPaused = false;
    let startTime = null;
    let totalElapsed = 0; // To handle pauses
    let lastResumeTime = null;
    let timerInterval;

    // ASR & Audio variables
    let websocket = null;
    let stream = null;
    let audioContext = null;
    let workletNode = null;
    let recorderWorker = null;
    let asrRecorder = null; // MediaRecorder for ASR
    let serverUseAudioWorklet = false;

    let fullAudioRecorder = null;
    let fullAudioChunks = [];
    
    let currentTranscriptionData = []; // Store lines for POST
    
    // Set Meeting Title
    const now = new Date();
    const meetingTitleStr = `${now.getFullYear()}-${String(now.getMonth()+1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')} ${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')} 记录`;
    document.querySelector('.meeting-title').textContent = meetingTitleStr;

    function formatTime(ms) {
        const totalSeconds = Math.floor(ms / 1000);
        const hours = Math.floor(totalSeconds / 3600);
        const minutes = Math.floor((totalSeconds % 3600) / 60);
        const seconds = totalSeconds % 60;
        
        if (hours > 0) {
            return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
        }
        return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
    }

    function updateTimer() {
        if (isRecording && !isPaused) {
            const currentElapsed = totalElapsed + (Date.now() - lastResumeTime);
            recordingTime.textContent = `${formatTime(currentElapsed)} / 06:00:00`;
            durationText.textContent = `时长 ${formatTime(currentElapsed)}`;
        }
    }

    btnBack.addEventListener('click', function() {
        if (isRecording) {
            if (!confirm('正在录音中，返回将丢失当前记录。确定要返回吗？')) {
                return;
            }
        }
        window.location.href = 'meeting_home.html';
    });

    btnPause.addEventListener('mouseenter', function() {
        const pauseImg = btnPause.querySelector('img');
        if (pauseImg) pauseImg.src = 'img/runHover.svg';
    });

    btnPause.addEventListener('mouseleave', function() {
        const pauseImg = btnPause.querySelector('img');
        if (pauseImg) pauseImg.src = 'img/playNew.svg';
    });

    btnPause.addEventListener('click', function() {
        if (!isRecording) return;
        
        isPaused = !isPaused;
        const pauseTooltip = document.getElementById('pause-tooltip');
        const recordingSprite = document.querySelector('.recording-sprite');
        
        if (!isPaused) {
            // Resume
            lastResumeTime = Date.now();
            recordingTitle.textContent = '录音中...';
            btnPause.innerHTML = `
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <rect x="6" y="4" width="4" height="16" rx="1" fill="currentColor"/>
                    <rect x="14" y="4" width="4" height="16" rx="1" fill="currentColor"/>
                </svg>
            `;
            pauseTooltip.textContent = '暂停录音';
            if (recordingSprite) recordingSprite.classList.remove('paused');
            if (statusBadge) {
                statusBadge.textContent = '转录中';
                statusBadge.className = 'status-badge status-recording';
            }
            if (fullAudioRecorder && fullAudioRecorder.state === 'paused') {
                fullAudioRecorder.resume();
            }
            if (asrRecorder && asrRecorder.state === 'paused') {
                asrRecorder.resume();
            }
            showToast('继续录音', 'info');
        } else {
            // Pause
            totalElapsed += (Date.now() - lastResumeTime);
            recordingTitle.textContent = '已暂停';
            btnPause.innerHTML = `
                <img src="img/playNew.svg" alt="继续录音" style="width: 20px; height: 20px;">
            `;
            pauseTooltip.textContent = '继续录音';
            if (recordingSprite) recordingSprite.classList.add('paused');
            if (statusBadge) {
                statusBadge.textContent = '暂停';
                statusBadge.className = 'status-badge status-paused';
            }
            if (fullAudioRecorder && fullAudioRecorder.state === 'recording') {
                fullAudioRecorder.pause();
            }
            if (asrRecorder && asrRecorder.state === 'recording') {
                asrRecorder.pause();
            }
            showToast('已暂停录音', 'info');
        }
    });

    btnStop.addEventListener('click', async function() {
        if (!isRecording) return;
        if (confirm('确定要结束录音并保存会议记录吗？')) {
            clearInterval(timerInterval);
            isRecording = false;
            
            showToast('正在保存记录，请稍候...', 'info');
            btnStop.disabled = true;
            btnPause.disabled = true;
            statusBadge.textContent = '保存中';
            statusBadge.className = 'status-badge status-paused';
            recordingTitle.textContent = '处理中...';
            
            await finalizeRecording();
        }
    });

    const stopImg = document.querySelector('#btn-stop img');
    if (stopImg) {
        btnStop.addEventListener('mouseenter', () => stopImg.src = 'img/stopHover.svg');
        btnStop.addEventListener('mouseleave', () => stopImg.src = 'img/stop.svg');
    }

    function showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        
        // 配合 Element Plus 风格的图标和文字
        let iconHtml = '';
        if (type === 'success') {
            iconHtml = `<svg viewBox="0 0 1024 1024" width="16" height="16"><path d="M512 64a448 448 0 1 1 0 896 448 448 0 0 1 0-896zm-55.808 536.384-99.52-99.584a38.4 38.4 0 1 0-54.336 54.336l126.72 126.72a38.272 38.272 0 0 0 54.336 0l262.4-262.464a38.4 38.4 0 1 0-54.272-54.336L456.192 600.384z" fill="#67C23A"></path></svg>`;
        } else if (type === 'error') {
            iconHtml = `<svg viewBox="0 0 1024 1024" width="16" height="16"><path d="M512 64a448 448 0 1 1 0 896 448 448 0 0 1 0-896zm0 393.664L407.36 353.024a38.4 38.4 0 1 0-54.336 54.336L457.664 512 353.024 616.64a38.4 38.4 0 1 0 54.336 54.336L512 566.336 616.64 670.976a38.4 38.4 0 1 0 54.336-54.336L566.336 512 670.976 407.36a38.4 38.4 0 1 0-54.336-54.336L512 457.664z" fill="#F56C6C"></path></svg>`;
        } else {
            iconHtml = `<svg viewBox="0 0 1024 1024" width="16" height="16"><path d="M512 64a448 448 0 1 1 0 896 448 448 0 0 1 0-896zm0 192a58.432 58.432 0 0 0-58.24 63.744l23.36 256.384a35.072 35.072 0 0 0 69.76 0l23.296-256.384A58.432 58.432 0 0 0 512 256zm0 512a51.2 51.2 0 1 0 0-102.4 51.2 51.2 0 0 0 0 102.4z" fill="#909399"></path></svg>`;
        }

        toast.innerHTML = `<div style="display: flex; align-items: center; gap: 8px;">${iconHtml}<span>${message}</span></div>`;
        
        // Element Plus Toast 样式
        Object.assign(toast.style, {
            position: 'fixed',
            top: '20px',
            left: '50%',
            transform: 'translateX(-50%)',
            padding: '10px 15px',
            borderRadius: '4px',
            fontSize: '14px',
            zIndex: '9999',
            boxShadow: '0 6px 16px rgba(0, 0, 0, 0.08)',
            opacity: '0',
            transition: 'opacity 0.3s, top 0.3s, transform 0.3s',
            display: 'flex',
            alignItems: 'center',
            backgroundColor: '#ffffff',
            border: '1px solid #ebeef5'
        });

        if (type === 'success') {
            toast.style.backgroundColor = '#f0f9eb';
            toast.style.borderColor = '#e1f3d8';
            toast.style.color = '#67c23a';
        } else if (type === 'error') {
            toast.style.backgroundColor = '#fef0f0';
            toast.style.borderColor = '#fde2e2';
            toast.style.color = '#f56c6c';
        } else {
            toast.style.backgroundColor = '#f4f4f5';
            toast.style.borderColor = '#e9e9eb';
            toast.style.color = '#909399';
        }
        
        document.body.appendChild(toast);
        
        // 动画显示
        setTimeout(() => {
            toast.style.opacity = '1';
            toast.style.top = '40px';
        }, 10);

        // 3秒后移除
        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.top = '20px';
            setTimeout(() => {
                if (document.body.contains(toast)) {
                    document.body.removeChild(toast);
                }
            }, 300);
        }, 3000);
    }

    function parseTimeToSeconds(timeStr) {
        if (timeStr === undefined || timeStr === null) return null;
        // Handle both "11.5" (plain seconds) and "0:00:11" (HH:MM:SS) formats
        const str = String(timeStr);
        if (str.includes(':')) {
            const parts = str.split(':');
            let seconds = 0;
            if (parts.length === 3) {
                // HH:MM:SS
                seconds = parseInt(parts[0]) * 3600 + parseInt(parts[1]) * 60 + parseFloat(parts[2]);
            } else if (parts.length === 2) {
                // MM:SS
                seconds = parseInt(parts[0]) * 60 + parseFloat(parts[1]);
            }
            return seconds;
        }
        return parseFloat(str);
    }

    function formatTimeDisplay(seconds) {
        if (seconds === null || seconds === undefined) return '';
        const totalSeconds = Math.floor(seconds);
        const minutes = Math.floor(totalSeconds / 60).toString().padStart(2, '0');
        const secs = (totalSeconds % 60).toString().padStart(2, '0');
        return `${minutes}:${secs}`;
    }

    // ASR & Rendering logic
    function renderLines(lines, buffer_transcription, buffer_diarization) {
        currentTranscriptionData = lines || [];
        transcriptionList.innerHTML = '';
        
        let combinedLines = [...currentTranscriptionData];
        
        combinedLines.forEach(item => {
            if (item.speaker === -2 || item.speaker === 0) return; // Skip silence or loading
            const div = document.createElement('div');
            div.className = 'transcription-item';
            
            const startSec = parseTimeToSeconds(item.start);
            const endSec = parseTimeToSeconds(item.end);
            
            let timeStr = '';
            if (startSec !== null) {
                timeStr = formatTimeDisplay(startSec);
                if (endSec !== null) {
                    timeStr += ` - ${formatTimeDisplay(endSec)}`;
                }
            }
            
            div.innerHTML = `
                <div class="speaker-info">
                    <div class="speaker-avatar">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M20 21V19C20 17.9391 19.5786 16.9217 18.8284 16.1716C18.0783 15.4214 17.0609 15 16 15H8C6.93913 15 5.92172 15.4214 5.17157 16.1716C4.42143 16.9217 4 17.9391 4 19V21" stroke="currentColor" stroke-width="2"/>
                            <path d="M12 11C14.2091 11 16 9.20914 16 7C16 4.79086 14.2091 3 12 3C9.79086 3 8 4.79086 8 7C8 9.20914 9.79086 11 12 11Z" stroke="currentColor" stroke-width="2"/>
                        </svg>
                    </div>
                    <div class="speaker-name">发言人 ${item.speaker}</div>
                    <div class="speaker-time" data-time="${startSec !== null ? startSec : 0}" style="cursor: pointer; color: #6366f1;" title="点击跳转播放">${timeStr}</div>
                </div>
                <div class="transcription-text">
                    ${item.text || ''}
                </div>
            `;
            transcriptionList.appendChild(div);
        });

        let bufferText = "";
        if (buffer_diarization) bufferText += buffer_diarization + " ";
        if (buffer_transcription) bufferText += buffer_transcription;

        if (bufferText.trim()) {
            const div = document.createElement('div');
            div.className = 'transcription-item';
            div.innerHTML = `
                <div class="speaker-info">
                    <div class="speaker-avatar">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M20 21V19C20 17.9391 19.5786 16.9217 18.8284 16.1716C18.0783 15.4214 17.0609 15 16 15H8C6.93913 15 5.92172 15.4214 5.17157 16.1716C4.42143 16.9217 4 17.9391 4 19V21" stroke="currentColor" stroke-width="2"/>
                            <path d="M12 11C14.2091 11 16 9.20914 16 7C16 4.79086 14.2091 3 12 3C9.79086 3 8 4.79086 8 7C8 9.20914 9.79086 11 12 11Z" stroke="currentColor" stroke-width="2"/>
                        </svg>
                    </div>
                    <div class="speaker-name">正在识别...</div>
                    <div class="speaker-time"></div>
                </div>
                <div class="transcription-text transcription-text-editing">
                    ${bufferText.trim()}
                </div>
            `;
            transcriptionList.appendChild(div);
        }

        if (transcriptionContent) {
            transcriptionContent.scrollTop = transcriptionContent.scrollHeight;
        }
    }

    async function initAudioAndWebSocket() {
        try {
            const host = window.location.hostname || "localhost";
            const port = window.location.port;
            const protocol = window.location.protocol === "https:" ? "wss" : "ws";
            const wsUrl = `${protocol}://${host}${port ? ":" + port : ""}/asr`;
            
            websocket = new WebSocket(wsUrl);

            websocket.onopen = async () => {
                console.log("WebSocket connected.");
            };

            websocket.onmessage = async (event) => {
                const data = JSON.parse(event.data);
                
                if (data.type === "config") {
                    serverUseAudioWorklet = !!data.useAudioWorklet;
                    await startAudioCapture();
                    return;
                }

                if (data.type === "ready_to_stop") {
                    console.log("Ready to stop received.");
                    if (websocket) websocket.close();
                    return;
                }

                renderLines(data.lines, data.buffer_transcription, data.buffer_diarization);
            };

            websocket.onclose = () => {
                console.log("WebSocket closed.");
            };

            websocket.onerror = (err) => {
                console.error("WebSocket error", err);
                showToast("连接服务器失败，请检查网络", "error");
            };

        } catch (err) {
            console.error("Initialization error:", err);
            showToast("初始化录音失败", "error");
        }
    }

    async function startAudioCapture() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            
            // Setup Full Audio Recorder
            try {
                // Try recording as standard webm/opus or mp4
                fullAudioRecorder = new MediaRecorder(stream, { mimeType: "audio/webm;codecs=opus" });
            } catch (e1) {
                try {
                    // Fallback for some browsers (e.g. Safari)
                    fullAudioRecorder = new MediaRecorder(stream, { mimeType: "audio/mp4" });
                } catch (e2) {
                    // Ultimate fallback
                    fullAudioRecorder = new MediaRecorder(stream);
                }
            }
            fullAudioRecorder.ondataavailable = (e) => {
                if (e.data && e.data.size > 0) {
                    fullAudioChunks.push(e.data);
                }
            };
            fullAudioRecorder.start(1000); // chunk every 1s

            // Setup ASR Streaming
            if (serverUseAudioWorklet) {
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                const microphone = audioContext.createMediaStreamSource(stream);
                
                await audioContext.audioWorklet.addModule("/web/sdk/pcm_worklet.js");
                workletNode = new AudioWorkletNode(audioContext, "pcm-forwarder", { numberOfInputs: 1, numberOfOutputs: 0, channelCount: 1 });
                microphone.connect(workletNode);

                recorderWorker = new Worker("/web/sdk/recorder_worker.js");
                recorderWorker.postMessage({
                    command: "init",
                    config: { sampleRate: audioContext.sampleRate }
                });

                recorderWorker.onmessage = (e) => {
                    if (!isPaused && websocket && websocket.readyState === WebSocket.OPEN) {
                        websocket.send(e.data.buffer);
                    }
                };

                workletNode.port.onmessage = (e) => {
                    if (!isPaused) {
                        const data = e.data;
                        const ab = data instanceof ArrayBuffer ? data : data.buffer;
                        recorderWorker.postMessage({ command: "record", buffer: ab }, [ab]);
                    }
                };
            } else {
                try {
                    asrRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
                } catch (e) {
                    asrRecorder = new MediaRecorder(stream);
                }
                asrRecorder.ondataavailable = (e) => {
                    if (!isPaused && websocket && websocket.readyState === WebSocket.OPEN) {
                        if (e.data && e.data.size > 0) {
                            websocket.send(e.data);
                        }
                    }
                };
                asrRecorder.start(100);
            }

            // Start Timing
            isRecording = true;
            isPaused = false;
            startTime = Date.now();
            lastResumeTime = startTime;
            timerInterval = setInterval(updateTimer, 1000);
            
            showToast("录音已开始", "success");

        } catch (err) {
            console.error("Audio capture error:", err);
            showToast("无法获取麦克风权限", "error");
        }
    }

    async function finalizeRecording() {
        // Send empty blob to websocket to signal end
        if (websocket && websocket.readyState === WebSocket.OPEN) {
            const emptyBlob = new Blob([], { type: "audio/webm" });
            websocket.send(emptyBlob);
        }

        // Wait a short moment to allow final WebSocket messages to arrive
        await new Promise(resolve => setTimeout(resolve, 1500));

        // Stop Full Recorder and collect final blob
        return new Promise((resolve) => {
            if (fullAudioRecorder && fullAudioRecorder.state !== "inactive") {
                fullAudioRecorder.onstop = async () => {
                    await submitMeetingData();
                    resolve();
                };
                fullAudioRecorder.stop();
            } else {
                submitMeetingData().then(resolve);
            }
            
            // Clean up resources
            if (asrRecorder && asrRecorder.state !== "inactive") asrRecorder.stop();
            if (recorderWorker) recorderWorker.terminate();
            if (workletNode) workletNode.disconnect();
            if (audioContext) audioContext.close();
            if (stream) stream.getTracks().forEach(track => track.stop());
        });
    }

    async function submitMeetingData() {
        // Find correct mime type for Blob
        let mimeType = "audio/webm";
        if (fullAudioRecorder && fullAudioRecorder.mimeType) {
            mimeType = fullAudioRecorder.mimeType;
        }
        
        const audioBlob = new Blob(fullAudioChunks, { type: mimeType });
        const formData = new FormData();
        
        // Calculate duration in seconds
        let finalDuration = totalElapsed;
        if (!isPaused && lastResumeTime) {
            finalDuration += (Date.now() - lastResumeTime);
        }
        const durationSeconds = Math.floor(finalDuration / 1000);

        formData.append("title", meetingTitleStr);
        formData.append("start_time", new Date(startTime || Date.now()).toISOString());
        formData.append("duration", durationSeconds);
        
        // Filter out temporary/loading states and format for database
        const cleanTranscriptionData = currentTranscriptionData.filter(item => item.speaker > 0);
        formData.append("transcription_data", JSON.stringify(cleanTranscriptionData));
        
        // Use standard extension mapping based on mimeType
        let extension = "webm";
        if (mimeType.includes("mp4")) {
            extension = "mp4";
        } else if (mimeType.includes("ogg")) {
            extension = "ogg";
        }
        
        formData.append("audio_file", audioBlob, `record.${extension}`);

        try {
            const response = await fetch("/api/meetings", {
                method: "POST",
                body: formData
            });

            if (response.ok) {
                showToast("会议记录保存成功！", "success");
                setTimeout(() => {
                    window.location.href = "meeting_home.html";
                }, 1500);
            } else {
                throw new Error("Server returned " + response.status);
            }
        } catch (error) {
            console.error("Submit error:", error);
            showToast("保存失败，请重试", "error");
            btnStop.disabled = false;
            btnPause.disabled = false;
            statusBadge.textContent = '出错';
            statusBadge.className = 'status-badge status-error';
        }
    }

    // Init on load
    initAudioAndWebSocket();

    // Keyframes for toast
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideIn {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        @keyframes slideOut {
            from { transform: translateX(0); opacity: 1; }
            to { transform: translateX(100%); opacity: 0; }
        }
        .status-error { background: #ef4444; color: white; }
    `;
    document.head.appendChild(style);
});
