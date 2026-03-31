document.addEventListener('DOMContentLoaded', async function() {
    const btnBack = document.getElementById('btn-back');
    const meetingMeta = document.getElementById('meeting-meta');
    const meetingTitle = document.getElementById('meeting-title');
    const transcriptionList = document.getElementById('transcription-list');
    const meetingAudio = document.getElementById('meeting-audio');

    btnBack.addEventListener('click', function() {
        window.location.href = 'meeting_home.html';
    });

    // 1. 获取 URL 参数
    const urlParams = new URLSearchParams(window.location.search);
    const meetingId = urlParams.get('id');

    if (!meetingId) {
        showToast('未找到会议 ID', 'error');
        setTimeout(() => window.location.href = 'meeting_home.html', 1500);
        return;
    }

    try {
        // 2. 获取会议数据
        const response = await fetch(`/api/meetings/${meetingId}`);
        if (!response.ok) {
            throw new Error('会议不存在或加载失败');
        }
        
        const meeting = await response.json();
        
        // 3. 渲染页面标题和元数据
        meetingTitle.textContent = meeting.title || '未命名会议';
        
        const startTime = new Date(meeting.start_time).toLocaleString('zh-CN', {
            year: 'numeric', month: '2-digit', day: '2-digit',
            hour: '2-digit', minute: '2-digit'
        });
        
        const durationMinutes = Math.round(meeting.duration / 60);

        meetingMeta.innerHTML = `
            <span class="meta-item">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="2"/>
                    <polyline points="12,6 12,12 16,14" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
                ${startTime}
            </span>
            <span class="meta-item">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="2"/>
                    <polyline points="12,6 12,12 16,14" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
                时长 ${durationMinutes} 分钟
            </span>
            <span class="status-badge status-completed">已完成</span>
        `;

        // 4. 设置音频播放器
        if (meeting.audio_path) {
            // 将本地路径转换为 URL，假设后端有配置静态文件路由，或者使用 /records 代理
            // 确保路径以 / 开头，避免相对路径解析错误
            let audioUrl = meeting.audio_path.replace(/\\/g, '/');
            if (!audioUrl.startsWith('/')) {
                audioUrl = '/' + audioUrl;
            }
            meetingAudio.src = audioUrl;

            // Fix for WebM duration issue in Chrome/Edge
            meetingAudio.addEventListener('loadedmetadata', function() {
                if (meetingAudio.duration === Infinity || isNaN(meetingAudio.duration)) {
                    meetingAudio.currentTime = 1e101; // Set to a very large number
                    meetingAudio.addEventListener('timeupdate', function getDuration() {
                        meetingAudio.currentTime = 0;
                        meetingAudio.removeEventListener('timeupdate', getDuration);
                    });
                }
            });

        } else {
            meetingAudio.style.display = 'none';
        }

        // 5. 解析并渲染对话内容
        let transcriptions = [];
        try {
            transcriptions = typeof meeting.transcription_data === 'string' 
                ? JSON.parse(meeting.transcription_data) 
                : meeting.transcription_data;
        } catch (e) {
            console.error('解析转录数据失败:', e);
        }

        if (!Array.isArray(transcriptions) || transcriptions.length === 0) {
            transcriptionList.innerHTML = '<div style="text-align: center; padding: 40px; color: #6b7280;">暂无转录内容</div>';
            return;
        }

        transcriptionList.innerHTML = '';
        transcriptions.forEach((item, index) => {
            // 解析开始时间和结束时间，格式化为 mm:ss
            let timeStr = '00:00';
            let validStartTime = 0;
            
            // Handle both "11.5" (plain seconds) and "0:00:11" (HH:MM:SS) formats
            function parseTimeToSeconds(timeStr) {
                if (timeStr === undefined || timeStr === null) return null;
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
            
            const startSec = parseTimeToSeconds(item.start);
            const endSec = parseTimeToSeconds(item.end);
            
            if (startSec !== null) {
                validStartTime = startSec;
                const totalSeconds = Math.floor(startSec);
                const minutes = Math.floor(totalSeconds / 60).toString().padStart(2, '0');
                const seconds = (totalSeconds % 60).toString().padStart(2, '0');
                
                let endTimeStr = '';
                if (endSec !== null) {
                    const endSeconds = Math.floor(endSec);
                    const endMinutes = Math.floor(endSeconds / 60).toString().padStart(2, '0');
                    const endSecs = (endSeconds % 60).toString().padStart(2, '0');
                    endTimeStr = ` - ${endMinutes}:${endSecs}`;
                }
                
                timeStr = `${minutes}:${seconds}${endTimeStr}`;
            }

            const speakerName = item.speaker !== undefined && item.speaker !== null 
                ? String(item.speaker) 
                : `Speaker ${item.speaker_id || '?'}`;
            const avatarChar = speakerName.charAt(0).toUpperCase();
            
            // 为不同说话人分配不同的颜色类名 (speaker-1, speaker-2, etc.)
            const speakerIdNum = item.speaker_id !== undefined ? item.speaker_id : (index % 5 + 1);
            const speakerClass = `speaker-${speakerIdNum % 5 + 1}`; // 假设有 speaker-1 到 speaker-5 的样式

            const div = document.createElement('div');
            div.className = 'transcription-item';
            div.innerHTML = `
                <div class="speaker-info">
                    <div class="speaker-avatar ${speakerClass}">${avatarChar}</div>
                    <div class="speaker-name">${speakerName}</div>
                    <div class="speaker-time" data-time="${validStartTime}" style="cursor: pointer; color: #6366f1;" title="点击跳转播放">${timeStr}</div>
                </div>
                <div class="transcription-text">
                    ${item.text || ''}
                </div>
                <div class="transcription-actions">
                    <button class="mini-action-btn" title="复制">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <rect x="9" y="9" width="13" height="13" rx="2" stroke="currentColor" stroke-width="2"/>
                            <path d="M5 15H4C3.46957 15 2.96086 14.7893 2.58579 14.4142C2.21071 14.0391 2 13.5304 2 13V4C2 3.46957 2.21071 2.96086 2.58579 2.58579C2.96086 2.21071 3.46957 2 4 2H13C13.5304 2 14.0391 2.21071 14.4142 2.58579C14.7893 2.96086 15 3.46957 15 4V5" stroke="currentColor" stroke-width="2"/>
                        </svg>
                    </button>
                    <button class="mini-action-btn" title="标记">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M5 21V5C5 4.46957 5.21071 3.96086 5.58579 3.58579C5.96086 3.21071 6.46957 3 7 3H17C17.5304 3 18.0391 3.21071 18.4142 3.58579C18.7893 3.96086 19 4.46957 19 5V21L12 18L5 21Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                        </svg>
                    </button>
                    <button class="mini-action-btn" title="编辑">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M11 4H4C3.46957 4 2.96086 4.21071 2.58579 4.58579C2.21071 4.96086 2 5.46957 2 6V20C2 20.5304 2.21071 21.0391 2.58579 21.4142C2.96086 21.7893 3.46957 22 4 22H18C18.5304 22 19.0391 21.7893 19.4142 21.4142C19.7893 21.0391 20 20.5304 20 20V13" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                            <path d="M18.5 2.50001C18.8978 2.10219 19.4374 1.87869 20 1.87869C20.5626 1.87869 21.1022 2.10219 21.5 2.50001C21.8978 2.89784 22.1213 3.4374 22.1213 4.00001C22.1213 4.56262 21.8978 5.10219 21.5 5.50001L12 15L8 16L9 12L18.5 2.50001Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                        </svg>
                    </button>
                </div>
            `;

            // 绑定事件
            const copyBtn = div.querySelector('.mini-action-btn:nth-child(1)');
            const markBtn = div.querySelector('.mini-action-btn:nth-child(2)');
            const editBtn = div.querySelector('.mini-action-btn:nth-child(3)');
            const transcriptionText = div.querySelector('.transcription-text');
            const timeElement = div.querySelector('.speaker-time');

            // 点击时间跳转音频播放
            timeElement.addEventListener('click', () => {
                if (meetingAudio) {
                    const startNum = Number(timeElement.getAttribute('data-time'));
                    if (!isNaN(startNum) && isFinite(startNum)) {
                        meetingAudio.currentTime = startNum;
                        meetingAudio.play().catch(e => console.error("Play error:", e));
                    }
                }
            });

            copyBtn.addEventListener('click', function(e) {
                e.stopPropagation();
                const text = transcriptionText.textContent.trim();
                navigator.clipboard.writeText(text).then(function() {
                    showToast('已复制到剪贴板', 'success');
                }).catch(function() {
                    showToast('复制失败', 'error');
                });
            });

            markBtn.addEventListener('click', function(e) {
                e.stopPropagation();
                transcriptionText.classList.toggle('marked');
                if (transcriptionText.classList.contains('marked')) {
                    transcriptionText.style.backgroundColor = '#fef3c7';
                    showToast('已标记', 'info');
                } else {
                    transcriptionText.style.backgroundColor = '';
                    showToast('已取消标记', 'info');
                }
            });

            editBtn.addEventListener('click', function(e) {
                e.stopPropagation();
                const text = transcriptionText.textContent.trim();
                const textarea = document.createElement('textarea');
                textarea.value = text;
                textarea.style.cssText = `
                    width: 100%;
                    min-height: 80px;
                    padding: 18px 22px;
                    border: 2px solid #6366f1;
                    border-radius: 12px;
                    font-size: 15px;
                    line-height: 1.8;
                    font-family: inherit;
                    resize: vertical;
                `;
                
                const originalContent = transcriptionText.innerHTML;
                transcriptionText.innerHTML = '';
                transcriptionText.appendChild(textarea);
                textarea.focus();

                const saveEdit = function() {
                    const newText = textarea.value.trim();
                    if (newText) {
                        transcriptionText.textContent = newText;
                        showToast('已保存修改', 'success');
                        // TODO: 可选，将修改后的内容同步到后端
                    } else {
                        transcriptionText.innerHTML = originalContent;
                    }
                };

                textarea.addEventListener('blur', saveEdit);
                textarea.addEventListener('keydown', function(e) {
                    if (e.key === 'Enter' && e.ctrlKey) {
                        saveEdit();
                        textarea.blur();
                    }
                    if (e.key === 'Escape') {
                        transcriptionText.innerHTML = originalContent;
                    }
                });
            });

            transcriptionList.appendChild(div);
        });

    } catch (error) {
        console.error('Error fetching meeting:', error);
        showToast(error.message || '加载会议详情失败', 'error');
        setTimeout(() => window.location.href = 'meeting_home.html', 1500);
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

    // Add styles if not already present
    if (!document.getElementById('toast-styles')) {
        const style = document.createElement('style');
        style.id = 'toast-styles';
        style.textContent = `
            @keyframes slideIn {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
            @keyframes slideOut {
                from { transform: translateX(0); opacity: 1; }
                to { transform: translateX(100%); opacity: 0; }
            }
            .transcription-text.marked {
                background-color: #fef3c7 !important;
            }
        `;
        document.head.appendChild(style);
    }
});
