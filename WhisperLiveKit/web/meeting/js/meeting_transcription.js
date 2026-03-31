document.addEventListener('DOMContentLoaded', function() {
    const btnBack = document.getElementById('btn-back');
    const btnPause = document.getElementById('btn-pause');
    const btnStop = document.getElementById('btn-stop');
    const recordingTime = document.querySelector('.recording-time');
    const recordingTitle = document.querySelector('.recording-title');
    const statusBadge = document.getElementById('transcription-status-badge');
    const durationText = document.getElementById('duration-text');

    let isRecording = true;
    let startTime = Date.now() - 168000;
    let timerInterval;

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
        if (isRecording) {
            const elapsed = Date.now() - startTime;
            recordingTime.textContent = `${formatTime(elapsed)} / 06:00:00`;
            durationText.textContent = `时长 ${formatTime(elapsed)}`;
        }
    }

    updateTimer();
    timerInterval = setInterval(updateTimer, 1000);

    btnBack.addEventListener('click', function() {
        window.location.href = 'meeting_home.html';
    });

    btnPause.addEventListener('mouseenter', function() {
        const pauseImg = btnPause.querySelector('img');
        if (pauseImg) {
            pauseImg.src = 'img/runHover.svg';
        }
    });

    btnPause.addEventListener('mouseleave', function() {
        const pauseImg = btnPause.querySelector('img');
        if (pauseImg) {
            pauseImg.src = 'img/playNew.svg';
        }
    });

    btnPause.addEventListener('click', function() {
        isRecording = !isRecording;
        const pauseTooltip = document.getElementById('pause-tooltip');
        const recordingSprite = document.querySelector('.recording-sprite');
        
        if (isRecording) {
            recordingTitle.textContent = '录音中...';
            btnPause.innerHTML = `
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <rect x="6" y="4" width="4" height="16" rx="1" fill="currentColor"/>
                    <rect x="14" y="4" width="4" height="16" rx="1" fill="currentColor"/>
                </svg>
            `;
            pauseTooltip.textContent = '暂停录音';
            if (recordingSprite) {
                recordingSprite.classList.remove('paused');
            }
            if (statusBadge) {
                statusBadge.textContent = '转录中';
                statusBadge.className = 'status-badge status-recording';
            }
        } else {
            recordingTitle.textContent = '已暂停';
            btnPause.innerHTML = `
                <img src="img/playNew.svg" alt="继续录音" style="width: 20px; height: 20px;">
            `;
            pauseTooltip.textContent = '继续录音';
            if (recordingSprite) {
                recordingSprite.classList.add('paused');
            }
            if (statusBadge) {
                statusBadge.textContent = '暂停';
                statusBadge.className = 'status-badge status-paused';
            }
        }
        showToast(isRecording ? '继续录音' : '已暂停录音', 'info');
    });

    btnStop.addEventListener('click', function() {
        if (confirm('确定要结束录音吗？')) {
            clearInterval(timerInterval);
            showToast('录音已结束', 'success');
        }
    });

    const stopImg = document.querySelector('#btn-stop img');
    if (stopImg) {
        btnStop.addEventListener('mouseenter', function() {
            stopImg.src = 'img/stopHover.svg';
        });
        btnStop.addEventListener('mouseleave', function() {
            stopImg.src = 'img/stop.svg';
        });
    }

    function showToast(message, type = 'info') {
        const toast = document.createElement('div');
        const colors = {
            info: '#6366f1',
            success: '#10b981',
            warning: '#f59e0b',
            error: '#ef4444'
        };
        
        toast.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: white;
            padding: 12px 20px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            border-left: 4px solid ${colors[type]};
            z-index: 9999;
            font-size: 14px;
            color: #374151;
            animation: slideIn 0.3s ease;
        `;
        toast.textContent = message;
        document.body.appendChild(toast);

        setTimeout(() => {
            toast.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => {
                document.body.removeChild(toast);
            }, 300);
        }, 2500);
    }

    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideIn {
            from {
                transform: translateX(100%);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }
        @keyframes slideOut {
            from {
                transform: translateX(0);
                opacity: 1;
            }
            to {
                transform: translateX(100%);
                opacity: 0;
            }
        }
    `;
    document.head.appendChild(style);
});
