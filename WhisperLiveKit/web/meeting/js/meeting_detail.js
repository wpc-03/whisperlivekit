document.addEventListener('DOMContentLoaded', function() {
    const btnBack = document.getElementById('btn-back');
    const transcriptionItems = document.querySelectorAll('.transcription-item');

    btnBack.addEventListener('click', function() {
        window.location.href = 'meeting_home.html';
    });

    transcriptionItems.forEach(function(item) {
        const copyBtn = item.querySelector('.mini-action-btn:nth-child(1)');
        const markBtn = item.querySelector('.mini-action-btn:nth-child(2)');
        const editBtn = item.querySelector('.mini-action-btn:nth-child(3)');
        const transcriptionText = item.querySelector('.transcription-text');

        if (copyBtn) {
            copyBtn.addEventListener('click', function(e) {
                e.stopPropagation();
                const text = transcriptionText.textContent.trim();
                navigator.clipboard.writeText(text).then(function() {
                    showToast('已复制到剪贴板', 'success');
                }).catch(function() {
                    showToast('复制失败', 'error');
                });
            });
        }

        if (markBtn) {
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
        }

        if (editBtn) {
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
        }
    });

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
        .transcription-text.marked {
            background-color: #fef3c7 !important;
        }
    `;
    document.head.appendChild(style);
});
