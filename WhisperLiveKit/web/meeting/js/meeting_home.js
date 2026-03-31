document.addEventListener('DOMContentLoaded', function() {
    const navLinks = document.querySelectorAll('.sidebar ul li a');
    const homeContent = document.getElementById('home-content');
    const meetingsContent = document.getElementById('meetings-content');        
    const viewMeetingsBtn = document.querySelector('[data-action="view-meetings"]');
    const btnNewMeeting = document.getElementById('btn-new-meeting');
    const btnNewMeeting2 = document.getElementById('btn-new-meeting2');
    const btnLiveView = document.getElementById('btn-live-view');
    
    // 动态渲染相关 DOM 元素
    const recentMeetingsContainer = document.getElementById('recent-meetings-container');
    const meetingListContainer = document.getElementById('meeting-list-container');
    const searchInput = document.getElementById('search-input');
    const btnSearch = document.getElementById('btn-search');
    const btnPrevPage = document.getElementById('btn-prev-page');
    const btnNextPage = document.getElementById('btn-next-page');
    const pageInfo = document.getElementById('page-info');
    const statTotalMeetings = document.getElementById('stat-total-meetings');
    const statRecording = document.getElementById('stat-recording');
    const statCompleted = document.getElementById('stat-completed');

    // 显示 Toast 提示
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
    let currentPage = 1;
    const itemsPerPage = 5;

    // 获取会议数据
    async function fetchMeetings() {
        try {
            const response = await fetch('/api/meetings');
            if (response.ok) {
                const data = await response.json();
                allMeetings = data.meetings || [];
                filteredMeetings = [...allMeetings];
                currentPage = 1;
                
                updateStats();
                renderRecentMeetings();
                renderMeetingList();
            } else {
                console.error('Failed to fetch meetings');
            }
        } catch (error) {
            console.error('Error fetching meetings:', error);
        }
    }

    // 格式化时长 (秒 -> 文本)
    function formatDuration(seconds) {
        if (!seconds && seconds !== 0) return '未知';
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        if (mins > 0) {
            return `${mins}分钟 ${secs > 0 ? secs + '秒' : ''}`;
        }
        return `${secs}秒`;
    }

    // 更新统计卡片
    function updateStats() {
        if (statTotalMeetings) statTotalMeetings.textContent = allMeetings.length;
        if (statRecording) statRecording.textContent = '0'; // 目前没有正在录制的后端状态，写死为 0
        if (statCompleted) statCompleted.textContent = allMeetings.length;
    }

    // 渲染最近会议 (首页)
    function renderRecentMeetings() {
        if (!recentMeetingsContainer) return;
        
        recentMeetingsContainer.innerHTML = '';
        const recent = allMeetings.slice(0, 3); // 取前3个
        
        if (recent.length === 0) {
            recentMeetingsContainer.innerHTML = '<div style="padding: 16px; color: #999; text-align: center;">暂无会议记录</div>';
            return;
        }

        recent.forEach(meeting => {
            const item = document.createElement('div');
            item.className = 'recent-meeting-item';
            
            let displayTime = meeting.start_time || meeting.created_at;
            try {
                displayTime = new Date(displayTime).toLocaleString('zh-CN', {
                    year: 'numeric', month: '2-digit', day: '2-digit',
                    hour: '2-digit', minute: '2-digit'
                });
            } catch(e) {}
            
            item.innerHTML = `
                <div class="recent-meeting-title">${meeting.title || '无标题会议'}</div>
                <div class="recent-meeting-info">
                    <span class="status-completed">已完成</span>
                    <span>${displayTime}</span>
                </div>
            `;
            recentMeetingsContainer.appendChild(item);
        });
    }

    // 渲染会议列表页
    function renderMeetingList() {
        if (!meetingListContainer) return;
        
        meetingListContainer.innerHTML = '';
        
        const totalPages = Math.ceil(filteredMeetings.length / itemsPerPage) || 1;
        if (currentPage > totalPages) currentPage = totalPages;
        
        const startIndex = (currentPage - 1) * itemsPerPage;
        const endIndex = startIndex + itemsPerPage;
        const pageData = filteredMeetings.slice(startIndex, endIndex);

        if (pageData.length === 0) {
            meetingListContainer.innerHTML = '<div style="padding: 32px; color: #999; text-align: center;">没有找到匹配的会议记录</div>';
        } else {
            pageData.forEach(meeting => {
                const card = document.createElement('div');
                card.className = 'card meeting-card';
                card.dataset.meetingId = meeting.id;
                
                let displayTime = meeting.start_time || meeting.created_at;
                try {
                    displayTime = new Date(displayTime).toLocaleString('zh-CN', {
                        year: 'numeric', month: '2-digit', day: '2-digit',
                        hour: '2-digit', minute: '2-digit'
                    });
                } catch(e) {}
                
                card.innerHTML = `
                    <div class="meeting-header">
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <div class="meeting-title" contenteditable="false" data-id="${meeting.id}" style="outline: none; cursor: text; min-width: 100px; padding: 2px 4px; border-radius: 4px; transition: background 0.2s;">${meeting.title || '无标题会议'}</div>
                            <button class="btn-save-title" data-id="${meeting.id}" title="保存标题" style="background: none; border: none; cursor: pointer; opacity: 0.3; transition: opacity 0.2s; padding: 4px;">
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                    <path d="M19 21H5C4.46957 21 3.96086 20.7893 3.58579 20.4142C3.21071 20.0391 3 19.5304 3 19V5C3 4.46957 3.21071 3.96086 3.58579 3.58579C3.96086 3.21071 4.46957 3 5 3H16L21 8V19C21 19.5304 20.7893 20.0391 20.4142 20.4142C20.0391 20.7893 19.5304 21 19 21Z" stroke="#52c41a" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                    <path d="M17 21V13H7V21" stroke="#52c41a" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                    <path d="M7 3V8H15" stroke="#52c41a" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                </svg>
                            </button>
                        </div>
                        <div class="meeting-status status-completed">已完成</div>
                    </div>
                    <div class="meeting-info">
                        <div class="info-item">
                            <span>时间: ${displayTime}</span>
                        </div>
                        <div class="info-item">
                            <span>时长: ${formatDuration(meeting.duration)}</span>
                        </div>
                    </div>
                    <div class="meeting-actions">
                        <button class="btn btn-secondary btn-small btn-view-detail" data-id="${meeting.id}">查看详情</button>
                        ${meeting.audio_path ? `<a href="${meeting.audio_path}" download class="btn btn-primary btn-small" style="text-decoration:none; line-height:30px; display:inline-block; text-align:center;">下载音频</a>` : ''}
                        <button class="btn btn-danger btn-small btn-delete-meeting" data-id="${meeting.id}" style="background-color: #ff4d4f; color: white; border: none;">删除</button>
                    </div>
                `;
                meetingListContainer.appendChild(card);
            });
        }

        // 绑定查看详情按钮事件
        const viewDetailBtns = meetingListContainer.querySelectorAll('.btn-view-detail');
        viewDetailBtns.forEach(btn => {
            btn.addEventListener('click', function() {
                const id = this.getAttribute('data-id');
                window.location.href = `meeting_detail.html?id=${id}`;
            });
        });

        // 绑定保存标题按钮事件
        const saveTitleBtns = meetingListContainer.querySelectorAll('.btn-save-title');
        saveTitleBtns.forEach(btn => {
            btn.addEventListener('click', async function() {
                const id = this.getAttribute('data-id');
                const titleEl = meetingListContainer.querySelector(`.meeting-title[data-id="${id}"]`);
                if (!titleEl) return;
                const newTitle = titleEl.textContent.trim();
                if (!newTitle) {
                    alert('标题不能为空');
                    return;
                }
                try {
                    const response = await fetch(`/api/meetings/${id}`, {
                        method: 'PATCH',
                        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                        body: `title=${encodeURIComponent(newTitle)}`
                    });
                    if (response.ok) {
                        // 更新本地数据
                        const meeting = allMeetings.find(m => m.id === id);
                        if (meeting) meeting.title = newTitle;
                        titleEl.setAttribute('contenteditable', 'false');
                        titleEl.style.background = 'transparent';
                        this.style.opacity = '0.3';
                        // 更新最近会议列表
                        renderRecentMeetings();
                        showToast('修改成功！', 'success');
                    } else {
                        showToast('保存失败', 'error');
                    }
                } catch (e) {
                    console.error('Save title error:', e);
                    showToast('保存失败', 'error');
                }
            });
        });

        // 绑定标题点击编辑事件
        const titleEls = meetingListContainer.querySelectorAll('.meeting-title');
        titleEls.forEach(el => {
            el.addEventListener('click', function() {
                if (this.getAttribute('contenteditable') === 'false') {
                    this.setAttribute('contenteditable', 'true');
                    this.style.background = '#e8f4ff';
                    this.focus();
                    // 选中所有文字
                    const range = document.createRange();
                    range.selectNodeContents(this);
                    const sel = window.getSelection();
                    sel.removeAllRanges();
                    sel.addRange(range);
                }
            });
            el.addEventListener('blur', function() {
                this.setAttribute('contenteditable', 'false');
                this.style.background = 'transparent';
                // 触发保存按钮显示
                const saveBtn = meetingListContainer.querySelector(`.btn-save-title[data-id="${this.getAttribute('data-id')}"]`);
                if (saveBtn) saveBtn.style.opacity = '1';
            });
            el.addEventListener('keydown', function(e) {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    this.blur();
                }
            });
        });

        // 绑定删除按钮事件
        const deleteBtns = meetingListContainer.querySelectorAll('.btn-delete-meeting');
        deleteBtns.forEach(btn => {
            btn.addEventListener('click', async function() {
                const id = this.getAttribute('data-id');
                if (confirm('确定要删除这个会议记录吗？')) {
                    try {
                        const response = await fetch(`/api/meetings/${id}`, { method: 'DELETE' });
                        if (response.ok) {
                            fetchMeetings(); // 重新加载
                        } else {
                            alert('删除失败');
                        }
                    } catch (e) {
                        console.error('Delete error:', e);
                        alert('删除失败');
                    }
                }
            });
        });

        // 更新分页信息
        if (pageInfo) {
            pageInfo.textContent = `第 ${currentPage} 页 / 共 ${totalPages} 页`;
        }
        
        if (btnPrevPage) {
            btnPrevPage.disabled = currentPage === 1;
            btnPrevPage.style.opacity = currentPage === 1 ? '0.5' : '1';
        }
        
        if (btnNextPage) {
            btnNextPage.disabled = currentPage === totalPages;
            btnNextPage.style.opacity = currentPage === totalPages ? '0.5' : '1';
        }
    }

    // 搜索逻辑
    function handleSearch() {
        const query = (searchInput.value || '').trim().toLowerCase();
        if (!query) {
            filteredMeetings = [...allMeetings];
        } else {
            filteredMeetings = allMeetings.filter(m => 
                (m.title && m.title.toLowerCase().includes(query)) ||
                (m.start_time && m.start_time.toLowerCase().includes(query))
            );
        }
        currentPage = 1;
        renderMeetingList();
    }

    if (btnSearch) {
        btnSearch.addEventListener('click', handleSearch);
    }

    if (searchInput) {
        searchInput.addEventListener('keyup', function(e) {
            if (e.key === 'Enter') {
                handleSearch();
            }
        });
        // 实时搜索
        searchInput.addEventListener('input', handleSearch);
    }

    // 分页事件
    if (btnPrevPage) {
        btnPrevPage.addEventListener('click', function() {
            if (currentPage > 1) {
                currentPage--;
                renderMeetingList();
            }
        });
    }

    if (btnNextPage) {
        btnNextPage.addEventListener('click', function() {
            const totalPages = Math.ceil(filteredMeetings.length / itemsPerPage) || 1;
            if (currentPage < totalPages) {
                currentPage++;
                renderMeetingList();
            }
        });
    }

    // 导航切换
    function switchPage(pageName) {
        navLinks.forEach(function(link) {
            link.classList.remove('active');
            if (link.getAttribute('data-page') === pageName) {
                link.classList.add('active');
            }
        });

        if (pageName === 'home') {
            homeContent.style.display = 'block';
            meetingsContent.style.display = 'none';
        } else if (pageName === 'meetings') {
            homeContent.style.display = 'none';
            meetingsContent.style.display = 'block';
        }
    }

    navLinks.forEach(function(link) {
        link.addEventListener('click', function(e) {
            const pageName = this.getAttribute('data-page');
            if (pageName) {
                e.preventDefault();
                switchPage(pageName);
            }
        });
    });

    if (viewMeetingsBtn) {
        viewMeetingsBtn.addEventListener('click', function() {
            switchPage('meetings');
        });
    }

    // 新建会议按钮跳转
    function goToTranscription() {
        window.location.href = 'meeting_transcription.html';
    }

    if (btnNewMeeting) btnNewMeeting.addEventListener('click', goToTranscription);
    if (btnNewMeeting2) btnNewMeeting2.addEventListener('click', goToTranscription);
    if (btnLiveView) btnLiveView.addEventListener('click', goToTranscription);

    // 初始化获取数据
    fetchMeetings();
});
