// 唤醒词管理页面逻辑

// 加载用户信息
async function loadUserInfo() {
    const user = await getCurrentUser();
    if (user) {
        document.getElementById('username').textContent = user.username;
    }
}

// 加载唤醒词列表
async function loadWakewords() {
    try {
        const response = await fetchWithAuth('/api/wakewords');
        if (response && response.ok) {
            const wakewords = await response.json();
            renderWakewordsList(wakewords);
        } else {
            document.getElementById('wakewords-list').innerHTML = '<tr><td colspan="4">加载失败</td></tr>';
        }
    } catch (error) {
        console.error('加载唤醒词错误:', error);
        document.getElementById('wakewords-list').innerHTML = '<tr><td colspan="4">加载失败</td></tr>';
    }
}

// 渲染唤醒词列表
function renderWakewordsList(wakewords) {
    const wakewordsList = document.getElementById('wakewords-list');
    
    if (wakewords.length === 0) {
        wakewordsList.innerHTML = '<tr><td colspan="4">暂无唤醒词</td></tr>';
        return;
    }
    
    let html = '';
    wakewords.forEach((wakeword, index) => {
        html += `
            <tr>
                <td>${wakeword.word}</td>
                <td>${wakeword.boost || '-'}</td>
                <td>${wakeword.threshold || '-'}</td>
                <td>
                    <button class="btn btn-secondary" onclick="editWakeword('${wakeword.word}', ${wakeword.boost}, ${wakeword.threshold})">编辑</button>
                    <button class="btn btn-danger" onclick="deleteWakeword('${wakeword.word}')">删除</button>
                </td>
            </tr>
        `;
    });
    
    wakewordsList.innerHTML = html;
}

// 添加唤醒词
async function addWakeword(word, boost, threshold) {
    try {
        const response = await fetchWithAuth('/api/wakewords', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded'
            },
            body: new URLSearchParams({
                word: word,
                boost: boost || '',
                threshold: threshold || ''
            })
        });

        if (response && response.ok) {
            return true;
        } else {
            return false;
        }
    } catch (error) {
        console.error('添加唤醒词错误:', error);
        return false;
    }
}

// 更新唤醒词
async function updateWakeword(oldWord, newWord, boost, threshold) {
    try {
        const response = await fetchWithAuth(`/api/wakewords/${encodeURIComponent(oldWord)}`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded'
            },
            body: new URLSearchParams({
                new_word: newWord,
                boost: boost || '',
                threshold: threshold || ''
            })
        });

        if (response && response.ok) {
            return true;
        } else {
            return false;
        }
    } catch (error) {
        console.error('更新唤醒词错误:', error);
        return false;
    }
}

// 删除唤醒词
async function deleteWakeword(word) {
    if (!confirm(`确定要删除唤醒词 "${word}" 吗？`)) {
        return;
    }

    try {
        const response = await fetchWithAuth(`/api/wakewords/${encodeURIComponent(word)}`, {
            method: 'DELETE'
        });

        if (response && response.ok) {
            loadWakewords();
            // 自动执行转换功能
            convertWakewords();
        } else {
            alert('删除失败');
        }
    } catch (error) {
        console.error('删除唤醒词错误:', error);
        alert('删除失败');
    }
}

// 编辑唤醒词
function editWakeword(oldWord, boost, threshold) {
    const newWord = prompt('请输入新的唤醒词:', oldWord);
    if (newWord && newWord.trim() !== '') {
        const newBoost = parseFloat(prompt('请输入新的Boost值:', boost || '')) || null;
        const newThreshold = parseFloat(prompt('请输入新的Threshold值:', threshold || '')) || null;
        
        updateWakeword(oldWord, newWord.trim(), newBoost, newThreshold).then(success => {
            if (success) {
                loadWakewords();
                // 自动执行转换功能
                convertWakewords();
            } else {
                alert('更新失败');
            }
        });
    }
}

// 转换唤醒词格式
async function convertWakewords() {
    try {
        const response = await fetchWithAuth('/api/wakewords/convert', {
            method: 'POST'
        });

        if (response && response.ok) {
            document.getElementById('convert-message').textContent = '转换成功';
            document.getElementById('convert-message').className = 'message success';
        } else {
            document.getElementById('convert-message').textContent = '转换失败';
            document.getElementById('convert-message').className = 'message error';
        }
    } catch (error) {
        console.error('转换唤醒词错误:', error);
        document.getElementById('convert-message').textContent = '转换失败';
        document.getElementById('convert-message').className = 'message error';
    }
    
    // 3秒后清除消息
    setTimeout(() => {
        document.getElementById('convert-message').textContent = '';
        document.getElementById('convert-message').className = 'message';
    }, 3000);
}

// 页面加载时执行
window.addEventListener('load', async function() {
    // 检查登录状态
    if (!isLoggedIn()) {
        redirectToLogin();
        return;
    }
    
    // 加载用户信息
    await loadUserInfo();
    
    // 加载唤醒词列表
    await loadWakewords();
    
    // 添加唤醒词表单处理
    document.getElementById('add-wakeword-form').addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const word = document.getElementById('new-word').value.trim();
        const boost = document.getElementById('new-boost').value ? parseFloat(document.getElementById('new-boost').value) : null;
        const threshold = document.getElementById('new-threshold').value ? parseFloat(document.getElementById('new-threshold').value) : null;
        const messageElement = document.getElementById('add-message');
        
        if (!word) {
            messageElement.textContent = '唤醒词不能为空';
            messageElement.className = 'message error';
            return;
        }
        
        const success = await addWakeword(word, boost, threshold);
        if (success) {
            messageElement.textContent = '添加成功';
            messageElement.className = 'message success';
            document.getElementById('new-word').value = '';
            document.getElementById('new-boost').value = '';
            document.getElementById('new-threshold').value = '';
            loadWakewords();
            // 自动执行转换功能
            convertWakewords();
            setTimeout(() => {
                messageElement.textContent = '';
                messageElement.className = 'message';
            }, 2000);
        } else {
            messageElement.textContent = '添加失败';
            messageElement.className = 'message error';
        }
    });
    

});
