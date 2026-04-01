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
            convertWakewords();
            showToast('删除成功', 'success');
        } else {
            showToast('删除失败', 'error');
        }
    } catch (error) {
        console.error('删除唤醒词错误:', error);
        showToast('删除失败', 'error');
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
                convertWakewords();
                showToast('更新成功', 'success');
            } else {
                showToast('更新失败', 'error');
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
            showToast('转换成功', 'success');
        } else {
            showToast('转换失败', 'error');
        }
    } catch (error) {
        console.error('转换唤醒词错误:', error);
        showToast('转换失败', 'error');
    }
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
        
        if (!word) {
            showToast('唤醒词不能为空', 'warning');
            return;
        }
        
        const success = await addWakeword(word, boost, threshold);
        if (success) {
            showToast('添加成功', 'success');
            document.getElementById('new-word').value = '';
            document.getElementById('new-boost').value = '';
            document.getElementById('new-threshold').value = '';
            loadWakewords();
            convertWakewords();
        } else {
            showToast('添加失败', 'error');
        }
    });


});
