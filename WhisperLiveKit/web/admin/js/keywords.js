// 专业术语管理页面逻辑

// 加载用户信息
async function loadUserInfo() {
    const user = await getCurrentUser();
    if (user) {
        document.getElementById('username').textContent = user.username;
    }
}

// 加载专业术语列表
async function loadKeywords() {
    try {
        const response = await fetchWithAuth('/api/keywords');
        if (response && response.ok) {
            const keywords = await response.json();
            renderKeywordsList(keywords);
        } else {
            document.getElementById('keywords-list').innerHTML = '<tr><td colspan="2">加载失败</td></tr>';
        }
    } catch (error) {
        console.error('加载专业术语错误:', error);
        document.getElementById('keywords-list').innerHTML = '<tr><td colspan="2">加载失败</td></tr>';
    }
}

// 渲染专业术语列表
function renderKeywordsList(keywords) {
    const keywordsList = document.getElementById('keywords-list');
    
    if (keywords.length === 0) {
        keywordsList.innerHTML = '<tr><td colspan="2">暂无专业术语</td></tr>';
        return;
    }
    
    let html = '';
    keywords.forEach((keyword, index) => {
        html += `
            <tr>
                <td>${keyword}</td>
                <td>
                    <button class="btn btn-secondary" onclick="editKeyword('${keyword}')">编辑</button>
                    <button class="btn btn-danger" onclick="deleteKeyword('${keyword}')">删除</button>
                </td>
            </tr>
        `;
    });
    
    keywordsList.innerHTML = html;
}

// 添加专业术语
async function addKeyword(keyword) {
    try {
        const response = await fetchWithAuth('/api/keywords', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded'
            },
            body: new URLSearchParams({
                keyword: keyword
            })
        });

        if (response && response.ok) {
            return true;
        } else {
            return false;
        }
    } catch (error) {
        console.error('添加专业术语错误:', error);
        return false;
    }
}

// 更新专业术语
async function updateKeyword(oldKeyword, newKeyword) {
    try {
        const response = await fetchWithAuth(`/api/keywords/${encodeURIComponent(oldKeyword)}`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded'
            },
            body: new URLSearchParams({
                new_keyword: newKeyword
            })
        });

        if (response && response.ok) {
            return true;
        } else {
            return false;
        }
    } catch (error) {
        console.error('更新专业术语错误:', error);
        return false;
    }
}

// 删除专业术语
async function deleteKeyword(keyword) {
    if (!confirm(`确定要删除术语 "${keyword}" 吗？`)) {
        return;
    }

    try {
        const response = await fetchWithAuth(`/api/keywords/${encodeURIComponent(keyword)}`, {
            method: 'DELETE'
        });

        if (response && response.ok) {
            loadKeywords();
        } else {
            alert('删除失败');
        }
    } catch (error) {
        console.error('删除专业术语错误:', error);
        alert('删除失败');
    }
}

// 编辑专业术语
function editKeyword(keyword) {
    const newKeyword = prompt('请输入新的术语:', keyword);
    if (newKeyword && newKeyword.trim() !== '' && newKeyword !== keyword) {
        updateKeyword(keyword, newKeyword.trim()).then(success => {
            if (success) {
                loadKeywords();
            } else {
                alert('更新失败');
            }
        });
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
    
    // 加载专业术语列表
    await loadKeywords();
    
    // 添加术语表单处理
    document.getElementById('add-keyword-form').addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const keyword = document.getElementById('new-keyword').value.trim();
        const messageElement = document.getElementById('add-message');
        
        if (!keyword) {
            messageElement.textContent = '术语不能为空';
            messageElement.className = 'message error';
            return;
        }
        
        const success = await addKeyword(keyword);
        if (success) {
            messageElement.textContent = '添加成功';
            messageElement.className = 'message success';
            document.getElementById('new-keyword').value = '';
            loadKeywords();
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
