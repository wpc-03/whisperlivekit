// 主控制台页面逻辑

// 加载用户信息
async function loadUserInfo() {
    const user = await getCurrentUser();
    if (user) {
        document.getElementById('username').textContent = user.username;
    }
}

// 加载专业术语数量
async function loadKeywordsCount() {
    try {
        const response = await fetchWithAuth('/api/keywords');
        if (response && response.ok) {
            const keywords = await response.json();
            document.getElementById('keywords-count').textContent = keywords.length;
        } else {
            document.getElementById('keywords-count').textContent = '0';
        }
    } catch (error) {
        console.error('加载专业术语数量错误:', error);
        document.getElementById('keywords-count').textContent = '0';
    }
}

// 加载唤醒词数量
async function loadWakewordsCount() {
    try {
        const response = await fetchWithAuth('/api/wakewords');
        if (response && response.ok) {
            const wakewords = await response.json();
            document.getElementById('wakewords-count').textContent = wakewords.length;
        } else {
            document.getElementById('wakewords-count').textContent = '0';
        }
    } catch (error) {
        console.error('加载唤醒词数量错误:', error);
        document.getElementById('wakewords-count').textContent = '0';
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
    
    // 加载专业术语数量
    await loadKeywordsCount();
    
    // 加载唤醒词数量
    await loadWakewordsCount();
});
