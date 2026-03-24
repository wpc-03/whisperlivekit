// 认证相关功能

// 获取token
function getToken() {
    return localStorage.getItem('token');
}

// 设置token
function setToken(token) {
    localStorage.setItem('token', token);
}

// 清除token
function clearToken() {
    localStorage.removeItem('token');
}

// 检查是否已登录
function isLoggedIn() {
    return getToken() !== null;
}

// 重定向到登录页面
function redirectToLogin() {
    window.location.href = 'index.html';
}

// 重定向到主控制台
function redirectToDashboard() {
    window.location.href = 'dashboard.html';
}

// 登录
async function login(username, password) {
    try {
        const response = await fetch('/api/auth/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded'
            },
            body: new URLSearchParams({
                username: username,
                password: password
            })
        });

        if (!response.ok) {
            throw new Error('登录失败');
        }

        const data = await response.json();
        setToken(data.access_token);
        return true;
    } catch (error) {
        console.error('登录错误:', error);
        return false;
    }
}

// 登出
async function logout() {
    try {
        const response = await fetch('/api/auth/logout', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${getToken()}`
            }
        });

        clearToken();
        redirectToLogin();
    } catch (error) {
        console.error('登出错误:', error);
        clearToken();
        redirectToLogin();
    }
}

// 获取当前用户信息
async function getCurrentUser() {
    try {
        const response = await fetch('/api/auth/me', {
            headers: {
                'Authorization': `Bearer ${getToken()}`
            }
        });

        if (!response.ok) {
            throw new Error('获取用户信息失败');
        }

        return await response.json();
    } catch (error) {
        console.error('获取用户信息错误:', error);
        return null;
    }
}

// 发送带认证的请求
async function fetchWithAuth(url, options = {}) {
    const token = getToken();
    if (!token) {
        redirectToLogin();
        return null;
    }

    const headers = {
        'Authorization': `Bearer ${token}`,
        ...options.headers
    };

    try {
        const response = await fetch(url, {
            ...options,
            headers
        });

        if (response.status === 401) {
            // Token 过期或无效
            clearToken();
            redirectToLogin();
            return null;
        }

        return response;
    } catch (error) {
        console.error('请求错误:', error);
        return null;
    }
}

// 页面加载时检查登录状态
function checkLoginStatus() {
    if (window.location.pathname.includes('index.html')) {
        // 在登录页面，如果已登录则重定向到控制台
        if (isLoggedIn()) {
            redirectToDashboard();
        }
    } else {
        // 在其他页面，如果未登录则重定向到登录页面
        if (!isLoggedIn()) {
            redirectToLogin();
        }
    }
}

// 登录表单处理
if (document.getElementById('login-form')) {
    document.getElementById('login-form').addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const username = document.getElementById('username').value;
        const password = document.getElementById('password').value;
        const messageElement = document.getElementById('login-message');
        
        messageElement.textContent = '';
        messageElement.className = 'message';
        
        const success = await login(username, password);
        if (success) {
            messageElement.textContent = '登录成功，正在跳转...';
            messageElement.className = 'message success';
            setTimeout(redirectToDashboard, 1000);
        } else {
            messageElement.textContent = '登录失败，请检查用户名和密码';
            messageElement.className = 'message error';
        }
    });
}

// 登出按钮处理
if (document.getElementById('btn-logout')) {
    document.getElementById('btn-logout').addEventListener('click', logout);
}

// 页面加载时检查登录状态
window.addEventListener('load', checkLoginStatus);
