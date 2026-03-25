// 参数设置页面逻辑

// 加载用户信息
async function loadUserInfo() {
    const user = await getCurrentUser();
    if (user) {
        document.getElementById('username').textContent = user.username;
    }
}

// 加载当前设置
async function loadSettings() {
    try {
        const response = await fetchWithAuth('/api/settings');
        if (response && response.ok) {
            const settings = await response.json();
            populateSettingsForm(settings);
        } else {
            document.getElementById('settings-message').textContent = '加载设置失败';
            document.getElementById('settings-message').className = 'message error';
        }
    } catch (error) {
        console.error('加载设置错误:', error);
        document.getElementById('settings-message').textContent = '加载设置失败';
        document.getElementById('settings-message').className = 'message error';
    }
}

// 填充设置表单
function populateSettingsForm(settings) {
    // 模型设置
    if (settings.model) {
        document.getElementById('model').value = settings.model;
    }
    if (settings.model_path) {
        document.getElementById('model_path').value = settings.model_path;
    }
    if (settings.language) {
        document.getElementById('language').value = settings.language;
    }
    if (settings.backend) {
        document.getElementById('backend').value = settings.backend;
    }
    
    // 音频设置
    if (settings.min_chunk_size !== undefined) {
        document.getElementById('min_chunk_size').value = settings.min_chunk_size;
    }
    if (settings.buffer_trimming_sec !== undefined) {
        document.getElementById('buffer_trimming_sec').value = settings.buffer_trimming_sec;
    }
    if (settings.pcm_input !== undefined) {
        document.getElementById('pcm_input').value = settings.pcm_input.toString();
    }
    
    // 识别设置
    if (settings.confidence_validation !== undefined) {
        document.getElementById('confidence_validation').value = settings.confidence_validation.toString();
    }
    if (settings.beam_size !== undefined) {
        document.getElementById('beam_size').value = settings.beam_size;
    }
    if (settings.keywords_file) {
        document.getElementById('keywords_file').value = settings.keywords_file;
    }
    if (settings.warmup_file) {
        document.getElementById('warmup_file').value = settings.warmup_file;
    }
    
    // 说话人识别
    if (settings.diarization !== undefined) {
        document.getElementById('diarization').value = settings.diarization.toString();
    }
    if (settings.diarization_model) {
        document.getElementById('diarization_model').value = settings.diarization_model;
    }
    if (settings.punctuation_split !== undefined) {
        document.getElementById('punctuation_split').value = settings.punctuation_split.toString();
    }
    
    // 唤醒词设置
    if (settings.hotword_model_dir) {
        document.getElementById('hotword_model_dir').value = settings.hotword_model_dir;
    }
    if (settings.hotword_threshold !== undefined) {
        document.getElementById('hotword_threshold').value = settings.hotword_threshold;
    }
    if (settings.hotword_sample_rate !== undefined) {
        document.getElementById('hotword_sample_rate').value = settings.hotword_sample_rate;
    }
    if (settings.hotword_threads !== undefined) {
        document.getElementById('hotword_threads').value = settings.hotword_threads;
    }
}

// 保存设置
async function saveSettings(settings) {
    try {
        const response = await fetchWithAuth('/api/settings', {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(settings)
        });

        if (response && response.ok) {
            return true;
        } else {
            return false;
        }
    } catch (error) {
        console.error('保存设置错误:', error);
        return false;
    }
}

// 重置设置为默认值
async function resetSettings() {
    if (!confirm('确定要重置为默认值吗？')) {
        return;
    }

    try {
        const response = await fetchWithAuth('/api/settings/reset', {
            method: 'POST'
        });

        if (response && response.ok) {
            loadSettings();
            document.getElementById('settings-message').textContent = '重置成功';
            document.getElementById('settings-message').className = 'message success';
            setTimeout(() => {
                document.getElementById('settings-message').textContent = '';
                document.getElementById('settings-message').className = 'message';
            }, 2000);
        } else {
            document.getElementById('settings-message').textContent = '重置失败';
            document.getElementById('settings-message').className = 'message error';
        }
    } catch (error) {
        console.error('重置设置错误:', error);
        document.getElementById('settings-message').textContent = '重置失败';
        document.getElementById('settings-message').className = 'message error';
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
    
    // 加载设置
    await loadSettings();
    
    // 保存设置表单处理
    document.getElementById('settings-form').addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const settings = {
            model: document.getElementById('model').value,
            model_path: document.getElementById('model_path').value,
            language: document.getElementById('language').value,
            backend: document.getElementById('backend').value,
            min_chunk_size: parseFloat(document.getElementById('min_chunk_size').value),
            buffer_trimming_sec: parseFloat(document.getElementById('buffer_trimming_sec').value),
            pcm_input: document.getElementById('pcm_input').value === 'true',
            confidence_validation: document.getElementById('confidence_validation').value === 'true',
            beam_size: parseInt(document.getElementById('beam_size').value),
            keywords_file: document.getElementById('keywords_file').value,
            warmup_file: document.getElementById('warmup_file').value,
            diarization: document.getElementById('diarization').value === 'true',
            diarization_model: document.getElementById('diarization_model').value,
            punctuation_split: document.getElementById('punctuation_split').value === 'true',
            hotword_model_dir: document.getElementById('hotword_model_dir').value,
            hotword_threshold: parseFloat(document.getElementById('hotword_threshold').value),
            hotword_sample_rate: parseInt(document.getElementById('hotword_sample_rate').value),
            hotword_threads: parseInt(document.getElementById('hotword_threads').value)
        };
        
        const success = await saveSettings(settings);
        if (success) {
            document.getElementById('settings-message').textContent = '保存成功';
            document.getElementById('settings-message').className = 'message success';
            
            setTimeout(() => {
                document.getElementById('settings-message').textContent = '';
                document.getElementById('settings-message').className = 'message';
            }, 3000);
        } else {
            document.getElementById('settings-message').textContent = '保存失败';
            document.getElementById('settings-message').className = 'message error';
        }
    });
    
    // 重置按钮处理
    document.getElementById('btn-reset').addEventListener('click', resetSettings);
});



