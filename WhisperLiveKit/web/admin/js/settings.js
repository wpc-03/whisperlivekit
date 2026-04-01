// 参数设置页面逻辑

// 缓存唤醒词模型目录，用于禁用后再启用时恢复
let cachedHotwordModelDir = '';

// 加载用户信息
async function loadUserInfo() {
    const user = await getCurrentUser();
    if (user) {
        document.getElementById('username').textContent = user.username;
    }
}

// 更新唤醒词设置项的显示状态
function updateHotwordSettingsVisibility() {
    const hotwordEnabled = document.getElementById('hotword_enabled');
    const hotwordSettingsGroup = document.getElementById('hotword-settings-group');
    const hotwordModelDirInput = document.getElementById('hotword_model_dir');
    
    if (hotwordEnabled && hotwordSettingsGroup) {
        const isEnabled = hotwordEnabled.value === 'true';
        hotwordSettingsGroup.style.display = isEnabled ? 'block' : 'none';
        
        // 禁用时保存当前值到缓存，启用时从缓存恢复
        if (!isEnabled && hotwordModelDirInput) {
            cachedHotwordModelDir = hotwordModelDirInput.value;
        } else if (isEnabled && hotwordModelDirInput && cachedHotwordModelDir) {
            hotwordModelDirInput.value = cachedHotwordModelDir;
        }
    }
}

// 更新说话人识别设置项的显示状态
function updateDiarizationSettingsVisibility() {
    const diarizationSelect = document.getElementById('diarization');
    const diarizationSettingsGroup = document.getElementById('diarization-settings-group');
    
    if (diarizationSelect && diarizationSettingsGroup) {
        diarizationSettingsGroup.style.display = diarizationSelect.value === 'true' ? 'block' : 'none';
    }
}

// 加载当前设置
async function loadSettings() {
    try {
        const response = await fetchWithAuth('/api/config');
        if (response && response.ok) {
            const settings = await response.json();
            populateSettingsForm(settings);
        } else {
            showToast('加载设置失败', 'error');
        }
    } catch (error) {
        console.error('加载设置错误:', error);
        showToast('加载设置失败', 'error');
    }
}

// 根据选择的模型类型自动更新模型路径
function updateModelPathBasedOnSelection() {
    const modelSelect = document.getElementById('model');
    const modelPathInput = document.getElementById('model_path');
    
    if (!modelSelect || !modelPathInput) return;
    
    const selectedModel = modelSelect.value;
    let modelPath = modelPathInput.value;
    
    // 如果模型路径为空，使用默认路径
    if (!modelPath || modelPath.trim() === '') {
        modelPath = '/app/models/models--Systran--faster-whisper-tiny';
        modelPathInput.value = modelPath;
    }
    
    // 替换路径中的模型名称部分
    // 支持多种路径格式：
    // 1. /app/models/models--Systran--faster-whisper-tiny
    // 2. D:/python/FastWhisperTranscriber/model/models--Systran--faster-whisper-tiny
    // 3. models--Systran--faster-whisper-tiny
    
    // 查找 "faster-whisper-" 在路径中的位置（这是模型路径的固定部分）
    const searchPattern = 'faster-whisper-';
    const searchIndex = modelPath.lastIndexOf(searchPattern);
    
    if (searchIndex !== -1) {
        // 找到了固定模式，替换后面的模型类型
        const basePath = modelPath.substring(0, searchIndex + searchPattern.length);
        modelPathInput.value = basePath + selectedModel;
    } else {
        // 如果没有找到固定模式，尝试查找最后一个 "-" 的位置
        const lastHyphenIndex = modelPath.lastIndexOf('-');
        if (lastHyphenIndex !== -1) {
            // 获取最后一个 "-" 之前的部分
            const basePath = modelPath.substring(0, lastHyphenIndex + 1); // 包含 "-"
            modelPathInput.value = basePath + selectedModel;
        } else {
            // 如果没有找到 "-"，直接附加模型名称
            modelPathInput.value = modelPath + '-' + selectedModel;
        }
    }
}

// 填充设置表单
function populateSettingsForm(settings) {
    // 服务设置
    if (settings.host) {
        document.getElementById('host').value = settings.host;
    }
    if (settings.port !== undefined) {
        document.getElementById('port').value = settings.port;
    }
    if (settings.log_level) {
        document.getElementById('log_level').value = settings.log_level;
    }
    if (settings.backend_policy) {
        document.getElementById('backend_policy').value = settings.backend_policy;
    }
    
    // 模型设置
    if (settings.model) {
        document.getElementById('model').value = settings.model;
    }
    if (settings.model_path) {
        document.getElementById('model_path').value = settings.model_path;
    }
    // 填充后，确保模型路径与选择的模型类型一致
    updateModelPathBasedOnSelection();
    
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
        updateDiarizationSettingsVisibility();
    }
    if (settings.diarization_model) {
        document.getElementById('diarization_model').value = settings.diarization_model;
    }
    if (settings.punctuation_split !== undefined) {
        document.getElementById('punctuation_split').value = settings.punctuation_split.toString();
    }
    
    // 唤醒词设置
    const hotwordEnabled = document.getElementById('hotword_enabled');
    const hasHotwordModelDir = settings.hotword_model_dir && settings.hotword_model_dir.trim() !== '';
    
    // 缓存 hotword_model_dir 的值，用于禁用后再启用时恢复
    if (settings.hotword_model_dir) {
        cachedHotwordModelDir = settings.hotword_model_dir;
        document.getElementById('hotword_model_dir').value = settings.hotword_model_dir;
    }
    
    if (hotwordEnabled) {
        hotwordEnabled.value = hasHotwordModelDir ? 'true' : 'false';
        updateHotwordSettingsVisibility();
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
    
    // SSL设置
    if (settings.ssl_certfile) {
        document.getElementById('ssl_certfile').value = settings.ssl_certfile;
    }
    if (settings.ssl_keyfile) {
        document.getElementById('ssl_keyfile').value = settings.ssl_keyfile;
    }
    if (settings.forwarded_allow_ips) {
        document.getElementById('forwarded_allow_ips').value = settings.forwarded_allow_ips;
    }
}

// 保存设置
async function saveSettings(settings) {
    try {
        const response = await fetchWithAuth('/api/config', {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(settings)
        });

        if (response && response.ok) {
            const result = await response.json();
            return result;
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
        const response = await fetchWithAuth('/api/config/reset', {
            method: 'POST'
        });

        if (response && response.ok) {
            const result = await response.json();
            loadSettings();
            showToast('重置成功' + (result.restart_required ? '，需要重启服务使配置生效' : ''), 'success');
        } else {
            showToast('重置失败', 'error');
        }
    } catch (error) {
        console.error('重置设置错误:', error);
        showToast('重置失败', 'error');
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
    
    // 添加模型选择change事件监听器，自动更新模型路径
    const modelSelect = document.getElementById('model');
    if (modelSelect) {
        modelSelect.addEventListener('change', updateModelPathBasedOnSelection);
    }
    
    // 添加唤醒词启用开关change事件监听器
    const hotwordEnabledSelect = document.getElementById('hotword_enabled');
    if (hotwordEnabledSelect) {
        hotwordEnabledSelect.addEventListener('change', updateHotwordSettingsVisibility);
    }
    
    // 添加说话人识别开关change事件监听器
    const diarizationSelect = document.getElementById('diarization');
    if (diarizationSelect) {
        diarizationSelect.addEventListener('change', updateDiarizationSettingsVisibility);
    }
    
    // 收集表单数据
    function collectFormData() {
        return {
            host: document.getElementById('host').value,
            port: parseInt(document.getElementById('port').value),
            log_level: document.getElementById('log_level').value,
            backend_policy: document.getElementById('backend_policy').value,
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
            hotword_model_dir: document.getElementById('hotword_enabled').value === 'true' 
                ? document.getElementById('hotword_model_dir').value 
                : '',
            hotword_threshold: parseFloat(document.getElementById('hotword_threshold').value),
            hotword_sample_rate: parseInt(document.getElementById('hotword_sample_rate').value),
            hotword_threads: parseInt(document.getElementById('hotword_threads').value),
            ssl_certfile: document.getElementById('ssl_certfile').value,
            ssl_keyfile: document.getElementById('ssl_keyfile').value,
            forwarded_allow_ips: document.getElementById('forwarded_allow_ips').value
        };
    }
    
    // 重置按钮处理
    document.getElementById('btn-reset').addEventListener('click', resetSettings);
    
    // 应用配置 (热重载) 按钮处理 - 先保存再热重载
    document.getElementById('btn-reload').addEventListener('click', async () => {
        if (!confirm('确定要应用配置吗？将先保存设置再热重载服务。')) {
            return;
        }
        
        const btn = document.getElementById('btn-reload');
        
        // 禁用按钮，防止重复点击
        btn.disabled = true;
        btn.style.opacity = '0.5';
        showToast('正在保存设置...', 'info');
        
        try {
            // 1. 先保存设置
            const settings = collectFormData();
            const saveResult = await saveSettings(settings);
            
            if (!saveResult) {
                showToast('保存设置失败', 'error');
                btn.disabled = false;
                btn.style.opacity = '1';
                return;
            }
            
            showToast('设置已保存，正在热重载...', 'info');
            
            // 2. 再热重载
            const token = localStorage.getItem('token');
            const response = await fetch('/api/config/restart', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            });
            
            const data = await response.json();
            
            if (response.ok) {
                showToast(data.message || '配置已保存并热重载生效', 'success');
            } else {
                showToast('热重载失败: ' + (data.detail || '未知错误'), 'error');
            }
        } catch (error) {
            console.error('操作失败:', error);
            showToast('操作失败，请检查网络连接', 'error');
        } finally {
            setTimeout(() => {
                btn.disabled = false;
                btn.style.opacity = '1';
            }, 3000);
        }
    });
});



