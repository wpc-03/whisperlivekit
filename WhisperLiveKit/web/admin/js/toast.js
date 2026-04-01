function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    
    let iconHtml = '';
    if (type === 'success') {
        iconHtml = `<svg viewBox="0 0 1024 1024" width="16" height="16"><path d="M512 64a448 448 0 1 1 0 896 448 448 0 0 1 0-896zm-55.808 536.384-99.52-99.584a38.4 38.4 0 1 0-54.336 54.336l126.72 126.72a38.272 38.272 0 0 0 54.336 0l262.4-262.464a38.4 38.4 0 1 0-54.272-54.336L456.192 600.384z" fill="#67C23A"></path></svg>`;
    } else if (type === 'error') {
        iconHtml = `<svg viewBox="0 0 1024 1024" width="16" height="16"><path d="M512 64a448 448 0 1 1 0 896 448 448 0 0 1 0-896zm0 393.664L407.36 353.024a38.4 38.4 0 1 0-54.336 54.336L457.664 512 353.024 616.64a38.4 38.4 0 1 0 54.336 54.336L512 566.336 616.64 670.976a38.4 38.4 0 1 0 54.336-54.336L566.336 512 670.976 407.36a38.4 38.4 0 1 0-54.336-54.336L512 457.664z" fill="#F56C6C"></path></svg>`;
    } else if (type === 'warning') {
        iconHtml = `<svg viewBox="0 0 1024 1024" width="16" height="16"><path d="M512 64a448 448 0 1 1 0 896 448 448 0 0 1 0-896zm0 192a58.432 58.432 0 0 0-58.24 63.744l23.36 256.384a35.072 35.072 0 0 0 69.76 0l23.296-256.384A58.432 58.432 0 0 0 512 256zm0 512a51.2 51.2 0 1 0 0-102.4 51.2 51.2 0 0 0 0 102.4z" fill="#E6A23C"></path></svg>`;
    } else {
        iconHtml = `<svg viewBox="0 0 1024 1024" width="16" height="16"><path d="M512 64a448 448 0 1 1 0 896 448 448 0 0 1 0-896zm0 192a58.432 58.432 0 0 0-58.24 63.744l23.36 256.384a35.072 35.072 0 0 0 69.76 0l23.296-256.384A58.432 58.432 0 0 0 512 256zm0 512a51.2 51.2 0 1 0 0-102.4 51.2 51.2 0 0 0 0 102.4z" fill="#909399"></path></svg>`;
    }

    toast.innerHTML = `<div style="display: flex; align-items: center; gap: 8px;">${iconHtml}<span>${message}</span></div>`;
    
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
    } else if (type === 'warning') {
        toast.style.backgroundColor = '#fdf6ec';
        toast.style.borderColor = '#faecd8';
        toast.style.color = '#e6a23c';
    } else {
        toast.style.backgroundColor = '#f4f4f5';
        toast.style.borderColor = '#e9e9eb';
        toast.style.color = '#909399';
    }
    
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.style.opacity = '1';
        toast.style.top = '40px';
    }, 10);

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
