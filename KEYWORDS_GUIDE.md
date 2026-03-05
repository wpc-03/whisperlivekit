# 关键字管理机制使用指南

## 概述

WhisperLiveKit 现在支持通过文本文件管理关键字，这些关键字将被自动用作语音识别的初始提示（init_prompt），提高特定词汇的识别准确率。

## 快速开始

### 1. 创建关键字文件

创建一个文本文件（例如 `keywords.txt`），每行一个关键字：

```text
# 这是注释行，会被忽略
Whisper
ASR
语音识别
百度
阿里巴巴
腾讯
```

### 2. 启动时使用关键字文件

使用 `--keywords-file` 参数指定关键字文件路径：

```bash
wlk --model tiny --language zh --backend-policy localagreement --keywords-file keywords.txt
```

## 关键字文件格式

### 基本规则

- **每行一个关键字**：每个关键字独占一行
- **注释行**：以 `#` 开头的行会被忽略
- **空行**：空行会被自动跳过
- **编码格式**：文件必须使用 UTF-8 编码

### 示例文件

```text
# 关键字配置文件示例
# =======================

# 公司名称
百度
阿里巴巴
腾讯
字节跳动
华为

# 产品名称
WhisperLiveKit
ChatGPT
GPT-4
Claude

# 技术术语
API
WebSocket
HTTP
JSON
REST

# 专业领域词汇
人工智能
机器学习
深度学习
自然语言处理
```

## 参数优先级

当多个提示参数同时使用时，优先级如下（从高到低）：

1. **`--keywords-file`**：关键字文件中的内容
2. **`--init-prompt`**：命令行指定的初始提示
3. **动态上下文**：已识别的文本内容
4. **`--static-init-prompt`**：静态初始提示（始终在最前面）

### 组合使用示例

```bash
# 同时使用关键字文件和静态提示
wlk --model tiny --language zh \
    --backend-policy localagreement \
    --keywords-file keywords.txt \
    --static-init-prompt "这是一个重要的会议记录"
```

最终的提示格式为：
```
[static-init-prompt] [keywords] [动态上下文]
```

## API 说明

### KeywordsManager 类

位于 `whisperlivekit.keywords_manager` 模块。

#### 初始化

```python
from whisperlivekit.keywords_manager import KeywordsManager

# 创建管理器
manager = KeywordsManager("keywords.txt")

# 获取关键字列表
keywords = manager.get_keywords()

# 获取拼接后的字符串
keywords_str = manager.get_keywords_as_string(separator=" ")

# 检查是否有关键字
if manager.has_keywords():
    print("有关键字")

# 重新加载文件
manager.reload_keywords()
```

#### 创建示例文件

```python
KeywordsManager.create_example_file(
    "keywords_example.txt",
    keywords=["Whisper", "ASR", "语音识别"]
)
```

### 便捷函数

```python
from whisperlivekit.keywords_manager import load_keywords_from_file

keywords_list, keywords_string = load_keywords_from_file("keywords.txt")
```

## 错误处理

系统包含完善的错误处理机制：

| 错误情况 | 处理方式 | 日志级别 |
|---------|---------|---------|
| 文件不存在 | 继续运行，使用空关键字列表 | WARNING |
| 路径不是文件 | 继续运行，使用空关键字列表 | WARNING |
| UTF-8 解码失败 | 继续运行，使用空关键字列表 | ERROR |
| 读取IO错误 | 继续运行，使用空关键字列表 | ERROR |
| 其他异常 | 继续运行，使用空关键字列表 | ERROR |

## 最佳实践

### 1. 关键字选择

- ✅ 选择**专有名词**：公司名、产品名、人名、地名
- ✅ 选择**专业术语**：行业特定词汇
- ✅ 选择**易混淆词汇**：容易被误识别的词
- ❌ 避免添加**常用词汇**：系统已经能很好识别
- ❌ 避免添加**太长的句子**：保持单个关键字

### 2. 文件管理

- 将关键字文件放在项目根目录或配置目录
- 使用版本控制系统管理关键字文件
- 定期更新关键字列表
- 为不同场景创建不同的关键字文件

### 3. 性能考虑

- 关键字数量建议控制在 50-100 个以内
- 避免添加过长的关键字
- 关键字总长度建议不超过 500 字符

## 完整示例

### 场景：会议记录

创建 `meeting_keywords.txt`：

```text
# 会议相关关键字
产品经理
技术总监
UI设计师
前端开发
后端开发
数据库
API接口
用户体验
迭代
冲刺
里程碑
```

启动命令：

```bash
wlk --model base --language zh \
    --backend-policy localagreement \
    --keywords-file meeting_keywords.txt \
    --static-init-prompt "这是产品会议记录"
```

### 场景：医疗转录

创建 `medical_keywords.txt`：

```text
# 医疗相关关键字
高血压
糖尿病
冠心病
心电图
CT扫描
核磁共振
处方
诊断
治疗
随访
```

启动命令：

```bash
wlk --model medium --language zh \
    --backend-policy localagreement \
    --keywords-file medical_keywords.txt
```

## 常见问题

### Q: 关键字文件需要放在哪里？

A: 可以放在任何位置，只要通过 `--keywords-file` 参数指定正确的路径即可。

### Q: 可以同时使用关键字文件和 --init-prompt 吗？

A: 可以。关键字文件的内容会优先于 --init-prompt。

### Q: 关键字文件支持中文吗？

A: 完全支持！确保文件使用 UTF-8 编码即可。

### Q: 如何验证关键字是否被正确加载？

A: 查看启动日志，会显示加载的关键字数量。

### Q: 可以在运行时重新加载关键字文件吗？

A: 当前版本需要重启服务。未来版本可能支持热重载。

## 更新日志

### v1.0 (当前版本)
- 支持关键字文件管理
- 支持注释和空行
- 完善的错误处理
- 优先级：关键字文件 > init_prompt > 动态上下文
