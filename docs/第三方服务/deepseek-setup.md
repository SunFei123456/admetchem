# DeepSeek AI 配置指南（服务器端配置版）

> **重要更新**：API Key 现在在服务器端统一配置，用户无需手动输入。

---

## 📋 目录

- [概述](#概述)
- [管理员配置（后端）](#管理员配置后端)
- [用户使用（前端）](#用户使用前端)
- [API 接口说明](#api-接口说明)
- [常见问题](#常见问题)

---

## 🎯 概述

### 架构设计

```
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│   前端用户   │ ------> │  后端服务器  │ ------> │  DeepSeek   │
│  (无需配置)  │         │ (配置 API Key)│         │   API       │
└─────────────┘         └─────────────┘         └─────────────┘
```

### 优势

1. **🔒 安全性高** - API Key 不暴露给前端
2. **✨ 用户友好** - 用户无需配置，开箱即用
3. **🎯 统一管理** - 管理员统一管理 API Key 和额度
4. **💰 成本可控** - 可以限制使用量，避免滥用

---

## 👨‍💼 管理员配置（后端）

### 步骤 1：获取 DeepSeek API Key

1. 访问 [DeepSeek 官网](https://platform.deepseek.com/)
2. 注册并登录账号
3. 进入控制台 → API Keys → Create API Key
4. 复制生成的 API Key（格式：`sk-xxxxxx`）

### 步骤 2：配置后端

#### 方法 1：环境变量配置（推荐）

在 `backend/` 目录下创建或编辑 `.env` 文件：

```env
# DeepSeek AI 配置
DEEPSEEK_API_KEY=sk-your-api-key-here
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_TEMPERATURE=0.7
```

#### 方法 2：直接修改配置文件

编辑 `backend/app/core/config.py`：

```python
# DeepSeek AI 配置
DEEPSEEK_API_KEY: str = "sk-your-api-key-here"
DEEPSEEK_MODEL: str = "deepseek-chat"
DEEPSEEK_TEMPERATURE: float = 0.7
```

### 步骤 3：启动服务

```bash
cd backend
python run.py
```

### 步骤 4：验证配置

访问 http://localhost:8000/api/v1/ai/providers

返回结果：

```json
{
  "code": 0,
  "data": {
    "deepseek": {
      "name": "DeepSeek",
      "configured": true,  // ✅ 表示已配置
      "current_model": "deepseek-chat",
      "current_temperature": 0.7
    }
  }
}
```

---

## 👤 用户使用（前端）

### 访问 AI Chat

1. 启动前端：`npm run dev`
2. 访问 http://localhost:3000
3. 点击导航栏 **"AI Chat"**
4. 直接开始对话，无需配置！✨

### 调整参数（可选）

用户可以在前端设置中调整：

1. **模型选择**
   - `deepseek-chat`：通用对话
   - `deepseek-coder`：代码专用

2. **温度参数**
   - `0.0`：最精确
   - `0.7`：默认（平衡）
   - `1.0`：最有创造力

> ⚠️ **注意**：前端设置只影响当前会话，不会修改服务器端的默认配置。

---

## 🔌 API 接口说明

### 1. 获取配置信息

**请求**
```http
GET /api/v1/ai/providers
Authorization: Bearer <token>
```

**响应**
```json
{
  "code": 0,
  "data": {
    "deepseek": {
      "name": "DeepSeek",
      "configured": true,
      "models": ["deepseek-chat", "deepseek-coder"],
      "current_model": "deepseek-chat",
      "current_temperature": 0.7
    }
  }
}
```

### 2. 发送聊天消息（流式响应）

**请求**
```http
POST /api/v1/ai/chat/stream
Authorization: Bearer <token>
Content-Type: application/json

{
  "model": "deepseek-chat",  // 可选
  "temperature": 0.7,          // 可选
  "messages": [
    {"role": "user", "content": "What is ADMET?"}
  ]
}
```

**响应（SSE 流式）**
```
data: {"content": "ADMET"}
data: {"content": " stands"}
data: {"content": " for..."}
data: {"done": true}
```

### 3. 获取可用模型

**请求**
```http
GET /api/v1/ai/models
Authorization: Bearer <token>
```

**响应**
```json
{
  "code": 0,
  "data": ["deepseek-chat", "deepseek-coder"]
}
```

---

## ❓ 常见问题

### 管理员问题

#### Q1: 如何检查 API Key 是否有效？

访问 DeepSeek 控制台查看余额和使用情况：
https://platform.deepseek.com/usage

#### Q2: 如何更换 API Key？

1. 在 DeepSeek 控制台生成新的 API Key
2. 更新 `.env` 文件或 `config.py`
3. 重启后端服务

#### Q3: 如何限制用户使用量？

可以在后端添加中间件，限制：
- 每用户每天请求次数
- 每次请求的最大 Token 数
- 特定时间段的并发请求数

#### Q4: API 额度用尽怎么办？

1. **续费**：访问 DeepSeek 控制台充值
2. **申请免费额度**：新用户有 500 万 tokens 免费
3. **优化使用**：
   - 限制对话历史长度
   - 使用更低成本的模型
   - 实施用户配额制度

### 用户问题

#### Q1: 为什么不需要配置 API Key？

API Key 在服务器端统一配置，用户无需关心，直接使用即可。

#### Q2: 可以使用其他 AI 服务吗？

当前版本仅支持 DeepSeek，如需其他服务，请联系管理员。

#### Q3: 对话历史会保存吗？

当前版本不保存对话历史，刷新页面后会清空。
如需保存，可以使用浏览器的本地存储或联系管理员添加后端存储。

#### Q4: 消息发送失败怎么办？

1. 检查网络连接
2. 刷新页面重试
3. 联系管理员检查服务器状态

---

## 🔒 安全建议

### 管理员

1. **保护 API Key**
   - 不要提交到 Git 仓库
   - 使用环境变量
   - 定期轮换

2. **监控使用情况**
   - 定期检查 API 调用量
   - 设置余额预警
   - 记录异常使用

3. **访问控制**
   - 确保 AI 接口需要登录
   - 实施请求频率限制
   - 记录所有 API 调用日志

### 用户

1. **不要发送敏感信息**
   - 密码、密钥等
   - 个人隐私信息
   - 商业机密

2. **合理使用**
   - 避免发送过长的消息
   - 不要频繁发送重复请求
   - 尊重服务器资源

---

## 📊 技术架构

### 后端架构

```python
# 配置层 (config.py)
DEEPSEEK_API_KEY = "sk-xxx"

# 服务层 (ai_factory.py)
def create_deepseek_service():
    return DeepSeekService(api_key=settings.DEEPSEEK_API_KEY)

# API 层 (ai.py)
@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    service = AIServiceFactory.create_deepseek_service()
    return StreamingResponse(service.chat_stream(...))
```

### 前端架构

```javascript
// API 调用 (ai.js)
export async function sendChatMessageStream(data, onChunk, onError, onComplete) {
  const response = await fetch('/api/v1/ai/chat/stream', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${token}`  // 只需登录 Token
    },
    body: JSON.stringify({
      model: data.model,        // 可选
      messages: data.messages,
      temperature: data.temperature  // 可选
    })
  })
  // 处理流式响应...
}
```

---

## 📞 获取帮助

- **后端配置问题**：参考 `backend/README_DeepSeek_配置指南.md`
- **DeepSeek 官方文档**：https://platform.deepseek.com/api-docs/
- **项目文档**：`docs/` 目录

---

**更新时间**：2025-01-17
**版本**：v2.0（服务器端配置版）
