# AI 聊天功能配置说明

> **重要更新**：AI Chat 功能已简化，API Key 在服务器端统一配置，用户无需手动输入。

---

## 🎉 更新内容

### 后端改进

1. **✅ 配置集中化**
   - API Key 在 `backend/.env` 或 `backend/app/core/config.py` 中配置
   - 统一管理，更安全

2. **✅ 简化的 AI 工厂**
   - 移除多服务商支持，专注 DeepSeek
   - 直接从配置创建服务实例

3. **✅ 优化的 API 端点**
   - 不再需要前端传递 `provider` 和 `api_key`
   - 请求参数更简洁

### 前端改进

1. **✅ 移除 API Key 输入**
   - 用户无需配置 API Key
   - 设置面板更简洁

2. **✅ 简化的配置界面**
   - 只保留模型和温度选择
   - 更友好的用户体验

---

## 🚀 快速开始

### 第一步：配置 DeepSeek API Key

#### 获取 API Key

1. 访问 https://platform.deepseek.com/
2. 注册并登录
3. 创建 API Key（格式：`sk-xxxxxx`）

#### 配置后端

**方法 1：使用 .env 文件（推荐）**

在 `backend/` 目录下创建 `.env` 文件：

```env
DEEPSEEK_API_KEY=sk-your-api-key-here
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_TEMPERATURE=0.7
```

**方法 2：直接修改配置文件**

编辑 `backend/app/core/config.py`：

```python
# DeepSeek AI 配置
DEEPSEEK_API_KEY: str = "sk-your-api-key-here"
DEEPSEEK_MODEL: str = "deepseek-chat"
DEEPSEEK_TEMPERATURE: float = 0.7
```

### 第二步：启动服务

**启动后端**

```bash
cd backend
python run.py
```

**启动前端**

```bash
npm run dev
```

### 第三步：测试 AI Chat

1. 访问 http://localhost:3000
2. 点击导航栏 **"AI Chat"**
3. 发送消息测试

---

## 📋 配置参数说明

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `DEEPSEEK_API_KEY` | ✅ | 无 | DeepSeek API 密钥 |
| `DEEPSEEK_MODEL` | ❌ | `deepseek-chat` | 默认模型 |
| `DEEPSEEK_TEMPERATURE` | ❌ | `0.7` | 温度参数 (0-1) |

### 可用模型

- **deepseek-chat**：通用对话模型，适合日常对话
- **deepseek-coder**：代码专用模型，适合编程问题

### 温度参数

- **0.0**：最精确，每次回答一致
- **0.7**：默认值（推荐）
- **1.0**：最有创造力

---

## 🔧 项目结构

### 后端文件

```
backend/
├── .env                          # 环境变量配置（你需要创建）
├── app/
│   ├── core/
│   │   └── config.py            # ✅ 配置文件（新增 DeepSeek 配置）
│   ├── services/
│   │   ├── ai_factory.py        # ✅ AI 工厂（简化版）
│   │   ├── ai_service_base.py   # AI 服务基类
│   │   └── ai_providers/
│   │       └── deepseek_service.py  # DeepSeek 实现
│   └── api/
│       └── endpoints/
│           └── ai.py            # ✅ AI API 端点（简化版）
└── README_DeepSeek_配置指南.md  # 📄 详细配置指南
```

### 前端文件

```
src/
├── views/
│   └── AIChat.vue               # ✅ AI 聊天页面（简化版）
├── api/
│   └── ai.js                    # ✅ AI API 调用（简化版）
└── components/
    └── Navigation.vue           # 导航栏（包含 AI Chat 链接）
```

### 文档文件

```
docs/
└── deepseek-setup.md            # 📄 DeepSeek 配置指南（更新版）
```

---

## 🎨 前后端对比

### 之前的架构（复杂）

```
前端用户输入 API Key
     ↓
前端发送 API Key 到后端
     ↓
后端转发到 DeepSeek API
```

**问题：**
- 每个用户都要配置 API Key
- API Key 在前端暴露，不安全
- 用户体验差

### 现在的架构（简洁）

```
前端用户登录
     ↓
后端自动使用配置的 API Key
     ↓
后端转发到 DeepSeek API
```

**优势：**
- ✅ 用户无需配置，开箱即用
- ✅ API Key 不暴露，更安全
- ✅ 统一管理，成本可控
- ✅ 用户体验好

---

## 🔍 API 变化对比

### 之前的请求

```json
POST /api/v1/ai/chat/stream
{
  "provider": "deepseek",        // ❌ 不再需要
  "model": "deepseek-chat",
  "messages": [...],
  "temperature": 0.7,
  "api_key": "sk-xxx"            // ❌ 不再需要
}
```

### 现在的请求

```json
POST /api/v1/ai/chat/stream
{
  "model": "deepseek-chat",      // 可选
  "messages": [...],
  "temperature": 0.7             // 可选
}
```

---

## ❓ 常见问题

### Q1: 提示 "DeepSeek API Key 未配置"

**解决方法：**
1. 检查 `backend/.env` 是否存在且配置正确
2. 或检查 `backend/app/core/config.py` 中的 `DEEPSEEK_API_KEY`
3. 重启后端服务

### Q2: 如何验证配置是否成功？

访问 http://localhost:8000/api/v1/ai/providers

返回 `"configured": true` 即表示成功：

```json
{
  "code": 0,
  "data": {
    "deepseek": {
      "configured": true  // ✅
    }
  }
}
```

### Q3: 可以动态切换 API Key 吗？

当前版本不支持动态切换，需要：
1. 修改 `.env` 或 `config.py`
2. 重启后端服务

### Q4: 如何监控 API 使用量？

访问 DeepSeek 控制台：
https://platform.deepseek.com/usage

---

## 📦 依赖包（已包含）

```txt
# backend/requirements.txt
bcrypt>=4.0.0              # 密码加密
httpx>=0.25.0              # AI 服务 HTTP 客户端
python-jose[cryptography]  # JWT 认证
fastapi                    # Web 框架
```

---

## 🔐 安全建议

### 管理员

1. **保护 API Key**
   - ❌ 不要提交到 Git 仓库
   - ✅ 使用 `.env` 文件（已加入 .gitignore）
   - ✅ 定期轮换 API Key

2. **监控使用**
   - 定期检查 DeepSeek 控制台
   - 设置余额预警
   - 记录异常使用

3. **访问控制**
   - AI 接口需要登录（已实现）
   - 可以添加请求频率限制
   - 记录所有 API 调用日志

### 用户

1. **不要发送敏感信息**
   - 密码、密钥
   - 个人隐私
   - 商业机密

---

## 📞 获取更多帮助

### 文档

- **详细配置指南**：`backend/README_DeepSeek_配置指南.md`
- **用户使用指南**：`docs/deepseek-setup.md`

### 外部资源

- **DeepSeek 官网**：https://platform.deepseek.com/
- **DeepSeek 文档**：https://platform.deepseek.com/api-docs/

---

## ✅ 完成清单

配置完成后，请确认以下事项：

- [ ] 已获取 DeepSeek API Key
- [ ] 已配置 `backend/.env` 或 `backend/app/core/config.py`
- [ ] 已启动后端服务
- [ ] 已访问 `/api/v1/ai/providers` 确认配置成功
- [ ] 已启动前端服务
- [ ] 已测试 AI Chat 功能

---

**更新时间**：2025-01-17  
**版本**：v2.0（服务器端配置版）  
**状态**：✅ 已完成

---

祝你使用愉快！🎉

