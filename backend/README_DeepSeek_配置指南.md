# DeepSeek AI 配置指南

> ADMETchem 项目使用 DeepSeek AI 作为唯一的 AI 服务提供商，API Key 在服务器端配置，无需用户手动输入。

## 📋 目录

- [为什么选择 DeepSeek？](#为什么选择-deepseek)
- [快速开始](#快速开始)
- [配置步骤](#配置步骤)
- [环境变量说明](#环境变量说明)
- [常见问题](#常见问题)
- [API 使用限制](#api-使用限制)

---

## 🎯 为什么选择 DeepSeek？

1. **性价比最高** - 新用户赠送 500 万 tokens 免费额度
2. **国内可用** - 国内用户无需 VPN 即可访问
3. **功能强大** - 支持通用对话和代码专用模型
4. **响应快速** - 支持流式响应，打字机效果
5. **稳定可靠** - 国内 AI 服务商，服务稳定

---

## 🚀 快速开始

### 1. 获取 API Key

访问 [DeepSeek 官网](https://platform.deepseek.com/) 注册并获取 API Key：

1. **注册账号**
   - 访问：https://platform.deepseek.com/
   - 点击右上角"注册"
   - 使用邮箱或手机号注册

2. **创建 API Key**
   - 登录后进入控制台
   - 点击左侧菜单 "API Keys"
   - 点击 "Create API Key"
   - 复制生成的 API Key（格式：`sk-xxxxxx`）
   - **⚠️ 注意：API Key 只显示一次，请妥善保存**

3. **查看额度**
   - 新用户自动获得 **500 万 tokens** 免费额度
   - 在控制台"余额"页面可查看剩余额度

### 2. 配置后端

#### 方法 1：使用环境变量（推荐）

在 `backend/` 目录下创建 `.env` 文件（如果不存在）：

```bash
cd backend
touch .env
```

编辑 `.env` 文件，添加以下内容：

```env
# DeepSeek AI 配置
DEEPSEEK_API_KEY=sk-your-api-key-here
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_TEMPERATURE=0.7
```

**说明：**
- `DEEPSEEK_API_KEY`: 你的 DeepSeek API Key（必填）
- `DEEPSEEK_MODEL`: 默认模型（可选，默认 `deepseek-chat`）
- `DEEPSEEK_TEMPERATURE`: 温度参数（可选，默认 `0.7`，范围 `0-1`）

#### 方法 2：直接修改配置文件

如果不想使用 `.env` 文件，可以直接修改 `backend/app/core/config.py`：

```python
# DeepSeek AI 配置
DEEPSEEK_API_KEY: str = "sk-your-api-key-here"  # 替换为你的 API Key
DEEPSEEK_MODEL: str = "deepseek-chat"
DEEPSEEK_TEMPERATURE: float = 0.7
```

> ⚠️ **安全提示**：如果修改配置文件，请确保不要将 API Key 提交到 Git 仓库！

### 3. 启动服务

```bash
# 激活虚拟环境（如果使用）
cd backend
.\venv_admet_py311\Scripts\activate  # Windows
# source venv_admet_py311/bin/activate  # Linux/Mac

# 启动后端服务
python run.py
```

启动成功后，你会看到以下日志：

```
🚀 正在启动应用...
📊 正在同步数据库模型...
✅ 数据库同步完成！
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### 4. 测试 AI 功能

1. 启动前端：
   ```bash
   npm run dev
   ```

2. 访问 http://localhost:3000

3. 点击导航栏中的 **"AI Chat"**

4. 发送消息测试：
   - 输入："What is ADMET?"
   - 点击发送
   - 看到打字机效果的流式响应即表示配置成功！✅

---

## 🔧 环境变量说明

| 变量名 | 必填 | 默认值 | 说明 |
|--------|------|--------|------|
| `DEEPSEEK_API_KEY` | ✅ | 无 | DeepSeek API 密钥 |
| `DEEPSEEK_MODEL` | ❌ | `deepseek-chat` | 默认模型 |
| `DEEPSEEK_TEMPERATURE` | ❌ | `0.7` | 温度参数 (0-1) |

### 可用模型

| 模型名称 | 适用场景 | 特点 |
|----------|----------|------|
| `deepseek-chat` | 日常对话、知识问答 | 通用模型，适合大多数场景 |
| `deepseek-coder` | 代码生成、编程问答 | 代码专用，编程能力更强 |

### 温度参数 (Temperature)

- **0.0** - 最精确，每次回答一致，适合需要准确答案的场景
- **0.3-0.5** - 较为保守，适合技术问答
- **0.7** - 默认值，平衡创造性和准确性
- **0.9-1.0** - 最有创造性，适合创意写作

---

## ❓ 常见问题

### 1. 如何检查 API Key 是否配置成功？

访问 http://localhost:8000/api/v1/ai/providers，返回结果中 `configured: true` 表示已配置：

```json
{
  "code": 0,
  "data": {
    "deepseek": {
      "name": "DeepSeek",
      "configured": true,  // ✅ 已配置
      "current_model": "deepseek-chat",
      "current_temperature": 0.7
    }
  }
}
```

### 2. 提示 "DeepSeek API Key 未配置"

**原因：** 环境变量未正确加载或配置文件未修改。

**解决方法：**
1. 检查 `backend/.env` 文件是否存在
2. 确认 `DEEPSEEK_API_KEY` 已正确填写
3. 重启后端服务（环境变量需要重启才能生效）

### 3. API 调用失败，提示 401 Unauthorized

**原因：** API Key 无效或过期。

**解决方法：**
1. 访问 [DeepSeek 控制台](https://platform.deepseek.com/) 检查 API Key 是否有效
2. 重新生成一个新的 API Key
3. 更新 `.env` 文件
4. 重启后端服务

### 4. 消息发送后没有响应

**可能原因：**
- 网络连接问题
- API 额度用尽
- 请求超时

**排查步骤：**
1. 检查后端日志，查看错误信息
2. 访问 DeepSeek 控制台检查余额
3. 尝试发送更短的消息

### 5. 如何更换模型？

**方法 1：前端设置**
1. 点击 AI Chat 页面右上角的设置图标（⚙️）
2. 在"Model"下拉框中选择模型
3. 点击"Save"保存

**方法 2：修改默认配置**
- 修改 `.env` 文件中的 `DEEPSEEK_MODEL`
- 或修改 `config.py` 中的 `DEEPSEEK_MODEL`
- 重启后端服务

---

## 📊 API 使用限制

### 免费额度

- **新用户赠送**：500 万 tokens
- **有效期**：永久有效（只要不超额）
- **查询余额**：https://platform.deepseek.com/usage

### Token 计算

- **输入 Token**：用户发送的消息 + 历史对话上下文
- **输出 Token**：AI 返回的响应内容

**示例：**
- "What is ADMET?" ≈ 50 tokens（输入）
- AI 回答 ≈ 200 tokens（输出）
- **总计**：250 tokens

### 请求限制

| 限制类型 | 免费用户 | 付费用户 |
|----------|----------|----------|
| 请求频率 | 60 次/分钟 | 200 次/分钟 |
| 并发请求 | 3 | 10 |
| 最大 Token | 4096 | 8192 |

---

## 🔒 安全建议

1. **不要公开 API Key**
   - 不要提交到 Git 仓库
   - 不要在前端代码中硬编码
   - 使用环境变量或配置文件

2. **定期轮换 API Key**
   - 每 3-6 个月更换一次
   - 如果怀疑泄露，立即重新生成

3. **监控使用情况**
   - 定期检查 API 调用量
   - 设置余额预警
   - 避免滥用

4. **备份配置**
   - 保存 API Key 到安全的密码管理器
   - 记录配置文件的修改

---

## 📞 获取帮助

- **DeepSeek 官方文档**：https://platform.deepseek.com/api-docs/
- **DeepSeek 控制台**：https://platform.deepseek.com/
- **技术支持**：https://platform.deepseek.com/help

---

## 🎉 完成！

现在你已经成功配置了 DeepSeek AI，可以开始使用 AI Chat 功能了！

**下一步：**
- 尝试不同的模型和温度参数
- 探索 AI 对话功能
- 将 AI 集成到你的工作流中

---

**更新时间**：2025-01-17
**版本**：v1.0

