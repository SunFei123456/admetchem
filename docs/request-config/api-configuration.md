# API 配置说明

## 环境变量配置

项目使用 Vite 的环境变量功能来管理不同环境下的 API 地址。

### 配置文件

在项目根目录创建以下文件：

#### `.env.development` - 开发环境
```bash
# 开发环境配置
VITE_API_BASE_URL=http://localhost:8000
```

#### `.env.production` - 生产环境
```bash
# 生产环境配置
VITE_API_BASE_URL=http://8.152.194.158:16666
```

### 环境说明

- **开发环境**: 运行 `pnpm dev` 时自动加载 `.env.development`
- **生产环境**: 运行 `pnpm build` 时自动加载 `.env.production`

### API 模块结构

```
src/api/
├── request.js          # Axios 封装，统一请求配置
├── druglikeness.js     # 药物相似性评估 API
└── index.js            # API 统一导出
```

## 使用方法

### 1. 在组件中导入 API

```javascript
import { evaluateSmiles, analyzeSdfFile } from '@/api'
```

### 2. 调用 API

```javascript
// 评估 SMILES
const result = await evaluateSmiles({
  smiles: 'CCO',
  rules: ['Lipinski', 'Ghose']
})

// 分析 SDF 文件
const formData = new FormData()
formData.append('file', file)
const result = await analyzeSdfFile(formData, {
  selected_rule: 'Lipinski',
  isomeric_smiles: true,
  kekule_smiles: true,
  canonical: true
})
```

## 拦截器功能

### 请求拦截器
- 自动添加请求头
- 可添加 Token 认证
- 日志记录

### 响应拦截器
- 统一处理响应数据
- 自动显示错误消息
- 处理各种 HTTP 状态码

## 错误处理

所有 API 错误都会被 axios 拦截器自动捕获并显示友好的错误提示：

- 400: 请求参数错误
- 401: 未授权，请登录
- 403: 拒绝访问
- 404: 请求的资源不存在
- 500: 服务器内部错误
- 超时: 请求超时，请稍后重试
- 网络错误: 无法连接到服务器

## 添加新的 API

### 1. 创建 API 模块

在 `src/api/` 下创建新的 API 文件，例如 `admet.js`:

```javascript
import request from './request'

export function predictADMET(data) {
  return request({
    url: '/api/v1/admet/predict',
    method: 'post',
    data
  })
}
```

### 2. 在 index.js 中导出

```javascript
export * from './druglikeness'
export * from './admet'  // 新增
```

### 3. 在组件中使用

```javascript
import { predictADMET } from '@/api'

const result = await predictADMET({ ... })
```

## 注意事项

1. **不要提交 .env 文件到 Git**
   - `.env.development` 和 `.env.production` 已被添加到 `.gitignore`
   - 团队成员需要根据文档自行创建这些文件

2. **环境变量命名规范**
   - Vite 环境变量必须以 `VITE_` 开头
   - 使用大写字母和下划线

3. **开发调试**
   - 可以在 `src/api/request.js` 中查看请求和响应日志
   - 根据需要调整 timeout 时间（默认 30 秒）

4. **生产环境**
   - 确保生产环境的 API 地址正确
   - 考虑添加请求加密或签名机制
