# 综合评估 API 使用文档

## 概述

新版API支持**同时评估多个类药性规则和分子性质**，用户可以勾选任意组合的评估项目，后端只计算用户选择的内容。

## API端点

```
POST /api/v1/druglikeness/evaluate
```

## 支持的评估项目

### 类药性规则（5个）
- `Lipinski` - Lipinski五规则
- `Ghose` - Ghose规则
- `Oprea` - Oprea规则
- `Veber` - Veber规则
- `Varma` - Varma规则

### 分子性质（5个）
- `QED` - 类药性定量评估
- `SAscore` - 合成可及性
- `Fsp3` - 碳饱和度
- `MCE18` - 结构复杂性
- `NPscore` - 天然产物相似性

## 请求格式

### 新版API（推荐）

```json
{
  "smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
  "selected_items": ["Lipinski", "QED", "SAscore"]
}
```

### 旧版API（向后兼容）

```json
{
  "smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
  "rules": ["Lipinski", "Veber"]
}
```

## 响应格式

### 新版API响应

```json
{
  "code": 200,
  "message": "综合评估成功",
  "data": {
    "selected_items": ["Lipinski", "QED", "SAscore"],
    "druglikeness_rules": {
      "metrics": {
        "Lipinski": {
          "mw": 206.28,
          "logp": 3.5,
          "hbd": 1,
          "hba": 2
        }
      },
      "matches": {
        "Lipinski": 1.0
      },
      "total_score": 1.0
    },
    "molecular_properties": {
      "QED": 0.8216,
      "QED_status": "excellent",
      "SAscore": 2.6464,
      "SAscore_status": "excellent"
    },
    "smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
  }
}
```

## 使用示例

### Python 示例

```python
import requests

# API地址
url = "http://localhost:8000/api/v1/druglikeness/evaluate"

# 场景1：只评估类药性规则
data = {
    "smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "selected_items": ["Lipinski", "Veber"]
}
response = requests.post(url, json=data)
print(response.json())

# 场景2：只计算分子性质
data = {
    "smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "selected_items": ["QED", "SAscore", "Fsp3"]
}
response = requests.post(url, json=data)
print(response.json())

# 场景3：混合评估
data = {
    "smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "selected_items": ["Lipinski", "Ghose", "QED", "SAscore", "Fsp3"]
}
response = requests.post(url, json=data)
print(response.json())
```

### JavaScript 示例（前端）

```javascript
import { evaluateComprehensive } from '@/api'

// 用户勾选的评估项目
const selectedRules = ['lipinski', 'qed', 'sascore']

// 将前端ID转换为API格式
const ruleMapping = {
  'lipinski': 'Lipinski',
  'ghose': 'Ghose',
  'oprea': 'Oprea',
  'veber': 'Veber',
  'varma': 'Varma',
  'qed': 'QED',
  'sascore': 'SAscore',
  'fsp3': 'Fsp3',
  'mce18': 'MCE18',
  'npscore': 'NPscore'
}

const selectedItems = selectedRules.map(id => ruleMapping[id])

// 调用API
const result = await evaluateComprehensive({
  smiles: 'CC(C)Cc1ccc(cc1)C(C)C(=O)O',
  selected_items: selectedItems
})

console.log('评估结果:', result.data)
```

## 响应字段说明

### `selected_items`
用户选择的评估项目列表

### `druglikeness_rules`（可选）
如果用户选择了类药性规则，则包含此字段：
- `metrics`: 各规则的详细指标值
- `matches`: 各规则的匹配度（0-1）
- `total_score`: 加权总分（0-1）

### `molecular_properties`（可选）
如果用户选择了分子性质，则包含此字段：
- 每个性质的值（如 `QED`: 0.82）
- 每个性质的状态（如 `QED_status`: "excellent"）

## 错误处理

### 无效的SMILES
```json
{
  "code": 400,
  "message": "Invalid SMILES string",
  "data": null
}
```

### 无效的评估项目
```json
{
  "code": 400,
  "message": "Invalid items: ['InvalidRule']. Available items: ['Lipinski', ...]",
  "data": null
}
```

### 缺少数据文件
```json
{
  "code": 500,
  "message": "服务器内部错误: 请下载fpscores.pkl.gz并放在backend目录",
  "data": null
}
```

## 迁移指南

### 从旧版API迁移

**旧版代码：**
```javascript
const result = await evaluateSmiles({
  smiles: smilesString,
  rules: ['Lipinski']
})
```

**新版代码：**
```javascript
const result = await evaluateComprehensive({
  smiles: smilesString,
  selected_items: ['Lipinski', 'QED', 'SAscore']
})
```

### 响应格式变化

**旧版响应：**
```json
{
  "metrics": {...},
  "matches": {...},
  "smiles": "..."
}
```

**新版响应：**
```json
{
  "selected_items": [...],
  "druglikeness_rules": {
    "metrics": {...},
    "matches": {...},
    "total_score": ...
  },
  "molecular_properties": {...},
  "smiles": "..."
}
```

## 性能优化建议

1. **按需选择**：只勾选需要的评估项目，可以提高响应速度
2. **批量处理**：如需评估多个分子，建议分批请求
3. **数据文件**：确保 `fpscores.pkl.gz` 和 `publicnp.model.gz` 已正确部署

## 常见问题

### Q: 如何获取所有评估项目？
A: 不传 `selected_items` 参数，或传 `null`，系统会计算所有10个项目。

### Q: 可以只选择一个项目吗？
A: 可以，例如 `selected_items: ["QED"]`

### Q: 旧版API还能用吗？
A: 可以，旧版API（使用 `rules` 参数）完全向后兼容。

### Q: NPscore和SAscore需要什么文件？
A: 需要 `publicnp.model.gz` 和 `fpscores.pkl.gz`，从RDKit官方下载。

---

更新时间: 2025-09-30
