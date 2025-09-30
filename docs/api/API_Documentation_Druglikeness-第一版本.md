# 药物类药性评估接口文档

## 概述

本接口用于评估分子的类药性（Drug-likeness），基于多种经典的药物筛选规则对输入的SMILES字符串进行分析，返回详细的分子指标和匹配度评分。

## 基础信息

- **基础URL**: `http://localhost:8000`
- **接口路径**: `/api/v1/druglikeness/evaluate`
- **请求方法**: `POST`
- **Content-Type**: `application/json`

## 接口详情

### 请求参数

#### 请求体 (JSON)

```json
{
  "smiles": "string",
  "rules": ["string"] // 可选
}
```

#### 参数说明

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| smiles | string | 是 | 分子的SMILES字符串表示 |
| rules | array[string] | 否 | 要评估的规则列表，如果不提供则评估所有规则 |

#### 可选规则列表

| 规则名称 | 说明 | 评估标准 |
|----------|------|----------|
| Lipinski | Lipinski五规则 | 分子量≤500, LogP≤5, 氢键供体≤5, 氢键受体≤10 |
| Ghose | Ghose规则 | 分子量160-480, LogP(-0.4-5.6), 摩尔折射率40-130, 原子数20-70 |
| Oprea | Oprea规则 | 可旋转键≤15, 刚性键≥2, 至少一个环结构 |
| Veber | Veber规则 | 可旋转键≤10, 极性表面积≤140, 氢键供体≤5, 氢键受体≤10 |
| Varma | Varma规则 | 分子量300-500, 极性表面积≤150, LogD 0-4, 氢键供体≤5, 氢键受体≤10, 可旋转键≤10 |

### 响应格式

#### 成功响应

**HTTP状态码**: 200

```json
{
  "code": 0,
  "message": "药物类药性评估成功",
  "data": {
    "metrics": {
      "Lipinski": {
        "mw": 508.00,
        "logp": 4.09,
        "hbd": 1.00,
        "hba": 8.00
      },
      "Ghose": {
        "mw": 508.00,
        "logp": 4.09,
        "mr": 129.04,
        "atom_count": 34.00
      },
      "Oprea": {
        "rot_bonds": 10.00,
        "rigid_bonds": 26.00,
        "ring_count": 3.00
      },
      "Veber": {
        "rot_bonds": 10.00,
        "tpsa": 118.92,
        "hbd": 1.00,
        "hba": 8.00
      },
      "Varma": {
        "molecular_weight": 508.00,
        "tpsa": 118.92,
        "logd": 4.09,
        "h_bond_donor": 1.00,
        "h_bond_acceptor": 8.00,
        "rotatable_bonds": 10.00
      }
    },
    "matches": {
      "Lipinski": 0.75,
      "Ghose": 0.75,
      "Oprea": 1.0,
      "Veber": 1.0,
      "Varma": 1.0
    },
    "smiles": "CCN(CC)CCCOC1=NC2=C(C=CC(=C2Cl)C(=O)OC)C(=N1)C3=CC=CC=C3S(=O)(=O)O"
  },
  "timestamp": 1703123456
}
```

#### 错误响应

**HTTP状态码**: 400 (客户端错误) 或 500 (服务器错误)

```json
{
  "code": 400,
  "message": "Invalid SMILES string",
  "data": null,
  "timestamp": 1703123456
}
```

### 响应字段说明

#### 标准响应字段

| 字段名 | 类型 | 说明 |
|--------|------|------|
| code | integer | 响应状态码，0表示成功，非0表示错误 |
| message | string | 响应消息 |
| data | object | 响应数据，成功时包含评估结果 |
| timestamp | integer | 响应时间戳 |

#### 数据字段 (data)

| 字段名 | 类型 | 说明 |
|--------|------|------|
| metrics | object | 各规则的详细指标值 |
| matches | object | 各规则的匹配度评分 (0-1) |
| smiles | string | 输入的SMILES字符串 |

#### 指标字段详解 (metrics)

**Lipinski规则指标**:
- `mw`: 分子量 (Molecular Weight)
- `logp`: 脂水分配系数 (LogP)
- `hbd`: 氢键供体数量 (Hydrogen Bond Donors)
- `hba`: 氢键受体数量 (Hydrogen Bond Acceptors)

**Ghose规则指标**:
- `mw`: 分子量
- `logp`: 脂水分配系数
- `mr`: 摩尔折射率 (Molar Refractivity)
- `atom_count`: 原子总数

**Oprea规则指标**:
- `rot_bonds`: 可旋转键数量
- `rigid_bonds`: 刚性键数量
- `ring_count`: 环结构数量

**Veber规则指标**:
- `rot_bonds`: 可旋转键数量
- `tpsa`: 拓扑极性表面积 (Topological Polar Surface Area)
- `hbd`: 氢键供体数量
- `hba`: 氢键受体数量

**Varma规则指标**:
- `molecular_weight`: 分子量
- `tpsa`: 拓扑极性表面积
- `logd`: 分布系数 (LogD, pH=7.4)
- `h_bond_donor`: 氢键供体数量
- `h_bond_acceptor`: 氢键受体数量
- `rotatable_bonds`: 可旋转键数量

#### 匹配度评分 (matches)

每个规则的匹配度评分范围为 0-1：
- 1.0: 完全符合该规则的所有标准
- 0.75: 符合该规则75%的标准
- 0.5: 符合该规则50%的标准
- 0.25: 符合该规则25%的标准
- 0.0: 完全不符合该规则

## 请求示例

### 示例1: 评估所有规则

```bash
curl -X POST "http://localhost:8000/api/v1/druglikeness/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "smiles": "CCN(CC)CCCOC1=NC2=C(C=CC(=C2Cl)C(=O)OC)C(=N1)C3=CC=CC=C3S(=O)(=O)O"
  }'
```

### 示例2: 仅评估Lipinski和Veber规则

```bash
curl -X POST "http://localhost:8000/api/v1/druglikeness/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "smiles": "CCN(CC)CCCOC1=NC2=C(C=CC(=C2Cl)C(=O)OC)C(=N1)C3=CC=CC=C3S(=O)(=O)O",
    "rules": ["Lipinski", "Veber"]
  }'
```

### JavaScript示例

```javascript
// 使用fetch API
const evaluateDruglikeness = async (smiles, rules = null) => {
  const requestBody = { smiles };
  if (rules) {
    requestBody.rules = rules;
  }

  try {
    const response = await fetch('http://localhost:8000/api/v1/druglikeness/evaluate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody)
    });

    const result = await response.json();
    
    if (result.code === 0) {
      console.log('评估成功:', result.data);
      return result.data;
    } else {
      console.error('评估失败:', result.message);
      throw new Error(result.message);
    }
  } catch (error) {
    console.error('请求错误:', error);
    throw error;
  }
};

// 使用示例
evaluateDruglikeness('CCN(CC)CCCOC1=NC2=C(C=CC(=C2Cl)C(=O)OC)C(=N1)C3=CC=CC=C3S(=O)(=O)O')
  .then(data => {
    console.log('分子指标:', data.metrics);
    console.log('匹配度:', data.matches);
  })
  .catch(error => {
    console.error('评估失败:', error);
  });
```

## 错误处理

### 常见错误码

| 错误码 | HTTP状态码 | 说明 | 解决方案 |
|--------|------------|------|----------|
| 400 | 400 | 无效的SMILES字符串 | 检查SMILES字符串格式是否正确 |
| 400 | 400 | 无效的规则名称 | 检查rules参数中的规则名称是否正确 |
| 500 | 500 | 服务器内部错误 | 联系技术支持或稍后重试 |

### 错误响应示例

```json
{
  "code": 400,
  "message": "Invalid rules: ['InvalidRule']. Available rules: ['Lipinski', 'Ghose', 'Oprea', 'Veber', 'Varma']",
  "data": null,
  "timestamp": 1703123456
}
```

## 注意事项

1. **SMILES格式**: 确保输入的SMILES字符串格式正确，系统会验证SMILES的有效性
2. **规则选择**: 如果不指定rules参数，系统将评估所有五种规则
3. **性能考虑**: 评估所有规则比仅评估部分规则需要更多计算时间
4. **数值精度**: 所有数值结果保留2位小数显示，但内部计算使用完整精度
5. **并发限制**: 建议控制并发请求数量以确保服务稳定性

## 技术支持

如有技术问题或需要进一步的接口定制，请联系开发团队。

---

**文档版本**: v1.0  
**最后更新**: 2024年12月  
**维护者**: ADMETchem开发团队

---

# SDF 批量评估接口文档（新增）

## 概述

前端上传 SDF 文件，后端提取全部 SMILES，并对每条分子进行药物类药性评估。默认仅使用 Lipinski 规则；前端可通过 selected_rule 指定其他规则。

## 基础信息

- **基础URL**: `http://localhost:8000/api/v1/`
- **接口路径**: `/sdf/analyze`
- **请求方法**: `POST`
- **Content-Type**: `multipart/form-data`

## 接口详情

### 请求参数

表单字段（multipart/form-data）

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| file | file | 是 | SDF 文件 |

查询参数（Query）

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| isomeric_smiles | boolean | 否 | 是否保留立体化学信息，默认 true |
| kekule_smiles | boolean | 否 | 是否使用凯库勒式，默认 true |
| canonical | boolean | 否 | 是否标准化 SMILES，默认 true |
| selected_rule | string | 否 | 单个评估规则名；不传则默认 'Lipinski' |

可选规则列表：`Lipinski`, `Ghose`, `Oprea`, `Veber`, `Varma`

### 成功响应

HTTP 200

```json
{
  "code": 0,
  "message": "提取成功",
  "data": {
    "filename": "input.sdf",
    "items": [
      {
        "smiles": "COCCCCC(=NOCCN)C1=CC=C(C(F)(F)F)C=C1",
        "metrics": { "Lipinski": { "mw": 508.0, "logp": 4.09, "hbd": 1, "hba": 8 } },
        "matches": { "Lipinski": 0.75 },
        "total_score": 0.75
      }
    ]
  },
  "timestamp": 1703123456
}
```

说明：
- `items` 为逐分子结果数组；单个分子评估失败会被静默跳过，不影响整体返回。
- 当全部分子均无法评估时，返回 400。

### 错误响应

HTTP 400 / 500，与全局错误响应一致。

```json
{ "code": 400, "message": "SDF文件中未找到可评估分子", "data": null, "timestamp": 1703123456 }
```

## 请求示例

### 示例1：仅默认规则（Lipinski）

```bash
curl -X POST "http://localhost:8000/api/v1/sdf/analyze" \
  -F "file=@input.sdf"
```

### 示例2：指定单条规则（Veber）

```bash
curl -X POST "http://localhost:8000/api/v1/sdf/analyze?selected_rule=Veber" \
  -F "file=@input.sdf"
```

### 示例3：指定全部参数

```bash
curl -X POST "http://localhost:8000/api/v1/sdf/analyze?isomeric_smiles=true&kekule_smiles=true&canonical=true&selected_rule=Lipinski" \
  -F "file=@input.sdf"