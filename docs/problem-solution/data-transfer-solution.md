# 数据传递方案优化

## 问题背景

原先的实现方式通过URL参数传递评估结果数据，存在以下问题：

1. **URL过长且丑陋**：大量数据编码后的URL参数使地址栏变得冗长难看
2. **数据暴露**：敏感的评估结果直接暴露在URL中，不够安全
3. **浏览器限制**：URL长度限制可能导致数据截断
4. **用户体验差**：用户可能意外复制包含数据的URL

## 解决方案

### 1. SessionStorage + 临时键方案

采用浏览器的 `sessionStorage` 存储评估结果，URL中只传递临时的结果键。

**优势：**
- ✅ URL简洁美观
- ✅ 数据不暴露在URL中
- ✅ 支持大量数据存储
- ✅ 会话级别的数据隔离
- ✅ 自动过期机制

### 2. 实现细节

#### 数据存储流程
```javascript
// 1. 生成唯一的结果键
const resultKey = `druglikeness_result_${Date.now()}`

// 2. 存储数据到sessionStorage
sessionStorage.setItem(resultKey, JSON.stringify({
  data: result.data,
  rule: selectedRule,
  source: 'smiles',
  timestamp: Date.now()
}))

// 3. 跳转时只传递结果键
router.push({
  path: '/druglikeness-result',
  query: { resultId: resultKey }
})
```

#### 数据获取流程
```javascript
// 1. 从URL获取结果键
const resultId = route.query.resultId

// 2. 从sessionStorage获取数据
const storedData = sessionStorage.getItem(resultId)

// 3. 验证数据有效性和过期时间
if (storedData && !isExpired(data)) {
  // 使用数据
  resultData.value = parsedData.data
}
```

### 3. 安全特性

#### 数据过期机制
- 数据存储时添加时间戳
- 读取时检查是否超过1小时
- 自动清理过期数据

#### 数据清理策略
- 页面初始化时清理过期数据
- 返回评估页面时清理当前数据
- 提供手动清理所有数据的功能

### 4. 工具函数模块

创建了 `src/utils/storage.js` 模块，提供统一的数据管理接口：

```javascript
// 存储数据
const resultKey = storeResultData({
  data: result.data,
  rule: selectedRule,
  source: 'smiles'
})

// 获取数据
const data = getResultData(resultId)

// 删除数据
removeResultData(resultId)

// 清理过期数据
cleanupExpiredData()
```

### 5. URL对比

#### 优化前
```
localhost:3001/druglikeness-result?data=%257B%2522filename%2522%253A%2522input.sdf%2522%252C%2522total_molecules%2522%253A3%252C%2522results%2522%253A%255B%257B%2522smiles%2522%253A%2522CCN%2528CC%2529CCCOC1%253DNC2%253DC%2528C%253DCC%2528%253DC2Cl%2529C%2528%253DO%2529OC%2529C%2528%253DN1%2529C3%253DCC%253DCC%253DC3S%2528%253DO%2529%2528%253DO%2529O%2522%252C%2522metrics%2522%253A%257B%2522Lipinski%2522%253A%257B%2522mw%2522%253A408.88%252C%2522logp%2522%253A3.49%252C%2522hbd%2522%253A1%252C%2522hba%2522%253A7%257D%257D%252C%2522matches%2522%253A%257B%2522Lipinski%2522%253A0.75%257D%257D%255D%257D&rule=Lipinski&source=file
```

#### 优化后
```
localhost:3001/druglikeness-result?resultId=druglikeness_result_1703123456789
```

### 6. 其他可选方案

#### 方案A：Vue Router State
```javascript
// 优点：Vue原生支持，数据不持久化
// 缺点：刷新页面数据丢失
router.push({
  path: '/druglikeness-result',
  state: { resultData: data }
})
```

#### 方案B：Vuex/Pinia状态管理
```javascript
// 优点：全局状态管理，响应式
// 缺点：增加复杂度，需要额外配置
store.commit('setResultData', data)
router.push('/druglikeness-result')
```

#### 方案C：LocalStorage
```javascript
// 优点：持久化存储
// 缺点：数据不会自动清理，可能泄露到其他会话
localStorage.setItem('resultData', JSON.stringify(data))
```

### 7. 选择理由

选择 SessionStorage 方案的原因：

1. **会话隔离**：数据只在当前标签页有效，关闭标签页自动清理
2. **适中的持久性**：支持页面刷新，但不会永久保存
3. **简单实现**：无需额外的状态管理库
4. **安全性好**：数据不会泄露到其他会话或标签页
5. **性能优秀**：读写速度快，不影响页面加载

### 8. 注意事项

1. **浏览器兼容性**：SessionStorage在所有现代浏览器中都支持
2. **存储限制**：SessionStorage通常有5-10MB的存储限制
3. **数据格式**：只能存储字符串，需要JSON序列化
4. **错误处理**：需要处理存储失败和数据损坏的情况

### 9. 未来扩展

可以考虑的进一步优化：

1. **数据压缩**：对大型数据集进行压缩存储
2. **分片存储**：将大数据分片存储，避免单个键值过大
3. **缓存策略**：实现LRU缓存，自动清理最少使用的数据
4. **加密存储**：对敏感数据进行加密后存储

## 总结

通过采用 SessionStorage + 临时键的方案，成功解决了URL参数传递数据的问题，提升了用户体验和数据安全性，同时保持了实现的简洁性和可维护性。