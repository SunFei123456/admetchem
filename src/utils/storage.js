/**
 * 数据存储工具函数
 * 用于管理应用中的临时数据存储，避免URL参数暴露敏感数据
 */

// 存储键前缀
const STORAGE_PREFIX = 'druglikeness_result_'

// 数据过期时间（1小时）
const MAX_AGE = 60 * 60 * 1000

/**
 * 存储结果数据到sessionStorage
 * @param {Object} data - 要存储的数据
 * @param {string} data.data - API返回的结果数据
 * @param {string} data.rule - 选择的规则
 * @param {string} data.source - 数据来源（'smiles' 或 'file'）
 * @returns {string} 存储键
 */
export const storeResultData = (data) => {
  const resultKey = `${STORAGE_PREFIX}${Date.now()}`
  const storageData = {
    ...data,
    timestamp: Date.now()
  }
  
  try {
    sessionStorage.setItem(resultKey, JSON.stringify(storageData))
    console.log('数据已存储到sessionStorage:', resultKey)
    return resultKey
  } catch (error) {
    console.error('存储数据失败:', error)
    throw new Error('数据存储失败，请稍后重试')
  }
}

/**
 * 从sessionStorage获取结果数据
 * @param {string} resultId - 存储键
 * @returns {Object|null} 存储的数据或null
 */
export const getResultData = (resultId) => {
  try {
    const storedData = sessionStorage.getItem(resultId)
    
    if (!storedData) {
      console.warn('未找到结果数据:', resultId)
      return null
    }
    
    const parsedData = JSON.parse(storedData)
    
    // 检查数据是否过期
    const currentTime = Date.now()
    const dataAge = currentTime - parsedData.timestamp
    
    if (dataAge > MAX_AGE) {
      console.warn('结果数据已过期:', resultId)
      sessionStorage.removeItem(resultId)
      return null
    }
    
    return parsedData
  } catch (error) {
    console.error('获取数据失败:', error)
    // 如果解析失败，删除损坏的数据
    sessionStorage.removeItem(resultId)
    return null
  }
}

/**
 * 删除指定的结果数据
 * @param {string} resultId - 存储键
 */
export const removeResultData = (resultId) => {
  try {
    sessionStorage.removeItem(resultId)
    console.log('已删除结果数据:', resultId)
  } catch (error) {
    console.error('删除数据失败:', error)
  }
}

/**
 * 清理所有过期的结果数据
 */
export const cleanupExpiredData = () => {
  const keys = Object.keys(sessionStorage)
  const currentTime = Date.now()
  let cleanedCount = 0
  
  keys.forEach(key => {
    if (key.startsWith(STORAGE_PREFIX)) {
      try {
        const data = JSON.parse(sessionStorage.getItem(key))
        if (data.timestamp && (currentTime - data.timestamp) > MAX_AGE) {
          sessionStorage.removeItem(key)
          cleanedCount++
        }
      } catch (error) {
        // 如果解析失败，也删除这个键
        sessionStorage.removeItem(key)
        cleanedCount++
      }
    }
  })
  
  if (cleanedCount > 0) {
    console.log(`清理了 ${cleanedCount} 个过期数据项`)
  }
}

/**
 * 获取当前存储的数据统计信息
 * @returns {Object} 统计信息
 */
export const getStorageStats = () => {
  const keys = Object.keys(sessionStorage)
  const resultKeys = keys.filter(key => key.startsWith(STORAGE_PREFIX))
  const currentTime = Date.now()
  
  let validCount = 0
  let expiredCount = 0
  
  resultKeys.forEach(key => {
    try {
      const data = JSON.parse(sessionStorage.getItem(key))
      if (data.timestamp && (currentTime - data.timestamp) > MAX_AGE) {
        expiredCount++
      } else {
        validCount++
      }
    } catch (error) {
      expiredCount++
    }
  })
  
  return {
    total: resultKeys.length,
    valid: validCount,
    expired: expiredCount
  }
}

/**
 * 清理所有结果数据（包括未过期的）
 */
export const clearAllResultData = () => {
  const keys = Object.keys(sessionStorage)
  let clearedCount = 0
  
  keys.forEach(key => {
    if (key.startsWith(STORAGE_PREFIX)) {
      sessionStorage.removeItem(key)
      clearedCount++
    }
  })
  
  console.log(`清理了 ${clearedCount} 个结果数据项`)
  return clearedCount
}