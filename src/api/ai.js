/**
 * AI 聊天 API
 */
import request from './request.js'
import { getToken } from '@/utils/auth.js'

/**
 * 发送聊天消息（从服务器配置读取 API Key）
 * @param {Object} data - 聊天数据
 * @param {string} data.model - 模型名称（可选）
 * @param {Array} data.messages - 消息列表 [{role: 'user', content: '...'}]
 * @param {number} data.temperature - 温度参数（可选）
 * @returns {Promise}
 */
export function sendChatMessage(data) {
  return request({
    url: '/ai/chat',
    method: 'POST',
    data
  })
}

/**
 * 发送聊天消息（流式响应）
 * @param {Object} data - 聊天数据
 * @param {Function} onChunk - 接收到数据块的回调 (chunk: string) => void
 * @param {Function} onError - 错误回调 (error) => void
 * @param {Function} onComplete - 完成回调 () => void
 */
export async function sendChatMessageStream(data, onChunk, onError, onComplete) {
  const baseURL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
  const token = getToken()
  
  try {
    const response = await fetch(`${baseURL}/api/v1/ai/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
      },
      body: JSON.stringify(data)
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder()

    while (true) {
      const { done, value } = await reader.read()
      
      if (done) {
        break
      }

      const chunk = decoder.decode(value, { stream: true })
      const lines = chunk.split('\n')

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const dataStr = line.slice(6)
          
          try {
            const data = JSON.parse(dataStr)
            
            if (data.error) {
              onError?.(new Error(data.error))
              return
            }
            
            if (data.done) {
              onComplete?.()
              return
            }
            
            if (data.content) {
              onChunk?.(data.content)
            }
          } catch (e) {
            // 忽略 JSON 解析错误
            console.warn('Failed to parse SSE data:', dataStr)
          }
        }
      }
    }
  } catch (error) {
    onError?.(error)
  }
}

/**
 * 获取所有 AI 服务提供商
 * @returns {Promise}
 */
export function getAIProviders() {
  return request({
    url: '/ai/providers',
    method: 'GET'
  })
}

/**
 * 获取指定提供商的模型列表
 * @param {string} provider - 提供商名称
 * @returns {Promise}
 */
export function getProviderModels(provider) {
  return request({
    url: `/ai/models/${provider}`,
    method: 'GET'
  })
}

