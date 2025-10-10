/**
 * Axios 请求封装
 * 统一管理 API 请求配置、拦截器等
 */
import axios from 'axios'
import message from '@/utils/message.js'

// 根据环境变量设置 baseURL
const getBaseURL = () => {
  // 开发环境
  if (import.meta.env.MODE === 'development') {
    return import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
  }
  
  // 生产环境
  if (import.meta.env.MODE === 'production') {
    return import.meta.env.VITE_API_BASE_URL || 'http://8.152.194.158:16666'
  }
  
  // 默认返回开发环境地址
  return 'http://localhost:8000'
}

// 创建 axios 实例
const request = axios.create({
  baseURL: getBaseURL() + '/api/v1',
  timeout: 30000, // 30秒超时
  headers: {
    'Content-Type': 'application/json'
  }
})

// 请求拦截器
request.interceptors.request.use(
  (config) => {
    // 在发送请求之前做些什么
    // 可以在这里添加 token 等认证信息
    // const token = localStorage.getItem('token')
    // if (token) {
    //   config.headers.Authorization = `Bearer ${token}`
    // }
    
    console.log('发送请求:', config.method?.toUpperCase(), config.url)
    return config
  },
  (error) => {
    // 对请求错误做些什么
    console.error('请求错误:', error)
    return Promise.reject(error)
  }
)

// 响应拦截器
request.interceptors.response.use(
  (response) => {
    // 2xx 范围内的状态码都会触发该函数
    const res = response.data
    
    // 根据后端返回的 code 判断请求是否成功
    if (res.code === 0) {
      // 请求成功
      return res
    } else {
      // 请求失败，显示错误消息
      const errorMsg = res.message || '请求失败'
      message.error({
        title: '错误',
        message: errorMsg,
        duration: 4000
      })
      
      return Promise.reject(new Error(errorMsg))
    }
  },
  (error) => {
    // 超出 2xx 范围的状态码都会触发该函数
    console.error('响应错误:', error)
    
    // 处理不同的错误情况
    let errorMsg = '未知错误'
    
    if (error.response) {
      // 服务器返回了错误状态码
      const status = error.response.status
      switch (status) {
        case 400:
          errorMsg = '请求参数错误'
          break
        case 401:
          errorMsg = '未授权，请登录'
          break
        case 403:
          errorMsg = '拒绝访问'
          break
        case 404:
          errorMsg = '请求的资源不存在'
          break
        case 500:
          errorMsg = '服务器内部错误'
          break
        case 502:
          errorMsg = '网关错误'
          break
        case 503:
          errorMsg = '服务不可用'
          break
        default:
          errorMsg = `服务器错误 (${status})`
      }
    } else if (error.request) {
      // 请求已发出，但没有收到响应
      if (error.code === 'ECONNABORTED') {
        errorMsg = '请求超时，请稍后重试'
      } else if (error.code === 'ERR_NETWORK') {
        errorMsg = '网络连接失败，请检查网络'
      } else {
        errorMsg = '无法连接到服务器，请检查服务器是否运行'
      }
    } else {
      // 发生了触发请求错误的问题
      errorMsg = error.message || '请求配置错误'
    }
    
    message.error({
      title: '请求失败',
      message: errorMsg,
      duration: 5000
    })
    
    return Promise.reject(error)
  }
)

export default request
