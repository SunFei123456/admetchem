/**
 * 认证相关工具函数
 */

const TOKEN_KEY = 'admet_token'
const USER_KEY = 'admet_user'

/**
 * 获取token
 * @returns {string|null}
 */
export function getToken() {
  return localStorage.getItem(TOKEN_KEY)
}

/**
 * 设置token
 * @param {string} token
 */
export function setToken(token) {
  localStorage.setItem(TOKEN_KEY, token)
}

/**
 * 移除token
 */
export function removeToken() {
  localStorage.removeItem(TOKEN_KEY)
}

/**
 * 获取用户信息
 * @returns {Object|null}
 */
export function getUser() {
  const userStr = localStorage.getItem(USER_KEY)
  try {
    return userStr ? JSON.parse(userStr) : null
  } catch (e) {
    return null
  }
}

/**
 * 设置用户信息
 * @param {Object} user
 */
export function setUser(user) {
  localStorage.setItem(USER_KEY, JSON.stringify(user))
}

/**
 * 移除用户信息
 */
export function removeUser() {
  localStorage.removeItem(USER_KEY)
}

/**
 * 清除所有认证信息
 */
export function clearAuth() {
  removeToken()
  removeUser()
}

/**
 * 检查是否已登录
 * @returns {boolean}
 */
export function isLoggedIn() {
  return !!getToken()
}

