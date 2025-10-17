/**
 * 用户相关 API
 */
import request from './request.js'

/**
 * 用户注册
 * @param {Object} data - 注册数据
 * @param {string} data.email - 邮箱
 * @param {string} data.username - 用户名
 * @param {string} data.password - 密码
 * @returns {Promise}
 */
export function register(data) {
  return request({
    url: '/users/register',
    method: 'POST',
    data
  })
}

/**
 * 用户登录
 * @param {Object} data - 登录数据
 * @param {string} data.email - 邮箱
 * @param {string} data.password - 密码
 * @returns {Promise}
 */
export function login(data) {
  return request({
    url: '/users/login',
    method: 'POST',
    data
  })
}

/**
 * 获取当前用户信息
 * @returns {Promise}
 */
export function getCurrentUser() {
  return request({
    url: '/users/me',
    method: 'GET'
  })
}

/**
 * 更新当前用户信息
 * @param {Object} data - 更新数据
 * @param {string} [data.username] - 用户名
 * @param {string} [data.old_password] - 旧密码
 * @param {string} [data.new_password] - 新密码
 * @returns {Promise}
 */
export function updateCurrentUser(data) {
  return request({
    url: '/users/me',
    method: 'PUT',
    data
  })
}

/**
 * 获取用户列表
 * @param {Object} params - 查询参数
 * @param {number} [params.skip=0] - 跳过的记录数
 * @param {number} [params.limit=100] - 返回的记录数
 * @returns {Promise}
 */
export function getUserList(params) {
  return request({
    url: '/users/',
    method: 'GET',
    params
  })
}

/**
 * 获取指定用户信息
 * @param {number} userId - 用户ID
 * @returns {Promise}
 */
export function getUserById(userId) {
  return request({
    url: `/users/${userId}`,
    method: 'GET'
  })
}

