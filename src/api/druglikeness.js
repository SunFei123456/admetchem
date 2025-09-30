/**
 * 药物相似性评估 API
 */
import request from './request'

/**
 * 评估 SMILES 字符串的药物相似性（旧版，向后兼容）
 * @param {Object} data - 请求数据
 * @param {string} data.smiles - SMILES字符串
 * @param {Array<string>} data.rules - 规则列表，如 ['Lipinski', 'Ghose']
 * @returns {Promise}
 */
export function evaluateSmiles(data) {
  return request({
    url: '/api/v1/druglikeness/evaluate',
    method: 'post',
    data
  })
}

/**
 * 综合评估 SMILES 字符串（新版，推荐使用）
 * @param {Object} data - 请求数据
 * @param {string} data.smiles - SMILES字符串
 * @param {Array<string>} data.selected_items - 选择的项目列表
 *   - 类药性规则: 'Lipinski', 'Ghose', 'Oprea', 'Veber', 'Varma'
 *   - 分子性质: 'QED', 'SAscore', 'Fsp3', 'MCE18', 'NPscore'
 * @returns {Promise}
 */
export function evaluateComprehensive(data) {
  return request({
    url: '/api/v1/druglikeness/evaluate',
    method: 'post',
    data
  })
}

/**
 * 分析 SDF 文件的药物相似性
 * @param {FormData} formData - 包含文件的 FormData
 * @param {Object} params - 查询参数
 * @param {string} params.selected_rule - 选择的规则
 * @param {boolean} params.isomeric_smiles - 是否保留立体化学信息
 * @param {boolean} params.kekule_smiles - 是否使用凯库勒式
 * @param {boolean} params.canonical - 是否标准化
 * @returns {Promise}
 */
export function analyzeSdfFile(formData, params) {
  return request({
    url: '/api/v1/sdf/analyze',
    method: 'post',
    data: formData,
    params,
    headers: {
      'Content-Type': 'multipart/form-data'
    }
  })
}

/**
 * 转换 SDF 文件为 SMILES
 * @param {FormData} formData - 包含文件的 FormData
 * @param {Object} params - 查询参数
 * @returns {Promise}
 */
export function convertSdfToSmiles(formData, params = {}) {
  return request({
    url: '/api/v1/sdf/to-smiles',
    method: 'post',
    data: formData,
    params: {
      isomeric_smiles: params.isomeric_smiles ?? true,
      kekule_smiles: params.kekule_smiles ?? true,
      canonical: params.canonical ?? true
    },
    headers: {
      'Content-Type': 'multipart/form-data'
    }
  })
}
