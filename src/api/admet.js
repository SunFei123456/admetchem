/**
 * ADMET预测相关API
 */
import request from './request'

/**
 * 单个SMILES的ADMET预测
 * @param {Object} data - 请求数据
 * @param {string} data.smiles - SMILES字符串
 * @param {Array<string>} data.property_ids - 属性ID列表
 * @returns {Promise}
 */
export function predictADMET(data) {
  const formData = new FormData()
  formData.append('smiles', data.smiles)
  formData.append('property_ids', data.property_ids.join(','))
  
  return request({
    url: '/admet/predict',
    method: 'post',
    data: formData,
    headers: {
      'Content-Type': 'multipart/form-data'
    }
  })
}

/**
 * 批量SMILES的ADMET预测
 * @param {Object} data - 请求数据
 * @param {Array<string>} data.smiles_list - SMILES列表
 * @param {Array<string>} data.property_ids - 属性ID列表
 * @returns {Promise}
 */
export function predictADMETBatch(data) {
  const formData = new FormData()
  formData.append('smiles_list', data.smiles_list.join('\n'))
  formData.append('property_ids', data.property_ids.join(','))
  
  return request({
    url: '/admet/predict-batch',
    method: 'post',
    data: formData,
    headers: {
      'Content-Type': 'multipart/form-data'
    }
  })
}

/**
 * SDF文件上传并进行ADMET预测
 * @param {FormData} formData - 包含文件和属性的FormData
 * @returns {Promise}
 */
export function analyzeSdfFileADMET(formData) {
  return request({
    url: '/admet/analyze-sdf',
    method: 'post',
    data: formData,
    headers: {
      'Content-Type': 'multipart/form-data'
    }
  })
}

/**
 * 获取所有支持的ADMET属性列表
 * @returns {Promise}
 */
export function getADMETProperties() {
  return request({
    url: '/admet/properties',
    method: 'get'
  })
}

