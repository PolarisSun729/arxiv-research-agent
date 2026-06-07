import axios from 'axios'
import { ApiError, normalizeApiError } from './errors'

const request = axios.create({
  baseURL: '/api',
  timeout: 600000
})

request.interceptors.response.use(
  response => response.data,
  error => {
    console.error('API request error:', error)
    // API 层统一把后端错误契约转成 ApiError，页面只需要按 code 展示提示。
    return Promise.reject(new ApiError(normalizeApiError(error, '请求失败，请稍后重试。')))
  }
)

export default request
