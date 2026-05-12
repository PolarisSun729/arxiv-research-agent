import axios from 'axios'

const request = axios.create({
  baseURL: '/api',
  timeout: 600000
})

request.interceptors.response.use(
  response => response.data,
  error => {
    console.error('API request error:', error)
    return Promise.reject(error)
  }
)

export default request
