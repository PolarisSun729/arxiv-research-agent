// 构建产物只包含公开地址；登录凭据必须在运行时获取，测试与 SSR 无 import.meta.env 时使用同源根路径。
export const API_BASE_URL = (import.meta.env?.VITE_API_BASE_URL || '/api').replace(/\/+$/, '')
