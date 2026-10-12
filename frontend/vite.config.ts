import { defineConfig, loadEnv } from 'vite'
import vue from '@vitejs/plugin-vue'
import { resolve } from 'path'
import { Agent } from 'http'

export default defineConfig(({ mode }) => {
  const publicEnv = loadEnv(mode, __dirname, 'VITE_')
  // VITE_ 配置会进入浏览器构建产物，拒绝沿用旧方案中的共享密钥或供应商凭据。
  if (Object.keys(publicEnv).some(name => /(?:API_?KEY|TOKEN|SECRET|PASSWORD)/i.test(name))) {
    throw new Error('VITE_ 环境变量不能包含密钥，请移除凭据配置并在登录页输入访问密钥。')
  }
  return {
    plugins: [vue()],
    resolve: {
      alias: {
        '@': resolve(__dirname, 'src')
      }
    },
    server: {
      proxy: {
        '/api': {
          target: 'http://127.0.0.1:8001',
          changeOrigin: true,
          // 后端已挂载 /api 前缀，代理必须保留原路径。
          // http-proxy 默认每个请求都带 Connection: close；Windows 上 uvicorn 按 close 关连接时会截断响应
          // （常卡在 64KiB 或 ECONNRESET），前端请求随之挂起。复用 keep-alive 连接可避开；
          // 空闲超时 4s 短于 uvicorn 默认的 5s keep-alive，避免复用到后端已关闭的连接。
          agent: new Agent({ keepAlive: true, timeout: 4000 })
        }
      }
    }
  }
})
