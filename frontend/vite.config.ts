import { defineConfig, loadEnv } from 'vite'
import vue from '@vitejs/plugin-vue'
import { resolve } from 'path'

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
        }
      }
    }
  }
})
