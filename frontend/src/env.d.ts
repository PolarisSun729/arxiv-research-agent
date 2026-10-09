/// <reference types="vite/client" />

// Vite 会在运行时注入 import.meta.env；这里补齐类型声明，避免 vue-tsc 把现有环境变量读取误判为类型错误。
interface ImportMetaEnv {
  readonly VITE_API_BASE_URL?: string
  readonly VITE_ENABLE_DEBUG_ROUTES?: string
}
