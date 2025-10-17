import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import tailwindcss from '@tailwindcss/vite'
import { resolve } from 'path'

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue(), tailwindcss()],
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
      '@/components': resolve(__dirname, 'src/components'),
      '@/views': resolve(__dirname, 'src/views'),
      '@/utils': resolve(__dirname, 'src/utils'),
      '@/api': resolve(__dirname, 'src/api'),
      '@/assets': resolve(__dirname, 'src/assets'),
      '@/router': resolve(__dirname, 'src/router'),
      '@/stores': resolve(__dirname, 'src/stores'),
      '@/constants': resolve(__dirname, 'src/constants')
    }
  },
  server: {
    port: 3000,
    open: true,
    // 开发环境禁用缓存
    headers: {
      'Cache-Control': 'no-store'
    }
  }
})
