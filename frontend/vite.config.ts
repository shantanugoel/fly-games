import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

const backend = process.env.FLY_GAMES_BACKEND || 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': { target: backend, changeOrigin: false },
      '/ws': { target: backend.replace(/^http/, 'ws'), ws: true },
    },
  },
  build: { chunkSizeWarningLimit: 1200 },
})
