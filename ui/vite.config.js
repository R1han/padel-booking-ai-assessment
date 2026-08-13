import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Built output is committed so the documented run command works without Node installed.
export default defineConfig({
  plugins: [react()],
  build: { outDir: 'dist', emptyOutDir: true },
  server: { proxy: { '/api': 'http://localhost:3000' } },
})
