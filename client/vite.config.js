import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    strictPort: true,
    // Add history API fallback for client-side routing
    historyApiFallback: true,
    // Configure CORS if needed
    cors: true
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
    extensions: ['.js', '.jsx', '.ts', '.tsx']
  },
  root: path.resolve(__dirname, './'),
  publicDir: path.resolve(__dirname, './public'),
  build: {
    outDir: path.resolve(__dirname, './dist'),
    // Generate a SPA-friendly build
    assetsDir: 'assets',
  },
  esbuild: {
    loader: 'jsx',
    include: /\.jsx?$/,
    exclude: []
  },
  optimizeDeps: {
    esbuildOptions: {
      loader: {
        '.js': 'jsx'
      }
    }
  }
})