import { defineConfig } from 'vite';

export default defineConfig({
  server: { port: 5173, strictPort: false, host: '127.0.0.1' },
  // Pre-bundle deps so the dev server never force-reloads the page mid-session (breaks headless captures).
  optimizeDeps: { include: ['tweakpane', 'wgpu-matrix'] },
  preview: { port: 4173, host: '127.0.0.1' },
  build: { target: 'es2022', sourcemap: true, chunkSizeWarningLimit: 4000 },
  test: {
    include: ['tests/**/*.test.ts', 'src/**/*.test.ts'],
    environment: 'node',
  },
} as any);
