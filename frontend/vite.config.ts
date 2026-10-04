import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// In development, /api is proxied to the FastAPI backend
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": process.env.VITE_API_PROXY ?? "http://localhost:8000",
    },
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
  },
});
