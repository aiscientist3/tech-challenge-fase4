import tailwindcss from "@tailwindcss/vite";
import { defineConfig, loadEnv } from "vite";

export default defineConfig(({ mode }) => {
  // Empty prefix: API_PROXY_TARGET stays on the dev server, never in the bundle.
  const env = loadEnv(mode, ".", "");
  return {
    plugins: [tailwindcss()],
    server: {
      port: 5173,
      proxy: {
        "/chat": {
          target: env.API_PROXY_TARGET || "http://localhost:8000",
          changeOrigin: true,
        },
      },
    },
  };
});
