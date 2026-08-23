import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// dev: `npm run dev` proxies API calls to the python service on :8000
// build: `npm run build` writes ui/dist, which the FastAPI app serves at /
export default defineConfig({
  plugins: [react()],
  base: "/",
  build: { outDir: "dist", emptyOutDir: true },
  server: {
    port: 5173,
    proxy: {
      "/recommend": "http://localhost:8000",
      "/health": "http://localhost:8000",
      "/ready": "http://localhost:8000",
    },
  },
});
