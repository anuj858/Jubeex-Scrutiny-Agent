import path from "node:path";
import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";

function normalizeBasePath(basePath: string | undefined): string {
  if (!basePath) return "/";
  return basePath.endsWith("/") ? basePath : `${basePath}/`;
}

/**
 * LlamaCloud often serves index.html (text/html) for JS chunks that are not
 * listed from the entry HTML. FilePreview lazy-loads pdf-preview-impl, which
 * then fails MIME checks. Inline those dynamic imports and preload every JS
 * asset from index.html so the item page can load.
 */
function preloadBuiltJsChunks(): Plugin {
  return {
    name: "preload-built-js-chunks",
    transformIndexHtml(_html, ctx) {
      if (!ctx.bundle) return [];
      return Object.keys(ctx.bundle)
        .filter((file) => file.endsWith(".js"))
        .map((file) => ({
          tag: "link",
          attrs: {
            rel: "modulepreload",
            href: file,
            crossorigin: "anonymous",
          },
          injectTo: "head" as const,
        }));
    },
  };
}

// https://vitejs.dev/config/
export default defineConfig(({}) => {
  const deploymentName = process.env.LLAMA_DEPLOY_DEPLOYMENT_NAME;
  const basePath = normalizeBasePath(
    process.env.LLAMA_DEPLOY_DEPLOYMENT_BASE_PATH,
  );
  const projectId = process.env.LLAMA_DEPLOY_PROJECT_ID;
  const port = process.env.PORT ? Number(process.env.PORT) : 3000;
  const serverPort = process.env.LLAMA_DEPLOY_SERVER_PORT;
  const baseUrl = process.env.LLAMA_CLOUD_BASE_URL;
  const apiKey = process.env.PUBLIC_LLAMA_CLOUD_API_KEY;
  return {
    plugins: [react(), preloadBuiltJsChunks()],
    resolve: {
      alias: {
        "@": path.resolve(__dirname, "./src"),
      },
    },
    server: {
      port: port,
      host: true,
      hmr: {
        port: port,
        clientPort: serverPort ? parseInt(serverPort) : undefined,
      },
    },
    build: {
      outDir: "dist",
      sourcemap: false,
      cssCodeSplit: false,
      chunkSizeWarningLimit: 4000,
      modulePreload: true,
      rollupOptions: {
        output: {
          inlineDynamicImports: true,
        },
      },
    },
    base: basePath,
    define: {
      // Primary define uses NAME
      "import.meta.env.VITE_LLAMA_DEPLOY_DEPLOYMENT_NAME": JSON.stringify(
        deploymentName
      ),
      "import.meta.env.VITE_LLAMA_DEPLOY_DEPLOYMENT_BASE_PATH": JSON.stringify(basePath),
      ...(projectId && {
        "import.meta.env.VITE_LLAMA_DEPLOY_PROJECT_ID":
          JSON.stringify(projectId),
      }),
      ...(baseUrl && {
        "import.meta.env.VITE_LLAMA_CLOUD_BASE_URL": JSON.stringify(baseUrl),
      }),
      ...(apiKey && {
        "import.meta.env.VITE_LLAMA_CLOUD_API_KEY": JSON.stringify(apiKey),
      }),
    },
  };
});
