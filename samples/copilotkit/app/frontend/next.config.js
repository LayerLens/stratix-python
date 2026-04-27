/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: false,
  // Allow HMR / dev resources to load when serving on 127.0.0.1.
  // Without this, Next 16 blocks cross-origin webpack-hmr WebSocket
  // requests with no client-side error -- which can leave React in a
  // half-hydrated state on first load.
  allowedDevOrigins: ["127.0.0.1", "localhost"],
  // Hide the Next.js dev "N" badge so it doesn't sit on top of the chat UI.
  devIndicators: false,
  // The CopilotKit runtime is used in a server route; transpile to keep
  // module resolution predictable across node + bundler.
  transpilePackages: [
    "@copilotkit/runtime",
    "@copilotkit/react-core",
    "@copilotkit/react-ui",
  ],
};

module.exports = nextConfig;
