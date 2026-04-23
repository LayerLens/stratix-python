/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: false,
  // The CopilotKit runtime is used in a server route; transpile to keep
  // module resolution predictable across node + bundler.
  transpilePackages: [
    "@copilotkit/runtime",
    "@copilotkit/react-core",
    "@copilotkit/react-ui",
  ],
};

module.exports = nextConfig;
