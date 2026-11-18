/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  experimental: {
    agentUpgrade: "security",
  },
  async rewrites() {
    const backendUrl = (process.env.BACKEND_API_URL ?? "http://localhost:8000").replace(/\/+$/, "");
    return [
      {
        source: "/api/:path*",
        destination: `${backendUrl}/:path*`,
      },
    ];
  },
};

export default nextConfig;
