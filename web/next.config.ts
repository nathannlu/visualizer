import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  // next.config.js
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    domains: [], // Add your CDN domain here
  },
};

export default nextConfig;
