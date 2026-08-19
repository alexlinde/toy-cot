import path from "node:path";
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Pin the workspace root (a stray lockfile in $HOME otherwise makes
  // Turbopack guess wrong and warn).
  turbopack: {
    root: path.join(__dirname, ".."),
  },
};

export default nextConfig;
