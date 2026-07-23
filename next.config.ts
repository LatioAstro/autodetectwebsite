import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  outputFileTracingIncludes: {
    "/*": ["data/src_data/**/*.json"],
  },
};

export default nextConfig;
