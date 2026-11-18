import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{js,ts,jsx,tsx,mdx}", "./components/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: { extend: { colors: { canvas: "#17191d", shell: "#101113", panel: "#191b1f", cyan: "#12d9f5" } } },
  plugins: [],
};

export default config;
