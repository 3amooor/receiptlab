import { execFileSync } from "node:child_process";
execFileSync(
  process.execPath,
  [
    "node_modules/vite/bin/vite.js",
    "build",
    "--outDir",
    "../docs",
    "--emptyOutDir",
    "false",
  ],
  {
    stdio: "inherit",
    env: { ...process.env, VITE_DEMO_MODE: "1", VITE_BASE_PATH: "./" },
  },
);
