import { defineConfig, devices } from "@playwright/test";
export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  workers: 1,
  retries: process.env.CI ? 1 : 0,
  reporter: [
    ["list"],
    ["json", { outputFile: "../reports/browser-tests.json" }],
  ],
  use: {
    baseURL: "http://127.0.0.1:8810",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [
    {
      name: "chromium",
      use: {
        ...devices["Desktop Chrome"],
        channel: process.env.USE_SYSTEM_CHROME ? "chrome" : undefined,
      },
    },
  ],
  webServer: {
    command: "python -m http.server 8810 --bind 127.0.0.1 --directory ../docs",
    url: "http://127.0.0.1:8810",
    reuseExistingServer: !process.env.CI,
  },
});
