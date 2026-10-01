import { test, expect } from "@playwright/test";
import { readFile } from "node:fs/promises";

test.beforeEach(async ({ page }) => {
  await page.goto("/");
  await expect(page.getByLabel("Grand total", { exact: true })).toHaveValue(
    "12.42",
  );
});

test("explicit sample scope, actual source, evidence and zoom", async ({
  page,
}) => {
  await expect(
    page.getByText("Sample workspace.", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Upload receipt", exact: true }).first(),
  ).toBeDisabled();
  await expect(page.locator(".doc-card")).toHaveCount(3);
  await expect(
    page.getByRole("img", { name: "Original receipt: sample-a.png" }),
  ).toBeVisible();
  await page.getByLabel("Tax", { exact: true }).focus();
  await expect(page.locator(".evidence-box")).toBeVisible();
  await expect(page.locator(".evidence-box")).toContainText("Tax");
  await page.getByRole("button", { name: "Zoom in", exact: true }).click();
  await expect(page.locator(".viewer-tools")).toContainText("125%");
  await page
    .getByRole("button", { name: "Toggle evidence highlights" })
    .click();
  await expect(page.locator(".evidence-box")).toHaveCount(0);
});

test("approval enforces arithmetic and saved edits survive refresh", async ({
  page,
}) => {
  await page.getByLabel("Grand total", { exact: true }).fill("13.42");
  await page
    .getByRole("button", { name: "Approve receipt", exact: true })
    .click();
  await expect(page.getByRole("status")).toContainText(
    "Amounts do not balance",
  );
  await expect(page.locator(".review-state")).toContainText("Needs review");
  await page.getByRole("button", { name: "Save edits", exact: true }).click();
  await expect(page.getByRole("status")).toContainText("Corrections saved");
  await page.reload();
  await expect(page.getByLabel("Grand total", { exact: true })).toHaveValue(
    "13.42",
  );
  await page.getByLabel("Grand total", { exact: true }).fill("12.42");
  await page
    .getByRole("button", { name: "Approve receipt", exact: true })
    .click();
  await expect(page.locator(".review-state")).toContainText("Approved");
  await page.getByRole("button", { name: "Activity", exact: true }).click();
  await expect(page.locator(".audit-list")).toContainText(
    "sample review approved",
  );
  await expect(page.locator(".audit-list")).toContainText(
    "sample review saved",
  );
});

test("unsaved edits prevent document changes and can be discarded", async ({
  page,
}) => {
  await page.getByLabel("Grand total", { exact: true }).fill("99.00");
  await page.locator(".doc-card").nth(1).click();
  await expect(page.getByRole("status")).toContainText("Save or discard edits");
  await expect(page.getByLabel("Grand total", { exact: true })).toHaveValue(
    "99.00",
  );
  await page.getByRole("button", { name: "Discard", exact: true }).click();
  await expect(page.getByLabel("Grand total", { exact: true })).toHaveValue(
    "12.42",
  );
  await page.locator(".doc-card").nth(1).click();
  await expect(page.getByLabel("Grand total", { exact: true })).toHaveValue(
    "24.84",
  );
  await page.getByLabel("Find a receipt").fill("does-not-exist");
  await expect(page.getByText("No matching receipts.")).toBeVisible();
});

test("JSON and CSV downloads preserve saved data and escape formulas", async ({
  page,
}, testInfo) => {
  await page.getByLabel("Item 1 description").fill("=SUM(1,2)");
  await page.getByRole("button", { name: "Save edits", exact: true }).click();
  await expect(page.getByRole("status")).toContainText("Corrections saved");
  await page.locator('summary[aria-label="Export document"]').click();
  const jsonDownload = page.waitForEvent("download");
  await page
    .getByRole("button", { name: "Download JSON", exact: true })
    .click();
  const json = await jsonDownload;
  const jsonPath = testInfo.outputPath("receipt.json");
  await json.saveAs(jsonPath);
  const data = JSON.parse(await readFile(jsonPath, "utf8"));
  expect(data.document.extraction.items[0].description).toBe("=SUM(1,2)");
  const csvDownload = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download CSV", exact: true }).click();
  const csv = await csvDownload;
  const csvPath = testInfo.outputPath("receipt.csv");
  await csv.saveAs(csvPath);
  expect(await readFile(csvPath, "utf8")).toContain("'=SUM(1,2)");
});

test("report distinguishes extraction, OCR and assistance", async ({
  page,
}) => {
  await page.getByRole("button", { name: "Model lab", exact: true }).click();
  await expect(page.getByText("LINE ACCURACY", { exact: true })).toBeVisible();
  await expect(
    page.getByText("Ground-truth text + bounding boxes", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "And when OCR has to read it?" }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", {
      name: "Text anchors assist the review pipeline.",
    }),
  ).toBeVisible();
  const downloadPromise = page.waitForEvent("download");
  await page
    .getByRole("button", { name: "Download report", exact: true })
    .click();
  expect((await downloadPromise).suggestedFilename()).toBe(
    "receiptlab-evaluation.json",
  );
});

test("mobile workspace and navigation fit the viewport", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(
    page.getByRole("img", { name: "Original receipt: sample-a.png" }),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBeTruthy();
  await page.getByRole("button", { name: "Open navigation" }).click();
  await page.getByRole("button", { name: "Model lab", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Rules meet a learned model." }),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBeTruthy();
});

test("no browser runtime errors or missing assets", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  const failures: string[] = [];
  page.on("response", (r) => {
    if (r.status() >= 400) failures.push(r.url());
  });
  await page.reload();
  await expect(page.getByLabel("Grand total", { exact: true })).toHaveValue(
    "12.42",
  );
  await page.getByRole("button", { name: "Next sample", exact: true }).click();
  await expect(page.getByLabel("Grand total", { exact: true })).toHaveValue(
    "24.84",
  );
  await page.getByRole("button", { name: "Model lab", exact: true }).click();
  expect(errors).toEqual([]);
  expect(failures).toEqual([]);
});
