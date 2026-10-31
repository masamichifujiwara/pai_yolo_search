// Render report/report.html to PDF with headless Chromium (Playwright).
//   node tools/render_report.js report/report.html out.pdf
const path = require('path');
const { chromium } = require('playwright');
(async () => {
  const [, , html, pdf] = process.argv;
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto('file://' + path.resolve(html), { waitUntil: 'networkidle' });
  await page.pdf({ path: pdf, format: 'A4', printBackground: true, preferCSSPageSize: true });
  await browser.close();
  console.log('wrote', pdf);
})();
