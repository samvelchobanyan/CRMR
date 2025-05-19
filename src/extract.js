// src/index.js
const path = require('path');
const fs = require('fs-extra');
const { extractTextFromPDF } = require('./extractText');

const INPUT_DIR  = path.resolve(__dirname, '..', 'PDFS');
const OUTPUT_DIR = path.resolve(__dirname, '..', 'Outputs');

async function processDirectory(srcDir, destDir) {
  const entries = await fs.readdir(srcDir, { withFileTypes: true });
  // Ensure the destination folder exists
  await fs.ensureDir(destDir);

  for (const entry of entries) {
    const srcPath  = path.join(srcDir, entry.name);
    const destPath = path.join(destDir, entry.name);

    if (entry.isDirectory()) {
      // Recurse into subfolder
      await processDirectory(srcPath, destPath);
    } else if (entry.isFile() && srcPath.toLowerCase().endsWith('.pdf')) {
      // Extract text and write .txt
      try {
        console.log(`Extracting: ${srcPath}`);
        const text = await extractTextFromPDF(srcPath);
        const outFile = destPath.replace(/\.pdf$/i, '.txt');
        await fs.writeFile(outFile, text, 'utf8');
        console.log(`   → Written: ${outFile}`);
      } catch (err) {
        console.error(`Failed on ${srcPath}:`, err.message);
      }
    }
  }
}

async function main() {
  console.log(`Reading PDFs from  ${INPUT_DIR}`);
  console.log(`Writing text to     ${OUTPUT_DIR}`);
  await processDirectory(INPUT_DIR, OUTPUT_DIR);
  console.log('Done.');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});