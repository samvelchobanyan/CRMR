const fs    = require('fs-extra');
const path  = require('path');
const glob  = require('glob');
const { createObjectCsvWriter } = require('csv-writer');

// 1) Columns (CSV header)
const FIELDS = [
  { id: 'product',              title: 'Product' },
  { id: 'loanCurrency',         title: 'Վարկի արժույթը' },
  { id: 'minLoanAmount',        title: 'Վարկի նվազագույն գումար' },
  { id: 'maxLoanAmount',        title: 'Վարկի առավելագույն գումար' },
  { id: 'annualNominalInterest',title: 'Տարեկան անվանական տոկոսադրույք' },
];

// 2) Helper: normalize digits
function normalizeNumber(str) {
  return str.replace(/\s+/g, '').replace(/,/g, '');
}

// 3) Extractors

// 3A) Currency: look for "Վարկի արժույթը"
function extractCurrency(text) {
  const lines = text.split(/\r?\n/);
  for (let i = 0; i < lines.length; i++) {
    if (/Վարկի\s*արժույթը/i.test(lines[i])) {
      // same line after synonyms
      const parts = lines[i].split(/Վարկի\s*արժույթը[:\s]*/i);
      if (parts[1] && parts[1].trim()) return parts[1].trim();
      // else next non-blank line
      if (lines[i+1] && lines[i+1].trim()) return lines[i+1].trim();
    }
  }
  return '';
}

// 3B) Min amount: "Վարկի նվազագույն գումար"
function extractMinLoan(text) {
  const lines = text.split(/\r?\n/);
  for (let i = 0; i < lines.length; i++) {
    if (/Վարկի\s*նվազագույն\s*գումար/i.test(lines[i])) {
      // join this line + next
      const block = lines[i] + ' ' + (lines[i+1]||'');
      const m = block.match(/([\d,]+)/);
      if (m) return normalizeNumber(m[1]);
    }
  }
  return '';
}

// 3C) Max amount: "Վարկի առավելագույն գումար"
function extractMaxLoan(text) {
  const lines = text.split(/\r?\n/);
  for (let i = 0; i < lines.length; i++) {
    if (/Վարկի\s*առավելագույն\s*գումար/i.test(lines[i])) {
      // block across up to 3 lines
      const block = 
        lines[i] + ' ' + (lines[i+1]||'') + ' ' + (lines[i+2]||'');
      const m = block.match(/([\d,]+)/);
      if (m) return normalizeNumber(m[1]);
    }
  }
  return '';
}

// 3D) Annual nominal interest: find the line "Տարեկան անվանական տոկոսադրույք" then search ahead for percentages
function extractAnnualNominal(text) {
  const lines = text.split(/\r?\n/);
  for (let i = 0; i < lines.length; i++) {
    if (/Տարեկան\s*ավանակ(?:ան)?ական\s*տոկոսադրույք/i.test(lines[i])) {
      // scan next few lines for percent patterns
      for (let j = i; j < Math.min(i + 5, lines.length); j++) {
        const m = lines[j].match(/(\d+(?:\.\d+)?%\s*(?:[\d.]+%)?)/);
        if (m) return m[1].trim();
      }
    }
  }
  return '';
}

// 4) Parse one .txt file
async function parseFile(txtPath) {
  const text = await fs.readFile(txtPath, 'utf8');
  const rel  = path.relative(path.resolve('Outputs'), txtPath);
  return {
    product: rel,
    loanCurrency: extractCurrency(text),
    minLoanAmount: extractMinLoan(text),
    maxLoanAmount: extractMaxLoan(text),
    annualNominalInterest: extractAnnualNominal(text),
  };
}

// 5) Main: walk, parse, write CSV
async function main() {
  const files = glob.sync('Outputs/**/*.txt');
  console.log(`Parsing ${files.length} files…`);
  const records = [];
  for (const f of files) {
    console.log(' →', f);
    records.push(await parseFile(f));
  }

  const csvWriter = createObjectCsvWriter({
    path: 'loan_summary.csv',
    header: FIELDS
  });

  await csvWriter.writeRecords(records);
  console.log('✅ loan_summary.csv created');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});