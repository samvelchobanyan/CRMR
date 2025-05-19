// src/extractText.js
const fs = require('fs');
const pdf = require('pdf-parse');

async function extractTextFromPDF(inputPath) {
  const dataBuffer = fs.readFileSync(inputPath);
  const { text } = await pdf(dataBuffer);
  // Optional: further normalize whitespace here
  return text;
}

module.exports = { extractTextFromPDF };