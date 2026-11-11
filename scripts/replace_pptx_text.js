const fs = require('fs');
const path = require('path');
const JSZip = require('jszip');

const [inputPath, outputPath, fromText, toText] = process.argv.slice(2);
if (!inputPath || !outputPath || !fromText || !toText) {
  throw new Error('Usage: node replace_pptx_text.js INPUT.pptx OUTPUT.pptx FROM TO');
}

async function main() {
  const zip = await JSZip.loadAsync(fs.readFileSync(inputPath));
  let replacements = 0;
  for (const name of Object.keys(zip.files)) {
    if (!/^ppt\/(slides|slideMasters)\/.*\.xml$/.test(name)) continue;
    const entry = zip.files[name];
    const xml = await entry.async('string');
    const matches = xml.split(fromText).length - 1;
    if (matches === 0) continue;
    replacements += matches;
    zip.file(name, xml.split(fromText).join(toText));
  }
  if (replacements === 0) throw new Error(`Text not found: ${fromText}`);
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, await zip.generateAsync({ type: 'nodebuffer', compression: 'DEFLATE' }));
  console.log(JSON.stringify({ inputPath, outputPath, replacements }));
}

main().catch((error) => {
  console.error(error.stack || error.message);
  process.exit(1);
});
