// Tokenizes text with tokenizers.js (@huggingface/tokenizers).
//
// usage: node encode.mjs MODULE TOKENIZER_DIR
// MODULE is the package's ESM entry: the installed npm package or a checkout's dist/tokenizers.mjs.
// Reads one JSON array of strings per line on stdin and writes one JSON array of token id arrays per line
// on stdout, so the tokenizer is loaded once per run.
import fs from "node:fs";
import path from "node:path";
import readline from "node:readline";
import { pathToFileURL } from "node:url";

const [modulePath, dir] = process.argv.slice(2);
const { Tokenizer } = await import(pathToFileURL(path.resolve(modulePath)).href);
const read = (name) => {
  const file = path.join(dir, name);
  return fs.existsSync(file) ? JSON.parse(fs.readFileSync(file, "utf-8")) : {};
};
const tokenizer = new Tokenizer(read("tokenizer.json"), read("tokenizer_config.json"));

const lines = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
for await (const line of lines) {
  if (!line) continue;
  const texts = JSON.parse(line);
  const ids = texts.map((t) => Array.from(tokenizer.encode(t, { add_special_tokens: false }).ids, Number));
  process.stdout.write(JSON.stringify(ids) + "\n");
}
