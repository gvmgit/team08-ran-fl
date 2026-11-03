// Writes initial.safetensors for model.py's build_net(): PyTorch-default initialisation (uniform +-1/sqrt(fan_in))
// with a fixed seed, float32, keys 0/2/4 .weight/.bias, exactly the model's state_dict() keys.
// Usage: node make_initial_weights.js <out.safetensors> [seed]
const fs = require("fs");

const out = process.argv[2];
const seed = Number(process.argv[3] || 20261102);
if (!out) { console.error("usage: node make_initial_weights.js <out.safetensors> [seed]"); process.exit(1); }

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rand = mulberry32(seed);
const layers = [["0", 540, 128], ["2", 128, 64], ["4", 64, 2]];

const header = {};
const chunks = [];
let offset = 0;
for (const [name, fanIn, fanOut] of layers) {
  const bound = 1 / Math.sqrt(fanIn);
  for (const [suffix, shape] of [["weight", [fanOut, fanIn]], ["bias", [fanOut]]]) {
    const n = shape.reduce((a, b) => a * b, 1);
    const buf = Buffer.alloc(n * 4);
    for (let i = 0; i < n; i++) buf.writeFloatLE((rand() * 2 - 1) * bound, i * 4);
    header[`${name}.${suffix}`] = { dtype: "F32", shape, data_offsets: [offset, offset + buf.length] };
    chunks.push(buf);
    offset += buf.length;
  }
}
header["__metadata__"] = { format: "pt" };
let json = JSON.stringify(header);
while (Buffer.byteLength(json) % 8 !== 0) json += " ";
const lenBuf = Buffer.alloc(8);
lenBuf.writeBigUInt64LE(BigInt(Buffer.byteLength(json)));
fs.writeFileSync(out, Buffer.concat([lenBuf, Buffer.from(json), ...chunks]));

// Read it back and check.
const raw = fs.readFileSync(out);
const hlen = Number(raw.readBigUInt64LE(0));
const h = JSON.parse(raw.subarray(8, 8 + hlen).toString());
const base = 8 + hlen;
let total = 0;
for (const [k, v] of Object.entries(h)) {
  if (k === "__metadata__") continue;
  const n = v.shape.reduce((a, b) => a * b, 1);
  total += n;
  let min = Infinity, max = -Infinity, bad = 0;
  for (let i = 0; i < n; i++) {
    const x = raw.readFloatLE(base + v.data_offsets[0] + i * 4);
    if (!Number.isFinite(x)) bad++;
    if (x < min) min = x; if (x > max) max = x;
  }
  console.log(`${k.padEnd(10)} shape=${JSON.stringify(v.shape).padEnd(10)} min=${min.toFixed(4)} max=${max.toFixed(4)} nonfinite=${bad}`);
}
console.log(`file: ${out}  bytes=${raw.length}  parameters=${total}  seed=${seed}`);
if (base + offset !== raw.length) { console.error("size mismatch"); process.exit(2); }
