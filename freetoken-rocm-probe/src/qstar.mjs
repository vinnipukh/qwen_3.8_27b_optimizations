#!/usr/bin/env node
// qstar.mjs — FreeToken q* policy projector
//
// Paper §3.2: split each decode step's m expert misses between PCIe cache
// fills (B_P) and in-place CPU execution (leftover host bandwidth B_H - B_P):
//
//   q* = m * B_P / B_H          (fills)      m-q*  (CPU exec)
//   T_layer(m) = max(q*S/B_P, (m-q)*S/(B_H-B_P))   ~ balanced at optimum
//
// Decode throughput projection (first-order):
//   tok/s ~= 1 / (L * T_layer(m_avg) + overhead)
// with m_avg = k * (1 - hit_rate), S = bytes per (layer,expert).
//
// Usage:
//   node qstar.mjs --bp 24 --bh 40 --preset qwen30a3
//   node qstar.mjs --bp 24 --bh 40 --pool-gb 13.5 --layers 48 --experts 128 \
//                  --k-active 8 --hits 0.5,0.7,0.85 --label mymodel

const args = process.argv.slice(2);
const opt = {};
for (let i = 0; i < args.length; i += 2) opt[args[i].replace(/^--/, "")] = args[i + 1];

// ---- presets: [layers, experts, k_active, expert_pool_GB] ----
const PRESETS = {
  qwen30a3: { layers: 48, experts: 128, k: 8, poolGB: 13.5,
              note: "Qwen3-30B-A3B-class MoE, MXFP4 routed experts" },
  qwen36a3: { layers: 45, experts: 128, k: 8, poolGB: 16.0,
              note: "paper's Qwen3.6-35B-A3B tier, MXFP4/NVFP4 (approx)" },
};
const p = PRESETS[opt.preset || "qwen30a3"] || PRESETS.qwen30a3;

const BP  = parseFloat(opt.bp ?? 25);            // GB/s PCIe fill bandwidth
const BH  = parseFloat(opt.bh ?? 40);            // GB/s host streaming bandwidth
const L   = parseInt(opt.layers ?? p.layers);
const E   = parseInt(opt.experts ?? p.experts);
const K   = parseInt(opt.kActive ?? opt["k-active"] ?? p.k);
const POOL= parseFloat(opt.poolGb ?? opt["pool-gb"] ?? p.poolGB); // GB
const HITS= (opt.hits ?? "0.5,0.7,0.85").split(",").map(Number);
const OVH = (parseFloat(opt.overheadMs ?? opt["overhead-ms"] ?? 2.5)) / 1000; // attn+shared+router s/tok
const LABEL = opt.label ?? opt.preset ?? "custom";

const S_GB = POOL / (L * E);                     // GB per expert (one layer)
const S = S_GB * 1e9;                            // bytes per expert

function exposedSec(m) {
  if (m <= 0) return 0;
  const qF = Math.max(1, Math.round(m * BP / BH));       // always keep >=1 fill warming cache
  const q  = Math.min(qF, m);
  const tFill = (q * S_GB) / BP;                         // s
  const resid = Math.max(BH - BP, 1e-3);                 // GB/s left while PCIe saturated
  const tCpu  = ((m - q) * S_GB) / resid;                // s
  return Math.max(tFill, tCpu);                          // concurrent branches
}

console.log(`== q* projector: ${LABEL} ==`);
console.log(`${p.note ?? ""}`);
console.log(`inputs: B_P=${BP} GB/s | B_H=${BH} GB/s | layers=${L} | experts/layer=${E}`);
console.log(`        k=${K} active | expert pool=${POOL} GB -> S=${(S / 1048576).toFixed(2)} MiB/expert`);
console.log(`q* ratio B_P/(B_P+B_H) = ${(BP / BH).toFixed(3)}  (share of misses sent to cache fill)\n`);

// per-m table for one layer
console.log("per-layer exposed miss time by #misses:");
process.stdout.write("  m     q*    T_fill   T_cpu   exposed\n");
for (let m = 1; m <= K; ++m) {
  const q = Math.min(Math.max(1, Math.round(m * BP / BH)), m);
  const tf = (q * S_GB) / BP * 1e3;
  const tc = ((m - q) * S_GB) / Math.max(BH - BP, 1e-3) * 1e3;
  console.log(`  ${String(m).padStart(2)}  ${String(q).padStart(4)}   ${tf.toFixed(3)}ms  ${tc.toFixed(3)}ms  ${(Math.max(tf, tc)).toFixed(3)}ms`);
}

console.log("\ndecode throughput projection (batch 1):");
console.log("  hit%   avg_miss/layer   tok/s     vs all-misses-on-CPU");
for (const r of HITS) {
  const mAvg = Math.round(K * (1 - r) * 100) / 100;
  const t = L * exposedSec(Math.max(1, Math.round(K * (1 - r)))) + OVH;
  const tokps = 1 / t;
  // baseline: every miss is a PCIe transfer (prediction-only systems)
  const tBase = L * (Math.max(1, Math.round(K * (1 - r))) * S_GB) / BP + OVH;
  console.log(`  ${(r * 100).toFixed(0).padStart(3)}%   ${mAvg.toFixed(2).padStart(8)}       ${tokps.toFixed(1).padStart(5)}     ${(tokps / (1 / tBase)).toFixed(2)}x`);
}
console.log("\nassumptions: batch-1 decode, balanced-branch q*, fixed hit rate across");
console.log("layers, constant per-token overhead (attention/router/shared-expert).");
console.log("First-order only — real engines add scheduling jitter; use as a bound.");
