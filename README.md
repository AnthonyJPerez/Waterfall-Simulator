# Creekside — Waterfall Simulator

An ultra-realistic, real-time, web-based simulator for small water features: little waterfalls in
creeks and streams, riffles and small rapids, meant to be viewed up close.

- Hybrid physics: GPU shallow-water solver for the stream + 3-D particle fluid for water that leaves
  the bed (over ledges, down rock faces, splitting sheets, curving trickles, splashes).
- Physically based water rendering: refraction, reflection, absorption, caustics, foam, spray.
- Procedural, simulation-driven audio: bubble resonances, plunging roar, droplet plinks, spatialized.
- Interactive: add, move, rotate, scale and delete rocks and logs; change flow rate and velocity live.

Requires a WebGPU-capable browser (recent Chrome/Edge, Safari 26+, Firefox 141+).

```bash
npm install
npm run dev
```

See `docs/ARCHITECTURE.md` for the design.
