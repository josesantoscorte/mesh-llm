# Design Context

### Users

Engineers and developers who pool spare GPU capacity across machines for distributed LLM inference. They work in terminal-first environments, understand concepts like VRAM splits, RPC latency, and QUIC tunnels. They reach for the web console to verify mesh state, monitor node health, and debug routing — not to be wowed by marketing aesthetics.

**Job to be done**: Understand the mesh at a glance, trust it is running correctly, and intervene confidently when something looks off.

### Brand Personality

**Powerful, clean, reliable.**

The tool is technically sophisticated but never shows off. Like the best infrastructure software — it earns your trust by being precise, not loud. The README's "built with caffeine and anger" energy is personality, not aesthetic direction. The UI projects competence.

### Aesthetic Direction

**Reference**: Vercel dashboard — crisp, developer-focused, high information density without clutter.

**Visual tone**: Light-and-dark friendly. Neutral backgrounds, high-contrast type, blue accent family (`hsl(211, 82%, 55%)` primary / `hsl(204, 99%, 61%)` accent). Information at a glance. Tight spacing, sharp corners, subtle borders.

**Anti-references** (explicitly avoid):

- Generic SaaS gloss — soft gradients, pastel cards, rounded-everything
- Enterprise dashboard bloat — Grafana widget walls, excessive chrome, nested sidebars
- Playful / consumer — bright color explosions, heavy animations, non-technical framing

### Design Principles

1. **Density with clarity** — surface a lot of information (node count, VRAM %, latency, routing, model names) without feeling overwhelming. Hierarchy and spacing create order, not simplicity through omission.
2. **Trust through restraint** — remove decorative elements that don't carry data. Clean typography, consistent spacing, and calm neutrals signal that the tool knows what it's doing.
3. **Status is always readable** — mesh health, peer states, inference load, and error conditions must be scannable in under 2 seconds. Color semantics (green = serving, blue = worker, grey = client) are consistent throughout.
4. **Purposeful delight** — small moments of cleverness are welcome: animated mesh edges showing data flow, witty empty states, satisfying micro-interactions. Never at the expense of legibility or load time.
5. **Dark-first quality** — all design decisions are made in dark mode first. Contrast ratios, glow effects on edges, and color selections are calibrated for dark backgrounds. Light mode inherits, not vice versa.
