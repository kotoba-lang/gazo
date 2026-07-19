# etzhayyim-project-gazo — Browser SD Image Generation

**URL**: `https://gazo.etzhayyim.com`

## Architecture

Browser-local Stable Diffusion image generation via ONNX Runtime WebGPU. All inference runs in the user's browser — no server-side GPU required.

### Sequential Load/Unload Pipeline

Peak VRAM = UNet alone (~1.7GB FP16). Same budget as Gemma 4 E2B text LLM.

```
User clicks Generate
  → diffusion-worker.ts (Web Worker, off main thread)
    1. CLIP text encoder (~470MB, 3 R2 parts) → tokenize + encode → release (VRAM freed)
    2. UNet denoiser (~1.7GB FP16, 9 R2 parts + 1.2MB graph) → N denoising steps → release
    3. VAE decoder (~189MB) → latent → 512x512 RGB → release
    → RGBA ImageData → blob URL → <img>
```

### B2 CDN Model Files

ONNX files hosted on `cdn.etzhayyim.com/models/sd15/`. Files >300MB are split into 200MB parts (wrangler B2 PUT limit).

| Component | B2 Path | Size | Format |
|---|---|---|---|
| CLIP text encoder | `text_encoder/model.onnx.part{0,1,2}` | 470 MB (3 parts) | FP32 ONNX |
| UNet graph | `unet/model.onnx` | 1.2 MB | ONNX graph (external data) |
| UNet weights | `unet/weights.pb.part{aa..ai}` | 1.7 GB (9 parts) | FP16 external data |
| VAE decoder | `vae_decoder/model.onnx` | 189 MB | FP32 ONNX |
| CLIP tokenizer | `tokenizer/{merges,vocab,config,tokenizer}*` | ~1 MB | BPE vocab |

### OPFS Cache

All downloaded model files are cached in Origin Private File System. Second run loads from cache (0 network).

### CLIP BPE Tokenizer

Uses `@huggingface/transformers` `AutoTokenizer.from_pretrained('openai/clip-vit-base-patch32')` for correct BPE tokenization. Loaded in Web Worker.

## Runtime

| Item | Value |
|---|---|
| nanoid | `g4z0img1` |
| Runtime | Worker (TS Native, Workers Assets) |
| UI | Svelte 5 CSR (Vite) |
| Dependencies | `onnxruntime-web`, `@huggingface/transformers`, Tailwind CSS |
| Model source | SD 1.5 (`stable-diffusion-v1-5/stable-diffusion-v1-5`) ONNX exported via `torch.onnx.export(dynamo=False)` |

## ONNX Conversion

waiREALCN v14 (`votepurchase/waiREALCN_v14`) is **SDXL architecture** (cross_attention_dim=2048, dual CLIP, text_embeds+time_ids). Phase 1 uses SD 1.5 base. SDXL browser support = Phase 2.

Conversion script: `/tmp/convert-waireal.py` (individual component export with `dynamo=False`, FP16 UNet).

## Build & Deploy

```bash
cd 60-apps/etzhayyim-project-gazo/wasm/etzhayyim-wasm-gazo-g4z0img1/svelte
pnpm install && pnpm build
cd .. && etzhayyim deploy

# Dev
pnpm dev --port 5180
```

## Browser Requirements

- WebGPU (Chrome 113+, Edge 113+)
- ~2.4 GB download (first run, cached after)
- ~1.7 GB VRAM (UNet peak)
