<!--
  gazo.etzhayyim.com — Browser-local Stable Diffusion verification page.
  ONNX Runtime WebGPU + Sequential CLIP → UNet → VAE.
-->
<script lang="ts">
  import { useDiffusion } from './lib/diffusion.svelte.js';

  const diff = useDiffusion();

  let prompt = $state('1girl, beautiful anime face, cherry blossom, detailed eyes, high quality, masterpiece');
  let negPrompt = $state('low quality, blurry, deformed, ugly, bad anatomy');
  let steps = $state(20);
  let cfgScale = $state(7.5);
  let seed = $state(Math.floor(Math.random() * 2147483647));
  let showAdvanced = $state(false);

  /** Percentage for progress bar. */
  const progressPct = $derived(
    diff.progress && diff.progress.totalSteps > 0
      ? Math.round((diff.progress.step / diff.progress.totalSteps) * 100)
      : 0,
  );

  /** Human-readable stage label. */
  function stageLabel(stage: string): string {
    return stage
      .replace('loading-', 'Loading ')
      .replace('clip', 'CLIP')
      .replace('unet', 'UNet')
      .replace('vae', 'VAE')
      .replace('scheduler', 'Scheduler');
  }

  async function handleGenerate() {
    await diff.generate({ prompt, negativePrompt: negPrompt, steps, cfgScale, seed });
  }
</script>

<div class="mx-auto flex min-h-screen max-w-lg flex-col gap-4 p-4 pb-20">
  <!-- Header -->
  <header class="flex items-center gap-3 pt-2">
    <div class="flex h-10 w-10 items-center justify-center rounded-xl bg-purple-600 text-lg font-bold">G</div>
    <div>
      <h1 class="text-lg font-bold">Gazo</h1>
      <p class="text-xs text-neutral-400">Browser SD Image Generation — WebGPU ONNX Runtime</p>
    </div>
  </header>

  <!-- Model Info -->
  <div class="rounded-xl border border-neutral-700 bg-neutral-900 p-3">
    <div class="flex items-center justify-between">
      <span class="text-sm font-semibold">{diff.selectedModel.label}</span>
      <span class="rounded-full bg-purple-600/20 px-2 py-0.5 text-xs text-purple-400">{diff.selectedModel.arch}</span>
    </div>
    <p class="mt-1 text-xs text-neutral-400">
      Sequential load/unload: CLIP (~470MB) → UNet (~1.7GB FP16) → VAE (~189MB).
      Peak VRAM = UNet only. First run downloads ~2.4GB + compiles shaders.
    </p>
  </div>

  <!-- Prompt -->
  <div class="flex flex-col gap-1.5">
    <label class="text-xs font-semibold uppercase text-neutral-400" for="prompt">Prompt</label>
    <textarea
      id="prompt"
      class="min-h-[100px] w-full resize-none rounded-xl border border-neutral-700 bg-neutral-900 px-3 py-2.5 text-sm text-white placeholder:text-neutral-500 focus:border-purple-500 focus:outline-none"
      placeholder="Describe the image you want to generate..."
      bind:value={prompt}
      disabled={diff.isGenerating}
    ></textarea>
  </div>

  <!-- Advanced Toggle -->
  <button
    type="button"
    class="self-start text-xs text-neutral-400 hover:text-neutral-200"
    onclick={() => (showAdvanced = !showAdvanced)}
  >
    {showAdvanced ? '▾ Hide' : '▸ Show'} advanced options
  </button>

  {#if showAdvanced}
    <div class="flex flex-col gap-3 rounded-xl border border-neutral-700 bg-neutral-900 p-3">
      <div class="flex flex-col gap-1">
        <label class="text-xs font-semibold uppercase text-neutral-400" for="neg">Negative Prompt</label>
        <input
          id="neg"
          type="text"
          class="w-full rounded-lg border border-neutral-700 bg-neutral-800 px-3 py-2 text-sm text-white focus:border-purple-500 focus:outline-none"
          bind:value={negPrompt}
          disabled={diff.isGenerating}
        />
      </div>
      <div class="grid grid-cols-3 gap-3">
        <div class="flex flex-col gap-1">
          <label class="text-xs font-semibold uppercase text-neutral-400" for="steps">Steps</label>
          <input id="steps" type="number" min="1" max="50"
            class="w-full rounded-lg border border-neutral-700 bg-neutral-800 px-3 py-2 text-sm text-white tabular-nums focus:border-purple-500 focus:outline-none"
            bind:value={steps} disabled={diff.isGenerating} />
        </div>
        <div class="flex flex-col gap-1">
          <label class="text-xs font-semibold uppercase text-neutral-400" for="cfg">CFG Scale</label>
          <input id="cfg" type="number" min="1" max="20" step="0.5"
            class="w-full rounded-lg border border-neutral-700 bg-neutral-800 px-3 py-2 text-sm text-white tabular-nums focus:border-purple-500 focus:outline-none"
            bind:value={cfgScale} disabled={diff.isGenerating} />
        </div>
        <div class="flex flex-col gap-1">
          <label class="text-xs font-semibold uppercase text-neutral-400" for="seed">Seed</label>
          <div class="flex gap-1">
            <input id="seed" type="number"
              class="w-full min-w-0 rounded-lg border border-neutral-700 bg-neutral-800 px-2 py-2 text-sm text-white tabular-nums focus:border-purple-500 focus:outline-none"
              bind:value={seed} disabled={diff.isGenerating} />
            <button type="button"
              class="flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-lg border border-neutral-700 bg-neutral-800 text-neutral-400 hover:text-white"
              title="Random seed"
              onclick={() => { seed = Math.floor(Math.random() * 2147483647); }}>
              ↻
            </button>
          </div>
        </div>
      </div>
    </div>
  {/if}

  <!-- Progress -->
  {#if diff.isGenerating && diff.progress}
    <div class="rounded-xl border border-purple-500/30 bg-purple-500/10 p-3">
      <div class="flex items-center justify-between text-sm">
        <span class="font-semibold text-purple-400">{stageLabel(diff.progress.stage)}</span>
        <span class="text-xs text-purple-300">{diff.progress.step}/{diff.progress.totalSteps}</span>
      </div>
      <div class="mt-2 h-2 w-full overflow-hidden rounded-full bg-neutral-700">
        <div
          class="h-full rounded-full bg-purple-500 transition-all duration-300"
          style="width: {progressPct}%"
        ></div>
      </div>
      <p class="mt-1 text-xs text-purple-300/70">{diff.progress.label}</p>
    </div>
  {/if}

  <!-- Error -->
  {#if diff.state === 'error' && diff.error}
    <div class="rounded-xl border border-red-500/30 bg-red-500/10 p-3">
      <p class="text-sm font-semibold text-red-400">Error</p>
      <p class="mt-1 text-xs text-red-300">{diff.error}</p>
    </div>
  {/if}

  <!-- Generated Image -->
  {#if diff.lastImageUrl}
    <div class="overflow-hidden rounded-xl border border-neutral-700">
      <img src={diff.lastImageUrl} alt="SD output" class="w-full" />
    </div>
    <div class="flex gap-2">
      <a
        href={diff.lastImageUrl}
        download="gazo-{seed}.png"
        class="flex flex-1 items-center justify-center rounded-full border border-neutral-600 py-2.5 text-sm font-semibold text-white hover:bg-neutral-800"
      >Download PNG</a>
      <button
        type="button"
        class="flex-1 rounded-full border border-red-500/30 py-2.5 text-sm font-semibold text-red-400 hover:bg-red-500/10"
        onclick={() => diff.reset()}
      >Clear</button>
    </div>
  {/if}

  <!-- Generate Button -->
  <button
    class="mt-2 w-full rounded-full bg-purple-600 py-3.5 text-sm font-bold text-white active:opacity-80 disabled:opacity-40"
    disabled={diff.isGenerating || !prompt.trim()}
    onclick={handleGenerate}
  >
    {#if diff.isGenerating}
      Generating...
    {:else}
      Generate ({diff.selectedModel.outputSize[0]}x{diff.selectedModel.outputSize[1]})
    {/if}
  </button>

  <!-- WebGPU Check -->
  <p class="text-center text-xs text-neutral-500">
    Requires WebGPU (Chrome 113+, Edge 113+). All inference runs locally in your browser.
  </p>
</div>
