/**
 * diffusion.svelte.ts — Svelte 5 state manager for browser diffusion inference.
 *
 * Singleton pattern: module-level `$state` survives component re-mounts.
 *
 * @module
 */

import { DIFFUSION_MODELS, type DiffusionModel } from './models.js';
import type { DiffusionWorkerMessage } from './diffusion-worker.js';

export type DiffusionState = 'idle' | 'generating' | 'error';

export interface DiffusionProgress {
	stage: string;
	step: number;
	totalSteps: number;
	label: string;
}

export interface GenerateOpts {
	prompt: string;
	negativePrompt?: string;
	steps?: number;
	cfgScale?: number;
	seed?: number;
}

/* ── Module-level singleton state ── */

let _state = $state<DiffusionState>('idle');
let _progress = $state<DiffusionProgress | null>(null);
let _selectedModelId = $state<string>(DIFFUSION_MODELS[0].id);
let _lastImageUrl = $state<string | null>(null);
let _error = $state<string | null>(null);
let _worker: Worker | null = null;

/** Spawn or reuse the diffusion Web Worker. */
function ensureWorker(): Worker {
	if (_worker) return _worker;

	_worker = new Worker(
		new URL('./diffusion-worker.ts', import.meta.url),
		{ type: 'module' },
	);

	_worker.onmessage = (ev: MessageEvent<DiffusionWorkerMessage>) => {
		const msg = ev.data;
		switch (msg.type) {
			case 'progress':
				if (msg.progress) {
					_progress = {
						stage: msg.progress.stage,
						step: msg.progress.step,
						totalSteps: msg.progress.totalSteps,
						label: msg.progress.label,
					};
				}
				break;
			case 'image':
				if (msg.image) {
					const imageData = new ImageData(new Uint8ClampedArray(msg.image.data), msg.image.width, msg.image.height);
					const canvas = new OffscreenCanvas(msg.image.width, msg.image.height);
					const ctx = canvas.getContext('2d')!;
					ctx.putImageData(imageData, 0, 0);
					canvas.convertToBlob({ type: 'image/png' }).then((blob) => {
						if (_lastImageUrl) URL.revokeObjectURL(_lastImageUrl);
						_lastImageUrl = URL.createObjectURL(blob);
					});
					_state = 'idle';
					_progress = null;
				}
				break;
			case 'error':
				_state = 'error';
				_error = msg.error ?? 'Unknown error';
				_progress = null;
				break;
		}
	};

	_worker.onerror = (err) => {
		_state = 'error';
		_error = err.message || 'Worker error';
		_progress = null;
	};

	return _worker;
}

/** Run image generation. */
async function generate(opts: GenerateOpts): Promise<void> {
	if (_state === 'generating') return;

	const model = DIFFUSION_MODELS.find((m) => m.id === _selectedModelId) ?? DIFFUSION_MODELS[0];

	_state = 'generating';
	_error = null;
	_progress = { stage: 'scheduler', step: 0, totalSteps: opts.steps ?? model.defaultSteps, label: 'Initializing...' };

	const worker = ensureWorker();

	const msg: DiffusionWorkerMessage = {
		type: 'generate',
		params: {
			prompt: opts.prompt,
			negativePrompt: opts.negativePrompt ?? '',
			steps: opts.steps ?? model.defaultSteps,
			cfgScale: opts.cfgScale ?? model.defaultCfg,
			width: model.outputSize[0],
			height: model.outputSize[1],
			seed: opts.seed ?? Math.floor(Math.random() * 2147483647),
			cdnBase: model.cdnBase,
			clipParts: [...model.clipParts],
			unetPath: model.unetPath,
			unetWeightsParts: model.unetWeightsParts ? [...model.unetWeightsParts] : null,
			vaePath: model.vaePath,
			tokenizerModel: model.tokenizerModel,
		},
	};

	worker.postMessage(msg);

	return new Promise<void>((resolve) => {
		const check = setInterval(() => {
			if (_state !== 'generating') { clearInterval(check); resolve(); }
		}, 100);
	});
}

/** Composable accessor. */
export function useDiffusion() {
	return {
		get state() { return _state; },
		get isGenerating() { return _state === 'generating'; },
		get progress() { return _progress; },
		get error() { return _error; },
		get models() { return DIFFUSION_MODELS; },
		get selectedModelId() { return _selectedModelId; },
		set selectedModelId(id: string) {
			if (DIFFUSION_MODELS.some((m) => m.id === id)) _selectedModelId = id;
		},
		get selectedModel(): DiffusionModel {
			return DIFFUSION_MODELS.find((m) => m.id === _selectedModelId) ?? DIFFUSION_MODELS[0];
		},
		get lastImageUrl() { return _lastImageUrl; },
		generate,
		reset() {
			if (_worker) { _worker.terminate(); _worker = null; }
			_state = 'idle';
			_progress = null;
			_error = null;
			if (_lastImageUrl) { URL.revokeObjectURL(_lastImageUrl); _lastImageUrl = null; }
		},
	};
}
