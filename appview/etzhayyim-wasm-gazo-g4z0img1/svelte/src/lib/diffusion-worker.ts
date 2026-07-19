/**
 * diffusion-worker.ts — Dedicated Web Worker for Stable Diffusion inference via ONNX Runtime WebGPU.
 *
 * **Sequential load/unload strategy**: To fit within browser WebGPU VRAM limits,
 * models are loaded one at a time: CLIP text encoder → (unload) → UNet denoiser ×N
 * steps → (unload) → VAE decoder. Peak VRAM = max(CLIP ~250MB, UNet ~1.7GB, VAE ~160MB),
 * not the sum. This trades speed for memory compatibility — same VRAM budget as Gemma 4 E2B.
 *
 * Communication with main thread via structured `DiffusionWorkerMessage` protocol.
 *
 * @module
 */

/* ── Message protocol ── */

export type DiffusionMsgType =
	| 'generate'     // main → worker: start generation
	| 'progress'     // worker → main: step progress update
	| 'image'        // worker → main: final RGBA image data
	| 'error'        // worker → main: error
	| 'ready';       // worker → main: pipeline ready

export interface DiffusionWorkerMessage {
	type: DiffusionMsgType;
	/** Generation request parameters (for 'generate' type). */
	params?: {
		prompt: string;
		negativePrompt: string;
		steps: number;
		cfgScale: number;
		width: number;
		height: number;
		seed: number;
		/** CDN base URL for ONNX files. */
		cdnBase: string;
		/** CLIP text encoder ONNX part URLs (relative to cdnBase). */
		clipParts: string[];
		/** Relative path to UNet ONNX graph. */
		unetPath: string;
		/** UNet external weights part URLs (relative to cdnBase), null if embedded. */
		unetWeightsParts: string[] | null;
		/** Relative path to VAE decoder ONNX. */
		vaePath: string;
		/** HuggingFace tokenizer model ID. */
		tokenizerModel: string;
	};
	/** Progress update (for 'progress' type). */
	progress?: {
		stage: 'clip' | 'unet' | 'vae' | 'loading-clip' | 'loading-unet' | 'loading-vae' | 'scheduler';
		step: number;
		totalSteps: number;
		label: string;
	};
	/** RGBA image data (for 'image' type). */
	image?: {
		data: Uint8ClampedArray;
		width: number;
		height: number;
	};
	/** Error message (for 'error' type). */
	error?: string;
}

/* ── ONNX Runtime lazy import ── */

let ort: typeof import('onnxruntime-web') | null = null;

async function getOrt() {
	if (!ort) {
		ort = await import('onnxruntime-web');
		// Prefer WebGPU, fallback to WASM
		ort.env.wasm.numThreads = 4;
		ort.env.wasm.simd = true;
	}
	return ort;
}

/* ── CDN model file fetcher with OPFS cache ── */

/**
 * Navigate to an OPFS subdirectory, creating intermediate dirs as needed.
 */
async function opfsDir(parts: string[]): Promise<FileSystemDirectoryHandle> {
	let dir = await navigator.storage.getDirectory();
	for (const p of parts) dir = await dir.getDirectoryHandle(p, { create: true });
	return dir;
}

/**
 * Fetch a model file from CDN (R2) with OPFS (Origin Private File System) caching.
 * Second load returns instantly from OPFS cache.
 *
 * @param url - Full URL to the ONNX model file on CDN.
 * @param onProgress - Optional progress callback (0-100).
 */
async function fetchModelFile(url: string, onProgress?: (pct: number) => void): Promise<ArrayBuffer> {
	// Derive OPFS cache key from URL path
	const cacheKey = new URL(url).pathname.replace(/^\//, '');
	const parts = cacheKey.split('/');
	const fileName = parts.pop()!;

	// Try OPFS cache first
	try {
		const dir = await opfsDir(parts);
		const fh = await dir.getFileHandle(fileName);
		const file = await fh.getFile();
		if (file.size > 0) {
			onProgress?.(100);
			return await file.arrayBuffer();
		}
	} catch {
		// Cache miss — download below
	}

	// Download from CDN
	const response = await fetch(url);
	if (!response.ok) throw new Error(`Failed to fetch ${url}: ${response.status}`);

	const contentLength = Number(response.headers.get('content-length') || 0);
	const reader = response.body!.getReader();
	const chunks: Uint8Array[] = [];
	let received = 0;

	while (true) {
		const { done, value } = await reader.read();
		if (done) break;
		chunks.push(value);
		received += value.byteLength;
		if (contentLength > 0) onProgress?.(Math.round((received / contentLength) * 100));
	}

	const buffer = new Uint8Array(received);
	let offset = 0;
	for (const chunk of chunks) {
		buffer.set(chunk, offset);
		offset += chunk.byteLength;
	}

	// Write to OPFS cache
	try {
		const dir = await opfsDir(parts);
		const fh = await dir.getFileHandle(fileName, { create: true });
		const writable = await fh.createWritable();
		await writable.write(buffer);
		await writable.close();
	} catch {
		// OPFS write failure is non-fatal
	}

	return buffer.buffer;
}

/**
 * Fetch multiple part files from CDN and concatenate into a single ArrayBuffer.
 * Used for files split to bypass R2's 300MB upload limit.
 */
async function fetchMultiPartFile(urls: string[], onProgress?: (pct: number) => void): Promise<ArrayBuffer> {
	const cacheKey = urls[0].replace(/\.part\d+$/, '').replace(/\.part[a-z]+$/, '');
	const cacheUrl = new URL(cacheKey);
	const parts = cacheUrl.pathname.replace(/^\//, '').split('/');
	const fileName = parts.pop()!;

	// Check OPFS cache for assembled file
	try {
		const dir = await opfsDir(parts);
		const fh = await dir.getFileHandle(fileName);
		const file = await fh.getFile();
		if (file.size > 0) {
			onProgress?.(100);
			return await file.arrayBuffer();
		}
	} catch { /* cache miss */ }

	// Download all parts
	const partBuffers: ArrayBuffer[] = [];
	let totalSize = 0;
	for (let i = 0; i < urls.length; i++) {
		const partPct = (p: number) => onProgress?.(Math.round(((i + p / 100) / urls.length) * 100));
		const buf = await fetchModelFile(urls[i], partPct);
		partBuffers.push(buf);
		totalSize += buf.byteLength;
	}

	// Concatenate
	const assembled = new Uint8Array(totalSize);
	let offset = 0;
	for (const buf of partBuffers) {
		assembled.set(new Uint8Array(buf), offset);
		offset += buf.byteLength;
	}

	// Cache assembled file in OPFS
	try {
		const dir = await opfsDir(parts);
		const fh = await dir.getFileHandle(fileName, { create: true });
		const writable = await fh.createWritable();
		await writable.write(assembled);
		await writable.close();
	} catch { /* non-fatal */ }

	onProgress?.(100);
	return assembled.buffer;
}

/* ── PNDM Scheduler (simplified) ── */

/**
 * Simplified PNDM (Pseudo Numerical methods for Diffusion Models) noise scheduler.
 * Generates the sigma schedule for denoising steps.
 */
function createPndmSchedule(numSteps: number): { alphasCumprod: Float32Array; timesteps: number[] } {
	const betaStart = 0.00085;
	const betaEnd = 0.012;
	const numTrainTimesteps = 1000;

	// Linear beta schedule
	const betas = new Float32Array(numTrainTimesteps);
	for (let i = 0; i < numTrainTimesteps; i++) {
		const t = i / (numTrainTimesteps - 1);
		const beta = (Math.sqrt(betaStart) + t * (Math.sqrt(betaEnd) - Math.sqrt(betaStart))) ** 2;
		betas[i] = beta;
	}

	// Cumulative product of alphas
	const alphasCumprod = new Float32Array(numTrainTimesteps);
	alphasCumprod[0] = 1 - betas[0];
	for (let i = 1; i < numTrainTimesteps; i++) {
		alphasCumprod[i] = alphasCumprod[i - 1] * (1 - betas[i]);
	}

	// Evenly spaced timesteps
	const stepSize = Math.floor(numTrainTimesteps / numSteps);
	const timesteps: number[] = [];
	for (let i = numTrainTimesteps - 1; i >= 0; i -= stepSize) {
		timesteps.push(i);
		if (timesteps.length >= numSteps) break;
	}

	return { alphasCumprod, timesteps };
}

/**
 * Single PNDM step: predict less-noisy latent from current latent + model output.
 */
function pndmStep(
	latent: Float32Array,
	modelOutput: Float32Array,
	alphasCumprod: Float32Array,
	timestep: number,
	nextTimestep: number,
): Float32Array {
	const alphaProdT = alphasCumprod[timestep];
	const alphaProdTPrev = nextTimestep >= 0 ? alphasCumprod[nextTimestep] : 1.0;

	const betaProdT = 1 - alphaProdT;
	const predOriginalSample = new Float32Array(latent.length);

	// x_0 prediction
	for (let i = 0; i < latent.length; i++) {
		predOriginalSample[i] = (latent[i] - Math.sqrt(betaProdT) * modelOutput[i]) / Math.sqrt(alphaProdT);
		// Clamp to [-1, 1]
		predOriginalSample[i] = Math.max(-1, Math.min(1, predOriginalSample[i]));
	}

	// Direction pointing to x_t
	const betaProdTPrev = 1 - alphaProdTPrev;
	const result = new Float32Array(latent.length);
	for (let i = 0; i < latent.length; i++) {
		result[i] = Math.sqrt(alphaProdTPrev) * predOriginalSample[i]
			+ Math.sqrt(betaProdTPrev) * modelOutput[i];
	}

	return result;
}

/* ── Seeded PRNG (xoshiro128**) for reproducible noise ── */

function xoshiro128ss(seed: number): () => number {
	let s0 = seed | 0;
	let s1 = (seed * 1664525 + 1013904223) | 0;
	let s2 = (s1 * 1664525 + 1013904223) | 0;
	let s3 = (s2 * 1664525 + 1013904223) | 0;
	return () => {
		const result = Math.imul(s1 * 5, 7) >>> 0;
		const t = s1 << 9;
		s2 ^= s0; s3 ^= s1; s1 ^= s2; s0 ^= s3;
		s2 ^= t;
		s3 = (s3 << 11) | (s3 >>> 21);
		return (result >>> 0) / 4294967296;
	};
}

/** Generate standard normal random Float32Array using Box-Muller transform. */
function randn(length: number, seed: number): Float32Array {
	const rng = xoshiro128ss(seed);
	const result = new Float32Array(length);
	for (let i = 0; i < length; i += 2) {
		const u1 = Math.max(1e-10, rng());
		const u2 = rng();
		const mag = Math.sqrt(-2 * Math.log(u1));
		result[i] = mag * Math.cos(2 * Math.PI * u2);
		if (i + 1 < length) result[i + 1] = mag * Math.sin(2 * Math.PI * u2);
	}
	return result;
}

/* ── CLIP BPE Tokenizer via @huggingface/transformers ── */

let _tokenizer: any = null;

/**
 * Load the CLIP BPE tokenizer via `@huggingface/transformers` AutoTokenizer.
 * Cached after first load (~2MB vocab + merges download from HuggingFace Hub).
 * Uses OPFS cache automatically via the transformers.js pipeline.
 */
async function getClipTokenizer() {
	if (_tokenizer) return _tokenizer;
	const { AutoTokenizer } = await import('@huggingface/transformers');
	_tokenizer = await AutoTokenizer.from_pretrained('openai/clip-vit-base-patch32');
	return _tokenizer;
}

/**
 * Tokenize text using the real CLIP BPE tokenizer.
 * Returns int64 tensor-compatible BigInt64Array padded to maxLength=77.
 */
async function clipTokenize(text: string, maxLength = 77): Promise<BigInt64Array> {
	const tokenizer = await getClipTokenizer();
	const encoded = tokenizer(text, {
		padding: true,
		max_length: maxLength,
		truncation: true,
	});

	// encoded.input_ids is a Tensor or typed array from transformers.js
	const ids: number[] = Array.from(encoded.input_ids.data ?? encoded.input_ids);

	// Pad to maxLength
	const PAD = 49407;
	while (ids.length < maxLength) ids.push(PAD);

	return BigInt64Array.from(ids.slice(0, maxLength).map((v: number) => BigInt(v)));
}

/* ── Main generation pipeline ── */

async function generate(params: NonNullable<DiffusionWorkerMessage['params']>) {
	const { prompt, negativePrompt, steps, cfgScale, width, height, seed,
		cdnBase, clipParts, unetPath, unetWeightsParts, vaePath, tokenizerModel } = params;
	const latentH = height / 8;
	const latentW = width / 8;
	const latentChannels = 4;
	const latentSize = latentChannels * latentH * latentW;

	const ortLib = await getOrt();
	const sessionOpts: import('onnxruntime-web').InferenceSession.SessionOptions = {
		executionProviders: ['webgpu', 'wasm'],
	};

	// ── Stage 1: CLIP Text Encoder (sequential load) ──
	postProgress('loading-clip', 0, steps, 'Downloading CLIP text encoder...');
	const clipBuffer = clipParts.length === 1
		? await fetchModelFile(`${cdnBase}/${clipParts[0]}`, (pct) => postProgress('loading-clip', 0, steps, `Downloading CLIP... ${pct}%`))
		: await fetchMultiPartFile(clipParts.map((p) => `${cdnBase}/${p}`), (pct) => postProgress('loading-clip', 0, steps, `Downloading CLIP... ${pct}%`));

	postProgress('clip', 0, steps, 'Loading CLIP tokenizer...');
	const promptTokens = await clipTokenize(prompt);
	const negTokens = await clipTokenize(negativePrompt || '');

	postProgress('clip', 0, steps, 'Running CLIP text encoder...');
	const clipSession = await ortLib.InferenceSession.create(clipBuffer, sessionOpts);

	const promptTensor = new ortLib.Tensor('int64', promptTokens, [1, 77]);
	const promptResult = await clipSession.run({ input_ids: promptTensor });
	const promptEmbedding = promptResult['last_hidden_state'];

	const negTensor = new ortLib.Tensor('int64', negTokens, [1, 77]);
	const negResult = await clipSession.run({ input_ids: negTensor });
	const negEmbedding = negResult['last_hidden_state'];

	// Unload CLIP to free VRAM
	await clipSession.release();
	postProgress('clip', 0, steps, 'CLIP done. Unloaded.');

	// ── Stage 2: UNet Denoising (sequential load, N steps) ──
	postProgress('loading-unet', 0, steps, 'Downloading UNet denoiser...');
	const unetOnnxBuffer = await fetchModelFile(
		`${cdnBase}/${unetPath}`,
		(pct) => postProgress('loading-unet', 0, steps, `Downloading UNet graph... ${pct}%`),
	);

	// UNet uses external data: model.onnx (graph, ~1MB) + weights.pb (~1.7GB FP16)
	let unetSession: import('onnxruntime-web').InferenceSession;
	if (unetWeightsParts && unetWeightsParts.length > 0) {
		postProgress('loading-unet', 0, steps, 'Downloading UNet weights (FP16)...');
		const weightsBuffer = unetWeightsParts.length === 1
			? await fetchModelFile(`${cdnBase}/${unetWeightsParts[0]}`, (pct) => postProgress('loading-unet', 0, steps, `Downloading UNet weights... ${pct}%`))
			: await fetchMultiPartFile(unetWeightsParts.map((p) => `${cdnBase}/${p}`), (pct) => postProgress('loading-unet', 0, steps, `Downloading UNet weights... ${pct}%`));

		postProgress('unet', 0, steps, 'Loading UNet into WebGPU...');
		unetSession = await ortLib.InferenceSession.create(unetOnnxBuffer, {
			...sessionOpts,
			externalData: [{ path: 'weights.pb', data: weightsBuffer }],
		} as any);
	} else {
		postProgress('unet', 0, steps, 'Loading UNet into WebGPU...');
		unetSession = await ortLib.InferenceSession.create(unetOnnxBuffer, sessionOpts);
	}

	// Initialize latent noise
	const schedule = createPndmSchedule(steps);
	let latent = randn(latentSize, seed);

	// Scale initial noise
	const initSigma = Math.sqrt((1 - schedule.alphasCumprod[schedule.timesteps[0]]) / schedule.alphasCumprod[schedule.timesteps[0]]);
	for (let i = 0; i < latent.length; i++) latent[i] *= initSigma;

	// Denoising loop
	for (let step = 0; step < steps; step++) {
		postProgress('unet', step + 1, steps, `Denoising step ${step + 1}/${steps}...`);

		const timestep = schedule.timesteps[step];
		const timestepTensor = new ortLib.Tensor('int64', BigInt64Array.from([BigInt(timestep)]), [1]);
		const latentTensor = new ortLib.Tensor('float32', latent, [1, latentChannels, latentH, latentW]);

		// CFG: run UNet twice (unconditional + conditional)
		const uncondResult = await unetSession.run({
			sample: latentTensor,
			timestep: timestepTensor,
			encoder_hidden_states: negEmbedding,
		});
		const uncondNoise = uncondResult['out_sample'].data as Float32Array;

		const condResult = await unetSession.run({
			sample: latentTensor,
			timestep: timestepTensor,
			encoder_hidden_states: promptEmbedding,
		});
		const condNoise = condResult['out_sample'].data as Float32Array;

		// Classifier-free guidance
		const guidedNoise = new Float32Array(latentSize);
		for (let i = 0; i < latentSize; i++) {
			guidedNoise[i] = uncondNoise[i] + cfgScale * (condNoise[i] - uncondNoise[i]);
		}

		// Scheduler step
		const nextTimestep = step + 1 < steps ? schedule.timesteps[step + 1] : -1;
		latent = pndmStep(latent, guidedNoise, schedule.alphasCumprod, timestep, nextTimestep);
	}

	// Unload UNet to free VRAM
	await unetSession.release();
	postProgress('unet', steps, steps, 'UNet done. Unloaded.');

	// ── Stage 3: VAE Decoder (sequential load) ──
	postProgress('loading-vae', steps, steps, 'Downloading VAE decoder...');
	const vaeBuffer = await fetchModelFile(
		`${cdnBase}/${vaePath}`,
		(pct) => postProgress('loading-vae', steps, steps, `Downloading VAE... ${pct}%`),
	);

	postProgress('vae', steps, steps, 'Decoding latent to image...');
	const vaeSession = await ortLib.InferenceSession.create(vaeBuffer, sessionOpts);

	// Scale latent for VAE
	const scaledLatent = new Float32Array(latentSize);
	for (let i = 0; i < latentSize; i++) scaledLatent[i] = latent[i] / 0.18215;

	const latentInput = new ortLib.Tensor('float32', scaledLatent, [1, latentChannels, latentH, latentW]);
	const vaeResult = await vaeSession.run({ latent_sample: latentInput });
	const decodedImage = vaeResult['sample'].data as Float32Array;

	await vaeSession.release();

	// ── Convert to RGBA Uint8ClampedArray ──
	const pixels = new Uint8ClampedArray(width * height * 4);
	for (let y = 0; y < height; y++) {
		for (let x = 0; x < width; x++) {
			const pixIdx = (y * width + x) * 4;
			// CHW → HWC, normalize [-1,1] → [0,255]
			for (let c = 0; c < 3; c++) {
				const val = decodedImage[c * height * width + y * width + x];
				pixels[pixIdx + c] = Math.max(0, Math.min(255, Math.round((val + 1) * 127.5)));
			}
			pixels[pixIdx + 3] = 255; // Alpha
		}
	}

	// Send final image back to main thread
	const msg: DiffusionWorkerMessage = {
		type: 'image',
		image: { data: pixels, width, height },
	};
	self.postMessage(msg, [pixels.buffer] as any);
}

/** Post a progress message to the main thread. */
function postProgress(stage: NonNullable<DiffusionWorkerMessage['progress']>['stage'], step: number, totalSteps: number, label: string) {
	const msg: DiffusionWorkerMessage = {
		type: 'progress',
		progress: { stage, step, totalSteps, label },
	};
	self.postMessage(msg);
}

/* ── Worker message handler ── */

self.onmessage = async (ev: MessageEvent<DiffusionWorkerMessage>) => {
	const msg = ev.data;
	if (msg.type === 'generate' && msg.params) {
		try {
			await generate(msg.params);
		} catch (err) {
			const errorMsg: DiffusionWorkerMessage = {
				type: 'error',
				error: err instanceof Error ? err.message : String(err),
			};
			self.postMessage(errorMsg);
		}
	}
};

// Signal readiness
self.postMessage({ type: 'ready' } satisfies DiffusionWorkerMessage);
