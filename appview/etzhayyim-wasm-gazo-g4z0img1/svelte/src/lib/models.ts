/**
 * models.ts — Diffusion model definitions for gazo.etzhayyim.com.
 *
 * @module
 */

/** Browser-local diffusion model definition. */
export interface DiffusionModel {
	/** Model identifier. */
	id: string;
	/** Human-readable display name. */
	label: string;
	/** Base architecture. */
	arch: string;
	/** Total model size in MB. Peak VRAM = UNet only (sequential load). */
	sizeMb: number;
	/** CDN base URL for ONNX model files. */
	cdnBase: string;
	/** CLIP text encoder ONNX paths (split parts if >300MB). */
	clipParts: string[];
	/** UNet denoiser ONNX path relative to cdnBase. */
	unetPath: string;
	/** UNet external weights paths (split parts if >300MB, null if embedded). */
	unetWeightsParts: string[] | null;
	/** VAE decoder ONNX path relative to cdnBase. */
	vaePath: string;
	/** HuggingFace tokenizer model ID. */
	tokenizerModel: string;
	/** Default denoising steps. */
	defaultSteps: number;
	/** Default guidance scale (CFG). */
	defaultCfg: number;
	/** Output image [width, height]. */
	outputSize: [number, number];
}

/** Available models. ONNX files hosted on R2 (cdn.etzhayyim.com). */
export const DIFFUSION_MODELS: readonly DiffusionModel[] = [
	{
		id: 'sd15-base',
		label: 'Stable Diffusion 1.5',
		arch: 'SD 1.5',
		sizeMb: 2400,
		cdnBase: 'https://cdn.etzhayyim.com/models/sd15',
		clipParts: [
			'text_encoder/model.onnx.part0',
			'text_encoder/model.onnx.part1',
			'text_encoder/model.onnx.part2',
		],
		unetPath: 'unet/model.onnx',
		unetWeightsParts: [
			'unet/weights.pb.partaa',
			'unet/weights.pb.partab',
			'unet/weights.pb.partac',
			'unet/weights.pb.partad',
			'unet/weights.pb.partae',
			'unet/weights.pb.partaf',
			'unet/weights.pb.partag',
			'unet/weights.pb.partah',
			'unet/weights.pb.partai',
		],
		vaePath: 'vae_decoder/model.onnx',
		tokenizerModel: 'openai/clip-vit-base-patch32',
		defaultSteps: 20,
		defaultCfg: 7.5,
		outputSize: [512, 512],
	},
] as const;
