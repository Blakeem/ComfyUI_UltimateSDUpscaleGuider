# ComfyUI_UltimateSDUpscaleGuider

> **This is a fork of [ComfyUI_UltimateSDUpscale](https://github.com/ssitu/ComfyUI_UltimateSDUpscale) with added Guider support.**

[ComfyUI](https://github.com/comfyanonymous/ComfyUI) nodes for performing the image-to-image diffusion process on large images in tiles. This approach improves the details that is commonly found on upscaled images while reducing hardware requirements and maintaining an image size that the diffusion model is trained on.

## Fork Changes

This fork adds **Guider input support** to enable the use of custom guiders (such as PerpNegGuider, CFGGuider, etc.) during tiled upscaling.

### New Nodes Added

| Node | Description |
|------|-------------|
| **Ultimate SD Upscale (Guider)** | Full upscaling with guider support |
| **Ultimate SD Upscale (No Upscale, Guider)** | Tile refinement without initial upscaling, with guider support |

### What's Different

The new Guider nodes accept:
- **GUIDER** - A guider that encapsulates model, conditioning, and CFG (from PerpNegGuider, CFGGuider, etc.)
- **SAMPLER** - From KSamplerSelect node
- **SIGMAS** - From BasicScheduler or other scheduler nodes (configure denoise here)

Instead of the standard nodes' separate inputs for:
- model, positive, negative, cfg, sampler_name, scheduler, steps, denoise

### Example Workflow

```
[KSamplerSelect] ──────────────────────────┐
[BasicScheduler (with denoise)] ───────────┤
[PerpNegGuider] ───────────────────────────┤
[Load VAE] ────────────────────────────────┼──► [Ultimate SD Upscale (Guider)] ──► IMAGE
[Load Upscale Model] ──────────────────────┤
[Image] ───────────────────────────────────┘
```

### Note on Conditioning

The Guider nodes skip per-tile conditioning cropping since conditioning is internal to the guider. This works perfectly for text-based guiders (PerpNegGuider, CFGGuider). For spatial conditioning (ControlNet, GLIGEN), use the original non-Guider nodes instead.

## Installation

### Using Git
1. Git must be installed on your system. Verify by running `git -v` in a terminal.
2. Enter the following command from the terminal starting in ComfyUI/custom_nodes/
    ```
    git clone https://github.com/your-username/ComfyUI_UltimateSDUpscaleGuider
    ```
    *(Replace `your-username` with your GitHub username if you've published this fork)*

### Manual Download
1. Download the zip file by clicking the green "Code" button on the GitHub repository page and selecting "Download ZIP".
2. Create a new folder in the `ComfyUI/custom_nodes/` directory (e.g. `ComfyUI/custom_nodes/ComfyUI_UltimateSDUpscaleGuider`).
3. Extract the contents of the zip file into that folder.

### Original Version (without Guider support)
If you don't need Guider support, you can install the original version:
- Via ComfyUI Manager: Search for "UltimateSDUpscale"
- Via comfy-cli: `comfy node install comfyui_ultimatesdupscale`
- Via Git: `git clone https://github.com/ssitu/ComfyUI_UltimateSDUpscale`


## Usage

Nodes can be found in the node menu under `image/upscaling`.

Documentation for the nodes can be found in the [`js/docs/`](js/docs/) folder, or viewed within the application by right-clicking the relevant node and selecting the info icon.

Details about most of the parameters can be found [here](https://github.com/Coyote-A/ultimate-upscale-for-automatic1111/wiki/FAQ#parameters-descriptions).

Example workflows can be found in the [`example_workflows/`](example_workflows/) folder. You can also find them in the ComfyUI application under the Templates menu, scroll down the left sidebar to find the Extensions section, then selecting this repository.

## References
* **Upstream fork**: https://github.com/ssitu/ComfyUI_UltimateSDUpscale
* Ultimate Stable Diffusion Upscale script for the Automatic1111 Web UI: https://github.com/Coyote-A/ultimate-upscale-for-automatic1111
* ComfyUI: https://github.com/comfyanonymous/ComfyUI