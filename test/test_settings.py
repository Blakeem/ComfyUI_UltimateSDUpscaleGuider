"""
Test for other settings included in the upscaling nodes.
"""

import logging
import pathlib
import pytest
import torch
from contextlib import nullcontext

from tensor_utils import img_tensor_mae, blur
from io_utils import save_image, load_image, image_name_format
from configs import DirectoryConfig
from fixtures_images import EXT

# Image file names
CATEGORY = pathlib.Path(pathlib.Path(__file__).stem.removeprefix("test_"))


@pytest.mark.parametrize("batch_size", [1, 2])
def test_minimal_tile_sizes(
    base_image,
    loaded_checkpoint,
    node_classes,
    seed,
    batch_size,
    test_dirs: DirectoryConfig,
):
    """Test upscaling with minimal tile sizes (non-uniform tiles)."""
    if "UltimateSDUpscale" not in node_classes:
        pytest.skip("Non-guider UltimateSDUpscale node not exported by this fork")

    image, positive, negative = base_image
    image = image[0:1]  # 1 image for simplicity
    model, clip, vae = loaded_checkpoint

    with torch.inference_mode():
        with pytest.raises(ValueError) if batch_size > 1 else nullcontext():
            usdu = node_classes["UltimateSDUpscale"]
            (upscaled,) = usdu().upscale(
                image=image,
                model=model,
                positive=positive,
                negative=negative,
                vae=vae,
                upscale_by=1.5,
                seed=seed,
                steps=5,
                cfg=8,
                sampler_name="euler",
                scheduler="normal",
                denoise=0.6,
                upscale_model=None,
                mode_type="Chess",
                tile_width=512,
                tile_height=512,
                mask_blur=8,
                tile_padding=8,
                seam_fix_mode="None",
                seam_fix_denoise=1.0,
                seam_fix_width=16,
                seam_fix_mask_blur=8,
                seam_fix_padding=4,
                force_uniform_tiles=False,  # This should trigger the ValueError for batch_size > 1
                tiled_decode=False,
                batch_size=batch_size,
            )

        if batch_size > 1:
            return  # Test passed if ValueError was raised

    # Save and reload sample image
    sample_dir = test_dirs.sample_images
    filename = CATEGORY / image_name_format("non_uniform_tiles", EXT, batch_size)
    save_image(upscaled[0], sample_dir / filename)
    upscaled = load_image(sample_dir / filename)

    # Compare with reference
    test_image_dir = test_dirs.test_images
    test_image = load_image(test_image_dir / filename)
    logger = logging.getLogger(__name__)
    diff = img_tensor_mae(blur(upscaled), blur(test_image))
    logger.info(f"{filename} MAE: {diff}")
    assert diff < 0.02, f"{filename} does not match reference (MAE {diff})"


def test_guider_context_only_batch_incompatible(
    base_image,
    loaded_checkpoint,
    upscale_model,
    node_classes,
    seed,
):
    """Test that batch_size > 1 with Context Only Overlap raises ValueError."""
    from setup_utils import execute

    image, positive, negative = base_image
    model, clip, vae = loaded_checkpoint

    with torch.inference_mode():
        # Setup guider
        custom_scheduler = node_classes["KarrasScheduler"]
        (sigmas,) = execute(custom_scheduler, 20, 14.614642, 0.0291675, 7.0)
        (_, sigmas) = execute(node_classes["SplitSigmasDenoise"], sigmas, 0.2)
        (sampler,) = execute(node_classes["KSamplerSelect"], "dpmpp_2m")
        (guider,) = execute(
            node_classes["CFGGuider"],
            model=model, positive=positive, negative=negative, cfg=8.0,
        )

        usdu = node_classes["UltimateSDUpscaleGuider"]
        with pytest.raises(ValueError, match="Context Only Overlap"):
            usdu().upscale(
                image=image[0:1],
                guider=guider,
                sampler=sampler,
                sigmas=sigmas,
                vae=vae,
                upscale_by=2.0,
                seed=seed,
                upscale_model=upscale_model,
                mode_type="Chess",
                tile_width=512,
                tile_height=512,
                mask_blur=8,
                tile_padding=32,
                seam_fix_mode="None",
                seam_fix_denoise=1.0,
                seam_fix_width=64,
                seam_fix_mask_blur=8,
                seam_fix_padding=16,
                tile_overlap_mode="Context Only Overlap",
                tiled_decode=False,
                batch_size=2,
            )
