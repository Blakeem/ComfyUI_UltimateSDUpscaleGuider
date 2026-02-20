"""
Tests a common workflow for UltimateSDUpscale with Guider support.
"""

import logging
import pathlib
import pytest
import torch

from setup_utils import execute
from tensor_utils import img_tensor_mae, blur
from io_utils import save_image, load_image, image_name_format
from configs import DirectoryConfig
from fixtures_images import EXT

# Image file names
CATEGORY = pathlib.Path("main_workflow")


class TestMainWorkflowGuider:
    """Integration tests for the guider-based upscaling workflow."""

    def test_base_image_matches_reference(self, base_image, test_dirs: DirectoryConfig):
        """
        Verify generated base images match reference images.
        This is just to check if the checkpoint and generation pipeline are as expected for the tests dependent on their behavior.
        """
        logger = logging.getLogger("test_base_image_matches_reference")
        image, _, _ = base_image
        test_image_dir = test_dirs.test_images
        im1 = image[0:1]
        im2 = image[1:2]

        from fixtures_images import BASE_IMAGE_1, BASE_IMAGE_2
        test_im1 = load_image(test_image_dir / BASE_IMAGE_1)
        test_im2 = load_image(test_image_dir / BASE_IMAGE_2)

        # Reduce high-frequency noise differences with gaussian blur. Using perceptual metrics are probably overkill.
        diff1 = img_tensor_mae(blur(im1), blur(test_im1))
        diff2 = img_tensor_mae(blur(im2), blur(test_im2))
        logger.info(f"Base Image Diff1: {diff1}, Diff2: {diff2}")
        assert diff1 < 0.05, "Image 1 does not match its test image."
        assert diff2 < 0.05, "Image 2 does not match its test image."

    @pytest.fixture(scope="class")
    def upscaled_image(
        self,
        base_image,
        loaded_checkpoint,
        upscale_model,
        node_classes,
        seed,
        test_dirs,
    ):
        """Generate upscaled images using the Guider node."""
        image, positive, negative = base_image
        model, clip, vae = loaded_checkpoint

        with torch.inference_mode():
            # Setup custom scheduler and sampler
            custom_scheduler = node_classes["KarrasScheduler"]
            (sigmas,) = execute(custom_scheduler, 20, 14.614642, 0.0291675, 7.0)
            (_, sigmas) = execute(node_classes["SplitSigmasDenoise"], sigmas, 0.2)

            custom_sampler = node_classes["KSamplerSelect"]
            (sampler,) = execute(custom_sampler, "dpmpp_2m")

            # Create a CFGGuider to wrap model/conditioning/cfg
            CFGGuider = node_classes["CFGGuider"]
            (guider,) = execute(
                CFGGuider,
                model=model,
                positive=positive,
                negative=negative,
                cfg=8.0,
            )

            # Run upscale using UltimateSDUpscaleGuider
            usdu = node_classes["UltimateSDUpscaleGuider"]
            (upscaled,) = usdu().upscale(
                image=image,
                guider=guider,
                sampler=sampler,
                sigmas=sigmas,
                vae=vae,
                upscale_by=2.00000004,  # Test small float difference doesn't add extra tiles
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
                tile_overlap_mode="Reprocess Overlap",
                tiled_decode=False,
            )

        # Save images
        im1_filename = "main1_sd15_upscaled" + EXT
        im2_filename = "main2_sd15_upscaled" + EXT
        sample_dir = test_dirs.sample_images
        upscaled_img1_path = sample_dir / CATEGORY / im1_filename
        upscaled_img2_path = sample_dir / CATEGORY / im2_filename
        save_image(upscaled[0], upscaled_img1_path)
        save_image(upscaled[1], upscaled_img2_path)
        # Load
        upscaled = torch.cat(
            [load_image(upscaled_img1_path), load_image(upscaled_img2_path)]
        )
        return upscaled

    def test_upscale_with_guider(
        self, upscaled_image, test_dirs: DirectoryConfig
    ):
        """Test upscaling with UltimateSDUpscaleGuider node."""
        logger = logging.getLogger("test_upscale_with_guider")
        # Verify results
        test_image_dir = test_dirs.test_images
        im1_upscaled = upscaled_image[0]
        im2_upscaled = upscaled_image[1]

        im1_filename = "main1_sd15_upscaled" + EXT
        im2_filename = "main2_sd15_upscaled" + EXT

        test_im1_upscaled = load_image(test_image_dir / CATEGORY / im1_filename)
        test_im2_upscaled = load_image(test_image_dir / CATEGORY / im2_filename)

        diff1 = img_tensor_mae(blur(im1_upscaled), blur(test_im1_upscaled))
        diff2 = img_tensor_mae(blur(im2_upscaled), blur(test_im2_upscaled))

        # This tolerance is enough to handle both cpu and gpu as the device, as well as jpg compression differences.
        logger.info(f"Diff1: {diff1}, Diff2: {diff2}")
        assert diff1 < 0.05, "Upscaled Image 1 doesn't match its test image."
        assert diff2 < 0.05, "Upscaled Image 2 doesn't match its test image."

    def test_save_sample_images(self, upscaled_image, test_dirs: DirectoryConfig):
        """Save sample images for visual inspection (optional utility test)."""
        sample_dir = test_dirs.sample_images

        im1_filename = "main1_sd15_upscaled" + EXT
        im2_filename = "main2_sd15_upscaled" + EXT

        # Save upscaled images
        save_image(upscaled_image[0], sample_dir / CATEGORY / im1_filename)
        save_image(upscaled_image[1], sample_dir / CATEGORY / im2_filename)


@pytest.mark.parametrize("batch_size", [1, 2])
class TestUpstreamMainWorkflow:
    """Integration tests for the upstream (non-guider) upscaling workflow.

    These tests reference UltimateSDUpscale which is NOT exported by this fork.
    They will be skipped if the upstream nodes are not available.
    """

    def test_upscale(
        self,
        base_image,
        loaded_checkpoint,
        upscale_model,
        node_classes,
        seed,
        batch_size,
        test_dirs: DirectoryConfig,
    ):
        """Generate upscaled images using standard workflow."""
        if "UltimateSDUpscale" not in node_classes:
            pytest.skip("Non-guider UltimateSDUpscale node not exported by this fork")

        image, positive, negative = base_image
        model, clip, vae = loaded_checkpoint

        with torch.inference_mode():
            usdu = node_classes["UltimateSDUpscale"]
            (upscaled,) = usdu().upscale(
                image=image,
                model=model,
                positive=positive,
                negative=negative,
                vae=vae,
                upscale_by=2.00000004,
                seed=seed,
                steps=5,
                cfg=8,
                sampler_name="euler",
                scheduler="normal",
                denoise=0.7,
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
                force_uniform_tiles=True,
                tiled_decode=False,
                batch_size=batch_size,
            )

        # Save images
        im1_filename = image_name_format("upscaled_image1", EXT, batch_size)
        im2_filename = image_name_format("upscaled_image2", EXT, batch_size)
        sample_dir = test_dirs.sample_images
        upscaled_img1_path = sample_dir / CATEGORY / im1_filename
        upscaled_img2_path = sample_dir / CATEGORY / im2_filename
        save_image(upscaled[0], upscaled_img1_path)
        save_image(upscaled[1], upscaled_img2_path)
        # Load to account for compression
        upscaled = torch.cat(
            [load_image(upscaled_img1_path), load_image(upscaled_img2_path)]
        )
        # Verify results
        logger = logging.getLogger("test_upscale")
        test_image_dir = test_dirs.test_images
        im1_upscaled = upscaled[0]
        im2_upscaled = upscaled[1]

        test_im1 = load_image(test_image_dir / CATEGORY / im1_filename)
        test_im2 = load_image(test_image_dir / CATEGORY / im2_filename)

        diff1 = img_tensor_mae(blur(im1_upscaled), blur(test_im1))
        diff2 = img_tensor_mae(blur(im2_upscaled), blur(test_im2))
        logger.info(f"Diff1: {diff1}, Diff2: {diff2}")
        assert diff1 < 0.01, "Upscaled Image 1 doesn't match its test image."
        assert diff2 < 0.01, "Upscaled Image 2 doesn't match its test image."


# Allow running directly for debugging
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
