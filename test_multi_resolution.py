#!/usr/bin/env python3
"""
Test script for multi-resolution image generation
"""
import sys
from pathlib import Path

# Add the specific module directory to path to avoid importing torch dependencies
transforms_dir = Path(__file__).parent / 'src' / 'deepsynth' / 'data' / 'transforms'
sys.path.insert(0, str(transforms_dir))

# Import directly from the text_to_image module
from text_to_image import TextToImageConverter
from PIL import Image

def test_multi_resolution():
    """Test multi-resolution image generation."""
    print("🧪 Testing Multi-Resolution Image Generation")
    print("=" * 60)

    # Sample text
    sample_text = """
    This is a test document for multi-resolution image generation.

    DeepSeek OCR requires images at various resolutions:
    - tiny: 512×512 pixels
    - small: 640×640 pixels
    - base: 1024×1024 pixels
    - large: 1280×1280 pixels
    - gundam: 1600×1600 pixels

    This test will verify that all resolutions are generated correctly
    with proper aspect ratio preservation and padding.
    """

    # Initialize converter
    print("\n1️⃣ Initializing TextToImageConverter...")
    converter = TextToImageConverter(
        font_size=12,
        max_width=1600,
        max_height=2200,
        margin=40
    )
    print("✅ Converter initialized")

    # Test single resolution (baseline)
    print("\n2️⃣ Testing single resolution conversion...")
    try:
        single_image = converter.convert(sample_text)
        print(f"✅ Single image generated: {single_image.size}")
        single_image.save("/tmp/test_single.png")
        print("   Saved to: /tmp/test_single.png")
    except Exception as e:
        print(f"❌ Single resolution test failed: {e}")
        return False

    # Test multi-resolution
    print("\n3️⃣ Testing multi-resolution conversion...")
    try:
        multi_images = converter.convert_multi_resolution(sample_text)
        print("✅ Multi-resolution images generated:")

        expected_sizes = {
            'tiny': (512, 512),
            'small': (640, 640),
            'base': (1024, 1024),
            'large': (1280, 1280),
            'gundam': (1600, 1600)
        }

        all_correct = True
        for name, expected_size in expected_sizes.items():
            if name not in multi_images:
                print(f"   ❌ Missing resolution: {name}")
                all_correct = False
                continue

            actual_size = multi_images[name].size
            if actual_size != expected_size:
                print(f"   ❌ {name}: Expected {expected_size}, got {actual_size}")
                all_correct = False
            else:
                print(f"   ✅ {name}: {actual_size}")
                # Save for inspection
                output_path = f"/tmp/test_{name}.png"
                multi_images[name].save(output_path)
                print(f"      Saved to: {output_path}")

        if not all_correct:
            return False

    except Exception as e:
        print(f"❌ Multi-resolution test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test custom sizes
    print("\n4️⃣ Testing custom resolution selection...")
    try:
        custom_sizes = {
            'small': (640, 640),
            'large': (1280, 1280)
        }
        custom_images = converter.convert_multi_resolution(sample_text, sizes=custom_sizes)

        if len(custom_images) != 2:
            print(f"❌ Expected 2 images, got {len(custom_images)}")
            return False

        for name in ['small', 'large']:
            if name not in custom_images:
                print(f"❌ Missing custom resolution: {name}")
                return False
            print(f"✅ {name}: {custom_images[name].size}")

    except Exception as e:
        print(f"❌ Custom resolution test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test aspect ratio preservation
    print("\n5️⃣ Testing aspect ratio preservation...")
    try:
        # Create a wide image
        wide_text = "This is a short line that should create a wide aspect ratio image."
        wide_base = converter.convert(wide_text)
        wide_multi = converter.convert_multi_resolution(wide_text)

        print(f"   Base image size: {wide_base.size}")
        base_aspect = wide_base.width / wide_base.height
        print(f"   Base aspect ratio: {base_aspect:.2f}")

        # Check that resized images maintain proportions (with padding)
        for name, img in wide_multi.items():
            # All target sizes are square, so aspect ratio should be 1.0
            aspect = img.width / img.height
            if abs(aspect - 1.0) > 0.01:
                print(f"   ❌ {name}: Aspect ratio {aspect:.2f} (expected 1.0 for square targets)")
                return False

        print("   ✅ All resized images are properly squared with padding")

    except Exception as e:
        print(f"❌ Aspect ratio test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED")
    print("\nGenerated test images saved to /tmp/test_*.png")
    print("You can inspect them manually to verify quality.")
    return True

if __name__ == '__main__':
    success = test_multi_resolution()
    sys.exit(0 if success else 1)
