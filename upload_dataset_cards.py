#!/usr/bin/env python3
"""
Upload comprehensive dataset cards to HuggingFace Hub

This script uploads the updated README.md files to the three PitVQA datasets
on HuggingFace Hub, replacing the auto-generated cards with comprehensive
documentation including LoRA/TRL/SFT training details.
"""

import os
from huggingface_hub import HfApi, login

def upload_dataset_cards():
    """Upload dataset cards to HuggingFace Hub."""

    # Login to HuggingFace
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("❌ HF_TOKEN environment variable not set")
        print("Set it with: export HF_TOKEN=your_token")
        return False

    login(token=token)
    print("✅ Logged in to HuggingFace\n")

    api = HfApi()

    # Dataset cards to upload
    datasets = [
        {
            "repo_id": "mmrech/pitvqa-comprehensive-spatial",
            "file": "pitvqa-comprehensive-spatial_README.md",
            "name": "PitVQA Comprehensive Spatial (Primary)"
        },
        {
            "repo_id": "mmrech/pitvqa-unified-vlm",
            "file": "pitvqa-unified-vlm_README.md",
            "name": "PitVQA Unified VLM (Classification)"
        },
        {
            "repo_id": "mmrech/pitvqa-spatial-vlm",
            "file": "pitvqa-spatial-vlm_README.md",
            "name": "PitVQA Spatial VLM (Early Version)"
        }
    ]

    print("📤 Uploading dataset cards to HuggingFace Hub...\n")

    for dataset in datasets:
        print(f"📂 {dataset['name']}")
        print(f"   Repository: {dataset['repo_id']}")
        print(f"   Source: {dataset['file']}")

        try:
            # Upload the README file
            api.upload_file(
                path_or_fileobj=dataset['file'],
                path_in_repo="README.md",
                repo_id=dataset['repo_id'],
                repo_type="dataset",
                commit_message=f"Add comprehensive dataset card with LoRA/TRL/SFT documentation\n\nIncludes:\n- Training methodology (TRL + SFT + LoRA)\n- 100% ground truth fidelity validation\n- Complete reproducibility guide\n- Citation information\n- Ethical considerations"
            )
            print(f"   ✅ Uploaded successfully!")
            print(f"   🔗 View at: https://huggingface.co/datasets/{dataset['repo_id']}\n")

        except Exception as e:
            print(f"   ❌ Upload failed: {e}\n")
            return False

    print("=" * 70)
    print("🎉 All dataset cards updated successfully!")
    print("=" * 70)
    print("\n📊 Updated Datasets:")
    for dataset in datasets:
        print(f"  ✓ {dataset['name']}")
        print(f"    https://huggingface.co/datasets/{dataset['repo_id']}")

    print("\n💡 Changes include:")
    print("  ✓ Proper YAML metadata (tags, license, task_categories)")
    print("  ✓ TRL + SFT + LoRA training documentation")
    print("  ✓ 100% ground truth fidelity validation details")
    print("  ✓ Complete reproducibility guides with code examples")
    print("  ✓ Citation information (BibTeX)")
    print("  ✓ Ethical considerations and limitations")

    return True

if __name__ == "__main__":
    success = upload_dataset_cards()
    exit(0 if success else 1)
