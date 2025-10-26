#!/usr/bin/env python3
"""
Test script for incremental HuggingFace processing with proper batch sizes.
"""

import os
import shutil
from pathlib import Path
from deepsynth.pipelines import IncrementalBuilder
from huggingface_hub import login

def test_incremental_processing():
    """Test incremental processing with proper batch sizes and limited samples."""
    
    print("🧪 Testing Incremental HuggingFace Processing")
    print("=" * 60)
    
    # Ensure HF token is available
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        print("❌ HF_TOKEN not found in environment")
        return False
    
    # Login to HuggingFace
    try:
        login(token=hf_token)
        print("✅ HuggingFace login successful")
    except Exception as e:
        print(f"❌ HuggingFace login failed: {e}")
        return False
    
    # Clean up any previous test data
    test_work_dir = Path("./test_work")
    if test_work_dir.exists():
        shutil.rmtree(test_work_dir)
        print("🧹 Cleaned up previous test data")
    
    # Create builder with proper batch size (1000 samples per batch)
    builder = IncrementalBuilder(work_dir="./test_work")
    
    print(f"📁 Work directory: {builder.work_dir}")
    print(f"📊 Initial progress: {builder.progress}")
    
    # Test with CNN/DailyMail - limit to first 2500 samples (2.5 batches)
    print("\n🎯 Testing CNN/DailyMail with 2500 samples (batch_size=1000)...")
    
    try:
        # Modify the process_dataset method to limit samples for testing
        original_process = builder.process_dataset
        
        def limited_process_dataset(name, subset, text_field, summary_field, batch_size=1000, max_samples=2500):
            """Process dataset with sample limit for testing."""
            
            if name in builder.progress['completed']:
                print(f"✅ {name} already completed")
                return

            print(f"\n📥 Processing: {name} (limited to {max_samples} samples)")

            from datasets import load_dataset
            
            # Load dataset
            if subset:
                dataset = load_dataset(name, subset, split='train')
            else:
                dataset = load_dataset(name, split='train')
            
            # Limit samples for testing
            dataset = dataset.select(range(min(max_samples, len(dataset))))
            total = len(dataset)
            print(f"    📊 Processing {total} samples with batch_size={batch_size}")

            batch_counter = len(list(builder.samples_dir.glob("batch_*.pkl")))
            batch = []
            
            for idx in range(total):
                builder.progress.update({'current': name, 'split': 'train', 'index': idx})

                example = dataset[idx]
                text, summary = builder._extract_text_and_summary(example, name, text_field, summary_field)

                if not text or not summary:
                    continue

                try:
                    image = builder.converter.convert(text)
                    batch.append({
                        'text': text, 'summary': summary, 'image': image,
                        'source_dataset': name, 'original_split': 'train', 'original_index': idx
                    })
                    builder.progress['total'] += 1
                except Exception as e:
                    print(f"      ❌ Sample {idx}: {e}")
                    continue

                # Save batch when it reaches batch_size
                if len(batch) >= batch_size:
                    builder.save_batch(batch, batch_counter)
                    batch_counter += 1
                    batch = []
                    builder.save_progress()
                    print(f"      💾 Saved batch {batch_counter-1}: {batch_size} samples")

                if (idx + 1) % 100 == 0:
                    print(f"      📈 Progress: {idx + 1}/{total} ({(idx + 1)/total*100:.1f}%)")

            # Save final batch
            if batch:
                builder.save_batch(batch, batch_counter)
                batch_counter += 1
                print(f"      💾 Saved final batch {batch_counter-1}: {len(batch)} samples")

            # Mark as completed
            builder.progress['completed'].append(name)
            builder.progress.update({'current': None, 'split': None, 'index': 0})
            builder.save_progress()
            print(f"✅ {name} completed")
        
        # Run limited processing
        limited_process_dataset('cnn_dailymail', '3.0.0', 'article', 'highlights', batch_size=1000, max_samples=2500)
        
        # Check results
        samples = builder.load_all_samples()
        print(f"\n📊 Test Results:")
        print(f"   Total samples processed: {len(samples)}")
        print(f"   Batches created: {len(list(builder.samples_dir.glob('batch_*.pkl')))}")
        
        if len(samples) > 0:
            sample = samples[0]
            print(f"   Sample text length: {len(sample['text'])} chars")
            print(f"   Sample summary length: {len(sample['summary'])} chars")
            print(f"   Sample has image: {'image' in sample}")
            print("✅ Incremental processing test PASSED")
            
            # Clean up test data
            shutil.rmtree(test_work_dir)
            print("🧹 Test cleanup completed")
            return True
        else:
            print("❌ No samples were processed")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_incremental_processing()
    if success:
        print("\n🎉 All tests passed! Ready for full pipeline execution.")
    else:
        print("\n❌ Tests failed. Please check the configuration.")