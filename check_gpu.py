import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # Show all logs

import tensorflow as tf

print("\n" + "="*80)
print("GPU DETECTION TEST")
print("="*80)

print(f"\nTensorFlow version: {tf.__version__}")

print("\n--- Physical Devices ---")
gpus = tf.config.list_physical_devices('GPU')
cpus = tf.config.list_physical_devices('CPU')

print(f"GPUs found: {len(gpus)}")
for gpu in gpus:
    print(f"  ✅ {gpu}")

print(f"CPUs found: {len(cpus)}")
for cpu in cpus:
    print(f"  ✅ {cpu}")

if gpus:
    print("\n--- GPU Test ---")
    try:
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            c = tf.matmul(a, b)
            print("✅ GPU computation successful!")
            print(f"   Result shape: {c.shape}")
    except Exception as e:
        print(f"❌ GPU computation failed: {e}")
else:
    print("\n⚠️  No GPU detected - will use CPU for training")
    print("   This is okay but will be slower (10-15x)")

print("\n--- Build Info ---")
print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
print(f"GPU available: {tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None) if hasattr(tf.test, 'is_gpu_available') else 'N/A (deprecated)'}")

print("\n" + "="*80)
print("RECOMMENDATION:")
if gpus:
    print("✅ GPU detected and working! Training will be 10-15x faster.")
else:
    print("⚠️  GPU not detected. You can still train but it will be slower.")
    print("   Option 1: Install CUDA toolkit from NVIDIA")
    print("   Option 2: Use Google Colab (free GPU)")
    print("   Option 3: Train on CPU with reduced dataset")
print("="*80 + "\n")
