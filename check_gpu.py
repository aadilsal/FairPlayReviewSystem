import torch

def check_gpu_setup():
    """
    Check if PyTorch can access the GPU
    Display GPU information
    """
    print("=" * 60)
    print("GPU SETUP VERIFICATION")
    print("=" * 60)
    
    # Check CUDA availability
    print(f"\n✓ PyTorch Version: {torch.__version__}")
    print(f"✓ CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        try:
            print(f"✓ CUDA Version: {torch.version.cuda}")
        except Exception:
            print("✓ CUDA Version: Unknown")

        try:
            print(f"✓ Number of GPUs: {torch.cuda.device_count()}")
            gpu_name = torch.cuda.get_device_name(0)
            print(f"\n✓ GPU Name: {gpu_name}")

            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"✓ Total GPU Memory: {total_memory:.2f} GB")

            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            cached = torch.cuda.memory_reserved(0) / (1024**3)
            print(f"✓ Currently Allocated: {allocated:.2f} GB")
            print(f"✓ Currently Cached: {cached:.2f} GB")
            print(f"✓ Available Memory: {total_memory - allocated:.2f} GB")

            # Test GPU with a small operation
            try:
                x = torch.rand(1000, 1000).cuda()
                y = torch.rand(1000, 1000).cuda()
                z = x @ y
                print("\n✓ GPU Test: SUCCESS - Matrix multiplication works!")
                del x, y, z
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"\n✗ GPU Test: FAILED - {str(e)}")

        except Exception as e:
            print(f"✗ Could not query GPU properties: {e}")
    else:
        print("\n✗ CUDA NOT AVAILABLE")
        print("Possible issues:")
        print("  1. PyTorch not installed with CUDA support")
        print("  2. NVIDIA drivers not installed")
        print("  3. CUDA toolkit not installed")
        print("\nInstall PyTorch with CUDA (example for CUDA 11.8):")
        print("  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

    print("=" * 60)
    return torch.cuda.is_available()


if __name__ == "__main__":
    gpu_available = check_gpu_setup()
    if not gpu_available:
        print("\n⚠️  WARNING: Training will run on CPU (very slow!)")
        print("Please install CUDA-enabled PyTorch before proceeding.")
