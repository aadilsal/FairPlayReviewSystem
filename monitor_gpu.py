import subprocess
import time
import os

def monitor_gpu(interval=2):
    """
    Monitor GPU usage in real-time
    Run this in a separate terminal while training
    """
    print("Monitoring GPU usage (Ctrl+C to stop)...\n")
    try:
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')
            print("=" * 70)
            print("GPU MONITORING - QUADRO P2000")
            print("=" * 70)

            result = subprocess.run([
                'nvidia-smi',
                '--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True)

            if result.returncode == 0:
                data = result.stdout.strip().split(',')
                if len(data) >= 5:
                    gpu_name = data[0].strip()
                    mem_used = float(data[1].strip())
                    mem_total = float(data[2].strip())
                    gpu_util = data[3].strip()
                    temp = data[4].strip()

                    mem_percent = (mem_used / mem_total) * 100

                    print(f"\nGPU: {gpu_name}")
                    print(f"Memory: {mem_used:.0f}MB / {mem_total:.0f}MB ({mem_percent:.1f}%)")
                    print(f"GPU Utilization: {gpu_util}%")
                    print(f"Temperature: {temp}°C")

                    if mem_percent > 90:
                        print("\n⚠️  WARNING: GPU memory usage > 90%!")
                        print("Consider reducing batch size")

                    if float(temp) > 80:
                        print(f"\n⚠️  WARNING: GPU temperature high ({temp}°C)")
                else:
                    print("nvidia-smi output parse error:", result.stdout)
            else:
                print("nvidia-smi not available or failed. Return code:", result.returncode)

            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped")


if __name__ == "__main__":
    monitor_gpu(interval=2)
