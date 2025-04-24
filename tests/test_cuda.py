import cupy as cp

def test_cupy_gpu():
    try:
        print("Checking CUDA configuration...\n")
        cp.show_config()

        print("\nRunning a simple GPU test...")

        a = cp.array([1, 2, 3], dtype=cp.float32)
        b = cp.array([4, 5, 6], dtype=cp.float32)
        c = a + b

        print("Result of a + b:", c)
        print("CuPy is working!")

    except Exception as e:
        print("CuPy test failed:", e)

if __name__ == "__main__":
    test_cupy_gpu()
