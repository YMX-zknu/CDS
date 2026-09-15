import importlib
import importlib.metadata
import platform
import sys


def main():
    print(f"Python {platform.python_version()} on {platform.platform()}")
    failures = []
    packages = {
        "torch": "torch", "torchvision": "torchvision", "numpy": "numpy",
        "pandas": "pandas", "matplotlib": "matplotlib", "scipy": "scipy",
        "h5py": "h5py", "Pillow": "PIL", "spikingjelly": "spikingjelly.activation_based.neuron",
    }
    for distribution, module in packages.items():
        try:
            importlib.import_module(module)
            print(f"{distribution}: {importlib.metadata.version(distribution)}")
        except Exception as exc:
            failures.append(distribution)
            print(f"{distribution}: FAILED ({exc})")
    if "torch" not in failures:
        import torch
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"PyTorch CUDA build: {torch.version.cuda}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    if failures:
        print("Install the environment described in README.md before training.")
        return 1
    print("Required imports succeeded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
