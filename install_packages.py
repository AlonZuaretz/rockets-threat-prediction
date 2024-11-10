import os
import sys
import subprocess


def install_packages(packages):
    """
    Install a list of packages using pip.
    """
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"{package} installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package}. Error: {e}")


def main():
    # List your packages here
    packages = [
        "numpy",  # Example package, you can add more
        "pandas",
        "torch",

    ]

    install_packages(packages)

if __name__ == "__main__":
    main()
