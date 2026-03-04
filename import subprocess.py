import subprocess
import sys

# List of required packages
required_packages = [
    'numpy',
    'scipy',
    'scikit-learn',
    'matplotlib',
    'pandas',
    'statsmodels',
    'jupyter',  # if you need Jupyter notebooks
    'seaborn'   # if you need seaborn for better plots
]

def install_packages(packages):
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError:
            print(f"Error installing {package}. Please check the package name or your environment.")
        else:
            print(f"{package} installed successfully.")

if __name__ == "__main__":
    install_packages(required_packages)