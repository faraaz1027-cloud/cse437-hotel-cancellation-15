"""Self-contained Colab bootstrap embedded verbatim in the five notebooks.

Uses only the standard library until the analysis dependencies are ready.
Does not modify a user's checkout, install Jupyter, or weaken scientific checks.
"""
import hashlib
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys


SOURCE_URL = 'https://github.com/faraaz1027-cloud/cse437-hotel-cancellation-15.git'
SOURCE_COMMIT = 'e01e785b78f2849b423c5be4a3fe5221a96f3e66'
ANALYSIS_PACKAGES = {
    'numpy': ('numpy', '2.3.5'),
    'pandas': ('pandas', '2.2.3'),
    'scipy': ('scipy', '1.17.0'),
    'scikit-learn': ('sklearn', '1.8.0'),
    'matplotlib': ('matplotlib', '3.10.8'),
    'seaborn': ('seaborn', '0.13.2'),
    'joblib': ('joblib', '1.5.3'),
}
PROTECTED_FILES = {
    'data/raw/hotel_bookings.csv':
        '7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06',
    'data/results/step12/final_selection.json':
        'e495222d6050492784b334110973b219dce2d3e9deaf516d311878462c6e47b6',
    'models/final_logistic_regression.joblib':
        '498112adf28d66f22f84f76101187c7a94eefeda62b7a92b45f1f9152790e097',
}


def in_colab():
    return 'google.colab' in sys.modules or bool(os.environ.get('COLAB_RELEASE_TAG'))


def installed_version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def ensure_analysis_packages():
    # Do not replace Colab's IPython, ipykernel, or notebook server.
    changed = {name for name, (_, version) in ANALYSIS_PACKAGES.items()
               if installed_version(name) != version}
    loaded_before = {name for name, (module, _) in ANALYSIS_PACKAGES.items()
                     if module in sys.modules}
    if changed:
        print('Installing pinned analysis packages. This may take a few minutes.', flush=True)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
                        *[f'{name}=={version}' for name, (_, version)
                          in ANALYSIS_PACKAGES.items()]], check=True)
    for name, (_, version) in ANALYSIS_PACKAGES.items():
        if installed_version(name) != version:
            raise RuntimeError(f'{name} installation did not reach {version}; stop and inspect pip output.')
    stale = changed & loaded_before
    for name, (module, version) in ANALYSIS_PACKAGES.items():
        loaded = sys.modules.get(module)
        if loaded is not None and getattr(loaded, '__version__', version) != version:
            stale.add(name)
    if stale:
        raise RuntimeError(
            'SETUP PAUSED: packages already loaded in memory need a restart: '
            + ', '.join(sorted(stale))
            + '. Choose Runtime > Restart session, then Runtime > Run all. '
              'Do not disconnect/delete the runtime. Installed packages are retained.')


def repository_at_or_above(folder):
    folder = Path(folder).resolve()
    for candidate in (folder, *folder.parents):
        if ((candidate / 'src/eligibility.py').is_file()
                and (candidate / 'data/splits/step6_split_plan.json').is_file()
                and (candidate / 'notebooks').is_dir()):
            return candidate
    return None


def checked_checkout(destination):
    destination = Path(destination)
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '--no-checkout', SOURCE_URL, str(destination)], check=True)
        subprocess.run(['git', '-C', str(destination), 'checkout', '--detach', SOURCE_COMMIT], check=True)
    elif not (destination / '.git').is_dir():
        raise RuntimeError('The setup folder already exists but is not a Git checkout. '
                           'Use a fresh Colab runtime; no existing files were overwritten.')
    head = subprocess.check_output(['git', '-C', str(destination), 'rev-parse', 'HEAD'], text=True).strip()
    origin = subprocess.check_output(['git', '-C', str(destination), 'remote', 'get-url', 'origin'], text=True).strip()
    if head != SOURCE_COMMIT or origin != SOURCE_URL:
        raise RuntimeError('Existing checkout has a different source/version. '
                           'Use a fresh runtime. Setup will not reset or overwrite it.')
    return destination


def verify_inputs(root):
    for relative, expected in PROTECTED_FILES.items():
        path = Path(root) / relative
        if not path.is_file():
            raise FileNotFoundError(f'Missing required project input: {relative}')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Protected input differs: {relative}. No replacement was attempted.')


def prepare_colab():
    if not in_colab():
        # The ordinary local/verification workflow already supplies its environment.
        return None
    if sys.version_info[:2] not in ((3, 12), (3, 13)):
        raise RuntimeError('These package pins need Python 3.12 or 3.13. '
                           'Use a compatible Colab runtime, or the documented local Python 3.12 setup.')
    ensure_analysis_packages()
    root = repository_at_or_above(Path.cwd())
    if root is None:
        root = checked_checkout(Path('/content/cse437-colab') / SOURCE_COMMIT)
        print('Analysis source commit:', SOURCE_COMMIT)
    else:
        print('Using the existing project working directory; no checkout reset or update.')
    verify_inputs(root)
    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    print('Setup ready:', root)
    print('CPU runtime is sufficient. No Google Drive mount or manual data upload is needed.')
    print('Development scores may vary; notebook 05 verifies the original cached test results.')
    return root


if __name__ == '__main__':
    prepare_colab()
