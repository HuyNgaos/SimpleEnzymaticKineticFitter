import PyInstaller.__main__

PyInstaller.__main__.run([
    'main.py',
    '--onefile',
    '--name=KineticFitter',
    '--icon=Icon.ico',  # Optional: remove this line if you don’t have an icon
    '--hidden-import=scipy._lib._uarray',
    '--hidden-import=scipy._lib.array_api_compat',
    '--hidden-import=scipy._lib.array_api_compat.numpy.fft',
    '--hidden-import=scipy.linalg._cythonized_array_utils',
    '--hidden-import=scipy.optimize._numdiff',
    '--hidden-import=scipy._cyutility',
])
