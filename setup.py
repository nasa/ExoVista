from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

ext_modules = cythonize(
    [
        Extension(
            "exovista.wrapImage",
            sources=["src/exovista/Image.cpp", "src/exovista/wrapImage.pyx"],
            include_dirs=["src/exovista"],
            language="c++",
        ),
        Extension(
            "exovista.wrapIntegrator",
            sources=["src/exovista/wrapIntegrator.pyx", "src/exovista/Integrator.cpp"],
            include_dirs=["src/exovista"],
            language="c++",
        ),
    ]
)

setup(
    name="exovista",
    version="2.4",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    ext_modules=ext_modules,
    package_data={"exovista": ["data/**/*"]},
    include_package_data=True,
)
