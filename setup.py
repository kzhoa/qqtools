import setuptools

setuptools.setup(
    name="qqtools",
    version="1.0.13",
    author="qq",
    author_email="qq@x1q.cc",
    description="A small tool package for qq",
    long_description="A small tool package for qq",
    long_description_content_type="text/markdown",
    url="https://github.com/kzhoa/qqtools",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    license="MIT",  # SPDX expression
    license_files=["LICENSE*"],
    python_requires=">=3.10",
    install_requires=["torch>=2.0"],
)
