from setuptools import setup, find_packages
import versioneer

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="exoorbit",
    author="Ansgar Wehrhahn",
    author_email="ansgar.wehrhahn@physics.uu.se",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AWehrhahn/ExoOrbits",
    packages=find_packages(),
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)
