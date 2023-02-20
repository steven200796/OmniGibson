# read the contents of your README file
import subprocess
import sys
import os
import re
import platform
import shutil
from distutils.version import LooseVersion
from setuptools import find_packages, setup, Extension
from setuptools.command.build_ext import build_ext

use_clang = False
here = os.path.abspath(os.path.dirname(__file__))

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r"version\s*([\d.]+)", out.decode()).group(1))
            if cmake_version < "3.1.0":
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")
                
        for ext in self.extensions:
            self.build_extension(ext)

        if platform.system() == "Windows":
            vr_dir = os.path.join(here, "omnigibson", "vr")
            release_dir = os.path.join(vr_dir, "Release")
            for f in os.listdir(release_dir):
                shutil.copy(os.path.join(release_dir, f), vr_dir)

            shutil.rmtree(release_dir)
            vr_dll = os.path.join(here, "omnigibson", "vr", "deps", "openvr", "bin", "win64", "openvr_api.dll")
            sr_ani_dir = os.path.join(here, "omnigibson", "vr", "deps", "sranipal", "bin")
            shutil.copy(vr_dll, vr_dir)

            for f in os.listdir(sr_ani_dir):
                if f.endswith("dll"):
                    shutil.copy(os.path.join(sr_ani_dir, f), vr_dir)


    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + os.path.join(extdir, "omnigibson", "vr"),
            "-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=" + os.path.join(extdir, "omnigibson", "vr", "build"),
            "-DPYTHON_EXECUTABLE=" + sys.executable,
        ]

        if os.getenv("USE_VR"):
            cmake_args += ["-DUSE_VR=TRUE"]

        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        if platform.system() == "Windows":
            cmake_args += ["-DCMAKE_LIBRARY_OUTPUT_DIRECTORY{}={}".format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ["-A", "x64"]
            build_args += ["--", "/m"]
        else:
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
            build_args += ["--", "-j2"]

        env = os.environ.copy()
        env["CXXFLAGS"] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get("CXXFLAGS", ""), self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=self.build_temp)


with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    lines = f.readlines()

# remove images from README
lines = [x for x in lines if ".png" not in x]
long_description = "".join(lines)

setup(
    name="omnigibson",
    version="0.0.5",
    author="Stanford University",
    long_description_content_type="text/markdown",
    long_description=long_description,
    url="https://github.com/StanfordVL/OmniGibson",
    zip_safe=False,
    packages=find_packages(),
    install_requires=[
        "gym>=0.26",
        "numpy>=1.20.0",
        "GitPython",
        "transforms3d>=0.3.1",
        "networkx>=2.0",
        "PyYAML",
        "addict",
        "ipython",
        "future",
        "trimesh",
        "h5py",
        "cryptography",
        "bddl>=3.0.0b1",
        "opencv-python",
        "nest_asyncio",
    ],
    ext_modules=[CMakeExtension("VRSys", sourcedir="omnigibson/vr")],
    cmdclass=dict(build_ext=CMakeBuild),
    tests_require=[],
    python_requires=">=3",
    package_data={"": ["omnigibson/global_config.yaml"]},
    include_package_data=True,
)  # yapf: disable
