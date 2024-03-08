from setuptools import setup, find_packages


version = '1.0.0'
print(version)

setup(
    name='deepseek_vlm',
    version=version,
    description='DeekSeel-VLM',
    author='HFAiLab',
    license='MIT',
    url='https://gitlab.deepseek.com/liuwen/deepseek_vl',
    python_requires='>=3.8',
    install_requires=['torch>=2.0'],
    packages=find_packages(exclude=("images",)),
)
