from setuptools import setup, find_packages


version = '1.0.0'
print(version)

setup(
    name='deepseek_vl',
    version=version,
    description='DeekSeel-VL',
    author='DeepSeek-AI',
    license='MIT',
    url='https://github.com/deepseek-ai/DeepSeek-VL',
    python_requires='>=3.8',
    install_requires=['torch>=2.0'],
    packages=find_packages(exclude=("images",)),
)
