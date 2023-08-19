from setuptools import find_packages,setup
from typing import List

def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open('requirements.txt') as f:
        requirements = f.readlines()
        requirements=[req.replace('\n','') for req in requirements]
        if '-e .' in requirements:
            requirements.remove('-e .')
    return requirements

setup(
    name="imdb_movies_review_classification",
    version='0.0.1',
    author='Aaditya',
    author_email='aadityakomerwar@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')

)