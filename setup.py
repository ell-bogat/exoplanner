from setuptools import setup, find_packages
import re

# copied from orbitize setup.py
def get_requires():
    reqs = []
    for line in open('requirements.txt', 'r').readlines():
        reqs.append(line)
    return reqs

# also from orbitize setup.py, auto-updating version code stolen from RadVel
def get_property(prop, project):
    result = re.search(
        r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop),
        open(project + "/__init__.py").read(),
    )
    return result.group(1)

setup(
    name = "exoplanner",
    version=get_property("__version__", "exoplanner"),
    packages = find_packages(),
    install_requires = get_requires()
)