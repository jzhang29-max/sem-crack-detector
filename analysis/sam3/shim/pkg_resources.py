"""Minimal pkg_resources shim.

setuptools >= 81 removed pkg_resources; Meta's sam3 package still imports it, but only
for resource_filename() to locate a bundled BPE vocab. This provides exactly that,
backed by importlib.resources. Lives on PYTHONPATH for this experiment only — nothing
is installed into the user's site-packages.
"""
import os
import importlib


def resource_filename(package, resource):
    mod = importlib.import_module(package)
    base = os.path.dirname(os.path.abspath(mod.__file__))
    return os.path.join(base, resource)


def resource_exists(package, resource):
    return os.path.exists(resource_filename(package, resource))


class DistributionNotFound(Exception):
    pass


def get_distribution(name):
    raise DistributionNotFound(name)
