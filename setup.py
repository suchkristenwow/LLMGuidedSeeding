from setuptools import setup, find_packages

setup(
    name="LLMGuidedSeeding_pkg",
    version="0.1.0",
    description="A very special project",
    author="Kristen Such",
    author_email="kristen.such@colorado.edu",
    packages=find_packages(include=[
        'LLMGuidedSeeding', 
        'LLMGuidedSeeding_pkg.utils', 
        'LLMGuidedSeeding_pkg.robot_client', 
        'UI_pkg', 
        'llm_robot_client'
    ]),
    classifiers=[
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
    ],
)
