from setuptools import setup, find_packages

setup(
    name="coco-langgraph",
    version="0.1.0",
    description="CoCo Blog Post Collaboration Tool using LangGraph",
    author="CoCo Team",
    author_email="info@cocoai.com",
    packages=find_packages(),
    install_requires=[
        "langchain",
        "langchain-openai",
        "langchain-anthropic",
        "langgraph>=0.0.20",
        "langsmith",
        "python-dotenv",
        "pydantic",
        "typing-extensions",
        "ipython",  # For visualizations
    ],
    entry_points={
        "console_scripts": [
            "coco-blog=langgraph.main:main",
        ],
    },
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
