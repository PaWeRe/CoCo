import os
import sys
import argparse
import asyncio
from dotenv import load_dotenv

# Try to load environment variables from multiple possible locations
env_paths = ["../backend/.env", "./.env", "../.env", "../../.env"]

env_loaded = False
for env_path in env_paths:
    if os.path.exists(env_path):
        load_dotenv(env_path)
        print(f"Loaded environment from {env_path}")
        env_loaded = True
        break

if not env_loaded:
    print(
        "Warning: Could not find .env file. Please ensure your API keys are set in environment variables."
    )


async def run_cli():
    """Run the CLI interface for the blog collaboration tool"""
    from cli import run_cli

    await run_cli()


async def main():
    """Main entry point for the application"""
    parser = argparse.ArgumentParser(description="CoCo Blog Collaboration Tool")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with verbose logging",
    )

    args = parser.parse_args()

    # Set debug flag if needed
    if args.debug:
        os.environ["DEBUG"] = "true"
        print("[DEBUG] Running in debug mode")

    # Run the CLI interface
    await run_cli()


if __name__ == "__main__":
    asyncio.run(main())
