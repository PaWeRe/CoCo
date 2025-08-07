#!/usr/bin/env python3
"""
Graph visualization script for CoCo Blog Collaboration with LangGraph.
This script generates a visual representation of the StateGraph.
"""

import os
import sys
from dotenv import load_dotenv

# Try to import optional display modules
try:
    from IPython.display import Image, display

    can_display = True
except ImportError:
    can_display = False

# Try to load environment variables from multiple possible locations
env_paths = ["../backend/.env", "./.env", "../.env", "../../.env"]

env_loaded = False
for env_path in env_paths:
    if os.path.exists(env_path):
        load_dotenv(env_path)
        env_loaded = True
        break

# Try to import required modules
try:
    from graph import create_blog_collaboration_graph
except ImportError as e:
    print(f"Error importing graph: {e}")
    print(
        "Please check that all dependencies are installed and the graph.py file exists."
    )
    sys.exit(1)


def visualize_graph():
    """
    Visualize the blog collaboration graph using Mermaid or ASCII as fallback.
    This function will save the graph visualization as a PNG file when possible.
    """
    try:
        print("Creating graph...")
        graph = create_blog_collaboration_graph()

        try:
            # Try to draw the graph using Mermaid
            output_path = "blog_graph.png"

            # Get the graph and try to draw it
            g = graph.get_graph()

            try:
                # Try to draw using Mermaid
                img_data = g.draw_mermaid_png()

                # Save the image
                with open(output_path, "wb") as f:
                    f.write(img_data)

                print(f"Graph visualization saved to {output_path}")

                # Display the image if we're in IPython
                if can_display:
                    display(Image(img_data))

            except Exception as e:
                print(f"Could not create Mermaid visualization: {e}")
                print("Falling back to ASCII visualization...")
                ascii_viz = g.draw_ascii()
                print(ascii_viz)

        except Exception as e:
            print(f"Error drawing graph: {e}")
            print("Please check that all dependencies are installed properly.")

    except Exception as e:
        print(f"Failed to create graph: {e}")
        print("Please check that the graph.py implementation is correct.")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    visualize_graph()
