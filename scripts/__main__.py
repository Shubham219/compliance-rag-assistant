"""
Main entry point for running the application.
"""

import sys
import argparse

from src.ui.gradio_app import main as ui_main
from src.api.routes import main as api_main
from src.utils.logger import app_logger


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Compliance RAG Assistant")
    parser.add_argument(
        "mode",
        choices=["ui", "api", "both"],
        default="ui",
        help="Run mode: ui (Gradio), api (FastAPI), or both"
    )
    
    args = parser.parse_args()
    
    app_logger.info(f"Starting in {args.mode} mode...")
    
    if args.mode == "ui":
        ui_main()
    elif args.mode == "api":
        api_main()
    elif args.mode == "both":
        # Run both (requires threading or multiprocessing)
        import threading
        
        api_thread = threading.Thread(target=api_main, daemon=True)
        api_thread.start()
        
        ui_main()


if __name__ == "__main__":
    main()