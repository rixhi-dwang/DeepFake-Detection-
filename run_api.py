"""
Entry point for the local-only Deepfake Detection API server.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config


def main():
    parser = argparse.ArgumentParser(description="Start local Deepfake Detection API server")
    parser.add_argument(
        "--host",
        type=str,
        default=Config.API_HOST,
        help=f"Host to bind (default: {Config.API_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=Config.API_PORT,
        help=f"Port to bind (default: {Config.API_PORT})",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    args = parser.parse_args()

    if args.host not in Config.LOCAL_ALLOWED_HOSTS:
        raise ValueError(
            f"Local-only mode is enabled. Host must be one of {sorted(list(Config.LOCAL_ALLOWED_HOSTS))}."
        )

    Config.summary()
    print("Starting Local Deepfake Detection API")
    print(f"URL: http://{args.host}:{args.port}")
    print(f"Dashboard: http://{args.host}:{args.port}/dashboard")
    print(f"Docs: http://{args.host}:{args.port}/docs")
    print(f"Health: http://{args.host}:{args.port}/health")

    import uvicorn

    uvicorn.run(
        "app.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
