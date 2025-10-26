"""Run the DeepSynth web application."""

from flask import Flask

from . import create_app


def main() -> None:
    """Create and run the Flask development server."""

    app: Flask = create_app()
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    main()
