"""Package entry point: python -m aquarov"""
from __future__ import annotations

import sys

def main() -> int:
    from PySide6.QtWidgets import QApplication

    from aquarov.ui.main_window import MainWindow # noqa: WPS433

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()

if __name__ == "__main__":
    raise SystemExit(main())
