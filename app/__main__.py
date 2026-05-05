from app.main_window import MainWindow
from PyQt6.QtWidgets import QApplication


def main() -> None:
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
