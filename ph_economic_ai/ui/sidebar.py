from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QFrame
from PyQt6.QtCore import pyqtSignal


class SidebarWidget(QWidget):
    page_changed = pyqtSignal(int)

    _NAV = [
        ('ANALYSIS', None),
        ('◈', 'Dashboard', 0),
        ('◉', 'Pressure Index', 1),
        ('⬡', 'Agent Network', 2),
        ('SYSTEM', None),
        ('⚙', 'Settings', 3),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(190)
        self.setStyleSheet('background: #FFFFFF;')
        self._buttons: list[tuple[QPushButton, int]] = []
        self._active_idx = 0
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Logo block
        logo = QFrame()
        logo.setStyleSheet('background:#FFFFFF; border-bottom: 1px solid #EAEAEA;')
        ll = QVBoxLayout(logo)
        ll.setContentsMargins(18, 14, 18, 14)
        ll.setSpacing(2)
        top = QLabel('PH ECONAI')
        top.setStyleSheet('font-size:11px; font-weight:700; color:#4A90E2; letter-spacing:1px;')
        sub = QLabel('Economic Advisor')
        sub.setStyleSheet('font-size:10px; color:#BBBBBB;')
        ll.addWidget(top)
        ll.addWidget(sub)
        layout.addWidget(logo)

        for item in self._NAV:
            if item[1] is None:
                lbl = QLabel(item[0])
                lbl.setStyleSheet(
                    'font-size:9px; font-weight:700; color:#CCCCCC;'
                    'letter-spacing:1px; padding:12px 18px 4px 18px;'
                )
                layout.addWidget(lbl)
            else:
                icon, text, page_idx = item
                btn = QPushButton(f'{icon}  {text}')
                btn.setFlat(True)
                btn.setCursor(self.cursor())
                btn.setStyleSheet(self._style(page_idx == 0))
                btn.clicked.connect(lambda _, idx=page_idx: self._on_click(idx))
                self._buttons.append((btn, page_idx))
                layout.addWidget(btn)

        layout.addStretch()

        footer = QLabel('  ●  Trained · Offline')
        footer.setStyleSheet(
            'font-size:10px; color:#4A90E2; font-weight:600;'
            'background:#EBF4FF; border-radius:10px;'
            'padding:4px 8px; margin:12px 14px;'
        )
        layout.addWidget(footer)

    def _style(self, active: bool) -> str:
        if active:
            return (
                'text-align:left; padding:9px 18px; font-size:13px;'
                'color:#4A90E2; background:#EBF4FF;'
                'border:none; border-left:3px solid #4A90E2; font-weight:600;'
            )
        return (
            'text-align:left; padding:9px 18px 9px 21px; font-size:13px;'
            'color:#666666; background:transparent; border:none;'
        )

    def _on_click(self, idx: int):
        self._active_idx = idx
        for btn, page_idx in self._buttons:
            btn.setStyleSheet(self._style(page_idx == idx))
        self.page_changed.emit(idx)
