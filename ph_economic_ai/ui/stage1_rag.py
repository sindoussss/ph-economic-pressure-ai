import concurrent.futures
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QFrame, QFileDialog, QProgressBar,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer

from ph_economic_ai.engine.rag import RagEngine


class _FetchThread(QThread):
    source_done = pyqtSignal(str, int)   # source_name, chunk_count
    all_done = pyqtSignal(dict)

    def __init__(self, rag: RagEngine, parent=None):
        super().__init__(parent)
        self._rag = rag

    def run(self):
        results = self._rag.fetch_all(
            on_progress=lambda name, status, count: self.source_done.emit(name, count)
        )
        self.all_done.emit(results)


class _SourceCard(QFrame):
    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            'QFrame{background:#F7F8FA;border:1px solid #EAECF0;'
            'border-radius:9px;padding:0px;}'
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)

        self._name_lbl = QLabel(name)
        self._name_lbl.setStyleSheet('font-size:10px;font-weight:600;color:#1C1E26;')

        self._status_lbl = QLabel('waiting...')
        self._status_lbl.setStyleSheet('font-size:9px;color:#9EA3AE;')
        self._status_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)

        self._bar = QProgressBar()
        self._bar.setRange(0, 0)  # indeterminate
        self._bar.setFixedHeight(3)
        self._bar.setTextVisible(False)
        self._bar.setStyleSheet(
            'QProgressBar{background:#EAECF0;border-radius:2px;border:none;}'
            'QProgressBar::chunk{background:#1C1E26;border-radius:2px;}'
        )

        info_col = QVBoxLayout()
        info_col.setSpacing(4)
        info_col.addWidget(self._name_lbl)
        info_col.addWidget(self._bar)

        layout.addLayout(info_col, stretch=1)
        layout.addWidget(self._status_lbl)

    def set_done(self, chunk_count: int):
        self._bar.setRange(0, 1)
        self._bar.setValue(1)
        self._status_lbl.setText(f'{chunk_count} chunks')
        self._status_lbl.setStyleSheet('font-size:9px;color:#1C1E26;font-weight:600;')

    def set_error(self):
        self._bar.setRange(0, 1)
        self._bar.setValue(0)
        self._status_lbl.setText('failed')
        self._status_lbl.setStyleSheet('font-size:9px;color:#E74C3C;')


class Stage1RagPanel(QWidget):
    def __init__(self, rag: RagEngine, parent=None):
        super().__init__(parent)
        self._rag = rag
        self._cards: dict[str, _SourceCard] = {}
        self._build()
        self._start_fetch()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(14)

        h = QLabel('Stage 1 — Graph Building')
        h.setStyleSheet('font-size:18px;font-weight:700;color:#1C1E26;')
        root.addWidget(h)

        sub = QLabel('Fetching live sources in parallel and indexing into TF-IDF knowledge base.')
        sub.setStyleSheet('font-size:11px;color:#9EA3AE;')
        root.addWidget(sub)

        status_row = QHBoxLayout()
        self._status_lbl = QLabel('Fetching 9 sources...')
        self._status_lbl.setStyleSheet('font-size:10px;color:#1C1E26;font-weight:600;')
        self._chunk_lbl = QLabel('0 chunks indexed')
        self._chunk_lbl.setStyleSheet('font-size:10px;color:#9EA3AE;')
        status_row.addWidget(self._status_lbl)
        status_row.addStretch()
        status_row.addWidget(self._chunk_lbl)
        root.addLayout(status_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet('background:transparent;')

        cards_widget = QWidget()
        self._cards_layout = QVBoxLayout(cards_widget)
        self._cards_layout.setSpacing(6)
        self._cards_layout.setContentsMargins(0, 0, 0, 0)

        # Use RagEngine.SOURCES class attribute for source names
        source_names = list(RagEngine.SOURCES.keys())
        for name in source_names:
            card = _SourceCard(name)
            self._cards[name] = card
            self._cards_layout.addWidget(card)

        self._cards_layout.addStretch()
        scroll.setWidget(cards_widget)
        root.addWidget(scroll, stretch=1)

        upload_btn = QPushButton('+ Upload PDF')
        upload_btn.setStyleSheet(
            'QPushButton{border:1.5px dashed #D1D5DB;border-radius:9px;'
            'padding:8px;font-size:10px;color:#9EA3AE;background:transparent;}'
            'QPushButton:hover{border-color:#9EA3AE;color:#6B7280;}'
        )
        upload_btn.clicked.connect(self._on_upload)
        root.addWidget(upload_btn)

    def _start_fetch(self):
        self._thread = _FetchThread(self._rag)
        self._thread.source_done.connect(self._on_source_done)
        self._thread.all_done.connect(self._on_all_done)
        self._thread.start()

        self._ticker = QTimer(self)
        self._ticker.timeout.connect(self._update_chunk_count)
        self._ticker.start(500)

    def _on_source_done(self, name: str, count: int):
        card = self._cards.get(name)
        if card:
            if count > 0:
                card.set_done(count)
            else:
                card.set_error()

    def _on_all_done(self, results: dict):
        self._ticker.stop()
        self._update_chunk_count()
        done = sum(1 for v in results.values() if v > 0)
        self._status_lbl.setText(f'{done}/9 sources fetched')

    def _update_chunk_count(self):
        self._chunk_lbl.setText(f'{self._rag.chunk_count} chunks indexed')

    def _on_upload(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Select PDF', '', 'PDF Files (*.pdf)'
        )
        if path:
            count = self._rag.add_pdf(path)
            self._update_chunk_count()
