"""Grounded entity enrichment over a built graph's evidence nodes. `enrich_with_
entities` is the pure, tested core; `EntityExtractWorker` runs it off the UI
thread. Both degrade to no-ops on any extraction failure (EntityExtractor already
returns empty on error)."""
from PyQt6.QtCore import QThread, pyqtSignal

from ph_economic_ai.engine.entity_extractor import extract
from ph_economic_ai.engine.kg_assemble import apply_extraction


def enrich_with_entities(builder, extract_fn=extract) -> int:
    nodes, _ = builder.snapshot()
    processed = 0
    for n in [x for x in nodes if x.kind == 'evidence']:
        result = extract_fn(n.payload.get('text', ''), n.payload.get('source', ''))
        apply_extraction(builder, n.id, n.payload.get('source', ''), result)
        processed += 1
    return processed


class EntityExtractWorker(QThread):
    progress = pyqtSignal(int)        # chunks done
    done = pyqtSignal()

    def __init__(self, builder, parent=None):
        super().__init__(parent)
        self._builder = builder

    def run(self):
        nodes, _ = self._builder.snapshot()
        for i, n in enumerate([x for x in nodes if x.kind == 'evidence'], start=1):
            result = extract(n.payload.get('text', ''), n.payload.get('source', ''))
            apply_extraction(self._builder, n.id, n.payload.get('source', ''), result)
            self.progress.emit(i)
        self.done.emit()
