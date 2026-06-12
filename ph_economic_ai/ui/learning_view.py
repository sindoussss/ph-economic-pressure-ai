"""The 'How it learns' view — surfaces Strata's real, honest learning (read-only)."""
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QScrollArea, QLabel, QFrame

from ph_economic_ai.ui import theme as _t
from ph_economic_ai.ui.agent_performance import AgentPerformancePanel
from ph_economic_ai.engine.evolution import _COLD_START_RUNS

_LAYERS = [
    ("Within a run — it adapts",
     "Agents debate over several rounds; each sees the previous rounds and revises. "
     "This resets every run."),
    ("Across runs — only on real outcomes",
     f"A background checker grades past forecasts against the real DOE pump price "
     f"(~5 days later); trust updates and the swarm evolves after a cold-start of "
     f"{_COLD_START_RUNS} runs. Same-day reruns change nothing."),
    ("The models — frozen",
     "The LLMs are not trained on your runs. What adapts is which agents are trusted, "
     "not the agents themselves."),
]


class LearningView(QWidget):
    def __init__(self, store, parent=None):
        super().__init__(parent)
        self._store = store
        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer.addWidget(scroll)
        body = QWidget(); body.setStyleSheet(f'background:{_t.SURFACE};')
        col = QVBoxLayout(body)
        col.setContentsMargins(28, 24, 28, 24); col.setSpacing(16)
        scroll.setWidget(body)

        # Block 1 — How it learns (static, always truthful)
        c1, l1 = _t.card('How it learns')
        for head, desc in _LAYERS:
            h = QLabel(head)
            h.setStyleSheet(f'color:{_t.INK};font-family:{_t.MONO};font-size:11px;font-weight:700;')
            l1.addWidget(h)
            d = QLabel(desc); d.setWordWrap(True)
            d.setStyleSheet(f'color:{_t.MUTED};font-family:{_t.MONO};font-size:10px;')
            l1.addWidget(d)
            l1.addWidget(_t.hairline())
        col.addWidget(c1)

        # Block 2 — This run's revisions
        c2, l2 = _t.card("This run's revisions")
        self._revisions_lbl = QLabel(); self._revisions_lbl.setWordWrap(True)
        self._revisions_lbl.setStyleSheet(f'color:{_t.INK};font-family:{_t.MONO};font-size:10px;')
        l2.addWidget(self._revisions_lbl)
        col.addWidget(c2)

        # Block 3 — Trust ladder (reuse AgentPerformancePanel)
        c3, l3 = _t.card('Trust ladder')
        cap = QLabel('Agents rise and fall as real outcomes grade their past calls.')
        cap.setWordWrap(True)
        cap.setStyleSheet(f'color:{_t.MUTED};font-family:{_t.MONO};font-size:9px;')
        l3.addWidget(cap)
        self._perf = AgentPerformancePanel(self._store)
        l3.addWidget(self._perf)
        col.addWidget(c3)

        # Block 4 — Track record (store-derived)
        c4, l4 = _t.card('Track record')
        self._track_lbl = QLabel(); self._track_lbl.setWordWrap(True)
        self._track_lbl.setStyleSheet(f'color:{_t.INK};font-family:{_t.MONO};font-size:10px;')
        l4.addWidget(self._track_lbl)
        col.addWidget(c4)

        col.addStretch(1)
        self.refresh(None)

    def refresh(self, run_id=None):
        rows = []
        try:
            if run_id is not None:
                rows = self._store.get_agent_responses(run_id)
        except Exception:
            rows = []
        self._revisions_lbl.setText(self._format_revisions(rows))
        try:
            self._perf.refresh()
        except Exception:
            pass
        self._track_lbl.setText(self._format_track())

    @staticmethod
    def _fmt_est(v):
        return f'{v:+.2f}' if isinstance(v, (int, float)) else '—'

    def _format_revisions(self, rows):
        if not rows:
            return "Run a simulation to see agents revise across debate rounds."
        by_agent: dict = {}
        for r in rows:
            by_agent.setdefault(r.get('agent_name', '?'), []).append(
                (r.get('round_num', 0), r.get('estimate')))
        lines = []
        for name in sorted(by_agent):
            seq = sorted(by_agent[name], key=lambda t: t[0])
            chain = '  ->  '.join(f"R{rn} {self._fmt_est(est)}" for rn, est in seq)
            lines.append(f'{name:<22} {chain}')
        return '\n'.join(lines)

    def _format_track(self):
        try:
            logged = self._store.total_runs()
            recent = self._store.get_recent_runs(limit=200)
            graded = [r for r in recent if r.get('actual_price_change') is not None]
            n_graded = len(graded)
            errs = [r['accuracy_error'] for r in graded if r.get('accuracy_error') is not None]
            mae = (sum(errs) / len(errs)) if errs else None
        except Exception:
            logged, n_graded, mae = 0, 0, None
        evo = 'active' if logged >= _COLD_START_RUNS else f'activates at {_COLD_START_RUNS}'
        out = [f'{logged} runs logged   .   {n_graded} graded   .   evolution {evo}']
        if n_graded == 0:
            out.append('No graded outcomes yet - grading waits ~5 days for real DOE pump prices.')
        elif mae is not None:
            out.append(f'mean abs error: ₱{mae:.2f}/L over {n_graded} graded runs')
        return '\n'.join(out)
