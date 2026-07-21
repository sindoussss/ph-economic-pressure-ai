from dataclasses import dataclass
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QScrollArea, QSlider, QDialog, QLineEdit, QDialogButtonBox,
)
from PyQt6.QtCore import Qt, pyqtSignal

from ph_economic_ai.engine import llm
from ph_economic_ai.engine.debate import Agent, DEFAULT_AGENTS
from ph_economic_ai.engine.swarm import (
    REGIONS, expected_call_counts, group_critical_path,
)

# Rough streamed-completion latency against a hosted free tier. The old
# estimate assumed 15s per call, which was a local-Ollama number — hosted
# inference is far quicker, and on free tiers the quota, not the model, is
# usually what sets the floor.
_FAST_CALL_SECS = 3
_DEEP_CALL_SECS = 7

# Reserved completion budget per call, mirroring the max_tokens the engine
# passes. Completions, not prompts, dominate the token bill.
_FAST_MAX_TOKENS = 750
_DEEP_MAX_TOKENS = 900
_TYPICAL_PROMPT_TOKENS = 650      # measured against a populated RAG index


def estimate_swarm_seconds(parallel_n: int) -> int:
    """Estimated wall-clock for one swarm run, in seconds.

    Three things can dominate, so the estimate takes whichever is worst:

    * **latency** — groups run `parallel_n` at a time; inside a group round 1
      is sequential (agents read each other) and later rounds are parallel.
      The judges then run sequentially.
    * **the request cap** — a free tier caps requests per minute, so a run
      cannot finish faster than `total_calls / RPM`.
    * **the token cap** — and usually this is the one that bites. A run spends
      roughly 44K fast-tier tokens against a 6K/min free-tier ceiling, so the
      token budget alone can set a multi-minute floor however fast the model
      responds.
    """
    counts = expected_call_counts()
    n_groups = max(1, len(REGIONS))
    parallel_n = max(1, parallel_n)

    batches = -(-n_groups // parallel_n)                      # ceil div
    group_secs = group_critical_path() * _FAST_CALL_SECS
    latency_secs = batches * group_secs + counts['deep'] * _DEEP_CALL_SECS

    request_floor = counts['total'] / max(1, llm.effective_rpm()) * 60
    token_floor = _token_floor_seconds(counts)

    return int(max(latency_secs, request_floor, token_floor))


def _token_floor_seconds(counts: dict) -> float:
    """Seconds implied by the per-minute token ceiling, worst tier wins."""
    try:
        provider = llm.active_provider()
    except llm.LLMError:
        return 0.0

    worst = 0.0
    for tier, n_calls, max_tokens in (
        (llm.FAST, counts['fast'], _FAST_MAX_TOKENS),
        (llm.DEEP, counts['deep'], _DEEP_MAX_TOKENS),
    ):
        tpm = llm.tpm_for(provider, tier)
        if not tpm:
            continue
        spend = n_calls * (_TYPICAL_PROMPT_TOKENS + max_tokens)
        worst = max(worst, spend / tpm * 60)
    return worst


@dataclass
class Scenario:
    oil_pct: float = 5.0
    usd_pct: float = 2.0
    bsp_rate: float = 6.5
    demand_index: float = 72.0

    def to_dict(self) -> dict:
        return {
            'oil_pct': self.oil_pct,
            'usd_pct': self.usd_pct,
            'bsp_rate': self.bsp_rate,
            'demand_index': self.demand_index,
        }


class _ScenarioPill(QFrame):
    value_changed = pyqtSignal()

    def __init__(self, label: str, default: float,
                 min_val: float, max_val: float, step: float = 0.5, parent=None):
        super().__init__(parent)
        self._step = step
        self._min = min_val
        self._max = max_val
        self._value = default
        self._label = label
        self.setStyleSheet(
            'QFrame{background:#FFFFFF;border:1px solid #1C1E26;'
            'border-radius:9px;}'
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)

        self._label_lbl = QLabel(label)
        self._label_lbl.setStyleSheet('font-size:8px;color:#9EA3AE;text-transform:uppercase;')

        self._val_lbl = QLabel(self._fmt(default))
        self._val_lbl.setStyleSheet('font-size:14px;font-weight:700;color:#1C1E26;')

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(int(min_val / step), int(max_val / step))
        self._slider.setValue(int(default / step))
        self._slider.setFixedHeight(14)
        self._slider.setStyleSheet(
            'QSlider::groove:horizontal{height:3px;background:#E5E7EB;border-radius:2px;}'
            'QSlider::handle:horizontal{width:10px;height:10px;margin:-4px 0;'
            'border-radius:5px;background:#1C1E26;}'
            'QSlider::sub-page:horizontal{background:#1C1E26;border-radius:2px;}'
        )
        self._slider.valueChanged.connect(self._on_slider)

        layout.addWidget(self._label_lbl)
        layout.addWidget(self._val_lbl)
        layout.addWidget(self._slider)

    def _fmt(self, v: float) -> str:
        return f'{v:+.1f}%' if '%' in self._label or 'pct' in self._label.lower() else f'{v:.1f}'

    def _on_slider(self, raw: int):
        self._value = raw * self._step
        self._val_lbl.setText(self._fmt(self._value))
        self.value_changed.emit()

    @property
    def value(self) -> float:
        return self._value


class _AgentCard(QFrame):
    def __init__(self, agent: Agent, parent=None):
        super().__init__(parent)
        self._agent = agent
        self.setStyleSheet(
            'QFrame{background:#FBFBFA;border:1px solid #E5E7EB;border-radius:10px;}'
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(5)

        name_lbl = QLabel(agent.name)
        name_lbl.setStyleSheet('font-size:11px;font-weight:700;color:#1C1E26;')

        role_lbl = QLabel(agent.role)
        role_lbl.setStyleSheet('font-size:9px;color:#9EA3AE;')

        prompt_lbl = QLabel(f'"{agent.system_prompt[:120]}..."')
        prompt_lbl.setWordWrap(True)
        prompt_lbl.setStyleSheet(
            'font-size:9px;color:#9EA3AE;font-style:italic;'
            'background:#FFFFFF;border:1px solid #E5E7EB;'
            'border-radius:7px;padding:5px 8px;'
        )

        sources_row = QHBoxLayout()
        sources_row.setSpacing(4)
        for src in agent.rag_sources[:5]:
            tag = QLabel(src)
            tag.setStyleSheet(
                'font-size:8px;font-weight:600;color:#FFFFFF;'
                'background:#1C1E26;border-radius:20px;padding:2px 7px;'
            )
            sources_row.addWidget(tag)
        sources_row.addStretch()

        layout.addWidget(name_lbl)
        layout.addWidget(role_lbl)
        layout.addLayout(sources_row)
        layout.addWidget(prompt_lbl)


class Stage2SetupPanel(QWidget):
    run_requested = pyqtSignal(object, list, bool, int)  # Scenario, list[Agent], swarm_mode, parallel_n

    def __init__(self, agents: list, parent=None):
        super().__init__(parent)
        self._agents = list(agents)
        self._pills: list[_ScenarioPill] = []
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(16)

        h = QLabel('Stage 2 — Environment Setup')
        h.setStyleSheet('font-size:18px;font-weight:700;color:#1C1E26;')
        root.addWidget(h)

        scenario_lbl = QLabel('SCENARIO INPUTS')
        scenario_lbl.setStyleSheet('font-size:9px;font-weight:600;color:#9EA3AE;letter-spacing:0.7px;')
        root.addWidget(scenario_lbl)

        pills_row = QHBoxLayout()
        pills_row.setSpacing(8)
        configs = [
            ('Oil shock %',    5.0,  -20.0, 30.0, 0.5),
            ('USD/PHP shift %', 2.0, -10.0, 15.0, 0.5),
            ('BSP rate %',     6.5,   3.0,  10.0, 0.25),
            ('Demand index',  72.0,  50.0, 100.0, 1.0),
        ]
        for label, default, mn, mx, step in configs:
            pill = _ScenarioPill(label, default, mn, mx, step)
            self._pills.append(pill)
            pills_row.addWidget(pill)
        root.addLayout(pills_row)

        # Swarm options row
        swarm_row = QHBoxLayout()
        swarm_row.setSpacing(8)

        self._swarm_btn = QPushButton('Swarm Mode')
        self._swarm_btn.setCheckable(True)
        self._swarm_btn.setChecked(True)
        self._swarm_btn.setFixedHeight(36)
        self._swarm_btn.setStyleSheet(
            'QPushButton:checked{background:#1C1E26;color:#FFFFFF;border-radius:9px;'
            'padding:0 14px;font-size:10px;font-weight:700;border:none;}'
            'QPushButton:!checked{background:#FFFFFF;color:#1C1E26;border-radius:9px;'
            'padding:0 14px;font-size:10px;font-weight:700;border:1.5px solid #D1D5DB;}'
        )
        self._swarm_btn.toggled.connect(self._on_swarm_toggled)

        parallel_frame = QFrame()
        parallel_frame.setStyleSheet('QFrame{background:#FFFFFF;border:1px solid #1C1E26;border-radius:9px;}')
        pf_layout = QHBoxLayout(parallel_frame)
        pf_layout.setContentsMargins(12, 6, 12, 6)
        pf_layout.setSpacing(8)
        pg_label = QLabel('Parallel Groups')
        pg_label.setStyleSheet('font-size:8px;color:#9EA3AE;')
        self._parallel_val_lbl = QLabel('4')
        self._parallel_val_lbl.setStyleSheet('font-size:12px;font-weight:700;color:#1C1E26;min-width:12px;')
        self._parallel_slider = QSlider(Qt.Orientation.Horizontal)
        self._parallel_slider.setRange(1, 8)
        self._parallel_slider.setValue(4)
        self._parallel_slider.setFixedWidth(120)
        self._parallel_slider.setStyleSheet(
            'QSlider::groove:horizontal{height:3px;background:#E5E7EB;border-radius:2px;}'
            'QSlider::handle:horizontal{width:10px;height:10px;margin:-4px 0;'
            'border-radius:5px;background:#1C1E26;}'
            'QSlider::sub-page:horizontal{background:#1C1E26;border-radius:2px;}'
        )
        self._parallel_slider.valueChanged.connect(lambda v: self._parallel_val_lbl.setText(str(v)))
        self._parallel_slider.valueChanged.connect(self._update_time_estimate)
        pf_layout.addWidget(pg_label)
        pf_layout.addWidget(self._parallel_slider)
        pf_layout.addWidget(self._parallel_val_lbl)

        swarm_row.addWidget(self._swarm_btn)
        swarm_row.addWidget(parallel_frame)
        swarm_row.addStretch()
        root.addLayout(swarm_row)

        agents_lbl = QLabel('AGENT ROSTER')
        agents_lbl.setStyleSheet('font-size:9px;font-weight:600;color:#9EA3AE;letter-spacing:0.7px;')
        root.addWidget(agents_lbl)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._agents_widget = QWidget()
        self._agents_layout = QVBoxLayout(self._agents_widget)
        self._agents_layout.setSpacing(8)
        self._agents_layout.setContentsMargins(0, 0, 0, 0)
        self._rebuild_agent_cards()
        self._agents_layout.addStretch()
        scroll.setWidget(self._agents_widget)
        root.addWidget(scroll, stretch=1)

        btn_row = QHBoxLayout()
        add_btn = QPushButton('+ Add custom agent')
        add_btn.setStyleSheet(
            'QPushButton{border:1.5px dashed #D1D5DB;border-radius:9px;'
            'padding:8px 16px;font-size:10px;color:#9EA3AE;background:transparent;}'
        )
        add_btn.clicked.connect(self._on_add_agent)

        self._run_btn = QPushButton('Run Simulation →')
        self._run_btn.setStyleSheet(
            'QPushButton{background:#1C1E26;color:#FFFFFF;border-radius:9px;'
            'padding:10px 20px;font-size:11px;font-weight:700;border:none;}'
            'QPushButton:hover{background:#374151;}'
        )
        self._run_btn.clicked.connect(self._on_run)

        self._time_lbl = QLabel('')
        self._time_lbl.setStyleSheet('font-size:9px;color:#9EA3AE;')

        btn_row.addWidget(add_btn)
        btn_row.addStretch()
        btn_row.addWidget(self._time_lbl)
        btn_row.addWidget(self._run_btn)
        root.addLayout(btn_row)

        for pill in self._pills:
            pill.value_changed.connect(self._update_time_estimate)
        self._swarm_btn.toggled.connect(self._update_time_estimate)
        self._update_time_estimate()

    def _rebuild_agent_cards(self):
        while self._agents_layout.count():
            item = self._agents_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for agent in self._agents:
            self._agents_layout.addWidget(_AgentCard(agent))

    def _update_time_estimate(self, *_):
        if self._swarm_btn.isChecked():
            secs = estimate_swarm_seconds(self._parallel_slider.value())
        else:
            rounds = 3 if len(self._agents) <= 7 else 2
            secs = len(self._agents) * rounds * _FAST_CALL_SECS
        label = f'~{secs // 60} min estimated' if secs >= 60 else f'~{secs}s estimated'
        if self._swarm_btn.isChecked():
            label += ' (swarm)'
        self._time_lbl.setText(label)

    def _on_swarm_toggled(self, checked: bool):
        self._parallel_slider.setEnabled(checked)

    def _on_add_agent(self):
        if len(self._agents) >= 10:
            return
        dlg = _AddAgentDialog(self)
        if dlg.exec():
            self._agents.append(dlg.agent())
            self._rebuild_agent_cards()
            self._update_time_estimate()

    def _on_run(self):
        scenario = Scenario(
            oil_pct=self._pills[0].value,
            usd_pct=self._pills[1].value,
            bsp_rate=self._pills[2].value,
            demand_index=self._pills[3].value,
        )
        swarm_mode = self._swarm_btn.isChecked()
        parallel_n = self._parallel_slider.value()
        self.run_requested.emit(scenario, list(self._agents), swarm_mode, parallel_n)

    def current_scenario(self) -> 'Scenario':
        return Scenario(
            oil_pct=self._pills[0].value,
            usd_pct=self._pills[1].value,
            bsp_rate=self._pills[2].value,
            demand_index=self._pills[3].value,
        )


class _AddAgentDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Add Custom Agent')
        self.setMinimumWidth(400)
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        self._name = QLineEdit('Custom Agent')
        self._role = QLineEdit('Specialist role description')
        self._prompt = QLineEdit('You are a specialist. Analyze the scenario and give a ₱/L estimate.')

        for lbl_text, widget in [('Name', self._name), ('Role', self._role), ('System prompt', self._prompt)]:
            layout.addWidget(QLabel(lbl_text))
            layout.addWidget(widget)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def agent(self) -> Agent:
        return Agent(
            name=self._name.text().strip() or 'Custom Agent',
            role=self._role.text().strip(),
            system_prompt=self._prompt.text().strip(),
            rag_sources=[],
        )
