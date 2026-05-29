from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QTabWidget, QScrollArea, QLineEdit, QSlider,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from ph_economic_ai.engine.rag import RagEngine
from ph_economic_ai.engine.debate import Agent, DebateEngine, AgentResponse


class _AskThread(QThread):
    token_received = pyqtSignal(str)
    done = pyqtSignal(str)

    def __init__(self, engine: DebateEngine, agent_name: str, question: str, parent=None):
        super().__init__(parent)
        self._engine = engine
        self._agent_name = agent_name
        self._question = question

    def run(self):
        answer = self._engine.ask(
            self._agent_name, self._question,
            on_token=lambda tok: self.token_received.emit(tok),
        )
        self.done.emit(answer)


class _AskContextThread(QThread):
    """Fallback LLM thread used in swarm mode (no single debate engine).
    Injects all three sector verdicts as context so the LLM has full economy data."""
    token_received = pyqtSignal(str)
    done = pyqtSignal(str)

    def __init__(self, gas: str, food: str, elec: str, question: str, parent=None):
        super().__init__(parent)
        self._gas = gas
        self._food = food
        self._elec = elec
        self._question = question

    def run(self):
        import ollama
        from ph_economic_ai.engine.debate import _MAIN_MODEL
        system = (
            'You are a Philippine macroeconomic analyst with access to the latest '
            'multi-agent simulation results for gas, food, and electricity prices. '
            'Answer questions about these forecasts accurately and concisely, '
            'citing specific numbers from the debate results when relevant.'
        )
        parts = []
        if self._gas:
            parts.append(f'GAS SECTOR (swarm debate result):\n{self._gas}')
        if self._food:
            parts.append(f'FOOD SECTOR (4-agent debate result):\n{self._food}')
        if self._elec:
            parts.append(f'ELECTRICITY SECTOR (4-agent debate result):\n{self._elec}')
        context = '\n\n'.join(parts)
        user_msg = f'{context}\n\nQuestion: {self._question}' if context else self._question
        full_text = ''
        for chunk in ollama.chat(
            model=_MAIN_MODEL,
            messages=[
                {'role': 'system', 'content': system},
                {'role': 'user',   'content': user_msg},
            ],
            stream=True,
            think=False,
        ):
            token = chunk['message']['content']
            full_text += token
            self.token_received.emit(token)
        self.done.emit(full_text)


class Stage5InteractPanel(QWidget):
    rerun_requested = pyqtSignal(object)  # scenario dict

    def __init__(self, rag: RagEngine, agents: list,
                 regressor, df, cv_rmse: float, parent=None):
        super().__init__(parent)
        self._rag = rag
        self._agents = list(agents)
        self._regressor = regressor
        self._df = df
        self._cv_rmse = cv_rmse
        self._debate_engine: DebateEngine | None = None
        self._last_scenario: dict = {}
        self._ask_thread: _AskThread | _AskContextThread | None = None
        self._current_answer_lbl = None
        # Sector verdict strings — populated as each debate completes
        self._gas_verdict:  str = ''
        self._food_verdict: str = ''
        self._elec_verdict: str = ''
        self._swarm_mode: bool = False
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        h = QLabel('Stage 5 — Deep Interaction')
        h.setStyleSheet('font-size:18px;font-weight:700;color:#1C1E26;padding:20px 24px 0 24px;')
        root.addWidget(h)

        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.setStyleSheet(
            'QTabBar::tab{padding:8px 16px;font-size:10px;font-weight:600;color:#9EA3AE;'
            'border:none;background:transparent;}'
            'QTabBar::tab:selected{color:#1C1E26;border-bottom:2px solid #1C1E26;}'
        )
        tabs.addTab(self._build_adjust_tab(), 'Adjust & Re-run')
        tabs.addTab(self._build_ask_tab(), 'Ask an Agent')
        tabs.addTab(self._build_toggle_tab(), 'Toggle Sources')
        root.addWidget(tabs, stretch=1)

    def _build_adjust_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(12)

        lbl = QLabel('Adjust scenario inputs and re-run the full simulation.')
        lbl.setStyleSheet('font-size:10px;color:#9EA3AE;')
        layout.addWidget(lbl)

        self._adjust_pills: list[tuple] = []
        configs = [
            ('Oil shock %',     'oil_pct',      5.0,  -20.0, 30.0,  0.5),
            ('USD/PHP shift %', 'usd_pct',      2.0,  -10.0, 15.0,  0.5),
            ('BSP rate %',      'bsp_rate',     6.5,    3.0, 10.0,  0.25),
            ('Demand index',    'demand_index', 72.0,  50.0, 100.0,  1.0),
        ]
        for label, key, default, mn, mx, step in configs:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(int(mn / step), int(mx / step))
            slider.setValue(int(default / step))
            val_lbl = QLabel(f'{default:.1f}')
            val_lbl.setFixedWidth(40)
            slider.valueChanged.connect(lambda v, lbl=val_lbl, s=step: lbl.setText(f'{v * s:.1f}'))
            row.addWidget(slider, stretch=1)
            row.addWidget(val_lbl)
            layout.addLayout(row)
            self._adjust_pills.append((key, slider, val_lbl, step, mn, mx))

        run_btn = QPushButton('Re-run Simulation →')
        run_btn.setStyleSheet(
            'QPushButton{background:#1C1E26;color:#FFFFFF;border-radius:9px;'
            'padding:10px;font-size:11px;font-weight:700;border:none;}'
        )
        run_btn.clicked.connect(self._on_rerun)
        layout.addWidget(run_btn)
        layout.addStretch()
        return w

    def _build_ask_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(10)

        selector_row = QHBoxLayout()
        self._agent_chips: list[QPushButton] = []
        self._selected_agent: str = self._agents[0].name if self._agents else ''
        for agent in self._agents:
            btn = QPushButton(agent.name)
            btn.setCheckable(True)
            btn.setChecked(agent.name == self._selected_agent)
            btn.clicked.connect(lambda _, name=agent.name: self._select_agent(name))
            btn.setStyleSheet(
                'QPushButton{border:1px solid #EAECF0;border-radius:8px;'
                'padding:5px 10px;font-size:9px;font-weight:600;color:#6B7280;background:#F7F8FA;}'
                'QPushButton:checked{background:#1C1E26;color:#FFFFFF;border-color:#1C1E26;}'
            )
            self._agent_chips.append(btn)
            selector_row.addWidget(btn)
        selector_row.addStretch()
        layout.addLayout(selector_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._chat_widget = QWidget()
        self._chat_layout = QVBoxLayout(self._chat_widget)
        self._chat_layout.setSpacing(6)
        self._chat_layout.setContentsMargins(0, 0, 0, 0)
        self._chat_layout.addStretch()
        scroll.setWidget(self._chat_widget)
        layout.addWidget(scroll, stretch=1)

        input_row = QHBoxLayout()
        self._chat_input = QLineEdit()
        self._chat_input.setPlaceholderText('Ask a follow-up question...')
        self._chat_input.setStyleSheet(
            'QLineEdit{border:1px solid #EAECF0;border-radius:8px;padding:7px 10px;font-size:10px;}'
        )
        self._chat_input.returnPressed.connect(self._on_ask)
        send_btn = QPushButton('Send')
        send_btn.setStyleSheet(
            'QPushButton{background:#1C1E26;color:#FFFFFF;border-radius:8px;'
            'padding:7px 14px;font-size:9px;font-weight:600;border:none;}'
        )
        send_btn.clicked.connect(self._on_ask)
        input_row.addWidget(self._chat_input, stretch=1)
        input_row.addWidget(send_btn)
        layout.addLayout(input_row)
        return w

    def _build_toggle_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(8)

        lbl = QLabel('Toggle RAG sources on/off. Re-run to see impact.')
        lbl.setStyleSheet('font-size:10px;color:#9EA3AE;')
        layout.addWidget(lbl)

        self._toggle_buttons: dict[str, QPushButton] = {}
        for src in self._rag.all_source_names:
            row = QHBoxLayout()
            toggle = QPushButton(src)
            toggle.setCheckable(True)
            toggle.setChecked(True)
            toggle.clicked.connect(lambda checked, s=src: self._on_toggle(s, checked))
            toggle.setStyleSheet(
                'QPushButton{border:1px solid #EAECF0;border-radius:8px;'
                'padding:6px 12px;font-size:10px;font-weight:600;background:#F7F8FA;color:#1C1E26;}'
                'QPushButton:checked{background:#1C1E26;color:#FFFFFF;border-color:#1C1E26;}'
            )
            self._toggle_buttons[src] = toggle
            row.addWidget(toggle)
            row.addStretch()
            layout.addLayout(row)

        rerun_btn = QPushButton('Re-run with current sources →')
        rerun_btn.setStyleSheet(
            'QPushButton{background:#1C1E26;color:#FFFFFF;border-radius:9px;'
            'padding:10px;font-size:11px;font-weight:700;border:none;}'
        )
        rerun_btn.clicked.connect(self._on_rerun)
        layout.addWidget(rerun_btn)
        layout.addStretch()
        return w

    def update_context(self, responses: list, scenario: dict):
        self._last_scenario = scenario

    def set_debate_engine(self, engine: DebateEngine):
        self._debate_engine = engine
        self._swarm_mode = False

    def set_swarm_context(self, master_verdict, scenario: dict):
        """Store swarm MasterVerdict context. Ask-an-Agent uses context fallback thread."""
        self._last_scenario = scenario
        self._debate_engine = None
        self._swarm_mode = True

    def update_gas_verdict(self, verdict: str):
        self._gas_verdict = verdict

    def update_food_verdict(self, verdict: str):
        self._food_verdict = verdict

    def update_elec_verdict(self, verdict: str):
        self._elec_verdict = verdict

    def _economy_context_prefix(self) -> str:
        """Build a context block from all available sector verdicts."""
        parts = []
        if self._gas_verdict:
            parts.append(f'GAS SECTOR:\n{self._gas_verdict}')
        if self._food_verdict:
            parts.append(f'FOOD SECTOR:\n{self._food_verdict}')
        if self._elec_verdict:
            parts.append(f'ELECTRICITY SECTOR:\n{self._elec_verdict}')
        if not parts:
            return ''
        return '[Economy Simulation Results]\n' + '\n\n'.join(parts) + '\n\n'

    def _select_agent(self, name: str):
        self._selected_agent = name
        for btn in self._agent_chips:
            btn.setChecked(btn.text() == name)

    def _on_ask(self):
        question = self._chat_input.text().strip()
        if not question:
            return
        self._chat_input.clear()
        self._add_bubble(f'You: {question}', user=True)

        if self._debate_engine is not None:
            # Debate mode: inject full economy context into the question then ask the agent
            context = self._economy_context_prefix()
            augmented = context + question if context else question
            self._current_answer_lbl = self._add_bubble(
                f'{self._selected_agent}: thinking...', user=False
            )
            self._ask_thread = _AskThread(
                self._debate_engine, self._selected_agent, augmented
            )
        else:
            # Swarm mode (no single engine): use standalone context-aware thread
            self._current_answer_lbl = self._add_bubble(
                'Economy Analyst: thinking...', user=False
            )
            self._ask_thread = _AskContextThread(
                self._gas_verdict, self._food_verdict, self._elec_verdict, question
            )
        self._ask_thread.done.connect(self._on_answer_done)
        self._ask_thread.start()

    def _add_bubble(self, text: str, user: bool) -> QLabel:
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setStyleSheet(
            'font-size:9px;padding:7px 9px;border-radius:8px;line-height:1.4;'
            + ('background:#1C1E26;color:#FFFFFF;margin-left:20px;'
               if user else
               'background:#F7F8FA;border:1px solid #EAECF0;color:#374151;margin-right:20px;')
        )
        self._chat_layout.insertWidget(self._chat_layout.count() - 1, lbl)
        return lbl

    def _on_answer_done(self, answer: str):
        if self._current_answer_lbl:
            label = self._selected_agent if self._debate_engine else 'Economy Analyst'
            self._current_answer_lbl.setText(f'{label}: {answer[:800]}')

    def _on_toggle(self, source: str, enabled: bool):
        self._rag.toggle_source(source, enabled)

    def _on_rerun(self):
        scenario = dict(self._last_scenario)
        for key, slider, _, step, _, _ in self._adjust_pills:
            scenario[key] = slider.value() * step
        self.rerun_requested.emit(scenario)
