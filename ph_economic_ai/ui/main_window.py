from pathlib import Path
from datetime import datetime

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QStackedWidget,
    QScrollArea, QFrame, QLabel, QPushButton,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal

from ph_economic_ai.engine.rag import RagEngine
from ph_economic_ai.engine.debate import (
    DEFAULT_AGENTS, FOOD_AGENTS, ELECTRICITY_AGENTS,
    SynthesizerThread, DebateEngine, DebateThread, _extract_percent,
)
from ph_economic_ai.engine.swarm import SwarmThread, fetch_live_retail_price, derive_regional_estimates, build_swarm_agents
from ph_economic_ai.engine.live_data import (
    LiveDataBrief, LiveBriefThread, CausalChainThread, PolicyRecoThread,
    derive_scenario_from_brief,
)
from ph_economic_ai.engine.store import AgentTrustStore
from ph_economic_ai.engine.quality_scorer import QualityScorer
from ph_economic_ai.engine.evolution import get_evolved_debate_agents, get_evolved_swarm_agents
from ph_economic_ai.engine.ground_truth import DOECheckerThread
from ph_economic_ai.ui.economy_overview import EconomyOverviewWidget
from ph_economic_ai.ui.landing import LandingPanel
from ph_economic_ai.ui.stage2_setup import Scenario  # dataclass still used downstream
from ph_economic_ai.ui.stage3_canvas import Stage3CanvasPanel
from ph_economic_ai.ui.stage3_swarm_canvas import Stage3SwarmPanel
from ph_economic_ai.ui.stage4_report import Stage4ReportPanel
from ph_economic_ai.ui.stage5_interact import Stage5InteractPanel
from ph_economic_ai.ui.agent_performance import AgentPerformancePanel
from ph_economic_ai.ui.accuracy_view import AccuracyView


# ─────────────────────────────────────────────────────────────────────────────
class _TopNavBar(QFrame):
    """Sticky global top nav. Replaces the left sidebar."""

    nav_clicked = pyqtSignal(int)   # stage index user wants to navigate to

    _BG     = '#FFFFFF'
    _INK    = '#0F1115'
    _TEXT_2 = '#525866'
    _TEXT_3 = '#8B95A7'
    _DIV    = '#E5E7EB'

    # (stage_index, label, locked_at_startup)
    # Overview (1) and Performance (5) are no longer navigable — their content
    # is already surfaced inside the Home scroll (live markets card, recent
    # runs strip). Keeping the panels in the stack at their old indices so
    # downstream code (DOE checker refresh, post-simulation overview updates)
    # keeps working without touching index math.
    _ITEMS: list[tuple[int, str, bool]] = [
        (0, 'Home',        False),
        (2, 'Simulation',  True),
        (3, 'Report',      True),
        (4, 'Interact',    True),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(58)
        self.setStyleSheet(
            f'QFrame{{background:{self._BG};border-bottom:1px solid {self._DIV};}}'
            f'QFrame QLabel{{background:transparent;border:none;}}'
        )
        self._buttons: dict[int, QPushButton] = {}
        self._active = 0
        self._locked: set[int] = {2, 3, 4}
        self._build()

    def _build(self):
        h = QHBoxLayout(self)
        h.setContentsMargins(40, 0, 40, 0)
        h.setSpacing(28)

        brand = QLabel('STRATA')
        brand.setStyleSheet(
            f'font-family:Consolas,monospace;font-size:12px;font-weight:700;'
            f'color:{self._INK};letter-spacing:5px;'
        )
        h.addWidget(brand)
        ver = QLabel('/ v2.0')
        ver.setStyleSheet(
            f'font-family:Consolas,monospace;font-size:10px;color:{self._TEXT_3};'
            f'letter-spacing:1.5px;'
        )
        h.addWidget(ver)
        h.addSpacing(8)

        # Nav links
        for idx, label, locked_init in self._ITEMS:
            btn = QPushButton(label)
            btn.setFlat(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _, i=idx: self._on_click(i))
            self._buttons[idx] = btn
            h.addWidget(btn)

        h.addStretch()

        # Clock on the right
        self._time_lbl = QLabel(datetime.now().strftime('%H:%M'))
        self._time_lbl.setStyleSheet(
            f'font-family:Consolas,monospace;font-size:10px;color:{self._TEXT_3};'
            f'letter-spacing:1.4px;'
        )
        h.addWidget(self._time_lbl)
        self._clock = QTimer(self)
        self._clock.setInterval(30_000)
        self._clock.timeout.connect(lambda: self._time_lbl.setText(
            datetime.now().strftime('%H:%M')))
        self._clock.start()

        self._refresh_styles()

    def _on_click(self, idx: int):
        if idx in self._locked:
            return
        self._active = idx
        self._refresh_styles()
        self.nav_clicked.emit(idx)

    def _refresh_styles(self):
        for idx, btn in self._buttons.items():
            if idx in self._locked:
                btn.setEnabled(False)
                btn.setStyleSheet(
                    f'QPushButton{{background:transparent;border:none;'
                    f'color:#CBD2DC;font-size:12px;padding:8px 6px;}}'
                )
            elif idx == self._active:
                btn.setEnabled(True)
                btn.setStyleSheet(
                    f'QPushButton{{background:transparent;border:none;'
                    f'color:{self._INK};font-size:12px;font-weight:700;'
                    f'padding:8px 6px;}}'
                    f'QPushButton:hover{{color:{self._INK};}}'
                )
            else:
                btn.setEnabled(True)
                btn.setStyleSheet(
                    f'QPushButton{{background:transparent;border:none;'
                    f'color:{self._TEXT_2};font-size:12px;padding:8px 6px;}}'
                    f'QPushButton:hover{{color:{self._INK};}}'
                )

    # Public API matching the old sidebar contract
    def set_active(self, idx: int):
        if idx in self._locked:
            return
        self._active = idx
        self._refresh_styles()

    def unlock_stages(self, indices: list[int]):
        for i in indices:
            self._locked.discard(i)
        self._refresh_styles()


class SimMainWindow(QMainWindow):
    def __init__(self, df, regressor, data_source: str = 'Live Data',
                 cv_rmse: float = 0.0, regressors: dict | None = None,
                 store: AgentTrustStore | None = None, parent=None):
        super().__init__(parent)
        self._df = df
        self._regressor = regressor
        self._regressors = regressors or {'gas': regressor}
        self._cv_rmse = cv_rmse
        self._rag = RagEngine()
        self._agents = list(DEFAULT_AGENTS)
        self._last_scenario: dict = {}
        self._last_swarm_mode: bool = False
        self._last_parallel_n: int = 4
        self._swarm_thread: SwarmThread | None = None
        self._food_thread: DebateThread | None = None
        self._elec_thread: DebateThread | None = None
        self._food_engine: DebateEngine | None = None
        self._elec_engine: DebateEngine | None = None
        self._synth_thread: SynthesizerThread | None = None
        self._chain_thread: CausalChainThread | None = None
        self._reco_thread: PolicyRecoThread | None = None
        self._brief_thread: LiveBriefThread | None = None
        self._brief_timer: QTimer | None = None
        self._gas_verdict: str = ''
        self._food_verdict: str = ''
        self._elec_verdict: str = ''
        self._gas_estimate = None
        self._food_estimate = None
        self._elec_estimate = None
        self._live_brief: LiveDataBrief | None = None
        self._last_scenario_obj = None   # Scenario dataclass saved for _on_brief_ready
        self._debates_started: bool = False

        self._store: AgentTrustStore = store or AgentTrustStore()
        self._current_run_id: int | None = None
        self._doe_checker: DOECheckerThread = DOECheckerThread(self._store)
        self._doe_checker.start()

        self.setWindowTitle('Strata · Philippine Economic Simulation')
        self.setMinimumSize(1200, 720)
        self.setStyleSheet('background:#F7F8FA;')

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)            # vertical now (was horizontal)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Global top nav (replaces the sidebar) ────────────────────────────
        self._sidebar = _TopNavBar()           # name preserved for downstream
        self._sidebar.nav_clicked.connect(self._on_stage_changed)
        root.addWidget(self._sidebar)

        self._stack = QStackedWidget()
        root.addWidget(self._stack, stretch=1)

        # ── Build the panels ────────────────────────────────────────────────
        self._economy_overview = EconomyOverviewWidget(self._df)

        self._landing = LandingPanel(store=self._store)
        self._stage3 = Stage3CanvasPanel(self._rag, self._agents, self._regressor,
                                         self._df, self._cv_rmse)
        self._stage4 = Stage4ReportPanel()
        self._stage5 = Stage5InteractPanel(self._rag, self._agents, self._regressor,
                                           self._df, self._cv_rmse)

        self._stage3_container = QStackedWidget()
        self._stage3_container.addWidget(self._stage3)
        self._stage3_swarm = Stage3SwarmPanel(store=self._store)
        self._stage3_container.addWidget(self._stage3_swarm)

        self._agent_perf = AgentPerformancePanel(self._store)
        self._accuracy_view = AccuracyView()

        # ── Wrap the landing in a vertical scroll area (website-style) ──────
        landing_scroll = QScrollArea()
        landing_scroll.setWidgetResizable(True)
        landing_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        landing_scroll.setStyleSheet(
            'QScrollArea{border:none;background:#FAFAF8;}'
            'QScrollBar:vertical{width:8px;background:transparent;border:none;}'
            'QScrollBar::handle:vertical{background:#D1D5DB;border-radius:4px;'
            '  min-height:30px;}'
            'QScrollBar::handle:vertical:hover{background:#9CA3AF;}'
            'QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0;}'
            'QScrollBar::add-page:vertical,QScrollBar::sub-page:vertical{background:transparent;}'
        )
        landing_scroll.setWidget(self._landing)
        self._landing_scroll = landing_scroll

        # Stack order: 0=Home(scroll), 1=Overview, 2=Simulation, 3=Report,
        #              4=Interact, 5=AgentPerf, 6=Methodology & Accuracy
        for widget in (landing_scroll, self._economy_overview,
                       self._stage3_container, self._stage4, self._stage5,
                       self._agent_perf, self._accuracy_view):
            self._stack.addWidget(widget)

        # Wire DOE checker → agent perf panel refresh
        self._doe_checker.grades_applied.connect(
            lambda _: self._agent_perf.refresh()
        )

        # Wire signals
        self._stage3.simulation_complete.connect(self._on_simulation_complete)
        self._stage3_swarm.swarm_complete.connect(self._on_swarm_complete)
        self._stage5.rerun_requested.connect(self._on_rerun_requested)
        # Landing page: one-click auto run with live data
        self._landing.run_requested.connect(self._on_landing_run)
        self._landing.view_overview_requested.connect(
            lambda: (self._sidebar.set_active(1), self._stack.setCurrentIndex(1)))
        self._landing.view_performance_requested.connect(
            lambda: (self._sidebar.set_active(5), self._stack.setCurrentIndex(5)))

        corpus_path = Path(__file__).parent.parent / 'assets' / 'corpus' / 'neda_2024_2026.txt'
        if corpus_path.exists():
            self._rag.add_text('neda_2024_2026', corpus_path.read_text(encoding='utf-8'))

    def closeEvent(self, event):
        self._doe_checker.stop()
        self._doe_checker.wait()
        super().closeEvent(event)

    def _on_stage_changed(self, idx: int):
        self._stack.setCurrentIndex(idx)
        if idx == 5:        # Agent Performance is now index 5
            self._agent_perf.refresh()

    def _on_landing_run(self):
        """Click on the landing 'RUN SIMULATION' button — auto-derive a scenario
        from sensible defaults (refined further once LiveDataBrief arrives) and
        kick off the swarm in default swarm mode + 4 parallel groups."""
        self._landing.set_busy(True)
        # Use defaults — real values get computed in _on_brief_ready once
        # LiveDataBrief lands. The brief drives oil/usd/weather; bsp_rate +
        # demand_index stay at sensible defaults.
        sc = Scenario(oil_pct=0.0, usd_pct=0.0, bsp_rate=6.5, demand_index=72.0)
        self._on_run_requested(sc, self._agents, swarm_mode=True, parallel_n=4)

    def _on_rerun_requested(self, scenario_dict: dict):
        sc = Scenario(
            oil_pct=scenario_dict.get('oil_pct', 0.0),
            usd_pct=scenario_dict.get('usd_pct', 0.0),
            bsp_rate=scenario_dict.get('bsp_rate', 6.5),
            demand_index=scenario_dict.get('demand_index', 72.0),
        )
        self._on_run_requested(sc, self._agents, self._last_swarm_mode, self._last_parallel_n)

    def _on_run_requested(self, scenario, agents, swarm_mode: bool = False,
                          parallel_n: int = 4):
        # ── Store state for use in _on_brief_ready ────────────────────────────
        self._last_scenario = scenario.to_dict()
        self._last_scenario_obj = scenario
        self._last_swarm_mode = swarm_mode
        self._last_parallel_n = parallel_n
        self._debates_started = False
        self._gas_verdict = ''
        self._food_verdict = ''
        self._elec_verdict = ''
        self._gas_estimate = None
        self._food_estimate = None
        self._elec_estimate = None
        self._live_brief = None

        # Fetch live PH retail gas price (fast, sync OK)
        if not swarm_mode:
            try:
                self._last_scenario['current_price'] = fetch_live_retail_price()
            except Exception:
                pass

        # Unlock the Simulation nav button the moment the run starts — so the
        # user can leave to Home and click back in to watch progress mid-run.
        # Report + Interact stay locked until the run completes (no valid
        # results to view yet).
        self._sidebar.unlock_stages([2])
        # Navigate to Simulation canvas immediately so the user sees something
        self._sidebar.set_active(2)
        self._stack.setCurrentIndex(2)
        if swarm_mode:
            self._stage3_container.setCurrentIndex(1)
            self._stage3_swarm.reset()
        else:
            self._stage3_container.setCurrentIndex(0)

        # Cancel any previous brief fetch
        if self._brief_timer is not None:
            self._brief_timer.stop()
        if self._brief_thread is not None:
            self._brief_thread.ready.disconnect()

        # Start live data brief fetch — debates begin when it's ready
        self._brief_thread = LiveBriefThread()
        self._brief_thread.ready.connect(self._on_brief_ready)
        self._brief_thread.start()

        # Fallback: if brief hasn't arrived in 9 seconds, start debates without it
        self._brief_timer = QTimer(self)
        self._brief_timer.setSingleShot(True)
        self._brief_timer.timeout.connect(lambda: self._on_brief_ready(None))
        self._brief_timer.start(9000)

    def _on_brief_ready(self, brief: LiveDataBrief | None):
        """Fires when LiveDataBrief fetch completes OR the 9s fallback fires.

        All debate/swarm startup lives here so the brief is guaranteed to be
        available (or gracefully absent) before any DebateEngine is constructed.
        """
        # Stop the fallback timer if brief arrived on time
        if self._brief_timer is not None:
            self._brief_timer.stop()
            self._brief_timer = None

        # Guard against double-fire (brief + timer race each other)
        if self._debates_started:
            return
        self._debates_started = True

        # Detach brief thread — if this is the fallback path it may still be fetching
        if self._brief_thread is not None:
            try:
                self._brief_thread.ready.disconnect()
            except TypeError:
                pass
            self._brief_thread.quit()
            self._brief_thread = None

        self._live_brief = brief

        # Auto-refine scenario from live data (overwrites defaults from landing-page
        # quick-run). This is what makes the "no inputs" flow possible — brent_hist
        # / fx_hist drive oil_pct / usd_pct, Manila max-temp drives demand_index.
        if brief is not None:
            derived = derive_scenario_from_brief(
                brief, current_price=self._last_scenario.get('current_price'))
            self._last_scenario.update(derived)

        if self._last_swarm_mode:
            # Gas swarm runs first; food/elec debates start in _on_swarm_complete
            base_swarm = build_swarm_agents()
            evolved_swarm = get_evolved_swarm_agents(self._store, base_swarm)
            thread = SwarmThread(self._rag, self._last_scenario,
                                 parallel_n=self._last_parallel_n,
                                 data_brief=brief,
                                 ml_baseline=self._compute_ml_baseline(),
                                 evolved_agents=evolved_swarm)
            self._stage3_swarm.connect_thread(thread)
            thread.error_occurred.connect(lambda msg: print(f'Swarm error: {msg}'))
            self._swarm_thread = thread
            thread.start()
        else:
            # Non-swarm: food/elec run in parallel with gas simulation
            self._agents = get_evolved_debate_agents(self._store, list(DEFAULT_AGENTS))
            self._start_sector_debates(self._last_scenario)
            self._stage3.start_simulation(
                self._last_scenario_obj, self._agents
            )

    def _compute_ml_baseline(self) -> str:
        """Run the trained HGB regressor on the current scenario and return a formatted anchor string."""
        try:
            from ph_economic_ai.utils.preprocessing import build_features
            X_gas, _, _, _ = build_features(self._df)
            feats = X_gas[-1].copy()
            s = self._last_scenario
            feats[0] *= (1 + s.get('oil_pct', 0) / 100)   # oil_price
            feats[1] *= (1 + s.get('usd_pct', 0) / 100)   # usd_php
            predicted = float(self._regressor.predict(feats.reshape(1, -1))[0])
            current = s.get('current_price', float(self._df['gas_price'].iloc[-1]))
            delta = predicted - current
            return (
                f"{delta:+.2f} ₱/L "
                f"(±{self._cv_rmse:.2f} uncertainty, HistGradientBoosting on historical PH data)"
            )
        except Exception:
            return ''

    def _start_sector_debates(self, scenario_dict: dict):
        """Run food and electricity sector debates in parallel with the gas simulation."""
        brief = self._live_brief  # may be None if still fetching — that's OK
        self._food_engine = DebateEngine(FOOD_AGENTS, self._rag, scenario_dict,
                                          price_extractor=_extract_percent,
                                          data_brief=brief)
        self._food_thread = DebateThread(self._food_engine, rounds=1)
        self._food_thread.debate_complete.connect(self._on_food_complete)
        self._food_thread.error_occurred.connect(
            lambda msg: self._on_food_complete([])
        )
        self._food_thread.start()

        self._elec_engine = DebateEngine(ELECTRICITY_AGENTS, self._rag, scenario_dict,
                                          data_brief=brief)
        self._elec_thread = DebateThread(self._elec_engine, rounds=1)
        self._elec_thread.debate_complete.connect(self._on_elec_complete)
        self._elec_thread.error_occurred.connect(
            lambda msg: self._on_elec_complete([])
        )
        self._elec_thread.start()

        # Wire food/electricity threads to the swarm canvas so their progress
        # is animated live on the food/electricity sector clusters.
        if self._last_swarm_mode:
            self._stage3_swarm.connect_food_thread(self._food_thread)
            self._stage3_swarm.connect_elec_thread(self._elec_thread)

    def _push_sector_forecasts(self):
        try:
            self._stage4.set_sector_forecasts(
                self._gas_estimate, self._food_estimate, self._elec_estimate)
        except Exception:
            pass

    def _on_food_complete(self, responses):
        if responses and self._food_engine:
            c = self._food_engine.consensus()
            avg = c.get('weighted_avg')
            self._food_estimate = avg
            conf = c.get('confidence_pct', 0)
            avg_str = f'{avg:+.2f}%' if avg is not None else 'N/A'
            self._food_verdict = (
                f'Food price index monthly change: {avg_str} '
                f'(confidence {conf}%, range {c.get("low", 0):+.2f}% to {c.get("high", 0):+.2f}%)'
            )
            if avg is not None and 'food_price_idx' in self._df.columns:
                food_hist = self._df['food_price_idx'].dropna().tail(6).tolist()
                current_food = food_hist[-1] if food_hist else 100.0
                delta_pts = current_food * avg / 100.0
                self._economy_overview.update_food({
                    'value': current_food + delta_pts,
                    'delta': delta_pts,
                    'history': food_hist,
                    'signal_text': f'Monthly est.: {avg:+.2f}%',
                    'pressure': 'Rising' if avg > 0 else 'Stable',
                })
        else:
            self._food_verdict = '(Food sector debate unavailable.)'
        self._stage5.update_food_verdict(self._food_verdict)
        self._push_sector_forecasts()
        self._run_synthesizer_if_ready()

    def _on_elec_complete(self, responses):
        if responses and self._elec_engine:
            c = self._elec_engine.consensus()
            avg = c.get('weighted_avg')
            self._elec_estimate = avg
            conf = c.get('confidence_pct', 0)
            avg_str = f'+₱{avg:.4f}/kWh' if avg is not None else 'N/A'
            self._elec_verdict = (
                f'Electricity rate monthly change: {avg_str} '
                f'(confidence {conf}%, range {c.get("low", 0):+.4f} to {c.get("high", 0):+.4f} ₱/kWh)'
            )
            if avg is not None and 'electricity_rate' in self._df.columns:
                elec_hist = self._df['electricity_rate'].dropna().tail(6).tolist()
                current_elec = elec_hist[-1] if elec_hist else 11.20
                self._economy_overview.update_electricity({
                    'value': current_elec + avg,
                    'delta': avg,
                    'history': elec_hist,
                    'pressure': 'Rising' if avg > 0 else 'Stable',
                })
        else:
            self._elec_verdict = '(Electricity sector debate unavailable.)'
        self._stage5.update_elec_verdict(self._elec_verdict)
        self._push_sector_forecasts()
        self._run_synthesizer_if_ready()

    def _on_simulation_complete(self, responses):
        consensus = self._stage3.engine.consensus()
        self._gas_verdict = str(consensus)
        if responses:
            estimates = [r.price_estimate for r in responses if r.price_estimate is not None]
            scores = QualityScorer.score_responses(responses, estimates)
            run_quality = QualityScorer.run_quality(responses, estimates)
            self._current_run_id = self._store.save_run(
                scenario=self._last_scenario,
                final_estimate=consensus.get('weighted_avg') if isinstance(consensus, dict) else None,
                confidence_pct=int(consensus.get('confidence_pct', 0)) if isinstance(consensus, dict) else 0,
            )
            self._store.update_run_quality(self._current_run_id, run_quality)
            response_dicts = []
            for r in responses:
                sc = scores.get(r.agent_name, {})
                response_dicts.append({
                    'agent_name': r.agent_name, 'round_num': r.round_num,
                    'estimate': r.price_estimate, 'statement': r.statement,
                    'citation_count': sc.get('citation_count', 0),
                    'has_causal_chain': sc.get('has_causal_chain', 0),
                    'internal_score': sc.get('overall', 0.5),
                    'model_used': next(
                        (a.model for a in self._agents if a.name == r.agent_name), ''),
                })
            self._store.save_agent_responses(self._current_run_id, response_dicts)
            for agent_name, sc in scores.items():
                self._store.update_trust(agent_name, internal_score=sc['overall'])
        self._stage4.populate(responses, consensus, self._regressor,
                              self._df, self._cv_rmse,
                              self._last_scenario)
        # Derive regional estimates using NCR (classic mode) as the single anchor
        self._stage4.set_regional_estimates(
            derive_regional_estimates(consensus.get('weighted_avg'))
        )
        self._stage5.update_context(responses, self._stage3.scenario())
        self._stage5.set_debate_engine(self._stage3.engine)
        self._stage5.update_gas_verdict(self._gas_verdict)
        # Sidebar indices: 2=Simulation, 3=Report, 4=Interact
        self._sidebar.unlock_stages([2, 3, 4])
        self._sidebar.set_active(3)
        self._stack.setCurrentIndex(3)
        # Update gas sector card on Economy Overview
        avg = consensus.get('weighted_avg') or 0.0
        live_price = self._last_scenario.get('current_price', float(self._df['gas_price'].iloc[-1]))
        gas_hist = self._df['gas_price'].dropna().tail(6).tolist()
        self._economy_overview.update_gas({
            'value': live_price + avg,
            'delta': avg,
            'history': gas_hist,
            'pressure': 'Rising' if avg > 0 else 'Stable',
        })
        self._run_synthesizer_if_ready()

    def _on_swarm_complete(self, master_verdict):
        all_responses = getattr(master_verdict, 'all_responses', [])
        if all_responses:
            estimates = [r.price_estimate for r in all_responses if r.price_estimate is not None]
            scores = QualityScorer.score_responses(all_responses, estimates)
            run_quality = QualityScorer.run_quality(all_responses, estimates)
            self._current_run_id = self._store.save_run(
                scenario=self._last_scenario,
                final_estimate=master_verdict.final_estimate,
                confidence_pct=master_verdict.confidence_pct,
            )
            self._store.update_run_quality(self._current_run_id, run_quality)
            response_dicts = []
            for r in all_responses:
                sc = scores.get(r.agent_name, {})
                response_dicts.append({
                    'agent_name': r.agent_name, 'round_num': r.round_num,
                    'estimate': r.price_estimate, 'statement': r.statement,
                    'citation_count': sc.get('citation_count', 0),
                    'has_causal_chain': sc.get('has_causal_chain', 0),
                    'internal_score': sc.get('overall', 0.5),
                    'model_used': '',
                })
            self._store.save_agent_responses(self._current_run_id, response_dicts)
            for agent_name, sc in scores.items():
                self._store.update_trust(agent_name, internal_score=sc['overall'])
        self._gas_verdict = str(master_verdict)
        self._gas_estimate = getattr(master_verdict, 'final_estimate', None)
        self._push_sector_forecasts()
        self._stage4.populate_swarm(
            master_verdict, self._regressor, self._df, self._cv_rmse,
            self._last_scenario,
        )
        # Swarm mode has per-group anchor estimates → full regional disaggregation
        self._stage4.set_regional_estimates(master_verdict.regional_estimates or {})
        self._stage5.set_swarm_context(master_verdict, self._last_scenario)
        self._stage5.update_gas_verdict(self._gas_verdict)
        # Sidebar indices: 2=Simulation, 3=Report, 4=Interact
        self._sidebar.unlock_stages([2, 3, 4])
        self._sidebar.set_active(3)
        self._stack.setCurrentIndex(3)
        self._landing.set_busy(False)
        # Update gas sector card on Economy Overview
        avg = master_verdict.final_estimate or 0.0
        live_price = self._last_scenario.get('current_price', float(self._df['gas_price'].iloc[-1]))
        gas_hist = self._df['gas_price'].dropna().tail(6).tolist()
        self._economy_overview.update_gas({
            'value': live_price + avg,
            'delta': avg,
            'history': gas_hist,
            'pressure': 'Rising' if avg > 0 else 'Stable',
        })
        # Gas is done — now start food and electricity debates
        self._start_sector_debates(self._last_scenario)
        self._run_synthesizer_if_ready()

    def _run_synthesizer_if_ready(self):
        """Launch Economy Synthesizer + Causal Chain once all three sector verdicts are in."""
        if not self._gas_verdict or not self._food_verdict or not self._elec_verdict:
            return
        self._synth_thread = SynthesizerThread(
            gas_verdict=self._gas_verdict,
            food_verdict=self._food_verdict,
            elec_verdict=self._elec_verdict,
        )
        self._synth_thread.finished.connect(self._economy_overview.update_summary)
        self._synth_thread.start()

        # BSP alert — compute CPI basket impact from all three verdicts
        self._run_bsp_alert()

        # Causal chain synthesis
        self._chain_thread = CausalChainThread(
            gas_verdict=self._gas_verdict,
            food_verdict=self._food_verdict,
            elec_verdict=self._elec_verdict,
            scenario=self._last_scenario,
        )
        self._chain_thread.chain_ready.connect(self._stage4.set_chain)
        self._chain_thread.error.connect(lambda msg: print(f'Chain error: {msg}'))

        # Policy reco — configured now, started only after chain releases the GPU
        self._reco_thread = PolicyRecoThread(
            gas_verdict=self._gas_verdict,
            food_verdict=self._food_verdict,
            elec_verdict=self._elec_verdict,
            scenario=self._last_scenario,
        )
        self._reco_thread.reco_ready.connect(self._stage4.set_policy_recos)
        self._reco_thread.error.connect(lambda msg: print(f'Reco error: {msg}'))
        self._chain_thread.finished.connect(self._reco_thread.start)

        self._chain_thread.start()

    def _run_bsp_alert(self):
        """Compute CPI basket impact and push BSP alert to Stage 4."""
        try:
            # Extract numeric estimates from stored verdict strings
            import re
            gas_m = re.search(r'([+\-])\s*₱(\d+\.?\d*)', self._gas_verdict)
            gas_v = (1 if gas_m.group(1) == '+' else -1) * float(gas_m.group(2)) if gas_m else None

            food_m = re.search(r'([+\-])\s*(\d+\.?\d*)\s*%', self._food_verdict)
            food_v = (1 if food_m.group(1) == '+' else -1) * float(food_m.group(2)) if food_m else None

            elec_m = re.search(r'([+\-])\s*₱(\d+\.?\d*)\s*/\s*kWh', self._elec_verdict)
            elec_v = (1 if elec_m.group(1) == '+' else -1) * float(elec_m.group(2)) if elec_m else None

            alert = LiveDataBrief.check_bsp_alert(gas_v, food_v, elec_v)
            self._stage4.set_bsp_alert(alert)
        except Exception:
            pass
