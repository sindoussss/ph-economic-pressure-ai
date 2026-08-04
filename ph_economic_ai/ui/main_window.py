import logging
from pathlib import Path
from datetime import datetime

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QStackedWidget,
    QScrollArea, QFrame, QLabel, QPushButton,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal

from ph_economic_ai.engine import anchoring, llm, recall, vintage
from ph_economic_ai.engine.rag import RagEngine
from ph_economic_ai.engine.debate import (
    DEFAULT_AGENTS, FOOD_AGENTS, ELECTRICITY_AGENTS,
    SynthesizerThread, DebateEngine, DebateThread, _extract_percent,
    _extract_electricity_change,
)
from ph_economic_ai.engine.swarm import SwarmThread, fetch_live_retail_price, derive_regional_estimates, build_swarm_agents
from ph_economic_ai.engine.live_data import (
    LiveDataBrief, LiveBriefThread, CausalChainThread, PolicyRecoThread,
    derive_scenario_from_brief,
)
from ph_economic_ai.engine import price_calendar as _calendar
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
from ph_economic_ai.ui.learning_view import LearningView
from ph_economic_ai.ui.pressure_monitor import PressureMonitorPanel
from ph_economic_ai.ui.quota_notice import QuotaNotice


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
        (7, 'Monitor',     False),
        (2, 'Simulation',  True),
        (3, 'Report',      True),
        (6, 'Learning',    False),
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
        self._locked: set[int] = {2, 3}
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


def _model_identity() -> str:
    """Which models actually answered this run, for the recall key.

    The provider is resolved from the environment, so exporting a GROQ_API_KEY
    silently switches provider and model without anything in the app changing. A
    3b local answer and a 70b hosted answer to the same question are not the same
    answer, and recalling one for the other would be the cache lying about what
    produced the number.

    `llm.provenance_id` reports what SERVED each tier, not what was configured
    for it. Those differ whenever a hosted tier hit its ceiling and fell back to
    local, and that run must not be recalled for a later one that did not fall
    back — the two verdicts came from different models and only the provenance
    says so.
    """
    try:
        return llm.provenance_id()
    except Exception:
        return 'unknown'


def _format_gas_verdict(
    estimate: float | None,
    agreement_pct: float | None = None,
    low: float | None = None,
    high: float | None = None,
    regional: list | None = None,
) -> str:
    """Human-readable gas verdict for the downstream LLM prompts.

    This used to be `str(master_verdict)` — a raw dataclass repr. Two things
    went wrong with that. The BSP banner regexed it for a peso figure and hit a
    nested regional verdict instead of the consensus, and the causal-chain
    prompt truncates its input to 600 characters, so the model was handed a
    mangled Python object and invented its own numbers (+₱13.70/L against a
    +₱2.54/L consensus). Food and electricity always formatted prose here; gas
    was the odd one out.
    """
    if estimate is None:
        return '(Gas sector verdict unavailable.)'
    parts = [f'Retail gasoline monthly change: {estimate:+.2f} ₱/L']
    if agreement_pct is not None:
        parts.append(f'agent agreement {agreement_pct:.0f}%')
    if low is not None and high is not None:
        parts.append(f'range {low:+.2f} to {high:+.2f} ₱/L')
    summary = f'{parts[0]} ({", ".join(parts[1:])})' if len(parts) > 1 else parts[0]

    for pair, value in (regional or []):
        if value is None:
            continue
        name = ' & '.join(pair) if isinstance(pair, (tuple, list)) else str(pair)
        summary += f'\n  {name}: {value:+.2f} ₱/L'
    return summary


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
        self._sector_responses: dict = {}
        self._gas_agreement = 0
        self._food_agreement = 0
        self._elec_agreement = 0
        self._live_brief: LiveDataBrief | None = None
        self._last_scenario_obj = None   # Scenario dataclass saved for _on_brief_ready
        self._debates_started: bool = False

        self._store: AgentTrustStore = store or AgentTrustStore()
        self._current_run_id: int | None = None
        # Recall state. `_run_key` is this run's vintage fingerprint;
        # `_force_fresh_run` is the user's explicit override, consumed once so a
        # forced re-run does not disable recall for the rest of the session.
        self._run_key: str = ''
        self._run_inputs: dict = {}
        self._force_fresh_run: bool = False
        self._recalled_from: int | None = None
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

        # Quota banner, directly under the nav so it is visible from any stage.
        # A 429 used to reach the user as a stalled progress bar and a traceback
        # in a console they may not have open.
        self._quota_notice = QuotaNotice()
        root.addWidget(self._quota_notice)
        llm.on_quota_event(self._quota_notice.report)

        self._stack = QStackedWidget()
        root.addWidget(self._stack, stretch=1)

        # ── Build the panels ────────────────────────────────────────────────
        self._economy_overview = EconomyOverviewWidget(self._df)

        self._landing = LandingPanel(store=self._store)
        self._stage3 = Stage3CanvasPanel(self._rag, self._agents, self._regressor,
                                         self._df, self._cv_rmse)
        self._stage5 = Stage5InteractPanel(self._rag, self._agents, self._regressor,
                                           self._df, self._cv_rmse)
        self._stage4 = Stage4ReportPanel(interact_panel=self._stage5)

        self._stage3_container = QStackedWidget()
        self._stage3_container.addWidget(self._stage3)
        self._stage3_swarm = Stage3SwarmPanel(store=self._store)
        self._stage3_container.addWidget(self._stage3_swarm)

        self._agent_perf = AgentPerformancePanel(self._store)
        # The store, so the tab can lead with THIS app's record before the
        # offline benchmark's. Without it the screen answers "how accurate is
        # this app" with a backtest of months the app never ran.
        self._accuracy_view = AccuracyView(store=self._store)
        self._learning = LearningView(self._store)
        self._monitor = PressureMonitorPanel(self._rag, store=self._store)

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

        # Stack order: 0=Home(scroll), 1=Overview, 2=Simulation, 3=Report(workbench),
        #              4=AgentPerf, 5=Methodology & Accuracy, 6=Learning, 7=Monitor
        for widget in (landing_scroll, self._economy_overview,
                       self._stage3_container, self._stage4,
                       self._agent_perf, self._accuracy_view, self._learning,
                       self._monitor):
            self._stack.addWidget(widget)

        # Wire DOE checker → agent perf panel refresh
        self._doe_checker.grades_applied.connect(
            lambda _: self._agent_perf.refresh()
        )

        # Wire signals
        self._stage3.simulation_complete.connect(self._on_simulation_complete)
        self._stage3_swarm.swarm_complete.connect(self._on_swarm_complete)
        self._stage3_swarm.view_report_requested.connect(self._goto_report)
        self._stage5.rerun_requested.connect(self._on_rerun_requested)
        # Landing page: one-click auto run with live data
        self._landing.run_requested.connect(self._on_landing_run)
        self._landing.view_overview_requested.connect(
            lambda: (self._sidebar.set_active(1), self._stack.setCurrentIndex(1)))
        self._landing.view_performance_requested.connect(
            lambda: (self._sidebar.set_active(4), self._stack.setCurrentIndex(4)))

        corpus_path = Path(__file__).parent.parent / 'assets' / 'corpus' / 'neda_2024_2026.txt'
        if corpus_path.exists():
            self._rag.add_text('neda_2024_2026', corpus_path.read_text(encoding='utf-8'))

    def closeEvent(self, event):
        self._doe_checker.stop()
        self._doe_checker.wait()
        super().closeEvent(event)

    def _on_stage_changed(self, idx: int):
        self._stack.setCurrentIndex(idx)
        if idx == 4:        # Agent Performance is now index 4
            self._agent_perf.refresh()
        if idx == 6:
            self._learning.refresh(getattr(self, '_current_run_id', None))

    def _on_landing_run(self):
        """Click on the landing 'RUN' button — run the Monitor forum first (the
        present-pressure read + live debate), then chain the tournament swarm the
        moment the Monitor finishes. Scenario defaults are refined from the live
        brief once it lands (see _on_brief_ready)."""
        self._landing.set_busy(True)
        self._pending_swarm_scenario = Scenario(
            oil_pct=0.0, usd_pct=0.0, bsp_rate=6.5, demand_index=72.0)

        # Stage 1: the Monitor forum. Show it so the user watches the debate.
        self._sidebar.unlock_stages([7])
        self._sidebar.set_active(7)
        self._stack.setCurrentIndex(7)

        # Chain the tournament to the Monitor finishing (one-shot connection).
        try:
            self._monitor.run_finished.disconnect(self._on_monitor_finished_run_swarm)
        except TypeError:
            pass
        self._monitor.run_finished.connect(self._on_monitor_finished_run_swarm)
        self._monitor.start()

    def _on_monitor_finished_run_swarm(self):
        """Stage 2: once the Monitor forum is done, run the tournament swarm."""
        try:
            self._monitor.run_finished.disconnect(self._on_monitor_finished_run_swarm)
        except TypeError:
            pass
        sc = getattr(self, '_pending_swarm_scenario', None) or Scenario(
            oil_pct=0.0, usd_pct=0.0, bsp_rate=6.5, demand_index=72.0)
        self._on_run_requested(sc, self._agents, swarm_mode=True, parallel_n=4)

    def _on_rerun_requested(self, scenario_dict: dict):
        sc = Scenario(
            oil_pct=scenario_dict.get('oil_pct', 0.0),
            usd_pct=scenario_dict.get('usd_pct', 0.0),
            bsp_rate=scenario_dict.get('bsp_rate', 6.5),
            demand_index=scenario_dict.get('demand_index', 72.0),
        )
        # An explicit re-run is the escape hatch from recall. The user asking for
        # the question to be answered again has said, in the only way the UI
        # offers, that they do not want the stored answer. Recall is a default,
        # not a policy.
        self._force_fresh_run = True
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
        self._sector_responses: dict = {}
        self._gas_agreement = 0
        self._food_agreement = 0
        self._elec_agreement = 0
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

        # What makes this run recallable: a coarse bucket (this day, this pricing
        # week, these models) and the exact inputs it was answered on. Computed
        # here because this is the first point at which the scenario is final.
        #
        # Provenance is cleared first, so this key describes the models the run
        # INTENDS to use. It is recomputed after the run from the models that
        # actually served — see `_on_swarm_complete`. The two differ exactly when
        # a hosted tier hit its ceiling and fell back, and a run decided by a
        # fallback judge must not be recalled for one decided by the real thing.
        llm.reset_provenance()
        self._run_key = vintage.vintage_key(model_id=_model_identity())
        self._run_inputs = vintage.input_snapshot(self._last_scenario, brief)

        recalled = None if self._force_fresh_run else recall.find_recall(
            self._store, self._run_key, self._run_inputs)
        if recalled is not None:
            # Nothing about the question has changed since that run, so re-asking
            # it would spend 39 model calls to sample a different answer to the
            # same inputs, which is what made eight runs on one afternoon disagree.
            logging.info('run: recalled run #%s for key %s',
                         recalled.run_id, self._run_key)
            self._replay_recalled_run(recalled)
            return
        self._force_fresh_run = False

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

    def _recent_food_mom_pcts(self) -> list:
        """Trailing month-on-month food inflation, %, for the persistence anchor."""
        if 'food_price_idx' not in self._df.columns:
            return []
        s = self._df['food_price_idx'].dropna()
        return (s.pct_change().dropna().tail(6) * 100.0).tolist()

    def _food_anchor(self, scenario: dict) -> float:
        return anchoring.food_persistence_anchor(
            self._recent_food_mom_pcts(), oil_pct=scenario.get('oil_pct', 0.0))

    def _elec_anchor(self, scenario: dict) -> float:
        return anchoring.electricity_passthrough_anchor(
            scenario.get('oil_pct', 0.0), scenario.get('usd_pct', 0.0))

    def _start_sector_debates(self, scenario_dict: dict):
        """Run food and electricity sector debates in parallel with the gas simulation.

        Each sector is anchored to the quantity carrying the right *scale*: food to
        its recent own-trend (commodity drivers are a null for it), electricity to
        the formulaic fuel pass-through in its generation charge. The anchor is
        injected as a prior so the weak agents start from the right magnitude — it
        is a guard, not a forecast, and under the corrected baseline pool (§4.7)
        neither sector is predictable beyond naive.
        """
        brief = self._live_brief  # may be None if still fetching — that's OK

        food_anchor = self._food_anchor(scenario_dict)
        elec_anchor = self._elec_anchor(scenario_dict)
        # Seed the sector forecasts with their anchors immediately. The anchor is
        # deterministic and known before the debate runs, so food/electricity
        # show a grounded provisional number right away instead of a blank dash
        # for the minute or two the debates take; each is refined in place when
        # its debate completes.
        self._food_estimate = food_anchor
        self._elec_estimate = elec_anchor
        self._push_sector_forecasts()

        food_note = (
            f"BASELINE: recent food inflation runs about {food_anchor:+.2f}% per "
            f"month (its own trend — commodity/oil prices are a poor predictor of "
            f"food here). Start from this and adjust only for a clear harvest, "
            f"weather, or import-price signal. Monthly food moves are rarely more "
            f"than ~1.5 percentage points from this trend."
        )
        self._food_engine = DebateEngine(FOOD_AGENTS, self._rag, scenario_dict,
                                          price_extractor=_extract_percent,
                                          data_brief=brief, anchor_note=food_note,
                                          sector='food')
        self._food_thread = DebateThread(self._food_engine, rounds=1)
        self._food_thread.debate_complete.connect(self._on_food_complete)
        self._food_thread.error_occurred.connect(
            lambda msg: self._on_food_complete([])
        )
        self._food_thread.start()

        elec_note = (
            f"MECHANICAL PASS-THROUGH: the fuel-indexed part of the Meralco "
            f"generation charge implies a rate change of about {elec_anchor:+.4f} "
            f"₱/kWh from this oil and FX move. This is a formulaic, observable "
            f"pass-through — start from it and adjust only for a demand or "
            f"regulatory factor it misses."
        )
        self._elec_engine = DebateEngine(ELECTRICITY_AGENTS, self._rag, scenario_dict,
                                          price_extractor=_extract_electricity_change,
                                          data_brief=brief, anchor_note=elec_note,
                                          sector='electricity')
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

    _SECTOR_UNITS = {'gas': '₱/L', 'food': '%', 'electricity': '₱/kWh'}

    def _sector_readings(self) -> list:
        """SectorReadings for the explanation layer, built from what actually ran.

        Drivers are quoted from the agents' own statements and sources are the RAG
        sources those agents were allowed to read, so the explanation is grounded
        in this run rather than in the model's general knowledge.
        """
        from ph_economic_ai.engine.pressure_brief import SectorReading

        estimates = {'gas': self._gas_estimate, 'food': self._food_estimate,
                     'electricity': self._elec_estimate}
        agreements = {'gas': getattr(self, '_gas_agreement', 0),
                      'food': getattr(self, '_food_agreement', 0),
                      'electricity': getattr(self, '_elec_agreement', 0)}
        engines = {'food': getattr(self, '_food_engine', None),
                   'electricity': getattr(self, '_elec_engine', None)}

        readings = []
        for sector, estimate in estimates.items():
            responses = (getattr(self, '_sector_responses', {}) or {}).get(sector, [])
            drivers = []
            for r in responses:
                stmt = (getattr(r, 'statement', '') or '').strip()
                if not stmt:
                    continue
                first = stmt.split('. ')[0].strip().rstrip('.')
                if 20 < len(first) < 200:
                    drivers.append(first)
                if len(drivers) >= 3:
                    break

            sources: list[str] = []
            engine = engines.get(sector)
            agents = getattr(engine, '_agents', None) or getattr(engine, 'agents', None)
            if agents:
                for a in agents:
                    for s in (getattr(a, 'rag_sources', None) or []):
                        if s not in sources:
                            sources.append(s)
            elif sector == 'gas':
                from ph_economic_ai.engine.swarm import _ROLE_RAG
                for lst in _ROLE_RAG.values():
                    for s in lst:
                        if s not in sources:
                            sources.append(s)

            direction = ('unknown' if estimate is None
                         else 'rising' if estimate > 0
                         else 'easing' if estimate < 0 else 'flat')
            readings.append(SectorReading(
                sector=sector, direction=direction, estimate=estimate,
                unit=self._SECTOR_UNITS[sector], confidence=agreements.get(sector, 0),
                drivers=drivers, sources=sources))
        return readings

    def _push_sector_forecasts(self):
        try:
            # Explanations first, so the rows render with their why/when/source
            # lines already attached rather than appearing a frame later.
            try:
                from ph_economic_ai.engine import llm as _llm
                self._stage4.set_sector_explanations(
                    self._sector_readings(), complete=_llm.complete)
            except Exception:
                # Logged, never silent: a bare `pass` here hid an AttributeError
                # that dropped every explanation from the report while the
                # numbers rendered normally, so nothing looked wrong.
                logging.exception('sector explanations could not be built')
            self._stage4.set_sector_forecasts(
                self._gas_estimate, self._food_estimate, self._elec_estimate)
        except Exception:
            pass

    def _on_food_complete(self, responses):
        scenario = self._last_scenario or {}
        anchor = self._food_anchor(scenario)
        raw_avg, conf = None, 0
        if responses and self._food_engine:
            c = self._food_engine.consensus()
            raw_avg = c.get('weighted_avg')
            conf = c.get('confidence_pct', 0)
        # Persistence-anchored: trust the agents near their own trend, clamp a
        # drift, and fall back to the trend outright when the debate produced
        # nothing — so food is never a blank, even on total debate failure.
        rec = anchoring.reconcile_estimate(
            raw_avg, anchor, tolerance=anchoring.FOOD_TOLERANCE_PCT)
        self._food_estimate = rec.value
        self._sector_responses['food'] = list(responses or [])
        self._food_agreement = conf
        note = anchoring.explain(rec, unit='%', anchor_label='own-trend persistence')
        self._food_verdict = (
            f'Food price index monthly change: {rec.value:+.2f}% '
            f'(confidence {conf}%). {note}'
        )
        if 'food_price_idx' in self._df.columns:
            food_hist = self._df['food_price_idx'].dropna().tail(6).tolist()
            current_food = food_hist[-1] if food_hist else 100.0
            delta_pts = current_food * rec.value / 100.0
            self._economy_overview.update_food({
                'value': current_food + delta_pts,
                'delta': delta_pts,
                'history': food_hist,
                'signal_text': f'Monthly est.: {rec.value:+.2f}%',
                'pressure': 'Rising' if rec.value > 0 else 'Stable',
            })
        self._stage5.update_food_verdict(self._food_verdict)
        self._push_sector_forecasts()
        if self._store is not None and self._current_run_id is not None:
            try:
                self._store.update_run_sectors(
                    self._current_run_id, self._food_estimate, self._elec_estimate)
            except Exception:
                pass
        self._attach_run_snapshot()
        self._run_synthesizer_if_ready()

    def _on_elec_complete(self, responses):
        scenario = self._last_scenario or {}
        anchor = self._elec_anchor(scenario)
        raw_avg, conf = None, 0
        if responses and self._elec_engine:
            c = self._elec_engine.consensus()
            raw_avg = c.get('weighted_avg')
            conf = c.get('confidence_pct', 0)
        # Electricity's fuel pass-through is the one sector channel the benchmark
        # confirmed as genuinely predictive, so the anchor is a validated signal,
        # not just a scale. Reconcile against it exactly as fuel does.
        rec = anchoring.reconcile_estimate(
            raw_avg, anchor, tolerance=anchoring.ELECTRICITY_TOLERANCE_PHP_KWH)
        self._elec_estimate = rec.value
        self._sector_responses['electricity'] = list(responses or [])
        self._elec_agreement = conf
        note = anchoring.explain(rec, unit='₱/kWh', anchor_label='fuel pass-through')
        self._elec_verdict = (
            f'Electricity rate monthly change: {rec.value:+.4f} ₱/kWh '
            f'(confidence {conf}%). {note}'
        )
        if 'electricity_rate' in self._df.columns:
            elec_hist = self._df['electricity_rate'].dropna().tail(6).tolist()
            current_elec = elec_hist[-1] if elec_hist else 11.20
            self._economy_overview.update_electricity({
                'value': current_elec + rec.value,
                'delta': rec.value,
                'history': elec_hist,
                'pressure': 'Rising' if rec.value > 0 else 'Stable',
            })
        self._stage5.update_elec_verdict(self._elec_verdict)
        self._push_sector_forecasts()
        if self._store is not None and self._current_run_id is not None:
            try:
                self._store.update_run_sectors(
                    self._current_run_id, self._food_estimate, self._elec_estimate)
            except Exception:
                pass
        self._attach_run_snapshot()
        self._run_synthesizer_if_ready()

    def _on_simulation_complete(self, responses):
        consensus = self._stage3.engine.consensus()
        self._gas_verdict = _format_gas_verdict(
            estimate=consensus.get('weighted_avg') if isinstance(consensus, dict) else None,
            agreement_pct=consensus.get('confidence_pct') if isinstance(consensus, dict) else None,
            low=consensus.get('low') if isinstance(consensus, dict) else None,
            high=consensus.get('high') if isinstance(consensus, dict) else None,
        )
        if responses:
            estimates = [r.price_estimate for r in responses if r.price_estimate is not None]
            scores = QualityScorer.score_responses(responses, estimates)
            run_quality = QualityScorer.run_quality(responses, estimates)
            self._current_run_id = self._store.save_run(
                scenario=self._last_scenario,
                final_estimate=consensus.get('weighted_avg') if isinstance(consensus, dict) else None,
                confidence_pct=int(consensus.get('confidence_pct', 0)) if isinstance(consensus, dict) else 0,
                horizon_days=self._fuel_horizon_days(),
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
                        (llm.describe_model(a.tier) for a in self._agents
                         if a.name == r.agent_name), ''),
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
        # Sidebar indices: 2=Simulation, 3=Report
        self._sidebar.unlock_stages([2, 3])
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

    def _attach_run_snapshot(self) -> None:
        """Store this run's answer well enough to reopen it without a model.

        Called after the gas verdict lands and again as each sector finishes, so
        the stored snapshot always reflects the most complete state of the run.
        Until it exists `find_run_by_key` will not recall the row, which is the
        right behaviour for a run still in flight.
        """
        if self._store is None or self._current_run_id is None:
            return
        try:
            self._store.attach_verdict(self._current_run_id, recall.build_snapshot(
                master_verdict=getattr(self, '_master_verdict', None),
                food_estimate=self._food_estimate,
                electricity_estimate=self._elec_estimate,
                scenario=self._last_scenario,
                inputs=self._run_inputs,
            ))
        except Exception:
            # A run that finished is worth more than a snapshot of it.
            logging.exception('recall: could not store the run snapshot')

    def _replay_recalled_run(self, recalled) -> None:
        """Put a previous run back on screen instead of re-running it.

        Deliberately re-uses the ordinary completion handlers rather than a
        parallel display path. A recalled report that renders through different
        code than a fresh one is a report that can disagree with itself, and the
        whole point of recall is that the two are the same answer.
        """
        self._recalled_from = recalled.run_id
        self._current_run_id = recalled.run_id
        self._master_verdict = recalled.master_verdict
        self._food_estimate = recalled.food_estimate
        self._elec_estimate = recalled.electricity_estimate

        self._stage4.set_recall_note(recalled.describe(), recalled.drift)
        self._on_swarm_complete_display(recalled.master_verdict, recalled=True)
        self._landing.set_busy(False)

    @staticmethod
    def _fuel_horizon_days() -> float:
        """Distance to the adjustment this run is actually forecasting.

        Without this a run defaults to a flat seven days, so a forecast made on a
        Monday for tomorrow's adjustment would be graded against a price a week
        later. The card already tells the user which Tuesday it means; this makes
        the grader agree with the card.
        """
        return _calendar.describe_next_fuel_adjustment()['horizon_days']

    def _on_swarm_complete(self, master_verdict):
        all_responses = getattr(master_verdict, 'all_responses', [])
        if all_responses:
            estimates = [r.price_estimate for r in all_responses if r.price_estimate is not None]
            scores = QualityScorer.score_responses(all_responses, estimates)
            run_quality = QualityScorer.run_quality(all_responses, estimates)
            # Record the price the run actually reasoned from, not the fallback
            # constant the scenario was built with. Grading subtracts this stored
            # baseline from the observed price, so a stale value does not produce
            # a slightly-off grade, it produces a fictional one: every graded run
            # recorded the same -14.44 PHP/L "actual change", which is 84.38 minus
            # the 98.82 fallback and never any week's move.
            used_price = getattr(master_verdict, 'current_price', None)
            if used_price is not None:
                self._last_scenario = {**self._last_scenario,
                                       'current_price': float(used_price)}
            else:
                # The fetch failed, so this run has no real baseline. Drop the
                # key rather than storing the fallback: grading skips a run with
                # no baseline, and a fallback stored here would be measured
                # against as though it were an observation.
                self._last_scenario = {k: v for k, v in self._last_scenario.items()
                                       if k != 'current_price'}
            # Re-key on what actually answered. If a judge fell back to local
            # after a hosted 429, this key no longer matches the one a clean run
            # would compute, so the two are never confused for each other.
            self._run_key = vintage.vintage_key(model_id=_model_identity())
            self._current_run_id = self._store.save_run(
                scenario=self._last_scenario,
                final_estimate=master_verdict.final_estimate,
                confidence_pct=master_verdict.confidence_pct,
                horizon_days=self._fuel_horizon_days(),
                run_key=self._run_key,
                temperature=llm.DEFAULT_TEMPERATURE,
            )
            self._store.update_run_quality(self._current_run_id, run_quality)
            self._master_verdict = master_verdict
            # A gas-only snapshot, so a recall works even if the app is closed
            # before the sector debates finish. `_attach_run_snapshot` overwrites
            # it with the complete one when they do.
            self._attach_run_snapshot()
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
        try:
            self._learning.refresh(self._current_run_id)
        except Exception:
            pass
        self._stage4.set_recall_note('')
        self._on_swarm_complete_display(master_verdict)

    def _on_swarm_complete_display(self, master_verdict, recalled: bool = False):
        """Put a gas verdict on screen. Shared by a fresh run and a recalled one.

        Split out of `_on_swarm_complete` so recall renders through exactly the
        same code as a live run. A recalled report drawn by a parallel display
        path is a report that can disagree with itself, which defeats the point.
        """
        self._gas_estimate = getattr(master_verdict, 'final_estimate', None)
        self._sector_responses['gas'] = list(
            getattr(master_verdict, 'all_responses', []) or [])
        self._gas_agreement = int(getattr(master_verdict, 'confidence_pct', 0) or 0)
        regional = [
            (rv.region_pair, rv.estimate)
            for rv in getattr(master_verdict, 'regional_verdicts', []) or []
        ]
        self._gas_verdict = _format_gas_verdict(
            estimate=self._gas_estimate,
            agreement_pct=getattr(master_verdict, 'confidence_pct', None),
            regional=regional,
        )
        self._push_sector_forecasts()
        self._stage4.populate_swarm(
            master_verdict, self._regressor, self._df, self._cv_rmse,
            self._last_scenario,
        )
        # Swarm mode has per-group anchor estimates → full regional disaggregation
        self._stage4.set_regional_estimates(master_verdict.regional_estimates or {})
        self._stage5.set_swarm_context(master_verdict, self._last_scenario)
        self._stage5.update_gas_verdict(self._gas_verdict)
        # Sidebar indices: 2=Simulation, 3=Report
        # Unlock Report tab so it is clickable, but do NOT auto-jump to it.
        # The user navigates there via "View report →" button.
        self._sidebar.unlock_stages([2, 3])
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
        if recalled:
            # The sector numbers came out of the snapshot, so there is nothing to
            # debate and nothing new to synthesise. Everything past this point
            # spends model calls, which is precisely what a recall avoids.
            self._push_sector_forecasts()
            return

        # Gas is done — now start food and electricity debates
        self._start_sector_debates(self._last_scenario)
        self._run_synthesizer_if_ready()

        # Guarded fallback: only build a post-run graph when the panel has no
        # live graph (i.e. the live-wiring path produced nothing).
        try:
            from ph_economic_ai.engine.kg_swarm_adapter import build_graph
            if not self._stage3_swarm.has_live_graph():
                price = self._last_scenario.get('current_price', 0.0)
                agents = build_swarm_agents(price)
                kg = build_graph(master_verdict, agents, self._last_scenario, self._rag)
                self._stage3_swarm.show_knowledge_graph(kg)
        except Exception as exc:
            import logging
            logging.warning('knowledge graph fallback failed: %s', exc)

    def _goto_report(self):
        """Navigate to the Report panel. Called by the 'View report →' button."""
        self._sidebar.unlock_stages([2, 3])
        self._sidebar.set_active(3)
        self._stack.setCurrentIndex(3)

    def _run_synthesizer_if_ready(self):
        """Launch Economy Synthesizer + Causal Chain once all three sector verdicts are in."""
        if not self._gas_verdict or not self._food_verdict or not self._elec_verdict:
            return
        # All three sectors are in (food/electricity estimates now written to the
        # store) — refresh the landing so its latest-forecast card shows them
        # (the post-gas refresh fired before these debates had even started).
        try:
            self._landing.refresh_recent()
        except Exception:
            pass
        self._synth_thread = SynthesizerThread(
            gas_verdict=self._gas_verdict,
            food_verdict=self._food_verdict,
            elec_verdict=self._elec_verdict,
        )
        self._synth_thread.finished.connect(self._economy_overview.update_summary)
        self._synth_thread.start()

        # BSP alert — compute CPI basket impact from all three verdicts. Reuse
        # the same deterministic projected CPI for the causal chain so the two
        # never disagree.
        alert = self._run_bsp_alert()
        projected_cpi = alert.get('projected_cpi') if alert else None

        # Causal chain synthesis
        self._chain_thread = CausalChainThread(
            gas_verdict=self._gas_verdict,
            food_verdict=self._food_verdict,
            elec_verdict=self._elec_verdict,
            scenario=self._last_scenario,
            projected_cpi=projected_cpi,
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
        """Compute CPI basket impact and push BSP alert to Stage 4.

        Uses the numeric estimates directly. This previously re-derived them by
        regexing the verdict *strings* — and `_gas_verdict` is `str(master_verdict)`,
        a dataclass repr that embeds every regional verdict, so a first-match
        scan picked up whichever region happened to appear first rather than the
        consensus. The banner then disagreed with the headline it sits above:
        a +₱2.54/L consensus was reported as +₱6.19/L worth of CPI impact.
        """
        try:
            alert = LiveDataBrief.check_bsp_alert(
                self._gas_estimate, self._food_estimate, self._elec_estimate,
            )
            self._stage4.set_bsp_alert(alert)
            return alert
        except Exception:
            return None
