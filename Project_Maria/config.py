"""
config.py — All global constants, semaphores, thread pools, and global state variables.
Dependency: none (no imports from other Maria modules).
"""
import os, re, json, warnings, time, sys, requests, subprocess, platform, ast, sqlite3, math

# Force UTF-8 stdout/stderr so emoji print() calls don't crash on Windows cp1252 consoles
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import urllib.parse
from collections import OrderedDict
from bs4 import BeautifulSoup
import html2text
import sympy as sp
import numpy as np
import fitz
import pickle
import hashlib
from datetime import datetime, timedelta
import pytz
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from reportlab.lib.colors import black, HexColor, white
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.platypus import KeepTogether
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# PyQt6 imports
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QScrollArea, QFrame, QLabel, QTextEdit, QPushButton, QScrollBar,
                             QDialog, QInputDialog, QMessageBox, QFileDialog, QColorDialog, QMenu, QSizePolicy,
                             QGraphicsDropShadowEffect, QProgressBar, QGridLayout, QLineEdit,
                             QComboBox, QSpinBox, QCheckBox, QToolButton, QTreeWidget, QTreeWidgetItem,
                             QTabWidget, QStackedWidget, QPlainTextEdit)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, pyqtProperty, QTimer, QPropertyAnimation, QEasingCurve, QRect, QRectF, QSize, QEvent, \
    QPoint, QSequentialAnimationGroup, QParallelAnimationGroup, QPointF, QObject
from PyQt6.QtGui import (QFont, QFontDatabase, QPalette, QColor, QTextCursor, QTextCharFormat,
                         QSyntaxHighlighter, QTextDocument, QPainter, QPen, QBrush, QMouseEvent,
                         QKeyEvent, QAction, QIcon, QLinearGradient, QRadialGradient, QGuiApplication,
                         QPixmap, QTextOption, QPainterPath, QCursor, QTextListFormat, QTextBlockFormat)
from PyQt6.QtWidgets import QGraphicsOpacityEffect

# Optional deps
try:
    import pyttsx3
    TTS_AVAILABLE = True
except Exception:
    pyttsx3 = None
    TTS_AVAILABLE = False

try:
    from langdetect import detect
except Exception:
    detect = None

import ollama
from requests.adapters import HTTPAdapter

# ── Concurrency primitives ─────────────────────────────────────────────────────
# Limits concurrent Ollama calls to 2 — prevents timeouts under multi-session load
_OLLAMA_SEMAPHORE = threading.Semaphore(2)

# Shared pool for fire-and-forget background tasks (retrain checks, SFT logging, etc.)
# Replaces ad-hoc threading.Thread().start() calls to avoid per-action thread creation overhead.
_BG_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="maria_bg")
_MARIA_PROCESSING = False   # True while a response is being generated; suppresses background critique

# Per-session language detection cache — avoids re-running langdetect every message
_SESSION_LANG_CACHE: Dict[str, Tuple[str, bool]] = {}
_SESSION_LANG_LOCK = threading.Lock()

# Stateless hashing vectorizer for web RAG reranking — created ONCE at startup.
_WEB_HASH_VEC = HashingVectorizer(
    stop_words='english',
    ngram_range=(1, 2),
    n_features=2 ** 14,      # 16 384 buckets — collision rate < 0.5% for short docs
    norm=None,               # we apply local-IDF then re-normalise manually below
    alternate_sign=False,    # keep all feature values >= 0
)

# ── Path constants ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Permanent folder for file attachments (images/PDFs) so they survive restarts
ATTACHMENTS_DIR = os.path.join(BASE_DIR, "maria_attachments")
os.makedirs(ATTACHMENTS_DIR, exist_ok=True)

# ── Model constants ────────────────────────────────────────────────────────────
CONTEXT_LIMIT = 8192    # Optimized for RTX 5050 VRAM — 32k caused KV-cache pressure & slowdowns
MAX_MEMORY = 50

LORA_FILE = os.path.join(BASE_DIR, "maria_training_data.json")
SESSION_FILE = os.path.join(BASE_DIR, "maria_sessions.json")

# ── Emergency config ───────────────────────────────────────────────────────────
EMERGENCY_CONFIG = {
    # Emergency keywords (case-insensitive)
    'KEYWORDS': {
        'FILIPINO': {
            'FIRE': ['sunog!', 'sunog ng', 'nag-aapoy', 'nasusunog', 'apoy'],
            'THIEF': ['magnanakaw!', 'holdap!', 'nakawan!', 'kawatan'],
            'MEDICAL': ['tulong!', 'saklolo!', 'emergency!', 'aksidente!', 'nasugatan!'],
            'POLICE': ['pulis!', 'krimen!', 'katarantaduhan!']
        },
        'ENGLISH': {
            'FIRE': ['fire!', 'burning!', 'flame!', 'blaze!'],
            'THIEF': ['thief!', 'robber!', 'robbery!', 'stealing!', 'burglar!'],
            'MEDICAL': ['help!', 'emergency!', 'accident!', 'injured!', 'bleeding!'],
            'POLICE': ['police!', 'crime!', 'danger!', 'attack!']
        }
    },

    'EMAIL': {
        'SENDER':      os.environ.get('MARIA_EMAIL_SENDER',   'claramindtest@gmail.com'),
        'PASSWORD':    os.environ.get('MARIA_EMAIL_PASSWORD', 'hqzp itzc pssf jkre'),
        'SMTP_SERVER': 'smtp.gmail.com',
        'SMTP_PORT':   587
    },

    # Emergency contacts (add your responders here)
    'RESPONDERS': [
        {
            'name': 'Local Police',
            'email': 'johncasili257@gmail.com',
            'type': ['THIEF', 'POLICE']
        },
        {
            'name': 'Fire Department',
            'email': 'sindousbuilding@gmail.com',
            'type': ['FIRE']
        },
        {
            'name': 'Medical Emergency',
            'email': 'simulationwithdaniel784@gmail.com',
            'type': ['MEDICAL']
        },
        {
            'name': 'Family Contact',
            'email': 'family@example.com',
            'type': ['FIRE', 'THIEF', 'MEDICAL', 'POLICE']
        }
    ],

    # Emergency response messages
    'RESPONSES': {
        'FILIPINO': {
            'FIRE': "🚨 **EMERGENCY ALERT - SUNOG DETECTED!** 🚨\nNagpadala na ako ng alert sa mga responders!\nMagsilabas na agad at tumawag sa 911!",
            'THIEF': "🚨 **EMERGENCY ALERT - MAGNANAKAW DETECTED!** 🚨\nNagpadala na ako ng alert sa mga responders!\nMagtago at tumawag agad sa pulis!",
            'MEDICAL': "🚨 **EMERGENCY ALERT - MEDICAL EMERGENCY!** 🚨\nNagpadala na ako ng alert sa mga responders!\nTumawag agad sa 911 o emergency hotline!",
            'GENERAL': "🚨 **EMERGENCY ALERT!** 🚨\nNagpadala na ako ng alert sa mga responders!\nManatiling ligtas at tumawag sa 911!"
        },
        'ENGLISH': {
            'FIRE': "🚨 **EMERGENCY ALERT - FIRE DETECTED!** 🚨\nI've sent alerts to responders!\nEvacuate immediately and call 911!",
            'THIEF': "🚨 **EMERGENCY ALERT - THIEF DETECTED!** 🚨\nI've sent alerts to responders!\nHide and call police immediately!",
            'MEDICAL': "🚨 **EMERGENCY ALERT - MEDICAL EMERGENCY!** 🚨\nI've sent alerts to responders!\nCall 911 or emergency hotline immediately!",
            'GENERAL': "🚨 **EMERGENCY ALERT!** 🚨\nI've sent alerts to responders!\nStay safe and call 911!"
        }
    },

    # ADD THIS NEW SECTION FOR SUCCESS MESSAGES
    'SUCCESS_MESSAGES': {
        'FILIPINO': "✅ **Email sent to the responders! Please be safe for now!**",
        'ENGLISH': "✅ **Email sent to the responders! Please be safe for now!**"
    },

    'FAILURE_MESSAGES': {
        'FILIPINO': "⚠️ **Could not send alerts automatically. Please call 911 immediately!**",
        'ENGLISH': "⚠️ **Could not send alerts automatically. Please call 911 immediately!**"
    }
}

# ── Theme dictionaries ─────────────────────────────────────────────────────────
THEME = {
    # === Midnight Precision — deep charcoal base with violet accent ===
    'bg_primary': '#0f1015',         # Near-black base
    'bg_secondary': '#13141a',       # Slightly elevated surface
    'bg_tertiary': '#1c1d26',        # Cards, hover, selected states
    'bg_surface': '#17181f',         # Input and surface components
    'bg_sidebar': '#0b0c12',         # Darkest layer — sidebar
    # Accent: rich violet — distinctive, premium AI feel
    'accent_primary': '#7c6ff5',     # Violet purple
    'accent_primary_hover': '#9a8fff',
    'accent_secondary': '#5b5fd1',
    'accent_blue': '#4f80ff',
    'accent_blue_hover': '#6b96ff',
    'accent_warm': '#f59e0b',        # Amber — kept for warnings
    'accent_warm_light': 'rgba(245, 158, 11, 0.15)',
    # Text hierarchy — warm-tinted whites
    'text_primary': '#e2e4ee',       # Warm near-white
    'text_secondary': '#8b8fa8',     # Blue-grey muted
    'text_tertiary': '#555770',      # Dimmed label text
    'text_placeholder': '#3d3f52',   # Very dimmed placeholder
    # Borders — dark, subtle
    'border_light': '#1e1f2b',
    'border_medium': '#252638',
    'border_dark': '#32334a',
    # Message bubbles
    'user_bubble': '#6c6ff5',        # Violet — user messages
    'assistant_bubble': '#17181f',   # Dark surface — assistant
    'user_border': 'transparent',
    'assistant_border': '#1e1f2b',
    'user_text': '#ffffff',
    'assistant_text': '#e2e4ee',
    'hover_light': '#1c1d28',
    'active_light': '#22233c',
    'success': '#22c55e',
    'warning': '#f59e0b',
    'error': '#ef4444',
    'shadow_light': 'rgba(0, 0, 0, 0.30)',
    'shadow_medium': 'rgba(0, 0, 0, 0.50)',
    'shadow_dark': 'rgba(0, 0, 0, 0.70)',
    'code_bg': '#0c0d12',
    'code_border': '#252638',
}

SYNTAX_THEME = {
    'keyword': '#d73a49',
    'string': '#032f62',
    'comment': '#6a737d',
    'number': '#005cc5',
    'function': '#6f42c1',
    'builtin': '#e36209',
    'operator': '#d73a49',
    'type': '#22863a',
}

MODERN_THEME = {
    'user_bubble': '#007AFF',  # iOS Blue
    'assistant_bubble': '#F2F2F7',  # iOS Gray
    'user_text': '#FFFFFF',
    'assistant_text': '#000000',
    'timestamp_user': 'rgba(255,255,255,0.6)',
    'timestamp_assistant': '#8E8E93',
    'link_user': '#5AC8FA',
    'link_assistant': '#007AFF',
    'shadow_user': 'rgba(0, 122, 255, 0.2)',
    'shadow_assistant': 'rgba(0, 0, 0, 0.08)',
}

SYNTAX_PATTERNS = {
    'keyword': re.compile(
        r'\b(def|class|if|else|elif|for|while|return|function|var|let|const|import|from|as|try|except|finally|with|lambda|in|is|and|or|not)\b'),
    'string': re.compile(r'(".*?"|\'.*?\'|`.*?`)'),
    'comment': re.compile(r'(#.*?$|//.*?$|/\*.*?\*/)', re.MULTILINE | re.DOTALL),
    'number': re.compile(r'\b\d+\.?\d*\b'),
    'function': re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\('),
    'builtin': re.compile(
        r'\b(print|len|range|type|str|int|float|list|dict|set|console\.log|document\.getElementById|Math\.)\b'),
    'operator': re.compile(r'(\+|\-|\*|/|=|==|!=|>|<|>=|<=|&&|\|\||!)'),
    'type': re.compile(r'\b(int|str|float|bool|list|dict|set|tuple|void|string|number|boolean|Array|Object)\b'),
}

LANGUAGE_ICONS = {
    'python': '🐍', 'javascript': '🌐', 'java': '☕', 'cpp': '⚙️', 'c': '🔧',
    'html': '📄', 'css': '🎨', 'sql': '💾', 'json': '📊', 'xml': '🔗',
    'bash': '⚙️', 'shell': '🐚', 'php': '🐘', 'ruby': '💎', 'go': '🔷',
    'rust': '🦀', 'swift': '📱', 'typescript': '⚡', 'markdown': '📝',
    'yaml': '⚙️', 'dockerfile': '🐳', 'generic': '💻'
}

# ── Async JSON writer globals ──────────────────────────────────────────────────
_ASYNC_WRITER_DEBOUNCE_MS  = 400          # coalesce writes within 400 ms
_ASYNC_WRITER_LOCK         = threading.Lock()      # protects _ASYNC_WRITER_REGISTRY
_ASYNC_WRITER_REGISTRY: Dict[str, dict] = {}       # path -> {timer, data, lock}

# ── Connectivity cache ─────────────────────────────────────────────────────────
_connectivity_cache: dict = {"result": None, "ts": 0.0}
_CONNECTIVITY_TTL = 30.0  # re-check at most every 30 seconds

# ── GPU layers (set by background probe at startup) ────────────────────────────
_NUM_GPU_LAYERS = 0   # default CPU; updated by background probe in maria_utils

# ── Specialist model refresh globals ──────────────────────────────────────────
_SPECIALIST_REFRESH_INTERVAL = 300  # re-probe Ollama every 5 min
_specialist_last_refresh     = 0.0
_specialist_lock             = threading.Lock()   # guards refresh + globals

# ── Lazy-initialized globals ───────────────────────────────────────────────────
_PROG_KNOWLEDGE = None   # ProgrammingKnowledgeSystem, lazy-init on first code query
