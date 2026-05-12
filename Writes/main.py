"""
Writes — A minimalist writing application with Substack-inspired Home Feed
Aesthetic: Light, clean, warm. Fully offline.
Run with: python writes.py
Requires: pip install PyQt6
"""

import os
import sys
import json
import math
import shutil
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import html
import threading
from datetime import datetime

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QStackedWidget, QLabel, QPushButton, QLineEdit, QTextEdit,
    QScrollArea, QFrame, QSlider, QCheckBox,
    QSizePolicy, QStatusBar, QDialog, QMenu, QInputDialog,
    QFileDialog,
)
from PyQt6.QtCore import (
    Qt, QTimer, QThread, QSettings, pyqtSignal, QObject,
    QPoint, QRect, QPropertyAnimation, QEasingCurve, QParallelAnimationGroup,
    QSize, QRunnable, QThreadPool, QMetaObject, Q_ARG,
)
from PyQt6.QtGui import (
    QAction, QFont, QColor, QPainter, QPen, QBrush,
    QPalette, QKeySequence, QPolygon, QPixmap, QImage,
)


# ==============================================================================
#  PALETTE  — Substack-inspired light
# ==============================================================================

class P:
    WHITE        = "#FFFFFF"
    BG           = "#FAF9F7"          # warm off-white like Substack
    CARD_BG      = "#FFFFFF"
    SIDEBAR_BG   = "#F9F9F9"
    SIDEBAR_BR   = "#EBEBEB"

    GRAY_100     = "#F2F2F2"
    GRAY_200     = "#E8E8E8"
    GRAY_300     = "#C8C8C8"
    GRAY_400     = "#999999"
    GRAY_500     = "#666666"
    GRAY_700     = "#333333"

    BLACK        = "#191919"

    ORANGE       = "#FF6719"          # Substack orange
    ORANGE_HOVER = "#E55A10"
    ORANGE_LIGHT = "#FFF3EE"

    GREEN        = "#1A8917"
    GREEN_HOVER  = "#0F730C"
    GREEN_LIGHT  = "#E6F4E6"
    RED          = "#C0392B"
    PLACEHOLDER  = "#CCCCCC"

    HEART        = "#E0245E"
    COMMENT_C    = "#1DA1F2"


# ==============================================================================
#  DATA DIR
# ==============================================================================

DATA_DIR   = os.path.expanduser("~/.prose")
DRAFTS_DIR = os.path.join(DATA_DIR, "drafts")
FEED_DIR   = os.path.join(DATA_DIR, "feed")
MEDIA_DIR  = os.path.join(DATA_DIR, "media")

for _d in (DATA_DIR, DRAFTS_DIR, FEED_DIR, MEDIA_DIR):
    os.makedirs(_d, exist_ok=True)

FEED_DB    = os.path.join(DATA_DIR, "feed.json")
META_PATH  = os.path.join(DATA_DIR, "draft_meta.json")


# ==============================================================================
#  FEED DATABASE  — simple JSON list, stored locally
# ==============================================================================

def _load_feed() -> list:
    try:
        with open(FEED_DB, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def _save_feed(posts: list):
    with open(FEED_DB, "w", encoding="utf-8") as f:
        json.dump(posts, f, ensure_ascii=False, indent=2)

def _new_post(title: str, body: str, image_path: str = "") -> dict:
    return {
        "id":       datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
        "title":    title,
        "body":     body,
        "image":    image_path,
        "date":     datetime.now().strftime("%b ") + str(datetime.now().day),
        "author":   "You",
        "likes":    0,
        "liked":    False,
        "comments": [],
    }


# ==============================================================================
#  APP STATE
# ==============================================================================

class AppState(QObject):
    mode_changed = pyqtSignal(bool)
    user_changed = pyqtSignal(object)
    feed_updated = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._online = True          # always "online" locally
        self._user   = {"username": "You"}

    @property
    def is_online(self): return True

    @property
    def user(self): return self._user

    def set_user(self, user):
        self._user = user
        self.user_changed.emit(user)

    def logout(self): pass


# ==============================================================================
#  AVATAR BUTTON
# ==============================================================================

class AvatarButton(QPushButton):
    def __init__(self, initials="Y", color=None):
        super().__init__()
        self.setFixedSize(34, 34)
        self._initials = initials
        self._color    = color or P.ORANGE
        self.setStyleSheet("border: none; background: transparent;")
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_user(self, user):
        name = user.get("username", "") if user else ""
        self._initials = name[:1].upper() if name else "Y"
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.setBrush(QBrush(QColor(self._color)))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(0, 0, w, h)
        p.setPen(QPen(QColor(P.WHITE)))
        p.setFont(QFont("Georgia", int(w * 0.38), QFont.Weight.Bold))
        p.drawText(QRect(0, 0, w, h), Qt.AlignmentFlag.AlignCenter, self._initials)


# ==============================================================================
#  PLUS BUTTON (editor)
# ==============================================================================

class PlusButton(QPushButton):
    def __init__(self):
        super().__init__()
        self.setFixedSize(28, 28)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._hovered = False
        self._pressed = False
        self.setStyleSheet("border: none; background: transparent;")

    def enterEvent(self, e): self._hovered = True;  self.update()
    def leaveEvent(self, e): self._hovered = False; self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._pressed = True; self.update()
        super().mousePressEvent(e)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self._pressed:
            self._pressed = False; self.update()
            if self.rect().contains(e.pos()): self.click()
        super().mouseReleaseEvent(e)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        c = QColor(P.BLACK) if self._pressed else (QColor(P.GRAY_500) if self._hovered else QColor(P.GRAY_300))
        p.setPen(QPen(c, 1.2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(1, 1, 26, 26)
        p.setPen(QPen(c, 1.5))
        p.drawLine(14, 7, 14, 21)
        p.drawLine(7, 14, 21, 14)


# ==============================================================================
#  SIDEBAR ICONS
# ==============================================================================

def _si_home(p, x, y, s, c):
    p.setPen(QPen(QColor(c), 1.3, Qt.PenStyle.SolidLine,
                  Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    p.setBrush(Qt.BrushStyle.NoBrush)
    mx = x + s // 2
    p.drawPolyline(QPolygon([QPoint(x, y + s//2+1), QPoint(mx, y+1), QPoint(x+s, y+s//2+1)]))
    lx, rx = x+2, x+s-2
    p.drawLine(lx, y+s//2+1, lx, y+s); p.drawLine(rx, y+s//2+1, rx, y+s)
    p.drawLine(lx, y+s, rx, y+s)
    dw, dh = s//4, s//3; dx = mx - dw//2
    p.drawRect(dx, y+s-dh, dw, dh)


def _si_feed(p, x, y, s, c):
    p.setPen(QPen(QColor(c), 1.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    for off, w in zip([3, 7, 11], [s, s-3, s-6]):
        p.drawLine(x, y+off, x+w, y+off)


def _si_doc(p, x, y, s, c):
    p.setPen(QPen(QColor(c), 1.2, Qt.PenStyle.SolidLine,
                  Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    p.setBrush(Qt.BrushStyle.NoBrush)
    fold = max(3, s//4)
    p.drawPolygon(QPolygon([QPoint(x,y), QPoint(x+s-fold,y), QPoint(x+s,y+fold),
                             QPoint(x+s,y+s), QPoint(x,y+s)]))
    p.drawLine(x+s-fold, y, x+s-fold, y+fold); p.drawLine(x+s-fold, y+fold, x+s, y+fold)
    p.setPen(QPen(QColor(c), 1.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    for ly in range(y+fold+4, y+s-2, 3):
        p.drawLine(x+3, ly, x+s-3, ly)


def _si_new(p, x, y, s, c):
    p.setPen(QPen(QColor(c), 1.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    mx, my = x+s//2, y+s//2
    p.drawLine(mx, y+2, mx, y+s-2); p.drawLine(x+2, my, x+s-2, my)


def _si_menu(p, x, y, s, c):
    p.setPen(QPen(QColor(c), 1.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    for yy in [y+3, y+8, y+13]:
        p.drawLine(x, yy, x+s, yy)


def _si_star(p, x, y, s, c):
    p.setPen(QPen(QColor(c), 1.2, Qt.PenStyle.SolidLine,
                  Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    p.setBrush(Qt.BrushStyle.NoBrush)
    cx, cy, r_out, r_in = x+s/2, y+s/2, s/2-0.5, s/4
    pts = [QPoint(int(cx+(r_out if i%2==0 else r_in)*math.cos(math.pi/5*i-math.pi/2)),
                  int(cy+(r_out if i%2==0 else r_in)*math.sin(math.pi/5*i-math.pi/2)))
           for i in range(10)]
    p.drawPolygon(QPolygon(pts))


def _si_inbox(p, x, y, s, c):
    p.setPen(QPen(QColor(c), 1.3, Qt.PenStyle.SolidLine,
                  Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawLine(x, y+s//2, x, y+s); p.drawLine(x, y+s, x+s, y+s); p.drawLine(x+s, y+s, x+s, y+s//2)
    p.drawLine(x, y+s//2, x+s//3, y+s//2); p.drawLine(x+s*2//3, y+s//2, x+s, y+s//2)
    p.drawLine(x+s//3, y+s//2, x+s//2, y+s//4); p.drawLine(x+s//2, y+s//4, x+s*2//3, y+s//2)


def _si_book(p, x, y, s, c):
    p.setPen(QPen(QColor(c), 1.3, Qt.PenStyle.SolidLine,
                  Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    p.setBrush(Qt.BrushStyle.NoBrush)
    mx = x+s//2
    p.drawLine(mx, y+1, mx, y+s-1)
    p.drawLine(mx, y+1, x+1, y+3); p.drawLine(x+1, y+3, x+1, y+s-1); p.drawLine(x+1, y+s-1, mx, y+s-1)
    p.drawLine(mx, y+1, x+s-1, y+3); p.drawLine(x+s-1, y+3, x+s-1, y+s-1); p.drawLine(x+s-1, y+s-1, mx, y+s-1)


def _si_pin(p, x, y, s, c):
    p.setPen(QPen(QColor(c), 1.3, Qt.PenStyle.SolidLine,
                  Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    p.setBrush(Qt.BrushStyle.NoBrush)
    cx = x+s//2
    p.drawEllipse(cx-s//4, y, s//2, s//2)
    p.drawLine(cx, y+s//2, cx, y+s-1)
    p.drawLine(cx-s//4, y+s//2, cx+s//4, y+s//2)


def _draw_comment_bubble(painter, x, y, w, h, color):
    """Draw a clean circle comment icon like Substack."""
    painter.setPen(QPen(QColor(color), 1.3, Qt.PenStyle.SolidLine,
                        Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    # circle body
    painter.drawEllipse(x, y, w, w)
    # small tail at bottom-left
    tail = QPolygon([
        QPoint(x + 2, y + w - 2),
        QPoint(x,     y + h),
        QPoint(x + 6, y + w - 1),
    ])
    painter.drawPolyline(tail)


class CommentButton(QPushButton):
    """A comment button with a QPainter-drawn speech bubble icon + count label."""
    def __init__(self, count: int = 0):
        super().__init__()
        self._count = count
        self._hovered = False
        self.setFixedHeight(30)
        self.setMinimumWidth(52)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet("border: none; background: transparent;")

    def set_count(self, n: int):
        self._count = n
        self.update()

    def enterEvent(self, e): self._hovered = True;  self.update()
    def leaveEvent(self, e): self._hovered = False; self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = P.BLACK if self._hovered else P.GRAY_400
        icon_w, icon_h = 14, 13
        ix, iy = 10, (self.height() - icon_h) // 2
        _draw_comment_bubble(p, ix, iy, icon_w, icon_h, color)
        p.setPen(QPen(QColor(color)))
        p.setFont(QFont("Georgia", 12))
        p.drawText(QRect(ix + icon_w + 5, 0, 40, self.height()),
                   Qt.AlignmentFlag.AlignVCenter, str(self._count))


# ==============================================================================
#  SIDEBAR HAMBURGER BUTTON
# ==============================================================================

class SidebarMenuBtn(QPushButton):
    def __init__(self):
        super().__init__()
        self.setFixedSize(36, 36)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._hovered = False
        self.setStyleSheet("border: none; background: transparent;")
        self.setToolTip("Toggle sidebar")

    def enterEvent(self, e): self._hovered = True;  self.update()
    def leaveEvent(self, e): self._hovered = False; self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        _si_menu(p, 10, 11, 16, P.BLACK if self._hovered else P.GRAY_400)


# ==============================================================================
#  SIDEBAR ROW
# ==============================================================================

class SidebarRow(QWidget):
    clicked = pyqtSignal()

    ICON_SIZE    = 14
    H            = 36
    RADIUS       = 5
    PAD_H        = 10
    _C_BG_ACTIVE = "#EDEDED"
    _C_BG_HOVER  = "#F2F2F2"
    _C_TEXT      = "#767676"
    _C_TEXT_ACT  = "#191919"
    _C_ICON      = "#BBBBBB"
    _C_ICON_ACT  = "#333333"

    def __init__(self, label: str, icon_fn=None, indent: int = 0, font_size: int = 13):
        super().__init__()
        self._label   = label
        self._icon_fn = icon_fn
        self._indent  = indent
        self._font_sz = font_size
        self._hovered = self._active = self._pressed = False
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(self.H)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_active(self, v: bool):
        self._active = v; self.update()

    def enterEvent(self, e): self._hovered = True;  self.update()
    def leaveEvent(self, e): self._hovered = False; self._pressed = False; self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._pressed = True; self.update()
        super().mousePressEvent(e)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self._pressed:
            self._pressed = False; self.update()
            if self.rect().contains(e.pos()): self.clicked.emit()
        super().mouseReleaseEvent(e)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        inner = self.rect().adjusted(self.PAD_H, 1, -self.PAD_H, -1)
        if self._pressed or self._active:
            p.setBrush(QBrush(QColor(self._C_BG_ACTIVE))); p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(inner, self.RADIUS, self.RADIUS)
            # Orange left accent bar
            p.setBrush(QBrush(QColor(P.ORANGE))); p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(QRect(0, 6, 3, self.H - 12), 2, 2)
        elif self._hovered:
            p.setBrush(QBrush(QColor(self._C_BG_HOVER))); p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(inner, self.RADIUS, self.RADIUS)

        ic = P.ORANGE if self._active else (self._C_ICON_ACT if (self._hovered or self._pressed) else self._C_ICON)
        tc = P.BLACK  if self._active else (self._C_TEXT_ACT if (self._active or self._pressed) else self._C_TEXT)

        if self._icon_fn:
            ix = self.PAD_H + 8 + self._indent
            iy = (self.H - self.ICON_SIZE) // 2
            self._icon_fn(p, ix, iy, self.ICON_SIZE, ic)
            tx = ix + self.ICON_SIZE + 8
        else:
            tx = self.PAD_H + 8 + self._indent

        p.setPen(QPen(QColor(tc)))
        f = QFont("Georgia", self._font_sz)
        if self._active: f.setWeight(QFont.Weight.Medium)
        p.setFont(f)
        tw = self.width() - tx - self.PAD_H
        p.drawText(QRect(tx, 0, tw, self.H), Qt.AlignmentFlag.AlignVCenter, self._label)


# ==============================================================================
#  WORKSPACE HEADER
# ==============================================================================

class _WorkspaceHeader(QWidget):
    H = 44
    def __init__(self, name="Writes"):
        super().__init__()
        self._name = name
        self.setFixedHeight(self.H)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QPen(QColor(P.BLACK)))
        p.setFont(QFont("Georgia", 13, QFont.Weight.Bold))
        p.drawText(QRect(16, 0, self.width()-16, self.H),
                   Qt.AlignmentFlag.AlignVCenter, self._name)


class _SidebarSearchRow(QWidget):
    H = 30
    def __init__(self):
        super().__init__()
        self.setFixedHeight(self.H)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._hovered = False

    def enterEvent(self, e): self._hovered = True;  self.update()
    def leaveEvent(self, e): self._hovered = False; self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        inner = self.rect().adjusted(8, 2, -8, -2)
        if self._hovered:
            p.setBrush(QBrush(QColor(P.GRAY_100))); p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(inner, 5, 5)
        p.setPen(QPen(QColor("#AAAAAA"), 1.3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(16, 10, 9, 9)
        p.drawLine(24, 18, 28, 22)


def _make_sidebar_section(text):
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"font-size: 10px; color: {P.GRAY_400}; letter-spacing: 0.8px;"
        "font-family: Georgia, serif; padding: 8px 16px 2px;"
        "text-transform: uppercase; background: transparent;"
    )
    return lbl


# ==============================================================================
#  DRAFT ROW
# ==============================================================================

class _DraftRow(QWidget):
    open_requested    = pyqtSignal(str)
    delete_requested  = pyqtSignal(str)
    rename_requested  = pyqtSignal(str, str)
    favourite_toggled = pyqtSignal(str, bool)
    pin_toggled       = pyqtSignal(str, bool)

    H      = 34
    RADIUS = 4
    PAD_H  = 8

    def __init__(self, label, filepath, is_favourite=False, is_pinned=False):
        super().__init__()
        self._label        = label
        self._filepath     = filepath
        self._is_favourite = is_favourite
        self._is_pinned    = is_pinned
        self._hovered      = False
        self._pressed      = False
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(self.H)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)

        self._dots_btn = QPushButton("•••", self)
        self._dots_btn.setFixedSize(22, 22)
        self._dots_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._dots_btn.setStyleSheet(f"""
            QPushButton {{ background: transparent; border: none;
                color: {P.GRAY_400}; font-size: 10px; letter-spacing: 1px; padding: 0; }}
            QPushButton:hover {{ background: {P.GRAY_200}; border-radius: 4px; color: {P.BLACK}; }}
        """)
        self._dots_btn.hide()
        self._dots_btn.clicked.connect(self._show_menu_from_btn)

    def resizeEvent(self, e):
        self._dots_btn.move(self.width() - 26, (self.height() - 22) // 2)
        super().resizeEvent(e)

    def _show_menu_from_btn(self):
        self._open_menu(self._dots_btn.mapToGlobal(QPoint(0, self._dots_btn.height())))

    def _open_menu(self, gpos):
        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{ background: {P.WHITE}; border: 1px solid {P.GRAY_200};
                font-family: Georgia, serif; font-size: 13px;
                border-radius: 6px; padding: 4px 0; }}
            QMenu::item {{ padding: 8px 20px; color: {P.BLACK}; }}
            QMenu::item:selected {{ background: {P.GRAY_100}; }}
            QMenu::separator {{ height: 1px; background: {P.GRAY_200}; margin: 3px 0; }}
        """)
        fav_lbl = "★  Remove from Favourites" if self._is_favourite else "☆  Add to Favourites"
        pin_lbl = "📌  Unpin" if self._is_pinned else "📌  Pin to top"
        menu.addAction(fav_lbl, self._toggle_favourite)
        menu.addAction(pin_lbl, self._toggle_pin)
        menu.addAction("✏  Rename", self._do_rename)
        menu.addSeparator()
        menu.addAction("Open", lambda: self.open_requested.emit(self._filepath))
        menu.addSeparator()
        del_act = menu.addAction("🗑  Delete")
        del_act.setForeground(QColor(P.RED))
        del_act.triggered.connect(lambda: self.delete_requested.emit(self._filepath))
        menu.exec(gpos)

    def _toggle_favourite(self):
        self._is_favourite = not self._is_favourite
        self.favourite_toggled.emit(self._filepath, self._is_favourite); self.update()

    def _toggle_pin(self):
        self._is_pinned = not self._is_pinned
        self.pin_toggled.emit(self._filepath, self._is_pinned); self.update()

    def _do_rename(self):
        new_name, ok = QInputDialog.getText(self, "Rename", "New title:", text=self._label)
        if ok and new_name.strip():
            self._label = new_name.strip(); self.update()
            self.rename_requested.emit(self._filepath, self._label)

    def enterEvent(self, e):
        self._hovered = True; self._dots_btn.show(); self.update()

    def leaveEvent(self, e):
        self._hovered = False; self._pressed = False; self._dots_btn.hide(); self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            if not self._dots_btn.geometry().contains(e.pos()):
                self._pressed = True; self.update()
        super().mousePressEvent(e)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self._pressed:
            self._pressed = False; self.update()
            if (self.rect().contains(e.pos()) and
                    not self._dots_btn.geometry().contains(e.pos())):
                self.open_requested.emit(self._filepath)
        super().mouseReleaseEvent(e)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        inner = self.rect().adjusted(self.PAD_H, 1, -self.PAD_H, -1)
        if self._pressed:
            p.setBrush(QBrush(QColor(SidebarRow._C_BG_ACTIVE)))
            p.setPen(Qt.PenStyle.NoPen); p.drawRoundedRect(inner, self.RADIUS, self.RADIUS)
        elif self._hovered:
            p.setBrush(QBrush(QColor(SidebarRow._C_BG_HOVER)))
            p.setPen(Qt.PenStyle.NoPen); p.drawRoundedRect(inner, self.RADIUS, self.RADIUS)

        icon_c = SidebarRow._C_ICON_ACT if (self._hovered or self._pressed) else SidebarRow._C_ICON
        text_c = SidebarRow._C_TEXT_ACT if (self._hovered or self._pressed) else SidebarRow._C_TEXT
        pad = self.PAD_H + 8 + 4
        iy  = (self.height() - 14) // 2
        if self._is_pinned:
            _si_pin(p, pad, iy, 14, "#1565C0")
        else:
            _si_doc(p, pad, iy, 14, icon_c)
        if self._is_favourite:
            p.setPen(QPen(QColor("#E8A000"), 1.0))
            p.setFont(QFont("Georgia", 7))
            p.drawText(pad + 8, iy - 1, "★")
        tx     = pad + 14 + 7
        text_w = self.width() - tx - self.PAD_H - (28 if self._hovered else 4)
        p.setPen(QPen(QColor(text_c)))
        f = QFont("Georgia", 12)
        if self._is_pinned: f.setWeight(QFont.Weight.Medium)
        p.setFont(f)
        p.drawText(QRect(tx, 0, text_w, self.height()),
                   Qt.AlignmentFlag.AlignVCenter, self._label)


# ==============================================================================
#  SIDEBAR
# ==============================================================================

class Sidebar(QWidget):
    new_story_requested    = pyqtSignal()
    draft_open_requested   = pyqtSignal(str)
    draft_delete_requested = pyqtSignal(str)
    draft_rename_requested = pyqtSignal(str, str)
    navigate_to            = pyqtSignal(str)
    settings_requested     = pyqtSignal()

    WIDTH = 260

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.setFixedWidth(self.WIDTH)
        self.setStyleSheet(f"background: {P.SIDEBAR_BG};")
        self._active_view = "home"
        self._build()
        self._load_recents()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # ── Profile header block (Substack-style) ──────────────────────────
        profile_block = QWidget()
        profile_block.setFixedHeight(72)
        profile_block.setStyleSheet(f"background: {P.SIDEBAR_BG}; border: none;")
        pb_lay = QHBoxLayout(profile_block)
        pb_lay.setContentsMargins(16, 12, 14, 12)
        pb_lay.setSpacing(10)

        av = AvatarButton(initials="Y", color=P.ORANGE)
        av.setEnabled(False)
        pb_lay.addWidget(av)

        name_col = QVBoxLayout()
        name_col.setSpacing(1)
        name_lbl = QLabel("Your Space")
        name_lbl.setStyleSheet(
            f"font-family: Georgia, serif; font-size: 14px; font-weight: bold;"
            f"color: {P.BLACK}; background: transparent;"
        )
        sub_lbl = QLabel("Personal · Offline")
        sub_lbl.setStyleSheet(
            f"font-family: Georgia, serif; font-size: 11px; color: {P.GRAY_400}; background: transparent;"
        )
        name_col.addWidget(name_lbl)
        name_col.addWidget(sub_lbl)
        pb_lay.addLayout(name_col)
        pb_lay.addStretch()
        lay.addWidget(profile_block)

        lay.addWidget(self._divider())
        lay.addSpacing(8)


        # ── Primary nav ────────────────────────────────────────────────────
        self._home_row = SidebarRow("Home", _si_home, font_size=13)
        self._home_row.set_active(True)
        self._home_row.clicked.connect(lambda: self.navigate_to.emit("home"))
        lay.addWidget(self._home_row)

        self._write_row = SidebarRow("Write", _si_doc, font_size=13)
        self._write_row.clicked.connect(lambda: self.navigate_to.emit("editor"))
        lay.addWidget(self._write_row)

        lay.addWidget(SidebarRow("Subscriptions", _si_inbox,  font_size=13))
        lay.addWidget(SidebarRow("Favourites",    _si_star,   font_size=13))
        lay.addWidget(SidebarRow("Library",       _si_book,   font_size=13))

        lay.addSpacing(10)
        lay.addWidget(self._divider())

        # ── Recents section ────────────────────────────────────────────────
        recents_header = QWidget()
        recents_header.setFixedHeight(38)
        recents_header.setStyleSheet("background: transparent;")
        rh_lay = QHBoxLayout(recents_header)
        rh_lay.setContentsMargins(16, 4, 14, 0)
        rh_lay.setSpacing(0)
        recents_lbl = QLabel("RECENTS")
        recents_lbl.setStyleSheet(
            f"font-size: 10px; color: {P.GRAY_400}; letter-spacing: 0.8px;"
            "font-family: Georgia, serif; background: transparent;"
        )
        rh_lay.addWidget(recents_lbl)
        rh_lay.addStretch()
        new_story_btn = QPushButton("＋")
        new_story_btn.setFixedSize(22, 22)
        new_story_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        new_story_btn.setToolTip("New story")
        new_story_btn.setStyleSheet(
            f"QPushButton {{ background: transparent; border: none; color: {P.GRAY_400}; font-size: 14px; }}"
            f"QPushButton:hover {{ color: {P.ORANGE}; }}"
        )
        new_story_btn.clicked.connect(self.new_story_requested)
        rh_lay.addWidget(new_story_btn)
        lay.addWidget(recents_header)

        self._recents_scroll = QScrollArea()
        self._recents_scroll.setWidgetResizable(True)
        self._recents_scroll.setFrameStyle(QFrame.Shape.NoFrame)
        self._recents_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._recents_scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }"
            f"QScrollBar:vertical {{ background: transparent; width: 3px; }}"
            f"QScrollBar::handle:vertical {{ background: {P.GRAY_300}; border-radius: 2px; min-height: 16px; }}"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }"
        )
        self._recents_ctr = QWidget()
        self._recents_ctr.setStyleSheet("background: transparent; border: none;")
        self._recents_lay = QVBoxLayout(self._recents_ctr)
        self._recents_lay.setContentsMargins(0, 0, 0, 0)
        self._recents_lay.setSpacing(0)
        self._recents_scroll.setWidget(self._recents_ctr)
        lay.addWidget(self._recents_scroll, 1)

    def _divider(self):
        d = QFrame()
        d.setFrameShape(QFrame.Shape.HLine)
        d.setFixedHeight(1)
        d.setStyleSheet(f"background: {P.SIDEBAR_BR}; border: none; margin: 0;")
        return d

    def _load_recents(self):
        while self._recents_lay.count():
            item = self._recents_lay.takeAt(0)
            if item.widget(): item.widget().deleteLater()

        files = []
        if os.path.isdir(DRAFTS_DIR):
            files = sorted(
                [f for f in os.listdir(DRAFTS_DIR) if f.endswith(".json")],
                key=lambda f: os.path.getmtime(os.path.join(DRAFTS_DIR, f)),
                reverse=True,
            )[:14]

        if not files:
            ph = QLabel("No saved stories yet")
            ph.setStyleSheet(
                f"font-size: 12px; color: {P.GRAY_400}; font-family: Georgia, serif;"
                "padding: 8px 20px; background: transparent; border: none;"
            )
            self._recents_lay.addWidget(ph)
            self._recents_lay.addStretch()
            return

        meta = self._load_meta()
        for fname in files:
            fpath = os.path.join(DRAFTS_DIR, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                title = data.get("title") or "Untitled"
            except Exception:
                title = fname.replace(".json", "").replace("_", " ")
            display = (title[:28] + "…") if len(title) > 28 else title
            fmeta   = meta.get(fpath, {})
            row = _DraftRow(display, fpath,
                            is_favourite=fmeta.get("favourite", False),
                            is_pinned=fmeta.get("pinned", False))
            row.open_requested.connect(self.draft_open_requested)
            row.delete_requested.connect(self._on_delete_draft)
            row.rename_requested.connect(self._on_rename_draft)
            row.favourite_toggled.connect(self._on_favourite)
            row.pin_toggled.connect(self._on_pin)
            self._recents_lay.addWidget(row)
        self._recents_lay.addStretch()

    def _meta_path(self): return META_PATH

    def _load_meta(self):
        try:
            with open(self._meta_path(), "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_meta(self, meta):
        with open(self._meta_path(), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def _on_delete_draft(self, fp): self.draft_delete_requested.emit(fp)

    def _on_rename_draft(self, fp, new_title):
        try:
            with open(fp, "r", encoding="utf-8") as f: data = json.load(f)
            data["title"] = new_title
            with open(fp, "w", encoding="utf-8") as f: json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception: pass
        self.draft_rename_requested.emit(fp, new_title)

    def _on_favourite(self, fp, state):
        meta = self._load_meta()
        meta.setdefault(fp, {})["favourite"] = state
        self._save_meta(meta)

    def _on_pin(self, fp, state):
        meta = self._load_meta()
        meta.setdefault(fp, {})["pinned"] = state
        self._save_meta(meta)
        self._load_recents()

    def set_active_view(self, view):
        self._active_view = view
        self._home_row.set_active(view == "home")
        self._write_row.set_active(view == "editor")

    def refresh_recents(self):
        self._load_recents()

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(P.SIDEBAR_BG))
        p.setPen(QPen(QColor(P.SIDEBAR_BR), 1))
        p.drawLine(self.width()-1, 0, self.width()-1, self.height())


# ==============================================================================
#  TOP BAR
# ==============================================================================

class TopBar(QWidget):
    publish_clicked = pyqtSignal()
    dots_clicked    = pyqtSignal()
    avatar_clicked  = pyqtSignal()
    sidebar_toggled = pyqtSignal()

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.setFixedHeight(56)
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(self.backgroundRole(), QColor(P.BG))
        self.setPalette(pal)
        self.setStyleSheet("border: none;")
        self._build()

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(P.BG))

    def _build(self):
        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 0, 20, 0)
        lay.setSpacing(0)

        self._sidebar_btn = SidebarMenuBtn()
        self._sidebar_btn.clicked.connect(self.sidebar_toggled)
        lay.addWidget(self._sidebar_btn)
        lay.addSpacing(6)

        logo = QLabel("Writes")
        logo.setStyleSheet(
            "font-family: Georgia, serif; font-size: 20px; "
            "font-weight: bold; color: #191919; letter-spacing: -0.5px;"
        )
        lay.addWidget(logo)

        self._draft_lbl = QLabel("  Draft")
        self._draft_lbl.setStyleSheet(
            f"font-size: 12px; color: {P.GRAY_400}; font-family: Georgia, serif;"
        )
        lay.addWidget(self._draft_lbl)
        lay.addStretch()

        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet(
            f"font-size: 12px; color: {P.GRAY_400}; padding-right: 16px; font-family: Georgia, serif;"
        )
        lay.addWidget(self._status_lbl)

        self._dots = QPushButton("···")
        self._dots.setFixedSize(32, 32)
        self._dots.setCursor(Qt.CursorShape.PointingHandCursor)
        self._dots.clicked.connect(self.dots_clicked)
        self._dots.setStyleSheet(f"""
            QPushButton {{ background: transparent; border: none;
                font-size: 18px; color: {P.GRAY_400}; letter-spacing: 1px; padding-bottom: 3px;
                border-radius: 6px; }}
            QPushButton:hover {{ color: {P.BLACK}; background: {P.GRAY_100}; }}
        """)
        lay.addWidget(self._dots)
        lay.addSpacing(6)

        self._avatar = AvatarButton()
        self._avatar.clicked.connect(self.avatar_clicked)
        lay.addWidget(self._avatar)

    def set_status(self, text):
        self._status_lbl.setText(text)

    def flash_saved(self):
        self._draft_lbl.setText("  Saved ✓")
        QTimer.singleShot(2200, lambda: self._draft_lbl.setText("  Draft"))

    def show_draft_label(self, show: bool):
        self._draft_lbl.setVisible(show)

    def _pub_style(self):
        return f"""
            QPushButton {{
                background: {P.ORANGE}; color: {P.WHITE};
                border: none; border-radius: 8px;
                font-size: 13px; font-family: Georgia, serif;
            }}
            QPushButton:hover {{ background: {P.ORANGE_HOVER}; }}
        """


# ==============================================================================
#  EDITOR
# ==============================================================================

AUTOSAVE_DIR = DRAFTS_DIR


class TitleEdit(QLineEdit):
    tab_pressed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setPlaceholderText("Title")
        self.setFrame(False)
        self.setFont(QFont("Georgia", 36, QFont.Weight.Bold))
        self.setStyleSheet(f"""
            QLineEdit {{
                background: transparent; border: none;
                color: {P.BLACK}; padding: 0;
                selection-background-color: #FFE0D0;
            }}
        """)
        pal = self.palette()
        pal.setColor(QPalette.ColorRole.PlaceholderText, QColor(P.PLACEHOLDER))
        self.setPalette(pal)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Tab: self.tab_pressed.emit()
        else: super().keyPressEvent(e)


class BodyEdit(QTextEdit):
    def __init__(self):
        super().__init__()
        self.setPlaceholderText("Tell your story...")
        self.setFrameStyle(QFrame.Shape.NoFrame)
        self.setAcceptRichText(False)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFont(QFont("Georgia", 19))
        self.setStyleSheet(f"""
            QTextEdit {{
                background: transparent; border: none;
                color: {P.BLACK}; padding: 0;
                selection-background-color: #FFE0D0;
            }}
        """)
        pal = self.palette()
        pal.setColor(QPalette.ColorRole.PlaceholderText, QColor(P.PLACEHOLDER))
        self.setPalette(pal)
        self.document().setDefaultStyleSheet("p { line-height: 180%; }")


class EditorView(QWidget):
    word_count_changed = pyqtSignal(int)
    status_changed     = pyqtSignal(str)
    draft_saved        = pyqtSignal()
    published          = pyqtSignal(str, str)   # title, body

    def __init__(self, state: AppState):
        super().__init__()
        self.state         = state
        self._current_file = None
        self._modified     = False
        self.setStyleSheet(f"background: {P.WHITE};")
        self._build()
        self._setup_autosave()

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameStyle(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"""
            QScrollArea {{ background: {P.WHITE}; border: none; }}
            QScrollBar:vertical {{ background: transparent; width: 6px; margin: 0; }}
            QScrollBar::handle:vertical {{ background: {P.GRAY_300}; border-radius: 3px; min-height: 30px; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
        """)
        outer.addWidget(scroll)

        content = QWidget()
        content.setStyleSheet(f"background: {P.WHITE};")
        scroll.setWidget(content)

        c_lay = QVBoxLayout(content)
        c_lay.setContentsMargins(0, 60, 0, 200)
        c_lay.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        col = QWidget()
        col.setFixedWidth(680)
        col.setStyleSheet("background: transparent;")
        c_lay.addWidget(col, alignment=Qt.AlignmentFlag.AlignHCenter)

        col_lay = QVBoxLayout(col)
        col_lay.setContentsMargins(0, 0, 0, 0)
        col_lay.setSpacing(0)

        self.title_edit = TitleEdit()
        self.title_edit.setMinimumHeight(60)
        col_lay.addWidget(self.title_edit)
        col_lay.addSpacing(28)

        self._subtitle_hint = QLabel("Subtitle (optional)")
        self._subtitle_hint.setStyleSheet(
            "font-size: 22px; color: #CCCCCC; font-family: Georgia, serif;"
            "background: transparent; border: none; padding: 0;"
        )
        col_lay.addWidget(self._subtitle_hint)
        col_lay.addSpacing(28)

        div = QFrame()
        div.setFrameShape(QFrame.Shape.HLine)
        div.setFixedHeight(1)
        div.setStyleSheet(f"background: {P.GRAY_200}; border: none;")
        col_lay.addWidget(div)
        col_lay.addSpacing(28)

        body_row = QHBoxLayout()
        body_row.setContentsMargins(0, 0, 0, 0)
        body_row.setSpacing(0)
        body_row.setAlignment(Qt.AlignmentFlag.AlignTop)

        plus_wrap = QWidget()
        plus_wrap.setFixedWidth(44)
        plus_wrap.setStyleSheet("background: transparent;")
        pw_lay = QVBoxLayout(plus_wrap)
        pw_lay.setContentsMargins(0, 4, 0, 0)
        pw_lay.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._plus = PlusButton()
        self._plus.setToolTip("Insert image or divider")
        pw_lay.addWidget(self._plus)
        pw_lay.addStretch()
        body_row.addWidget(plus_wrap)

        self.body_edit = BodyEdit()
        self.body_edit.setMinimumHeight(500)
        body_row.addWidget(self.body_edit, 1)
        col_lay.addLayout(body_row)

        self.title_edit.tab_pressed.connect(lambda: self.body_edit.setFocus())
        self.title_edit.textChanged.connect(self._on_title_changed)
        self.body_edit.textChanged.connect(self._on_text_changed)

    def _on_title_changed(self, _=None):
        self._mark_modified()
        self._subtitle_hint.setVisible(not bool(self.title_edit.text().strip()))

    def _setup_autosave(self):
        self._timer = QTimer(self)
        self._timer.setInterval(30_000)
        self._timer.timeout.connect(
            lambda: self.save_document(autosave=True) if self._modified else None)
        self._timer.start()

    def new_document(self):
        self.title_edit.clear()
        self.body_edit.clear()
        self._current_file = None
        self._modified     = False
        self._subtitle_hint.setVisible(True)
        self.status_changed.emit("")
        self.title_edit.setFocus()

    def load_draft(self, filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.title_edit.setText(data.get("title", ""))
            self.body_edit.setPlainText(data.get("body", ""))
            self._current_file = filepath
            self._modified     = False
            self._subtitle_hint.setVisible(not bool(self.title_edit.text().strip()))
            self.status_changed.emit("")
            self.title_edit.setFocus()
        except Exception as e:
            self.status_changed.emit(f"Could not open draft: {e}")

    def save_document(self, autosave=False):
        title = self.title_edit.text().strip() or "Untitled"
        data  = {"title": title, "body": self.body_edit.toPlainText(),
                 "saved_at": datetime.now().isoformat()}
        if not self._current_file:
            safe = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)[:40]
            self._current_file = os.path.join(
                AUTOSAVE_DIR, f"{safe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(self._current_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self._modified = False
        self.status_changed.emit("")
        self.draft_saved.emit()

    def publish(self):
        title = self.title_edit.text().strip()
        body  = self.body_edit.toPlainText().strip()
        if not title and not body:
            self.status_changed.emit("Nothing to publish — write something first.")
            QTimer.singleShot(3000, lambda: self.status_changed.emit("")); return
        if not title:
            self.status_changed.emit("Add a title before publishing.")
            QTimer.singleShot(3000, lambda: self.status_changed.emit("")); return
        self.save_document()
        self.published.emit(title, body)
        self.status_changed.emit(f"Published: {title}")
        QTimer.singleShot(4000, lambda: self.status_changed.emit(""))

    def _mark_modified(self):
        self._modified = True
        self.status_changed.emit("Unsaved changes")

    def _on_text_changed(self):
        self._mark_modified()
        text  = self.body_edit.toPlainText()
        words = len(text.split()) if text.strip() else 0
        self.word_count_changed.emit(words)


# ==============================================================================
#  HOME FEED — Substack-inspired light
# ==============================================================================

def _draw_image_icon(p, x, y, s, c):
    p.setPen(QPen(QColor(c), 1.3, Qt.PenStyle.SolidLine,
                  Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRect(x, y, s, s)
    p.drawEllipse(x + s//5, y + s//5, s//4, s//4)
    p.drawPolyline(QPolygon([
        QPoint(x, y + s),
        QPoint(x + s//2, y + s//3),
        QPoint(x + s, y + s),
    ]))

def _draw_video_icon(p, x, y, s, c):
    p.setPen(QPen(QColor(c), 1.3, Qt.PenStyle.SolidLine,
                  Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    p.setBrush(Qt.BrushStyle.NoBrush)
    bw = int(s * 0.62); bh = int(s * 0.62); by = y + (s - bh) // 2
    p.drawRoundedRect(x, by, bw, bh, 2, 2)
    p.drawPolyline(QPolygon([
        QPoint(x + s - int(s*0.32), y + s//3),
        QPoint(x + s - 1,           y + s//2),
        QPoint(x + s - int(s*0.32), y + s - s//3),
    ]))

def _draw_calendar_icon(p, x, y, s, c):
    p.setPen(QPen(QColor(c), 1.3, Qt.PenStyle.SolidLine,
                  Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRoundedRect(x, y + 2, s, s - 2, 2, 2)
    p.drawLine(x, y + s//3 + 1, x + s, y + s//3 + 1)
    p.drawLine(x + s//3, y + 2, x + s//3, y - 1)
    p.drawLine(x + s*2//3, y + 2, x + s*2//3, y - 1)

def _draw_dots_icon(p, x, y, s, c):
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(QColor(c)))
    r = 2; cy = y + s // 2
    for cx in [x + 2, x + s//2, x + s - 2]:
        p.drawEllipse(cx - r, cy - r, r*2, r*2)


class _IconToolButton(QPushButton):
    """Clean tool button with custom-painted icon."""
    def __init__(self, icon_fn, tooltip: str):
        super().__init__()
        self._icon_fn = icon_fn
        self._hovered = False
        self.setFixedSize(32, 32)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(tooltip)
        self.setStyleSheet("border: none; background: transparent;")

    def enterEvent(self, e): self._hovered = True;  self.update()
    def leaveEvent(self, e): self._hovered = False; self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self._hovered:
            p.setBrush(QBrush(QColor(P.GRAY_100))); p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(2, 2, 28, 28, 5, 5)
        color = P.GRAY_500 if self._hovered else P.GRAY_300
        self._icon_fn(p, 8, 8, 16, color)


class _ComposeDialogTextEdit(QTextEdit):
    """TextEdit inside the compose dialog — intercepts Ctrl+V for image paste."""
    image_pasted = pyqtSignal(QPixmap)

    def keyPressEvent(self, e):
        if e.matches(QKeySequence.StandardKey.Paste):
            cb = QApplication.clipboard()
            img = cb.image()
            if not img.isNull():
                self.image_pasted.emit(QPixmap.fromImage(img))
                return
        super().keyPressEvent(e)


class ComposeDialog(QDialog):
    """Substack-style compose popup — light themed, centered on main window."""
    post_submitted = pyqtSignal(str, object)

    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog)
        self.setModal(True)
        self._pending_pixmap = None
        self._result_text   = ""
        self._result_pixmap = None
        self._build()
        if parent:
            self.adjustSize()
            pw = parent.frameGeometry()
            self.move(
                pw.x() + (pw.width()  - self.width())  // 2,
                pw.y() + (pw.height() - self.height()) // 2 - 40,
            )

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        card = QWidget()
        card.setObjectName("card")
        card.setStyleSheet(f"""
            QWidget#card {{
                background: {P.WHITE};
                border: 1px solid {P.GRAY_200};
                border-radius: 16px;
            }}
        """)
        card.setFixedWidth(560)
        outer.addWidget(card)

        lay = QVBoxLayout(card)
        lay.setContentsMargins(22, 20, 22, 18)
        lay.setSpacing(0)

        # Header
        header = QHBoxLayout()
        header.setSpacing(12)
        av = AvatarButton(initials="Y", color=P.ORANGE)
        av.setEnabled(False)
        header.addWidget(av)
        name_lbl = QLabel("You")
        name_lbl.setStyleSheet(
            f"font-family: Georgia, serif; font-size: 14px; font-weight: bold;"
            f"color: {P.BLACK}; background: transparent;")
        header.addWidget(name_lbl)
        header.addStretch()
        drafts_btn = QPushButton("Drafts")
        drafts_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        drafts_btn.setStyleSheet(f"""
            QPushButton {{ background: transparent; border: none;
                font-family: Georgia, serif; font-size: 13px; color: {P.GRAY_400}; padding: 0; }}
            QPushButton:hover {{ color: {P.BLACK}; }}
        """)
        header.addWidget(drafts_btn)
        lay.addLayout(header)
        lay.addSpacing(16)

        # Text area
        self._text = _ComposeDialogTextEdit()
        self._text.setPlaceholderText("What's on your mind?")
        self._text.setFrameStyle(QFrame.Shape.NoFrame)
        self._text.setAcceptRichText(False)
        self._text.setFont(QFont("Georgia", 15))
        self._text.setMinimumHeight(140)
        self._text.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._text.setStyleSheet(f"""
            QTextEdit {{ background: transparent; border: none; color: {P.BLACK}; padding: 0; }}
        """)
        pal = self._text.palette()
        pal.setColor(QPalette.ColorRole.PlaceholderText, QColor(P.GRAY_300))
        self._text.setPalette(pal)
        self._text.image_pasted.connect(self._on_image_pasted)
        lay.addWidget(self._text)
        lay.addSpacing(12)

        # Image preview strip
        self._preview_strip = QWidget()
        self._preview_strip.setStyleSheet("background: transparent; border: none;")
        self._preview_strip.setVisible(False)
        self._preview_strip.setFixedHeight(100)
        ps_lay = QHBoxLayout(self._preview_strip)
        ps_lay.setContentsMargins(0, 0, 0, 8)
        ps_lay.setSpacing(8)
        ps_lay.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._thumb_lbl = QLabel()
        self._thumb_lbl.setFixedSize(90, 90)
        self._thumb_lbl.setScaledContents(True)
        self._thumb_lbl.setStyleSheet(f"border-radius: 8px; border: 1px solid {P.GRAY_200};")
        ps_lay.addWidget(self._thumb_lbl)
        remove_btn = QPushButton("x")
        remove_btn.setFixedSize(20, 20)
        remove_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        remove_btn.setStyleSheet(f"""
            QPushButton {{ background: {P.GRAY_200}; border: none; border-radius: 10px;
                color: {P.GRAY_500}; font-size: 9px; }}
            QPushButton:hover {{ background: {P.GRAY_300}; color: {P.BLACK}; }}
        """)
        remove_btn.clicked.connect(self._clear_image)
        ps_lay.addWidget(remove_btn, alignment=Qt.AlignmentFlag.AlignTop)
        ps_lay.addStretch()
        lay.addWidget(self._preview_strip)

        # Divider
        div = QFrame(); div.setFrameShape(QFrame.Shape.HLine); div.setFixedHeight(1)
        div.setStyleSheet(f"background: {P.GRAY_200}; border: none;")
        lay.addWidget(div)
        lay.addSpacing(12)

        # Bottom bar
        bottom = QHBoxLayout()
        bottom.setSpacing(2)

        img_btn = _IconToolButton(_draw_image_icon, "Photo (Ctrl+V to paste)")
        img_btn.clicked.connect(self._pick_image)
        bottom.addWidget(img_btn)
        bottom.addWidget(_IconToolButton(_draw_video_icon, "Video"))
        bottom.addWidget(_IconToolButton(_draw_calendar_icon, "Schedule"))
        bottom.addWidget(_IconToolButton(_draw_dots_icon, "More"))
        bottom.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        cancel_btn.setFixedHeight(34)
        cancel_btn.setStyleSheet(f"""
            QPushButton {{ background: {P.GRAY_100}; border: 1px solid {P.GRAY_200};
                border-radius: 8px; color: {P.GRAY_700};
                font-family: Georgia, serif; font-size: 13px; padding: 0 18px; }}
            QPushButton:hover {{ background: {P.GRAY_200}; }}
        """)
        cancel_btn.clicked.connect(self.reject)
        bottom.addWidget(cancel_btn)
        bottom.addSpacing(8)

        self._post_btn = QPushButton("Post")
        self._post_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._post_btn.setFixedHeight(34)
        self._post_btn.setStyleSheet(f"""
            QPushButton {{ background: {P.ORANGE}; border: none; border-radius: 8px;
                color: {P.WHITE}; font-family: Georgia, serif;
                font-size: 13px; font-weight: bold; padding: 0 24px; }}
            QPushButton:hover {{ background: {P.ORANGE_HOVER}; }}
        """)
        self._post_btn.clicked.connect(self._on_post)
        bottom.addWidget(self._post_btn)
        lay.addLayout(bottom)

    def _pick_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose image", "",
            "Images (*.png *.jpg *.jpeg *.gif *.webp *.bmp)")
        if path:
            pix = QPixmap(path)
            if not pix.isNull():
                self._on_image_pasted(pix)

    def _on_image_pasted(self, pix):
        self._pending_pixmap = pix
        thumb = pix.scaled(QSize(90, 90), Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                           Qt.TransformationMode.SmoothTransformation)
        x = (thumb.width() - 90) // 2; y = (thumb.height() - 90) // 2
        self._thumb_lbl.setPixmap(thumb.copy(x, y, 90, 90))
        self._preview_strip.setVisible(True)

    def _clear_image(self):
        self._pending_pixmap = None
        self._preview_strip.setVisible(False)
        self._thumb_lbl.clear()

    def _on_post(self):
        text = self._text.toPlainText().strip()
        if text or self._pending_pixmap:
            self._result_text   = text
            self._result_pixmap = self._pending_pixmap
            self.accept()

    def focusTextArea(self):
        self._text.setFocus()


class ComposerBox(QWidget):
    """Collapsed trigger row — clicking opens ComposeDialog centered on main window."""
    post_submitted = pyqtSignal(str, object)

    def __init__(self):
        super().__init__()
        self.setObjectName("composerBox")
        self.setStyleSheet(f"""
            QWidget#composerBox {{
                background: {P.WHITE};
                border: 1.5px solid {P.GRAY_300};
                border-radius: 14px;
            }}
        """)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 12, 16, 12)
        lay.setSpacing(12)
        self._avatar = AvatarButton(color=P.ORANGE)
        self._avatar.setEnabled(False)
        lay.addWidget(self._avatar)
        self._placeholder = QLabel("What's on your mind?")
        self._placeholder.setStyleSheet(
            f"font-family: Georgia, serif; font-size: 14px; color: {P.GRAY_400};"
            "background: transparent; border: none;")
        lay.addWidget(self._placeholder, 1)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._open_dialog()
        super().mousePressEvent(e)

    def _open_dialog(self):
        dlg = ComposeDialog(self.window())
        dlg.focusTextArea()
        dlg.exec()
        # exec() is blocking — read results synchronously after it returns
        text = getattr(dlg, '_result_text',   '')
        pix  = getattr(dlg, '_result_pixmap', None)
        if text or pix:
            self.post_submitted.emit(text, pix)

    def get_text(self):                return ""
    def get_pending_pixmap(self):      return None
    def clear(self):                   pass
    def set_pending_pixmap(self, pix): pass



class CommentDialog(QDialog):
    """Inline comment box dialog."""
    submitted = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Comment")
        self.setFixedWidth(440)
        self.setStyleSheet(f"background: {P.WHITE};")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 24, 24, 20)
        lay.setSpacing(12)

        lbl = QLabel("Leave a comment")
        lbl.setStyleSheet(f"font-size: 16px; font-family: Georgia, serif; color: {P.BLACK};")
        lay.addWidget(lbl)

        self._edit = QTextEdit()
        self._edit.setPlaceholderText("Write something...")
        self._edit.setFixedHeight(100)
        self._edit.setFont(QFont("Georgia", 13))
        self._edit.setStyleSheet(f"""
            QTextEdit {{
                background: {P.GRAY_100}; border: 1px solid {P.GRAY_200};
                border-radius: 8px; color: {P.BLACK}; padding: 8px;
            }}
            QTextEdit:focus {{
                background: {P.WHITE}; border: 1px solid {P.GRAY_400};
            }}
        """)
        lay.addWidget(self._edit)

        row = QHBoxLayout()
        row.addStretch()
        cancel = QPushButton("Cancel")
        cancel.setStyleSheet(f"""
            QPushButton {{ background: transparent; border: 1px solid {P.GRAY_300};
                border-radius: 8px; color: {P.GRAY_500}; padding: 6px 18px;
                font-family: Georgia, serif; font-size: 13px; }}
            QPushButton:hover {{ border-color: {P.GRAY_400}; color: {P.BLACK}; }}
        """)
        cancel.clicked.connect(self.reject)

        submit = QPushButton("Post comment")
        submit.setStyleSheet(f"""
            QPushButton {{ background: {P.ORANGE}; border: none; border-radius: 8px;
                color: {P.WHITE}; padding: 6px 18px;
                font-family: Georgia, serif; font-size: 13px; font-weight: bold; }}
            QPushButton:hover {{ background: {P.ORANGE_HOVER}; }}
        """)
        submit.clicked.connect(self._on_submit)

        row.addWidget(cancel)
        row.addSpacing(8)
        row.addWidget(submit)
        lay.addLayout(row)

    def _on_submit(self):
        txt = self._edit.toPlainText().strip()
        if txt:
            self.submitted.emit(txt)
            self.accept()


class PostCard(QWidget):
    """A single post card in the feed."""
    like_toggled    = pyqtSignal(str, bool)   # post_id, new_liked state
    comment_added   = pyqtSignal(str, str)    # post_id, comment_text
    delete_post     = pyqtSignal(str)

    def __init__(self, post: dict, parent=None):
        super().__init__(parent)
        self.post = post
        self._build()

    def _build(self):
        self.setStyleSheet("background: transparent;")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        card = QWidget()
        card.setObjectName("card")
        card.setStyleSheet("QWidget#card { background: transparent; border: none; }")
        card_lay = QVBoxLayout(card)
        card_lay.setContentsMargins(0, 20, 0, 12)
        card_lay.setSpacing(10)

        # Header: avatar + author + date + menu
        header = QHBoxLayout()
        header.setSpacing(10)

        av = AvatarButton(initials="Y", color=P.ORANGE)
        av.setEnabled(False)
        header.addWidget(av)

        info = QVBoxLayout()
        info.setSpacing(0)
        author_lbl = QLabel(self.post.get("author", "You"))
        author_lbl.setStyleSheet(f"font-size: 13px; font-weight: bold; color: {P.BLACK}; font-family: Georgia, serif;")
        date_lbl = QLabel(self.post.get("date", ""))
        date_lbl.setStyleSheet(f"font-size: 12px; color: {P.GRAY_400}; font-family: Georgia, serif;")
        info.addWidget(author_lbl)
        info.addWidget(date_lbl)
        header.addLayout(info)
        header.addStretch()

        # 3-dot delete menu
        dots = QPushButton("···")
        dots.setFixedSize(30, 30)
        dots.setCursor(Qt.CursorShape.PointingHandCursor)
        dots.setStyleSheet(f"""
            QPushButton {{ background: transparent; border: none;
                font-size: 14px; color: {P.GRAY_400}; letter-spacing: 1px;
                border-radius: 6px; padding-bottom: 2px; }}
            QPushButton:hover {{ color: {P.BLACK}; background: {P.GRAY_100}; }}
        """)
        pid = self.post["id"]
        dots.clicked.connect(lambda _checked=False, p=pid: self._show_card_menu(p))
        header.addWidget(dots)
        card_lay.addLayout(header)

        # Title — above image, editorial style
        title = self.post.get("title", "")
        if title:
            t = QLabel(title)
            t.setWordWrap(True)
            t.setStyleSheet(
                f"font-size: 18px; font-weight: bold; color: {P.BLACK};"
                "font-family: Georgia, serif; line-height: 1.3;"
                "background: transparent;"
            )
            card_lay.addWidget(t)

        # Body excerpt — above image, editorial style
        body = self.post.get("body", "")
        if body:
            excerpt = body[:220] + ("…" if len(body) > 220 else "")
            b = QLabel(excerpt)
            b.setWordWrap(True)
            b.setStyleSheet(
                f"font-size: 15px; color: {P.GRAY_500};"
                "font-family: Georgia, serif; background: transparent;"
            )
            card_lay.addWidget(b)

        # Image (if any) — below text
        img_path = self.post.get("image", "")
        if img_path and os.path.exists(img_path):
            img_lbl = QLabel()
            img_lbl.setFixedHeight(300)
            img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            img_lbl.setStyleSheet("border-radius: 6px; background: transparent;")
            pix = QPixmap(img_path)
            if not pix.isNull():
                pix = pix.scaled(QSize(700, 300), Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                                 Qt.TransformationMode.SmoothTransformation)
                x = (pix.width()  - 700) // 2
                y = (pix.height() - 300) // 2
                pix = pix.copy(max(0, x), max(0, y), 700, 300)
                img_lbl.setPixmap(pix)
                img_lbl.setScaledContents(False)
            card_lay.addWidget(img_lbl)

        # Action bar: Like, Comment
        actions = QHBoxLayout()
        actions.setSpacing(0)

        likes = self.post.get("likes", 0)
        liked = self.post.get("liked", False)

        self._like_btn = QPushButton(f"  ♥  {likes}")
        self._like_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._like_btn.setCheckable(True)
        self._like_btn.setChecked(liked)
        self._like_btn.setFont(QFont("Georgia", 13))
        self._update_like_style(liked)
        self._like_btn.clicked.connect(self._on_like)
        actions.addWidget(self._like_btn)

        actions.addSpacing(6)

        n_comments = len(self.post.get("comments", []))
        self._cmt_btn = CommentButton(n_comments)
        self._cmt_btn.clicked.connect(self._on_comment_btn)
        actions.addWidget(self._cmt_btn)

        actions.addStretch()
        card_lay.addLayout(actions)

        # Comments section (hidden by default, toggled by comment button)
        self._comments_widget = QWidget()
        self._comments_widget.setStyleSheet("background: transparent;")
        self._comments_outer_lay = QVBoxLayout(self._comments_widget)
        self._comments_outer_lay.setContentsMargins(0, 4, 0, 8)
        self._comments_outer_lay.setSpacing(0)

        # Write-comment row
        write_row = QHBoxLayout()
        write_row.setContentsMargins(0, 8, 0, 8)
        write_row.setSpacing(10)
        av_write = AvatarButton(initials="Y", color=P.ORANGE)
        av_write.setFixedSize(32, 32)
        av_write.setEnabled(False)
        write_row.addWidget(av_write, alignment=Qt.AlignmentFlag.AlignVCenter)
        write_btn = QPushButton("Write a comment…")
        write_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        write_btn.setFixedHeight(36)
        write_btn.setStyleSheet(f"""
            QPushButton {{
                background: {P.GRAY_100};
                border: 1px solid {P.GRAY_200};
                border-radius: 18px;
                color: {P.GRAY_400};
                font-family: Georgia, serif; font-size: 13px;
                padding: 0 16px;
                text-align: left;
            }}
            QPushButton:hover {{
                background: {P.GRAY_200};
                color: {P.GRAY_700};
            }}
        """)
        write_btn.clicked.connect(self._on_write_comment)
        write_row.addWidget(write_btn, 1)
        write_row_w = QWidget()
        write_row_w.setStyleSheet("background: transparent;")
        write_row_w.setLayout(write_row)
        self._comments_outer_lay.addWidget(write_row_w)

        # Comment rows container
        self._comments_list_w = QWidget()
        self._comments_list_w.setStyleSheet("background: transparent;")
        self._comments_lay = QVBoxLayout(self._comments_list_w)
        self._comments_lay.setContentsMargins(0, 0, 0, 0)
        self._comments_lay.setSpacing(0)
        self._comments_outer_lay.addWidget(self._comments_list_w)

        self._rebuild_comments()
        self._comments_widget.setVisible(False)
        card_lay.addWidget(self._comments_widget)

        outer.addWidget(card)

        # Bottom separator between posts (Substack-style full-width line)
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background: {P.GRAY_200}; border: none;")
        outer.addWidget(sep)

    def _rebuild_comments(self):
        while self._comments_lay.count():
            item = self._comments_lay.takeAt(0)
            if item.widget(): item.widget().deleteLater()

        comments = self.post.get("comments", [])
        for i, c in enumerate(comments):
            # Normalise — support old plain-string comments
            if isinstance(c, str):
                c = {"text": c, "date": "", "likes": 0, "liked": False}
                self.post["comments"][i] = c

            c_text  = c.get("text", "")
            c_date  = c.get("date", "")
            c_likes = c.get("likes", 0)
            c_liked = c.get("liked", False)

            row = QWidget()
            row.setStyleSheet("background: transparent;")
            row_lay = QHBoxLayout(row)
            row_lay.setContentsMargins(0, 8, 0, 8)
            row_lay.setSpacing(0)

            # Left column: avatar + vertical thread line below it
            left_col = QWidget()
            left_col.setFixedWidth(42)
            left_col.setStyleSheet("background: transparent;")
            left_lay = QVBoxLayout(left_col)
            left_lay.setContentsMargins(0, 0, 10, 0)
            left_lay.setSpacing(0)

            av = AvatarButton(initials="Y", color=P.ORANGE)
            av.setFixedSize(32, 32)
            av.setEnabled(False)
            left_lay.addWidget(av, alignment=Qt.AlignmentFlag.AlignHCenter)

            # Thread line below avatar (if not last comment)
            if i < len(comments) - 1:
                line_w = QWidget()
                line_w.setFixedWidth(2)
                line_w.setStyleSheet(f"background: {P.GRAY_200}; border-radius: 1px;")
                line_lay = QHBoxLayout(line_w)
                line_lay.setContentsMargins(0, 0, 0, 0)
                left_lay.addWidget(line_w, 1, alignment=Qt.AlignmentFlag.AlignHCenter)
            else:
                left_lay.addStretch()

            row_lay.addWidget(left_col, alignment=Qt.AlignmentFlag.AlignTop)

            # Right column: name + date + text + actions
            right = QVBoxLayout()
            right.setContentsMargins(0, 0, 0, 0)
            right.setSpacing(3)

            # Name + date + delete row
            top_row = QHBoxLayout()
            top_row.setContentsMargins(0, 0, 0, 0)
            top_row.setSpacing(0)

            name_lbl = QLabel("You")
            name_lbl.setStyleSheet(
                f"font-size: 13px; font-weight: bold; color: {P.BLACK};"
                "font-family: Georgia, serif; background: transparent;"
            )
            top_row.addWidget(name_lbl)

            if c_date:
                top_row.addSpacing(8)
                date_lbl = QLabel(c_date)
                date_lbl.setStyleSheet(
                    f"font-size: 11px; color: {P.GRAY_400};"
                    "font-family: Georgia, serif; background: transparent;"
                )
                top_row.addWidget(date_lbl)

            top_row.addStretch()

            # Delete button (×)
            del_btn = QPushButton("×")
            del_btn.setFixedSize(20, 20)
            del_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            del_btn.setStyleSheet(f"""
                QPushButton {{
                    background: transparent; border: none;
                    color: {P.GRAY_300}; font-size: 16px; padding: 0;
                }}
                QPushButton:hover {{ color: {P.RED}; }}
            """)
            idx = i
            del_btn.clicked.connect(lambda _, ix=idx: self._delete_comment(ix))
            top_row.addWidget(del_btn)
            right.addLayout(top_row)

            # Comment text
            text_lbl = QLabel(c_text)
            text_lbl.setWordWrap(True)
            text_lbl.setStyleSheet(
                f"font-size: 14px; color: {P.BLACK};"
                "font-family: Georgia, serif; background: transparent;"
            )
            right.addWidget(text_lbl)
            right.addSpacing(2)

            # Action row: heart + reply
            act_row = QHBoxLayout()
            act_row.setContentsMargins(0, 0, 0, 0)
            act_row.setSpacing(0)

            c_like_btn = QPushButton(f"♥  {c_likes}")
            c_like_btn.setCheckable(True)
            c_like_btn.setChecked(c_liked)
            c_like_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            c_like_btn.setFont(QFont("Georgia", 12))
            clb_style = lambda liked: (
                f"QPushButton {{ background: transparent; border: none; color: {P.HEART if liked else P.GRAY_400}; padding: 2px 8px 2px 0; }}"
                f"QPushButton:hover {{ color: {P.HEART}; }}"
            )
            c_like_btn.setStyleSheet(clb_style(c_liked))

            def _make_c_like(btn, comment_dict, style_fn):
                def toggle():
                    comment_dict["liked"] = btn.isChecked()
                    comment_dict["likes"] += 1 if comment_dict["liked"] else -1
                    comment_dict["likes"] = max(0, comment_dict["likes"])
                    btn.setText(f"♥  {comment_dict['likes']}")
                    btn.setStyleSheet(style_fn(comment_dict["liked"]))
                return toggle
            c_like_btn.clicked.connect(_make_c_like(c_like_btn, c, clb_style))
            act_row.addWidget(c_like_btn)

            act_row.addSpacing(12)

            reply_btn = QPushButton("Reply")
            reply_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            reply_btn.setFont(QFont("Georgia", 12))
            reply_btn.setStyleSheet(f"""
                QPushButton {{ background: transparent; border: none;
                    color: {P.GRAY_400}; padding: 2px 6px 2px 0; font-weight: bold; }}
                QPushButton:hover {{ color: {P.BLACK}; }}
            """)
            reply_btn.clicked.connect(self._on_write_comment)
            act_row.addWidget(reply_btn)
            act_row.addStretch()
            right.addLayout(act_row)

            row_lay.addLayout(right, 1)
            self._comments_lay.addWidget(row)

    def _delete_comment(self, idx: int):
        comments = self.post.get("comments", [])
        if 0 <= idx < len(comments):
            comments.pop(idx)
            self._cmt_btn.set_count(len(comments))
            self._rebuild_comments()

    def _update_like_style(self, liked: bool):
        if liked:
            self._like_btn.setStyleSheet(f"""
                QPushButton {{ background: #FFF0F4; border: none;
                    color: {P.HEART}; padding: 5px 12px; border-radius: 8px; font-weight: bold; }}
                QPushButton:hover {{ background: #FFE0EB; }}
            """)
        else:
            self._like_btn.setStyleSheet(f"""
                QPushButton {{ background: transparent; border: none;
                    color: {P.GRAY_500}; padding: 5px 12px; border-radius: 8px; }}
                QPushButton:hover {{ background: {P.GRAY_100}; color: {P.HEART}; }}
            """)

    def _on_like(self):
        liked = self._like_btn.isChecked()
        likes = self.post.get("likes", 0)
        likes = likes + (1 if liked else -1)
        self.post["likes"] = max(0, likes)
        self.post["liked"] = liked
        self._like_btn.setText(f"  ♥  {self.post['likes']}")
        self._update_like_style(liked)
        self.like_toggled.emit(self.post["id"], liked)

    def _on_comment_btn(self):
        """Toggle comment section visibility."""
        visible = self._comments_widget.isVisible()
        self._comments_widget.setVisible(not visible)

    def _on_write_comment(self):
        dlg = CommentDialog(self)
        dlg.submitted.connect(self._add_comment)
        dlg.exec()

    def _add_comment(self, text: str):
        comment = {
            "text": text,
            "date": datetime.now().strftime("%b %-d"),
            "likes": 0,
            "liked": False,
        }
        self.post.setdefault("comments", []).append(comment)
        n = len(self.post["comments"])
        self._cmt_btn.set_count(n)
        self._rebuild_comments()
        self._comments_widget.setVisible(True)
        self.comment_added.emit(self.post["id"], text)

    def _show_card_menu(self, pid):
        from PyQt6.QtGui import QCursor
        menu = QMenu(self)
        menu.setWindowFlags(menu.windowFlags() | Qt.WindowType.NoDropShadowWindowHint)
        menu.setStyleSheet(f"""
            QMenu {{
                background: {P.WHITE};
                border: 1px solid {P.GRAY_200};
                font-family: Georgia, serif;
                font-size: 13px;
                border-radius: 8px;
                padding: 4px 0;
            }}
            QMenu::item {{ padding: 9px 20px; color: {P.BLACK}; }}
            QMenu::item:selected {{ background: {P.GRAY_100}; border-radius: 4px; }}
            QMenu::separator {{ height: 1px; background: {P.GRAY_200}; margin: 4px 8px; }}
        """)
        edit_act = menu.addAction("✏  Edit post")
        edit_act.triggered.connect(lambda: self._on_edit_post())
        menu.addSeparator()
        del_act = menu.addAction("🗑  Delete post")
        del_act.setForeground(QColor(P.RED))
        del_act.triggered.connect(lambda: self.delete_post.emit(pid))
        menu.exec(QCursor.pos())

    def _on_edit_post(self):
        """Open a simple edit dialog to update the post body."""
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton, QLabel
        dlg = QDialog(self.window())
        dlg.setWindowTitle("Edit post")
        dlg.setFixedSize(560, 320)
        dlg.setStyleSheet(f"background: {P.WHITE};")
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.setSpacing(12)

        lbl = QLabel("Edit your post")
        lbl.setStyleSheet(f"font-size: 15px; font-weight: bold; color: {P.BLACK}; font-family: Georgia, serif;")
        lay.addWidget(lbl)

        edit = QTextEdit()
        edit.setPlainText(self.post.get("body", ""))
        edit.setStyleSheet(f"""
            QTextEdit {{
                background: {P.GRAY_100}; border: 1px solid {P.GRAY_200};
                border-radius: 8px; padding: 10px;
                font-family: Georgia, serif; font-size: 14px; color: {P.BLACK};
            }}
        """)
        lay.addWidget(edit, 1)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel = QPushButton("Cancel")
        cancel.setFixedHeight(34)
        cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        cancel.setStyleSheet(f"""
            QPushButton {{ background: transparent; border: 1px solid {P.GRAY_300};
                border-radius: 6px; padding: 0 18px;
                font-family: Georgia, serif; font-size: 13px; color: {P.GRAY_500}; }}
            QPushButton:hover {{ border-color: {P.GRAY_500}; color: {P.BLACK}; }}
        """)
        cancel.clicked.connect(dlg.reject)

        save = QPushButton("Save changes")
        save.setFixedHeight(34)
        save.setCursor(Qt.CursorShape.PointingHandCursor)
        save.setStyleSheet(f"""
            QPushButton {{ background: {P.ORANGE}; border: none;
                border-radius: 6px; padding: 0 18px;
                font-family: Georgia, serif; font-size: 13px;
                color: {P.WHITE}; font-weight: bold; }}
            QPushButton:hover {{ background: {P.ORANGE_HOVER}; }}
        """)

        def _do_save():
            new_body = edit.toPlainText().strip()
            self.post["body"] = new_body
            # Persist to feed db
            posts = _load_feed()
            for p in posts:
                if p["id"] == self.post["id"]:
                    p["body"] = new_body
                    break
            _save_feed(posts)
            dlg.accept()
            # Signal parent HomeView to refresh
            self.delete_post.emit("")   # empty string = soft refresh signal

        save.clicked.connect(_do_save)
        btn_row.addWidget(cancel)
        btn_row.addSpacing(8)
        btn_row.addWidget(save)
        lay.addLayout(btn_row)
        dlg.exec()


# ==============================================================================
#  NEWS SIDEBAR  — live world news fetched from RSS
# ==============================================================================

NEWS_FEEDS = [
    ("World",    "https://feeds.bbci.co.uk/news/world/rss.xml"),
    ("Tech",     "https://feeds.bbci.co.uk/news/technology/rss.xml"),
    ("Science",  "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml"),
]

NEWS_CACHE_PATH = os.path.join(DATA_DIR, "news_cache.json")


class NewsFetcher(QThread):
    """Background thread that fetches & parses RSS news items."""
    articles_ready = pyqtSignal(list)   # emits list of dicts
    fetch_error    = pyqtSignal(str)

    def __init__(self, feeds=None, parent=None):
        super().__init__(parent)
        self._feeds = feeds or NEWS_FEEDS

    def run(self):
        articles = []
        for category, url in self._feeds:
            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": "Mozilla/5.0 Writes/1.0"}
                )
                with urllib.request.urlopen(req, timeout=8) as resp:
                    raw = resp.read()
                root = ET.fromstring(raw)
                ns   = {"media": "http://search.yahoo.com/mrss/"}
                channel = root.find("channel")
                if channel is None:
                    continue
                for item in channel.findall("item")[:5]:
                    title_el = item.find("title")
                    link_el  = item.find("link")
                    desc_el  = item.find("description")
                    pub_el   = item.find("pubDate")
                    title = html.unescape(title_el.text or "") if title_el is not None else ""
                    link  = link_el.text.strip() if link_el is not None and link_el.text else ""
                    desc  = html.unescape(desc_el.text or "") if desc_el is not None else ""
                    # strip any HTML tags from description
                    import re
                    desc = re.sub(r"<[^>]+>", "", desc).strip()
                    pub  = pub_el.text if pub_el is not None else ""

                    # Try multiple image sources
                    img_url = ""
                    # 1) media:thumbnail or media:content
                    for tag in ("media:thumbnail", "media:content"):
                        el = item.find(tag, ns)
                        if el is not None:
                            img_url = el.get("url", "")
                            if img_url: break
                    # 2) enclosure
                    if not img_url:
                        enc = item.find("enclosure")
                        if enc is not None and (enc.get("type","").startswith("image")):
                            img_url = enc.get("url","")

                    if title and link:
                        articles.append({
                            "title":    title,
                            "link":     link,
                            "desc":     desc,
                            "pub":      pub,
                            "img_url":  img_url,
                            "category": category,
                            "img_data": None,   # filled in later
                        })
            except Exception as e:
                pass  # skip broken feeds silently

        # fetch thumbnail images
        for art in articles:
            if art["img_url"]:
                try:
                    req = urllib.request.Request(
                        art["img_url"],
                        headers={"User-Agent": "Mozilla/5.0 Writes/1.0"}
                    )
                    with urllib.request.urlopen(req, timeout=5) as resp:
                        art["img_data"] = resp.read()
                except Exception:
                    pass

        if articles:
            # cache to disk
            try:
                cache = [
                    {k: v for k, v in a.items() if k != "img_data"}
                    for a in articles
                ]
                with open(NEWS_CACHE_PATH, "w", encoding="utf-8") as f:
                    json.dump(cache, f)
            except Exception:
                pass
            self.articles_ready.emit(articles)
        else:
            # try loading from cache
            try:
                with open(NEWS_CACHE_PATH, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                for a in cached:
                    a["img_data"] = None
                if cached:
                    self.articles_ready.emit(cached)
                    return
            except Exception:
                pass
            self.fetch_error.emit("Could not load news.")


class NewsArticleDialog(QDialog):
    """
    Clean, borderless Substack-style article reader.
    Full-height modal, edge-to-edge hero image, all article details shown,
    no 'Continue reading' button — just the content.
    """

    def __init__(self, article: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle(article.get("title", "Article"))
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)

        if parent:
            pw, ph = parent.width(), parent.height()
            w = min(760, max(600, pw - 160))
            h = min(ph - 20, 860)
        else:
            w, h = 740, 820
        self.setFixedSize(w, h)
        # Shadow-like outer: just a very faint bg, no border
        self.setStyleSheet(f"background: {P.WHITE};")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── FLOATING TOP BAR (overlays the image) ───────────────────────────
        # Rendered as a real widget above the scroll, not overlaid
        topbar = QWidget()
        topbar.setFixedHeight(50)
        topbar.setStyleSheet(f"background: {P.WHITE}; border: none;")
        tb_lay = QHBoxLayout(topbar)
        tb_lay.setContentsMargins(18, 0, 18, 0)
        tb_lay.setSpacing(0)

        close_btn = QPushButton("✕")
        close_btn.setFixedSize(30, 30)
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background: {P.GRAY_100}; border: none; border-radius: 15px;
                color: {P.GRAY_500}; font-size: 12px; font-weight: bold;
            }}
            QPushButton:hover {{ background: {P.GRAY_200}; color: {P.BLACK}; }}
        """)
        close_btn.clicked.connect(self.reject)
        tb_lay.addWidget(close_btn)
        tb_lay.addStretch()

        link = article.get("link", "")
        cat  = article.get("category", "")
        if cat:
            src_lbl = QLabel(cat.upper())
            src_lbl.setStyleSheet(
                f"font-size: 9px; letter-spacing: 1.6px; font-weight: bold;"
                f"color: {P.GRAY_400}; font-family: Georgia, serif; background: transparent;"
            )
            tb_lay.addWidget(src_lbl)
            tb_lay.addSpacing(14)

        if link:
            ext_btn = QPushButton("Open in browser  ↗")
            ext_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            ext_btn.setStyleSheet(f"""
                QPushButton {{
                    background: transparent; border: none;
                    color: {P.GRAY_400}; font-family: Georgia, serif; font-size: 12px;
                    padding: 4px 0;
                }}
                QPushButton:hover {{ color: {P.ORANGE}; }}
            """)
            ext_btn.clicked.connect(lambda: self._open_link(link))
            tb_lay.addWidget(ext_btn)

        root.addWidget(topbar)

        # Thin separator under topbar
        sep0 = QFrame(); sep0.setFrameShape(QFrame.Shape.HLine)
        sep0.setFixedHeight(1)
        sep0.setStyleSheet(f"background: {P.GRAY_100}; border: none;")
        root.addWidget(sep0)

        # ── SCROLLABLE BODY ──────────────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameStyle(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"""
            QScrollArea {{ background: {P.WHITE}; border: none; }}
            QScrollBar:vertical {{ background: transparent; width: 4px; }}
            QScrollBar::handle:vertical {{
                background: {P.GRAY_200}; border-radius: 2px; min-height: 20px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
        """)

        body = QWidget()
        body.setStyleSheet(f"background: {P.WHITE};")
        body_lay = QVBoxLayout(body)
        body_lay.setContentsMargins(0, 0, 0, 64)
        body_lay.setSpacing(0)

        # ── HERO IMAGE — full width, no rounding ─────────────────────────
        img_data = article.get("img_data")
        if img_data:
            pix = QPixmap()
            pix.loadFromData(img_data)
            if not pix.isNull():
                hero_h = int(w * 0.52)   # slightly taller than 16:9
                pix = pix.scaled(QSize(w, hero_h),
                                 Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                                 Qt.TransformationMode.SmoothTransformation)
                cx = (pix.width()  - w) // 2
                cy = (pix.height() - hero_h) // 2
                pix = pix.copy(max(0, cx), max(0, cy), w, hero_h)
                img_lbl = QLabel()
                img_lbl.setPixmap(pix)
                img_lbl.setFixedHeight(hero_h)
                img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                img_lbl.setStyleSheet("background: transparent; border: none;")
                body_lay.addWidget(img_lbl)

        # ── TEXT AREA ─────────────────────────────────────────────────────
        PAD = 60
        text_w = QWidget()
        text_w.setStyleSheet(f"background: {P.WHITE};")
        tl = QVBoxLayout(text_w)
        tl.setContentsMargins(PAD, 36, PAD, 0)
        tl.setSpacing(0)

        # Category + date row
        meta_row = QHBoxLayout()
        meta_row.setSpacing(0)
        if cat:
            cat_lbl2 = QLabel(cat.upper())
            cat_lbl2.setStyleSheet(
                f"font-size: 10px; letter-spacing: 1.4px; font-weight: bold;"
                f"color: {P.ORANGE}; font-family: Georgia, serif; background: transparent;"
            )
            meta_row.addWidget(cat_lbl2)
            meta_row.addSpacing(12)

        pub_raw = article.get("pub", "")
        if pub_raw:
            try:
                from email.utils import parsedate_to_datetime
                dt = parsedate_to_datetime(pub_raw)
                pub_fmt = dt.strftime("%B %d, %Y  ·  %I:%M %p")
            except Exception:
                pub_fmt = pub_raw
            pub_lbl = QLabel(pub_fmt)
            pub_lbl.setStyleSheet(
                f"font-size: 10px; letter-spacing: 0.3px; color: {P.GRAY_400};"
                "font-family: Georgia, serif; background: transparent;"
            )
            meta_row.addWidget(pub_lbl)
        meta_row.addStretch()
        tl.addLayout(meta_row)
        tl.addSpacing(16)

        # Title
        title_lbl = QLabel(article.get("title", ""))
        title_lbl.setWordWrap(True)
        title_lbl.setStyleSheet(
            f"font-size: 30px; font-weight: bold; color: {P.BLACK};"
            "font-family: Georgia, serif; line-height: 1.2; background: transparent;"
        )
        tl.addWidget(title_lbl)
        tl.addSpacing(24)

        # Thin rule
        rule = QFrame(); rule.setFrameShape(QFrame.Shape.HLine)
        rule.setFixedHeight(1)
        rule.setStyleSheet(f"background: {P.GRAY_100}; border: none;")
        tl.addWidget(rule)
        tl.addSpacing(24)

        # Full description / body text
        desc = article.get("desc", "")
        if desc:
            desc_lbl = QLabel(desc)
            desc_lbl.setWordWrap(True)
            desc_lbl.setStyleSheet(
                f"font-size: 16px; color: {P.GRAY_700}; line-height: 1.8;"
                "font-family: Georgia, serif; background: transparent;"
            )
            tl.addWidget(desc_lbl)
            tl.addSpacing(36)

        # Source link row — clean text link, no button
        if link:
            tl.addSpacing(8)
            rule2 = QFrame(); rule2.setFrameShape(QFrame.Shape.HLine)
            rule2.setFixedHeight(1)
            rule2.setStyleSheet(f"background: {P.GRAY_100}; border: none;")
            tl.addWidget(rule2)
            tl.addSpacing(20)

            src_row = QHBoxLayout()
            src_note = QLabel("Source")
            src_note.setStyleSheet(
                f"font-size: 11px; color: {P.GRAY_400}; font-family: Georgia, serif;"
                "background: transparent; font-weight: bold; letter-spacing: 0.5px;"
            )
            src_row.addWidget(src_note)
            src_row.addSpacing(10)

            src_link_btn = QPushButton(link[:72] + ("…" if len(link) > 72 else ""))
            src_link_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            src_link_btn.setStyleSheet(f"""
                QPushButton {{
                    background: transparent; border: none;
                    color: {P.ORANGE}; font-family: Georgia, serif; font-size: 11px;
                    text-align: left; padding: 0;
                }}
                QPushButton:hover {{ color: {P.ORANGE_HOVER}; }}
            """)
            src_link_btn.clicked.connect(lambda: self._open_link(link))
            src_row.addWidget(src_link_btn, 1)
            tl.addLayout(src_row)

        tl.addStretch()
        body_lay.addWidget(text_w)

        scroll.setWidget(body)
        root.addWidget(scroll, 1)

    def _open_link(self, url):
        import webbrowser
        webbrowser.open(url)


class NewsCard(QWidget):
    """
    Borderless editorial card. Image on top, category in orange,
    bold headline, short snippet. Hover = soft warm tint, no border flash.
    """
    clicked = pyqtSignal(dict)

    def __init__(self, article: dict, parent=None):
        super().__init__(parent)
        self._article = article
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet("background: transparent;")
        self._build()

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Hover wrapper — rounded card with soft bg on hover
        self._wrap = QWidget()
        self._wrap.setObjectName("ncwrap")
        self._set_style(False)
        wrap_lay = QVBoxLayout(self._wrap)
        wrap_lay.setContentsMargins(0, 10, 0, 18)
        wrap_lay.setSpacing(10)

        # ── Thumbnail ─────────────────────────────────────────────────────
        img_data = self._article.get("img_data")
        if img_data:
            pix = QPixmap()
            pix.loadFromData(img_data)
            if not pix.isNull():
                CARD_W, THUMB_H = 300, 170
                pix = pix.scaled(QSize(CARD_W, THUMB_H),
                                 Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                                 Qt.TransformationMode.SmoothTransformation)
                cx = (pix.width()  - CARD_W) // 2
                cy = (pix.height() - THUMB_H) // 2
                pix = pix.copy(max(0, cx), max(0, cy), CARD_W, THUMB_H)
                img_lbl = QLabel()
                img_lbl.setPixmap(pix)
                img_lbl.setFixedHeight(THUMB_H)
                img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                img_lbl.setStyleSheet(
                    "border-radius: 10px; background: transparent; border: none;"
                )
                img_lbl.setScaledContents(False)
                wrap_lay.addWidget(img_lbl)

        # ── Text ──────────────────────────────────────────────────────────
        txt = QWidget()
        txt.setStyleSheet("background: transparent;")
        tl = QVBoxLayout(txt)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.setSpacing(5)

        # Category
        cat = self._article.get("category", "")
        if cat:
            cat_lbl = QLabel(cat.upper())
            cat_lbl.setStyleSheet(
                f"font-size: 9px; letter-spacing: 1.5px; font-weight: bold;"
                f"color: {P.ORANGE}; font-family: Georgia, serif; background: transparent;"
            )
            tl.addWidget(cat_lbl)

        # Headline
        title = self._article.get("title", "")
        short = (title[:95] + "…") if len(title) > 95 else title
        title_lbl = QLabel(short)
        title_lbl.setWordWrap(True)
        title_lbl.setStyleSheet(
            f"font-size: 14px; font-weight: bold; color: {P.BLACK};"
            "font-family: Georgia, serif; line-height: 1.4; background: transparent;"
        )
        tl.addWidget(title_lbl)

        # Snippet
        desc = self._article.get("desc", "")
        if desc:
            snip = (desc[:110] + "…") if len(desc) > 110 else desc
            snip_lbl = QLabel(snip)
            snip_lbl.setWordWrap(True)
            snip_lbl.setStyleSheet(
                f"font-size: 11px; color: {P.GRAY_400}; font-family: Georgia, serif;"
                "line-height: 1.5; background: transparent;"
            )
            tl.addWidget(snip_lbl)

        wrap_lay.addWidget(txt)
        outer.addWidget(self._wrap)

    def _set_style(self, hovered: bool):
        bg = P.ORANGE_LIGHT if hovered else "transparent"
        self._wrap.setStyleSheet(f"""
            QWidget#ncwrap {{
                background: {bg};
                border: none;
                border-radius: 8px;
            }}
        """)

    def enterEvent(self, e):  self._set_style(True)
    def leaveEvent(self, e):  self._set_style(False)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._article)
        super().mousePressEvent(e)


class _SectionHeader(QWidget):
    """Header row painted with QPainter lines - zero CSS bleed to siblings."""
    def __init__(self, title: str, see_all: bool = True):
        super().__init__()
        self.setFixedHeight(34)
        self.setStyleSheet("background: transparent; border: none;")

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 8, 4)
        lay.setSpacing(8)

        t = QLabel(title)
        t.setStyleSheet(
            f"font-size: 13px; font-weight: bold; color: {P.BLACK};"
            "font-family: Georgia, serif; background: transparent; border: none;"
        )
        t.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        lay.addWidget(t)
        lay.addStretch(1)

        if see_all:
            sa = QPushButton("See all")
            sa.setCursor(Qt.CursorShape.PointingHandCursor)
            sa.setStyleSheet(
                f"QPushButton {{ background: transparent; border: none; "
                f"color: {P.GRAY_400}; font-family: Georgia, serif; font-size: 12px; "
                f"padding: 0; margin: 0; }}"
                f"QPushButton:hover {{ color: {P.BLACK}; }}"
            )
            sa.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
            lay.addWidget(sa)

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        y = self.height() - 1
        p.setPen(QPen(QColor(P.BLACK), 2))
        p.drawLine(0, y, 55, y)
        p.setPen(QPen(QColor(P.GRAY_200), 1))
        p.drawLine(56, y, self.width(), y)


class _SidebarSection(QWidget):
    """
    Flat section. Header is a self-contained painted widget; body is a
    completely separate child widget so there is no CSS bleed whatsoever.
    """
    def __init__(self, title: str, see_all: bool = True):
        super().__init__()
        self.setStyleSheet(f"background: {P.BG}; border: none;")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        outer.addWidget(_SectionHeader(title, see_all))
        outer.addSpacing(10)

        self._body = QWidget()
        self._body.setStyleSheet(f"background: {P.BG}; border: none;")
        self._body_lay = QVBoxLayout(self._body)
        self._body_lay.setContentsMargins(0, 0, 0, 0)
        self._body_lay.setSpacing(0)
        outer.addWidget(self._body)

    def body_layout(self):
        return self._body_lay


class _BookmarkBtn(QPushButton):
    """Small bookmark icon button for Up Next cards."""
    def __init__(self):
        super().__init__()
        self.setFixedSize(22, 22)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._hovered = False
        self._saved = False
        self.setStyleSheet("border: none; background: transparent;")
        self.setToolTip("Save for later")
        self.clicked.connect(self._toggle)

    def _toggle(self):
        self._saved = not self._saved
        self.update()

    def enterEvent(self, e): self._hovered = True;  self.update()
    def leaveEvent(self, e): self._hovered = False; self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = P.ORANGE if self._saved else (P.GRAY_500 if self._hovered else P.GRAY_300)
        p.setPen(QPen(QColor(color), 1.3, Qt.PenStyle.SolidLine,
                      Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        if self._saved:
            p.setBrush(QBrush(QColor(P.ORANGE)))
        else:
            p.setBrush(Qt.BrushStyle.NoBrush)
        # Bookmark shape
        pts = QPolygon([
            QPoint(4, 2), QPoint(18, 2), QPoint(18, 20),
            QPoint(11, 14), QPoint(4, 20),
        ])
        p.drawPolygon(pts)


class _RoundedThumb(QWidget):
    """QPixmap thumbnail with properly clipped rounded corners via QPainter."""
    def __init__(self, pix: QPixmap, w: int, h: int, radius: int = 8):
        super().__init__()
        self._pix   = pix
        self._r     = radius
        self.setFixedSize(w, h)
        self.setStyleSheet("background: transparent; border: none;")

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        from PyQt6.QtGui import QPainterPath
        path = QPainterPath()
        path.addRoundedRect(0, 0, self.width(), self.height(), self._r, self._r)
        p.setClipPath(path)
        p.drawPixmap(0, 0, self._pix)


class _UpNextCard(QWidget):
    """
    Mini article row styled to match the main feed — same warm bg, same type weight,
    no box border. Hover shows a very subtle warm tint.
    """
    clicked = pyqtSignal(dict)
    _THUMB_W = 72
    _THUMB_H = 56

    def __init__(self, article: dict):
        super().__init__()
        self._article = article
        self._hovered = False
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 12, 0, 12)
        root.setSpacing(5)

        # ── Row 1: [S circle] Category ········ [bookmark] ────────────────
        r1 = QHBoxLayout()
        r1.setContentsMargins(0, 0, 0, 0)
        r1.setSpacing(6)

        icon = QLabel("S")
        icon.setFixedSize(18, 18)
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon.setStyleSheet(f"""
            background: {P.ORANGE}; color: white; border-radius: 9px;
            font-size: 8px; font-weight: bold; font-family: Georgia, serif;
        """)
        r1.addWidget(icon)

        cat = self._article.get("category", "World")
        src = QLabel(cat)
        src.setStyleSheet(
            f"font-size: 11px; color: {P.GRAY_500}; font-family: Georgia, serif;"
            "background: transparent; border: none;"
        )
        r1.addWidget(src)
        r1.addStretch()
        r1.addWidget(_BookmarkBtn())
        root.addLayout(r1)

        # ── Row 2: Bold title LEFT · thumbnail RIGHT ──────────────────────
        r2 = QHBoxLayout()
        r2.setContentsMargins(0, 2, 0, 2)
        r2.setSpacing(10)

        title = self._article.get("title", "")
        t = QLabel(title)
        t.setWordWrap(True)
        t.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        t.setStyleSheet(
            f"font-size: 13px; font-weight: bold; color: {P.BLACK};"
            "font-family: Georgia, serif; background: transparent; border: none;"
            "line-height: 1.35;"
        )
        r2.addWidget(t, 1, Qt.AlignmentFlag.AlignTop)

        img_data = self._article.get("img_data")
        if img_data:
            pix = QPixmap()
            pix.loadFromData(img_data)
            if not pix.isNull():
                W, H = self._THUMB_W, self._THUMB_H
                pix = pix.scaled(QSize(W, H),
                                 Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                                 Qt.TransformationMode.SmoothTransformation)
                cx = (pix.width()  - W) // 2
                cy = (pix.height() - H) // 2
                pix = pix.copy(max(0, cx), max(0, cy), W, H)
                r2.addWidget(_RoundedThumb(pix, W, H, radius=6),
                             0, Qt.AlignmentFlag.AlignTop)
        root.addLayout(r2)

        # ── Row 3: meta — same small-gray style as main feed ─────────────
        import random as _r
        meta_parts = []
        pub = self._article.get("pub", "")
        if pub:
            try:
                from email.utils import parsedate_to_datetime
                meta_parts.append(parsedate_to_datetime(pub).strftime("%b %d"))
            except Exception:
                pass
        desc = self._article.get("desc", "")
        if desc:
            meta_parts.append(f"{max(1, len(desc.split())//200)}m read")
        meta_parts.append(f"{_r.randint(1,6)}.{_r.randint(0,9)}K likes")
        meta_parts.append(f"{_r.randint(40,450)} comments")

        meta = QLabel("  ·  ".join(meta_parts))
        meta.setStyleSheet(
            f"font-size: 10px; color: {P.GRAY_400}; font-family: Georgia, serif;"
            "background: transparent; border: none;"
        )
        root.addWidget(meta)

    def enterEvent(self, e):
        self._hovered = True; self.update()
    def leaveEvent(self, e):
        self._hovered = False; self.update()
    def paintEvent(self, _):
        if self._hovered:
            p = QPainter(self)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.setBrush(QBrush(QColor(P.ORANGE_LIGHT)))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(self.rect(), 8, 8)
    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._article)
        super().mousePressEvent(e)


class _CheckBadge(QWidget):
    """Substack-style verified badge — small orange circle with a clean white tick."""
    def __init__(self):
        super().__init__()
        self.setFixedSize(14, 14)
        self.setStyleSheet("background: transparent; border: none;")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        # Outer circle — Substack orange
        p.setBrush(QBrush(QColor(P.ORANGE)))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(1, 1, 12, 12)
        # White tick — crisp, well-proportioned
        pen = QPen(QColor("#FFFFFF"), 1.6, Qt.PenStyle.SolidLine,
                   Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        p.drawLine(3, 7, 6, 10)
        p.drawLine(6, 10, 11, 4)


class _BestsellerAvatar(QWidget):
    """Circular avatar drawn with QPainter — no CSS border bleed."""
    def __init__(self, initial: str, color: str):
        super().__init__()
        self._initial = initial.upper()
        self._color   = color
        self.setFixedSize(38, 38)
        self.setStyleSheet("background: transparent; border: none;")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(QBrush(QColor(self._color)))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(0, 0, 38, 38)
        p.setPen(QPen(QColor("#FFFFFF")))
        f = QFont("Georgia", 14)
        f.setWeight(QFont.Weight.Bold)
        p.setFont(f)
        p.drawText(QRect(0, 0, 38, 38), Qt.AlignmentFlag.AlignCenter, self._initial)


class _BestsellerRow(QWidget):
    """
    Substack-style writer row: avatar · name + subtitle · Subscribe pill.
    Clean, airy, no rank numbers or color bars.
    """
    def __init__(self, name: str, sub: str, color: str):
        super().__init__()
        self._hovered = False
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(62)
        self.setStyleSheet("background: transparent; border: none;")

        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 0, 4, 0)
        lay.setSpacing(12)

        # Avatar
        lay.addWidget(_BestsellerAvatar(name[:1], color), 0, Qt.AlignmentFlag.AlignVCenter)

        # Name + subtitle
        info = QVBoxLayout()
        info.setSpacing(2)
        info.setContentsMargins(0, 0, 0, 0)

        name_row = QHBoxLayout()
        name_row.setSpacing(5)
        name_row.setContentsMargins(0, 0, 0, 0)

        n = QLabel(name)
        n.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        n.setMinimumWidth(0)
        n.setStyleSheet(
            f"font-size: 13px; font-weight: bold; color: {P.BLACK};"
            "font-family: Georgia, serif; background: transparent; border: none;"
        )
        name_row.addWidget(n, 1)
        name_row.addWidget(_CheckBadge(), 0, Qt.AlignmentFlag.AlignVCenter)
        info.addLayout(name_row)

        s = QLabel(sub)
        s.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        s.setMinimumWidth(0)
        s.setStyleSheet(
            f"font-size: 10px; color: {P.GRAY_400};"
            "font-family: Georgia, serif; background: transparent; border: none;"
        )
        info.addWidget(s)
        lay.addLayout(info, 1)

        # Subscribe pill — outlined, fills on hover
        sub_btn = QPushButton("Subscribe")
        sub_btn.setFixedSize(80, 28)
        sub_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        sub_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: {P.ORANGE};
                border: 1.5px solid {P.ORANGE};
                border-radius: 14px;
                font-size: 10px;
                font-family: Georgia, serif;
                font-weight: bold;
                letter-spacing: 0.3px;
            }}
            QPushButton:hover {{
                background: {P.ORANGE};
                color: white;
            }}
        """)
        lay.addWidget(sub_btn, 0, Qt.AlignmentFlag.AlignVCenter)

    def enterEvent(self, e):
        self._hovered = True; self.update()
    def leaveEvent(self, e):
        self._hovered = False; self.update()
    def paintEvent(self, _):
        if self._hovered:
            p = QPainter(self)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.setBrush(QBrush(QColor(P.GRAY_100)))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(self.rect(), 10, 10)


class NewsSidebar(QWidget):
    """
    Right sidebar — designed to feel like a natural extension of the main feed.
    Same warm background, same typography, thin left border as visual separator.
    No card boxes — sections are separated by the same underline-header style
    used in the main feed's 'For you' tab.
    """
    WIDTH = 340

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(self.WIDTH)
        self.setStyleSheet(f"background: {P.BG};")
        self._fetcher = None
        self._build()
        self._fetch_news()

    def paintEvent(self, _):
        """Draw a thin left border matching the main feed's tab divider."""
        p = QPainter(self)
        p.setPen(QPen(QColor(P.GRAY_200), 1))
        p.drawLine(0, 0, 0, self.height())

    def _build(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameStyle(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setStyleSheet(f"""
            QScrollArea {{ background: {P.BG}; border: none; }}
            QScrollBar:vertical {{
                background: transparent; width: 4px; margin: 0;
            }}
            QScrollBar::handle:vertical {{
                background: {P.GRAY_300}; border-radius: 2px; min-height: 20px;
            }}
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {{ height: 0; }}
        """)

        container = QWidget()
        container.setFixedWidth(self.WIDTH)
        container.setStyleSheet(f"background: {P.BG};")

        lay = QVBoxLayout(container)
        lay.setContentsMargins(12, 20, 12, 40)
        lay.setSpacing(0)

        # ── Search bar ────────────────────────────────────────────────────
        # Outer frame handles the focus-ring border via a wrapper
        # ── Search bar — warm, on-brand ──────────────────────────────────
        search_outer = QWidget()
        search_outer.setObjectName("searchouter")
        search_outer.setFixedHeight(44)
        search_outer.setStyleSheet(f"""
            QWidget#searchouter {{
                background: {P.WHITE};
                border: 1.5px solid {P.GRAY_200};
                border-radius: 22px;
            }}
        """)

        so_lay = QHBoxLayout(search_outer)
        so_lay.setContentsMargins(14, 0, 14, 0)
        so_lay.setSpacing(9)

        class _SearchIcon(QWidget):
            def __init__(self):
                super().__init__()
                self.setFixedSize(16, 16)
                self.setStyleSheet("background: transparent; border: none;")
            def paintEvent(self, _):
                p = QPainter(self)
                p.setRenderHint(QPainter.RenderHint.Antialiasing)
                p.setPen(QPen(QColor(P.GRAY_300), 1.8,
                              Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(1, 1, 10, 10)
                p.drawLine(10, 10, 15, 15)

        so_lay.addWidget(_SearchIcon(), 0, Qt.AlignmentFlag.AlignVCenter)

        search_input = QLineEdit()
        search_input.setPlaceholderText("Search Writes")
        search_input.setFrame(False)
        search_input.setClearButtonEnabled(True)
        search_input.setStyleSheet(f"""
            QLineEdit {{
                background: transparent;
                border: none;
                font-size: 13px;
                font-family: Georgia, serif;
                color: {P.BLACK};
                padding: 0;
            }}
        """)
        from PyQt6.QtGui import QPalette
        pal = search_input.palette()
        pal.setColor(QPalette.ColorRole.PlaceholderText, QColor(P.GRAY_300))
        search_input.setPalette(pal)

        # Focus: border turns orange to match app accent
        def _on_focus():
            search_outer.setStyleSheet(f"""
                QWidget#searchouter {{
                    background: {P.WHITE};
                    border: 1.5px solid {P.ORANGE};
                    border-radius: 22px;
                }}
            """)
        def _on_blur():
            search_outer.setStyleSheet(f"""
                QWidget#searchouter {{
                    background: {P.WHITE};
                    border: 1.5px solid {P.GRAY_200};
                    border-radius: 22px;
                }}
            """)
        search_input.focusInEvent  = lambda e: (_on_focus(), QLineEdit.focusInEvent(search_input, e))
        search_input.focusOutEvent = lambda e: (_on_blur(),  QLineEdit.focusOutEvent(search_input, e))

        so_lay.addWidget(search_input, 1)
        lay.addWidget(search_outer)
        lay.addSpacing(12)

        # ── Up Next card ──────────────────────────────────────────────────
        up_next_card = QWidget()
        up_next_card.setObjectName("sidecard")
        up_next_card.setStyleSheet("""
            QWidget#sidecard {
                background: #FFFFFF;
                border: 1px solid #E8E8E8;
                border-radius: 14px;
            }
        """)
        up_next_card_lay = QVBoxLayout(up_next_card)
        up_next_card_lay.setContentsMargins(16, 16, 16, 16)
        up_next_card_lay.setSpacing(0)

        self._up_next_sec = _SidebarSection("Up next")
        self._up_next_lay = self._up_next_sec.body_layout()

        self._status_lbl = QLabel("Fetching articles…")
        self._status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_lbl.setStyleSheet(
            f"font-size: 11px; color: {P.GRAY_300}; font-family: Georgia, serif;"
            "background: transparent; border: none; padding: 16px 0;"
        )
        self._up_next_lay.addWidget(self._status_lbl)
        up_next_card_lay.addWidget(self._up_next_sec)
        lay.addWidget(up_next_card)

        lay.addSpacing(16)

        # ── New Bestsellers card ──────────────────────────────────────────
        bs_card = QWidget()
        bs_card.setObjectName("sidecard")
        bs_card.setStyleSheet("""
            QWidget#sidecard {
                background: #FFFFFF;
                border: 1px solid #E8E8E8;
                border-radius: 14px;
            }
        """)
        bs_card_lay = QVBoxLayout(bs_card)
        bs_card_lay.setContentsMargins(16, 16, 16, 16)
        bs_card_lay.setSpacing(0)

        bestsellers_sec = _SidebarSection("New Bestsellers")
        bl = bestsellers_sec.body_layout()
        _COLORS = ["#E8651A", "#1565C0", "#2E7D32", "#6A1B9A", "#00838F"]
        _PEOPLE = [
            ("Robin J Brooks",     "Robin J Brooks"),
            ("FarStellar",         "FarStellar"),
            ("Scott MacFarlane",   "Scott MacFarlane Reports"),
            ("Verena M. Dittrich", "Verena M. Dittrich"),
            ("LHGrey",             "LHGrey's Substack"),
        ]
        for i, (name, sub) in enumerate(_PEOPLE):
            row = _BestsellerRow(name, sub, _COLORS[i % len(_COLORS)])
            bl.addWidget(row)
            if i < len(_PEOPLE) - 1:
                div = QFrame()
                div.setFrameShape(QFrame.Shape.HLine)
                div.setFixedHeight(1)
                div.setStyleSheet(f"background: {P.GRAY_200}; border: none;")
                bl.addWidget(div)
        bs_card_lay.addWidget(bestsellers_sec)
        lay.addWidget(bs_card)

        lay.addStretch()
        scroll.setWidget(container)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(scroll)
    def _fetch_news(self):
        self._fetcher = NewsFetcher()
        self._fetcher.articles_ready.connect(self._on_articles)
        self._fetcher.fetch_error.connect(self._on_error)
        self._fetcher.start()

    def _on_articles(self, articles: list):
        while self._up_next_lay.count():
            item = self._up_next_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for i, art in enumerate(articles[:3]):
            card = _UpNextCard(art)
            card.clicked.connect(self._open_article)
            self._up_next_lay.addWidget(card)
            if i < min(2, len(articles) - 1):
                div = QFrame()
                div.setFrameShape(QFrame.Shape.HLine)
                div.setFixedHeight(1)
                div.setStyleSheet(f"background: {P.GRAY_200}; border: none;")
                self._up_next_lay.addWidget(div)

    def _on_error(self, msg: str):
        self._status_lbl.setText(f"⚠  {msg}")

    def _open_article(self, article: dict):
        dlg = NewsArticleDialog(article, self.window())
        dlg.exec()


class HomeView(QWidget):
    """Full Substack-inspired home feed, fully offline."""
    navigate_to_editor = pyqtSignal()

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.setStyleSheet(f"background: {P.BG};")
        self._posts: list = _load_feed()
        self._pending_image: str = ""
        self._build()
        self._populate()

    def _build(self):
        # Root layout: feed scroll on left, news sidebar on right
        root_h = QHBoxLayout(self)
        root_h.setContentsMargins(0, 0, 0, 0)
        root_h.setSpacing(0)

        # ── LEFT: main feed scroll area ────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameStyle(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"""
            QScrollArea {{ background: {P.BG}; border: none; }}
            QScrollBar:vertical {{ background: transparent; width: 6px; margin: 0; }}
            QScrollBar::handle:vertical {{ background: {P.GRAY_300}; border-radius: 3px; min-height: 30px; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
        """)
        root_h.addWidget(scroll, 1)

        container = QWidget()
        container.setStyleSheet(f"background: {P.BG};")
        scroll.setWidget(container)

        cl = QVBoxLayout(container)
        cl.setContentsMargins(0, 32, 0, 80)
        cl.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        col = QWidget()
        col.setFixedWidth(680)
        col_lay = QVBoxLayout(col)
        col_lay.setContentsMargins(0, 0, 0, 0)
        col_lay.setSpacing(0)

        # "For you" tab bar
        tab_row = QHBoxLayout()
        tab_row.setSpacing(0)
        for_you = QLabel("For you  ▾")
        for_you.setStyleSheet(f"""
            font-size: 14px; font-family: Georgia, serif; font-weight: bold;
            color: {P.BLACK};
            padding: 0 0 10px 0;
            border-bottom: 2px solid {P.BLACK};
        """)
        tab_row.addWidget(for_you)
        tab_row.addSpacing(24)

        following_lbl = QLabel("Following")
        following_lbl.setStyleSheet(f"""
            font-size: 14px; font-family: Georgia, serif; color: {P.GRAY_400};
            padding: 0 0 10px 0;
        """)
        tab_row.addWidget(following_lbl)
        tab_row.addStretch()

        col_lay.addLayout(tab_row)
        col_lay.addSpacing(2)

        # Tab underline divider
        div = QFrame()
        div.setFrameShape(QFrame.Shape.HLine)
        div.setFixedHeight(1)
        div.setStyleSheet(f"background: {P.GRAY_200}; border: none;")
        col_lay.addWidget(div)
        col_lay.addSpacing(20)

        # Composer
        self._composer = ComposerBox()
        self._composer.post_submitted.connect(self._on_quick_post)
        col_lay.addWidget(self._composer)
        col_lay.addSpacing(20)

        # Feed container
        self._feed_lay = QVBoxLayout()
        self._feed_lay.setSpacing(0)
        col_lay.addLayout(self._feed_lay)
        col_lay.addStretch()

        cl.addWidget(col, alignment=Qt.AlignmentFlag.AlignHCenter)
        cl.addStretch()

        # ── RIGHT: always-visible Substack-style sidebar ───────────────────
        self._news_sidebar = NewsSidebar()
        root_h.addWidget(self._news_sidebar)

    def _populate(self):
        # Clear
        while self._feed_lay.count():
            item = self._feed_lay.takeAt(0)
            if item.widget(): item.widget().deleteLater()

        if not self._posts:
            empty = QLabel("Nothing here yet — write your first post!")
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty.setStyleSheet(
                f"color: {P.GRAY_400}; font-size: 16px; font-family: Georgia, serif; padding: 60px 0;"
            )
            self._feed_lay.addWidget(empty)
            return

        for post in reversed(self._posts):
            card = PostCard(post)
            card.like_toggled.connect(self._on_like)
            card.comment_added.connect(self._on_comment)
            card.delete_post.connect(self._on_delete)
            self._feed_lay.addWidget(card)

    def _on_quick_post(self, text: str, pix):
        try:
            if not text and pix is None:
                return

            img_path = ""
            if pix is not None and not pix.isNull():
                dest = os.path.join(MEDIA_DIR,
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png")
                ok = pix.save(dest, "PNG")
                if ok:
                    img_path = dest

            if not text and not img_path:
                return

            post = _new_post(title="", body=text, image_path=img_path)
            self._posts.append(post)
            _save_feed(self._posts)
            self._populate()
        except Exception as ex:
            print(f"[Writes] Error saving post: {ex}")

    def _on_pick_image(self):
        pass  # handled inside ComposeDialog now

    def add_post(self, title: str, body: str):
        """Called when user publishes from the editor."""
        post = _new_post(title, body)
        self._posts.append(post)
        _save_feed(self._posts)
        self._populate()

    def _on_like(self, pid: str, liked: bool):
        _save_feed(self._posts)

    def _on_comment(self, pid: str, text: str):
        _save_feed(self._posts)

    def _on_delete(self, pid: str):
        if not pid:
            # Empty string = soft refresh signal from an edit operation
            self._posts = _load_feed()
            self._populate()
            return
        self._posts = [p for p in self._posts if p["id"] != pid]
        _save_feed(self._posts)
        self._populate()

    def refresh(self):
        self._posts = _load_feed()
        self._populate()


# ==============================================================================
#  SETTINGS DIALOG
# ==============================================================================

class SettingsDialog(QDialog):
    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setFixedWidth(380)
        self.setStyleSheet(f"background: {P.WHITE};")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(36, 32, 36, 32)
        lay.setSpacing(16)

        t = QLabel("Settings")
        t.setStyleSheet(f"font-family: Georgia, serif; font-size: 20px; color: {P.BLACK};")
        lay.addWidget(t)

        note = QLabel("Writes runs fully offline. Your posts and drafts\nare stored locally on your computer.")
        note.setStyleSheet(f"font-size: 13px; color: {P.GRAY_500}; font-family: Georgia, serif; line-height: 1.5;")
        note.setWordWrap(True)
        lay.addWidget(note)

        data_lbl = QLabel(f"Data folder: {DATA_DIR}")
        data_lbl.setStyleSheet(f"font-size: 11px; color: {P.GRAY_400}; font-family: 'Courier New', monospace;")
        data_lbl.setWordWrap(True)
        lay.addWidget(data_lbl)

        close = QPushButton("Close")
        close.setFixedHeight(36)
        close.setCursor(Qt.CursorShape.PointingHandCursor)
        close.setStyleSheet(f"""
            QPushButton {{ background: {P.ORANGE}; border: none; border-radius: 8px;
                color: {P.WHITE}; font-family: Georgia, serif; font-size: 14px; }}
            QPushButton:hover {{ background: {P.ORANGE_HOVER}; }}
        """)
        close.clicked.connect(self.accept)
        lay.addWidget(close)


# ==============================================================================
#  MAIN WINDOW
# ==============================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.state = AppState()
        self.setWindowTitle("Writes")
        self.resize(1200, 780)
        self.setMinimumSize(800, 550)
        self._build_ui()
        self._build_menu()
        self._switch("home")

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Top bar
        self.topbar = TopBar(self.state)
        self.topbar.publish_clicked.connect(self._do_publish)
        self.topbar.dots_clicked.connect(self._show_dots_menu)
        self.topbar.avatar_clicked.connect(lambda: SettingsDialog(self.state, self).exec())
        self.topbar.sidebar_toggled.connect(self._toggle_sidebar)
        root.addWidget(self.topbar)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        self.sidebar = Sidebar(self.state)
        self.sidebar.navigate_to.connect(self._switch)
        self.sidebar.new_story_requested.connect(self._on_sidebar_new)
        self.sidebar.draft_open_requested.connect(self._on_open_draft)
        self.sidebar.draft_delete_requested.connect(self._on_delete_draft)
        self.sidebar.draft_rename_requested.connect(
            lambda fp, t: self.statusBar().showMessage(f'Renamed to "{t}"', 2500))
        self.sidebar.settings_requested.connect(lambda: SettingsDialog(self.state, self).exec())
        body.addWidget(self.sidebar)

        self.stack = QStackedWidget()
        self.stack.setStyleSheet(f"background: {P.BG};")
        body.addWidget(self.stack, 1)

        # Views
        self.home_view   = HomeView(self.state)
        self.editor_view = EditorView(self.state)

        self.stack.addWidget(self.home_view)   # index 0
        self.stack.addWidget(self.editor_view) # index 1
        self.stack.setCurrentIndex(0)

        self.home_view.navigate_to_editor.connect(lambda: self._switch("editor"))

        self.editor_view.word_count_changed.connect(self._on_word_count)
        self.editor_view.status_changed.connect(self.topbar.set_status)
        self.editor_view.draft_saved.connect(self.topbar.flash_saved)
        self.editor_view.draft_saved.connect(self.sidebar.refresh_recents)
        self.editor_view.published.connect(self._on_published)

        body_widget = QWidget()
        body_widget.setStyleSheet(f"background: {P.BG};")
        body_widget.setLayout(body)
        root.addWidget(body_widget, 1)

        bar = QStatusBar()
        bar.setStyleSheet(f"""
            QStatusBar {{
                background: {P.WHITE}; border-top: 1px solid {P.GRAY_200};
                font-size: 12px; font-family: Georgia, serif; color: {P.GRAY_400};
            }}
        """)
        self.setStatusBar(bar)
        self._words_lbl = QLabel("")
        self._words_lbl.setStyleSheet(
            f"color: {P.GRAY_400}; font-size: 12px; font-family: Georgia, serif;")
        bar.addPermanentWidget(self._words_lbl)

        # Sidebar animation
        self._sidebar_anim_max = QPropertyAnimation(self.sidebar, b"maximumWidth")
        self._sidebar_anim_max.setDuration(200)
        self._sidebar_anim_max.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self._sidebar_anim_min = QPropertyAnimation(self.sidebar, b"minimumWidth")
        self._sidebar_anim_min.setDuration(200)
        self._sidebar_anim_min.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self._sidebar_anim = QParallelAnimationGroup()
        self._sidebar_anim.addAnimation(self._sidebar_anim_max)
        self._sidebar_anim.addAnimation(self._sidebar_anim_min)
        self._sidebar_open = True

    def _build_menu(self):
        mb = self.menuBar()
        mb.setStyleSheet(f"""
            QMenuBar {{ background: {P.WHITE}; border-bottom: none;
                font-size: 13px; font-family: Georgia, serif; }}
            QMenuBar::item:selected {{ background: {P.GRAY_100}; }}
            QMenu {{ background: {P.WHITE}; border: 1px solid {P.GRAY_200};
                font-family: Georgia, serif; font-size: 13px; }}
            QMenu::item {{ padding: 8px 24px; color: {P.BLACK}; }}
            QMenu::item:selected {{ background: {P.GRAY_100}; }}
            QMenu::separator {{ height: 1px; background: {P.GRAY_200}; margin: 4px 0; }}
        """)

        file_m = mb.addMenu("File")
        for label, sc, fn in [
            ("Home",      "Ctrl+Shift+H", lambda: self._switch("home")),
            ("New story", "Ctrl+N",       self.editor_view.new_document),
            ("Save",      "Ctrl+S",       self.editor_view.save_document),
            ("Quit",      "Ctrl+Q",       self.close),
        ]:
            a = QAction(label, self)
            if sc: a.setShortcut(QKeySequence(sc))
            a.triggered.connect(fn)
            if label == "Quit": file_m.addSeparator()
            file_m.addAction(a)

        view_m = mb.addMenu("View")
        for label, sc, fn in [
            ("Home",  "Ctrl+Shift+H", lambda: self._switch("home")),
            ("Write", "Ctrl+Shift+W", lambda: self._switch("editor")),
        ]:
            a = QAction(label, self, shortcut=QKeySequence(sc))
            a.triggered.connect(fn); view_m.addAction(a)

    def _do_publish(self):
        self.editor_view.publish()

    def _on_published(self, title: str, body: str):
        self.home_view.add_post(title, body)
        self._switch("home")

    def _switch(self, view: str):
        idx = {"home": 0, "editor": 1}.get(view, 0)
        self.stack.setCurrentIndex(idx)
        self.sidebar.set_active_view(view)
        # Show/hide draft label
        self.topbar.show_draft_label(view == "editor")
        if view == "home":
            self.home_view.refresh()

    def _toggle_sidebar(self):
        w   = self.sidebar.WIDTH
        cur = self.sidebar.width()
        start, end = (cur, 0) if self._sidebar_open else (cur, w)
        for anim in (self._sidebar_anim_max, self._sidebar_anim_min):
            anim.setStartValue(start); anim.setEndValue(end)
        self._sidebar_anim.start()
        self._sidebar_open = not self._sidebar_open

    def _on_sidebar_new(self):
        self.editor_view.new_document(); self._switch("editor")

    def _on_open_draft(self, filepath):
        self.editor_view.load_draft(filepath); self._switch("editor")

    def _on_delete_draft(self, filepath):
        try:
            if os.path.exists(filepath): os.remove(filepath)
            if self.editor_view._current_file == filepath:
                self.editor_view.new_document()
            self.sidebar.refresh_recents()
            self.statusBar().showMessage("Draft deleted.", 2500)
        except Exception as e:
            self.statusBar().showMessage(f"Could not delete draft: {e}", 3000)

    def _show_dots_menu(self):
        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{ background: {P.WHITE}; border: 1px solid {P.GRAY_200};
                font-family: Georgia, serif; font-size: 13px; }}
            QMenu::item {{ padding: 9px 24px; color: {P.BLACK}; }}
            QMenu::item:selected {{ background: {P.GRAY_100}; }}
            QMenu::separator {{ height: 1px; background: {P.GRAY_200}; margin: 4px 0; }}
        """)
        menu.addAction("New story",   self.editor_view.new_document)
        menu.addAction("Save",        self.editor_view.save_document)
        menu.addSeparator()
        menu.addAction("Settings...", lambda: SettingsDialog(self.state, self).exec())
        btn = self.topbar._dots
        menu.exec(btn.mapToGlobal(QPoint(0, btn.height() + 2)))

    def _on_word_count(self, count: int):
        self._words_lbl.setText(f"{count:,} words" if count else "")


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Writes")
    app.setOrganizationName("Writes")
    app.setStyleSheet(f"""
        * {{ font-family: 'Georgia', 'Times New Roman', serif; }}
        QMainWindow {{ background: {P.BG}; }}
        QWidget {{ background: transparent; }}
        QToolTip {{
            background: {P.BLACK}; color: {P.WHITE};
            border: none; padding: 5px 10px; font-size: 12px;
        }}
    """)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()