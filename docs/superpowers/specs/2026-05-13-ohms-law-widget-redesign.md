# Ohm's Law Widget Redesign

**Date:** 2026-05-13  
**Status:** Approved

## Overview

Redesign the inline chat Ohm's Law widget (`_OhmsLawWidget` and `_OhmsLawVisualization`) to be simpler, more aesthetic, and narrower. The widget embeds in the chat stream when Maria explains Ohm's Law concepts.

## What Changes

### Removed
- `_OhmsLawVisualization` — the animated circuit diagram (battery + wire + resistor) is dropped entirely
- "Open large" / pop-out button from the card header
- Intensity badge (LOW / MED / HIGH CURRENT)
- Three-column layout with large 28px Georgian numbers and individual column separators
- Power label with emoji prefix

### New Design

**Card shell** — white background (`#ffffff`), 1px border (`#e8e6e0`), 14px border-radius, soft drop shadow. Maximum width follows the existing `_chat_column_width()` cap.

**Header** — single row, no button. Left-aligned title "Ohm's Law" (12px, 700 weight, `#1a1a1a`) with italic subtitle "I = V / R" (10px, `#bbb`) beside it. Bottom border `#eeece6`.

**Body** — fixed height ~138px, split horizontally into two panels divided by a 1px `#eeece6` line:

**Left panel (controls, flex: 1)**
- Formula `V = IR` in italic Georgia serif, 17px, muted grey (`#555`)
- Voltage slider row: label `Vs` (9px, `#bbb`), slim 3px groove (`#eeece6`), filled track (`#888`), circular handle (10px, white with `#666` border), value right-aligned (9px, `#555`)
- Resistance slider row: same style, label `R`
- Derived result line: `I = V/R = 9.0 / 3.0 = 3.00 A` — muted with the final value bolded dark (`#222`, 700 weight)
- No power label in the left panel

**Right panel (oscilloscope, fixed 200px wide)**
- Background `#fafaf9`
- Qt `QPainter`-drawn scrolling sine wave animation (replaces `_OhmsLawVisualization`)
- Subtle grid: horizontal and vertical dotted lines at `rgba(0,0,0,0.04)`
- Centre baseline at `rgba(0,0,0,0.06)`
- Wave: green stroke `rgba(60,140,100,0.75)`, 1.6px width, soft glow shadow
- Wave amplitude scales linearly with voltage (V / 24 × panel_height × 0.52)
- Wave scroll speed scales linearly with current (base 1.6 px/frame + I × 0.15)
- Power label `P = X.X W` bottom-right, 9px, `#ccc`
- Animation driven by existing `QTimer` pattern (33ms interval)

## Slider Ranges & Mapping

| Slider | Range (int) | Displayed value | Formula |
|--------|-------------|-----------------|---------|
| Voltage | 1–240 | `value / 10.0` V | — |
| Resistance | 1–200 | `value / 10.0` Ω | — |
| Current | — | derived | I = V / R |

Current is always derived (read-only). The left panel shows it as a formatted equation, not a slider.

## Classes Affected

| Class | Change |
|-------|--------|
| `_OhmsLawVisualization` | **Replace** with `_OhmsLawOscilloscope` — sine wave painter |
| `_OhmsLawWidget._build_ui()` | Full rewrite: two-panel layout, remove three-col layout |
| `_OhmsLawWidget._make_col()` | **Delete** — no longer used |
| `_OhmsLawWidget._update_display()` | Update: remove badge logic, update oscilloscope, update derived-I label |
| `_insert_ohms_law_chat_widget()` | Remove "Open large" button and its connect; simplify header |

## Behaviour Preserved

- `state_changed` signal still emits `{"voltage": v, "resistance": r, "current": i}` on every drag
- `widget_data()` and `apply_widget_data()` signatures unchanged — no impact on persistence or reload
- Animated value interpolation (`_animate_step`) kept for smooth voltage display
- The standalone "Open large" panel (`open_ohms_law_widget`) is left untouched — only the inline chat card changes
