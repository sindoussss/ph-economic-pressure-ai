def generate(oil_delta: float, usd_delta: float, demand_norm: float,
             pressure_index: float, current_price: float, predicted_price: float) -> dict:

    # ── Drivers ──────────────────────────────────────────────────────────────
    if oil_delta > 0.5:
        oil_label, oil_color = '↑ High', '#E07A4A'
    elif oil_delta > 0.0:
        oil_label, oil_color = '↑ Rising', '#E07A4A'
    else:
        oil_label, oil_color = '→ Neutral', '#888888'

    if usd_delta > 0.3:
        usd_label, usd_color = '↑ Rising', '#E07A4A'
    else:
        usd_label, usd_color = '→ Stable', '#888888'

    if demand_norm > 0.7:
        dem_label, dem_color = '↑ High', '#E07A4A'
    else:
        dem_label, dem_color = '→ Neutral', '#888888'

    drivers = [
        {
            'icon': '🛢', 'name': 'Crude Oil',
            'value': f'Δ {oil_delta:+.2f}σ · Weight 50%',
            'status': oil_label, 'color': oil_color,
        },
        {
            'icon': '💱', 'name': 'USD / PHP',
            'value': f'Δ {usd_delta:+.2f}σ · Weight 30%',
            'status': usd_label, 'color': usd_color,
        },
        {
            'icon': '📊', 'name': 'Demand Index',
            'value': f'{demand_norm * 100:.0f}/100 · Weight 20%',
            'status': dem_label, 'color': dem_color,
        },
    ]

    # ── Risk badge ────────────────────────────────────────────────────────────
    if pressure_index > 60:
        risk_badge = '⚠ High Pressure — Price rise likely'
        risk_color = '#E07A4A'
    elif pressure_index > 30:
        risk_badge = '⚡ Rising Pressure — Monitor closely'
        risk_color = '#E0A84A'
    else:
        risk_badge = '✓ Stable — No immediate risk'
        risk_color = '#4A90E2'

    # ── Summary ───────────────────────────────────────────────────────────────
    parts = []
    if oil_delta > 0.3:
        parts.append('Oil prices are pushing import costs higher.')
    if usd_delta > 0.3:
        parts.append('A stronger dollar makes fuel imports more expensive.')
    if demand_norm > 0.7:
        parts.append('High demand is adding to price pressure.')
    if not parts:
        parts.append('Economic indicators are broadly stable.')
    summary = ' '.join(parts)

    # ── Advisory ─────────────────────────────────────────────────────────────
    if pressure_index > 60:
        advisory, advisory_icon = 'Refuel within 48 hours', '⛽'
    elif pressure_index > 30:
        advisory, advisory_icon = 'Monitor prices this week', '👁'
    else:
        advisory, advisory_icon = 'No action needed — prices stable', '✓'

    price_change = predicted_price - current_price
    sign = '+' if price_change >= 0 else ''
    direction = 'increase' if price_change >= 0 else 'decrease'

    return {
        'drivers': drivers,
        'risk_badge': risk_badge,
        'risk_color': risk_color,
        'summary': summary,
        'advisory': advisory,
        'advisory_icon': advisory_icon,
        'expected_increase': f'{sign}₱{abs(price_change):.2f} / liter',
        'price_direction': direction,
    }
