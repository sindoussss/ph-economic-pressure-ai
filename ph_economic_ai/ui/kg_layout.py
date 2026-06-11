"""Pure render model for the knowledge graph: a seeded force-directed layout +
colour-by-kind. No Qt — returns plain dicts the canvas draws."""
import math
import random

_KIND_COLORS = {
    'source': '#3B6FD4', 'evidence': '#9AA1AC', 'entity': '#9AA1AC',
    'agent': '#E5484D', 'judge': '#3B6FD4', 'master': '#111111',
    'data_input': '#6B7280', 'claim': '#C0C4CC',
}
_SECTOR_COLORS = {'food': '#15A150', 'elec': '#E8920C', 'electricity': '#E8920C'}


def node_color(node) -> str:
    if getattr(node, 'cluster', None) in _SECTOR_COLORS:
        return _SECTOR_COLORS[node.cluster]
    return _KIND_COLORS.get(node.kind, '#9AA1AC')


def _layout(nodes, edges, width, height, iterations, seed):
    rng = random.Random(seed)
    pos = {n.id: [rng.uniform(0, width), rng.uniform(0, height)] for n in nodes}
    if not nodes:
        return pos
    k = math.sqrt((width * height) / max(len(nodes), 1)) * 0.8
    adj = [(e.src, e.dst) for e in edges if e.src in pos and e.dst in pos]
    temp = width / 10.0
    for _ in range(iterations):
        disp = {nid: [0.0, 0.0] for nid in pos}
        ids = list(pos)
        for i, a in enumerate(ids):                       # repulsion
            for b in ids[i + 1:]:
                dx, dy = pos[a][0] - pos[b][0], pos[a][1] - pos[b][1]
                d = math.hypot(dx, dy) or 0.01
                f = (k * k) / d
                ux, uy = dx / d, dy / d
                disp[a][0] += ux * f; disp[a][1] += uy * f
                disp[b][0] -= ux * f; disp[b][1] -= uy * f
        for s, t in adj:                                  # attraction
            dx, dy = pos[s][0] - pos[t][0], pos[s][1] - pos[t][1]
            d = math.hypot(dx, dy) or 0.01
            f = (d * d) / k
            ux, uy = dx / d, dy / d
            disp[s][0] -= ux * f; disp[s][1] -= uy * f
            disp[t][0] += ux * f; disp[t][1] += uy * f
        for nid in pos:                                   # apply, cooled + bounded
            dx, dy = disp[nid]
            d = math.hypot(dx, dy) or 0.01
            pos[nid][0] += (dx / d) * min(d, temp)
            pos[nid][1] += (dy / d) * min(d, temp)
            pos[nid][0] = min(width, max(0.0, pos[nid][0]))
            pos[nid][1] = min(height, max(0.0, pos[nid][1]))
        temp *= 0.95
    return pos


def render_model(nodes, edges, width=1000, height=700, iterations=80, seed=7) -> dict:
    pos = _layout(nodes, edges, width, height, iterations, seed)
    degree = {n.id: 0 for n in nodes}
    for e in edges:
        if e.src in degree:
            degree[e.src] += 1
        if e.dst in degree:
            degree[e.dst] += 1
    rnodes = []
    for n in nodes:
        x, y = pos[n.id]
        r = 2.0 + min(degree.get(n.id, 0), 12) * 0.6
        if n.kind in ('master', 'judge'):
            r = max(r, 5.0)
        rnodes.append({'id': n.id, 'x': x, 'y': y, 'r': r, 'color': node_color(n),
                       'kind': n.kind, 'label': n.label})
    redges = [{'src': e.src, 'dst': e.dst, 'kind': e.kind} for e in edges]
    return {'nodes': rnodes, 'edges': redges, 'pos': pos}
