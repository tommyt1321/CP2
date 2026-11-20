"""FOV pilot: RGB-only mask, and CP2's boundary-distance SLIC restricted to that mask.

This is a separate implementation for an experiment; it does not monkey-patch CP2.
The FOV is an image-footprint estimate, never a lesion or diagnosis annotation.
"""
from __future__ import annotations
import heapq
import math
import numpy as np
from scipy import ndimage as ndi

CROSS = ndi.generate_binary_structure(2, 1)


class MaskFailure(ValueError):
    pass


def fundus_fov(rgb):
    a = np.asarray(rgb)
    if a.shape != (384, 384, 3) or a.dtype != np.uint8:
        raise ValueError("Expected an unmodified 384 x 384 RGB uint8 image.")
    maximum = a.max(axis=2)
    frame = np.concatenate((maximum[0], maximum[-1], maximum[1:-1, 0], maximum[1:-1, -1]))
    # Use a low threshold for *exterior* detection; do not punch holes in dark tissue.
    floor = int(np.percentile(frame, 10))
    threshold = int(np.clip(floor + 6, 7, 28))
    seed = maximum > threshold
    padded = np.pad(seed, 3, mode="edge")
    closed = ndi.binary_closing(padded, structure=CROSS, iterations=2)[3:-3, 3:-3]
    cc, n = ndi.label(closed, structure=CROSS)
    if not n:
        raise MaskFailure("no detectable retinal footprint")
    sizes = np.bincount(cc.ravel())[1:]
    main = cc == (1 + int(sizes.argmax()))
    mask = ndi.binary_fill_holes(main, structure=CROSS)
    fraction = float(mask.mean())
    if mask.sum() < 1024 or float(maximum[mask].std()) < 2:
        raise MaskFailure("too little non-uniform tissue to estimate a footprint")
    y, x = np.nonzero(mask)
    second = 0 if len(sizes) < 2 else int(np.partition(sizes, -2)[-2])
    flags = []
    if fraction < .25 or fraction > .975: flags.append("extreme_coverage")
    if not mask[192, 192] or max(abs(y.mean()-191.5), abs(x.mean()-191.5)) > 75:
        flags.append("off_centre")
    if second > .04 * mask.sum(): flags.append("another_large_component")
    if floor > 22: flags.append("bright_frame")
    return mask, dict(fov_pixels=int(mask.sum()), fov_fraction=fraction,
        threshold=threshold, frame_p10=floor, flags=";".join(flags), second_component_pixels=second)


def connected_labels(raw, minimum):
    """Background = -1. Split disconnected foreground parts; merge only across FOV contacts."""
    import superpixels_v1 as sp
    mask = raw >= 0
    comp, _ = sp.components4(raw)
    ids, inv = np.unique(comp[mask], return_inverse=True)
    c = np.full(raw.shape, -1, np.int32)
    c[mask] = inv.ravel()
    n = len(ids)
    if n == 0: raise MaskFailure("empty masked partition")
    if minimum <= 1: return c
    sizes = np.bincount(c[mask], minlength=n)
    parent = np.arange(n)
    adj = [{} for _ in range(n)]
    pairs, lengths = sp.shared_edges(c)
    for (a, b), length in zip(pairs, lengths):
        if a < 0: continue
        a, b, length = int(a), int(b), int(length)
        adj[a][b] = length
        adj[b][a] = length
    small = [(int(sizes[i]), i) for i in range(n) if sizes[i] < minimum]
    heapq.heapify(small)
    while small:
        area, a = heapq.heappop(small)
        if a != parent[a] or sizes[a] != area or area >= minimum or not adj[a]: continue
        b = min(adj[a], key=lambda k: (-adj[a][k], k))
        parent[a] = b
        sizes[b] += sizes[a]
        adj[b].pop(a)
        for k, length in adj[a].items():
            if k == b: continue
            adj[k].pop(a)
            total = adj[b].get(k, 0) + length
            adj[b][k] = adj[k][b] = total
        adj[a].clear()
        if sizes[b] < minimum: heapq.heappush(small, (int(sizes[b]), b))
    for i in range(n):
        k = i
        while parent[k] != k: k = parent[k]
        parent[i] = k
    roots, first, ix = np.unique(parent[c[mask]], return_index=True, return_inverse=True)
    mapping = np.empty(len(roots), np.int32)
    mapping[np.argsort(first)] = np.arange(len(roots))
    c[mask] = mapping[ix.ravel()]
    return c


def _seeds(lab, mask, k):
    # Equal-area strips and bins avoid spending the 256 seeds on a rectangular black frame.
    side = math.isqrt(k)
    if side*side != k or mask.sum() < 4*k: raise MaskFailure("too little support for seed budget")
    y, x = np.nonzero(mask)
    out = []
    for band in np.array_split(np.arange(y.size), side):
        by_x = band[np.argsort(x[band], kind="stable")]
        for group in np.array_split(by_x, side):
            sy, sx = y[group], x[group]
            d = (sy - sy.mean())**2 + (sx - sx.mean())**2
            j = int(d.argmin())
            yy, xx = sy[j], sx[j]
            out.append([*lab[yy, xx], yy, xx])
    return np.asarray(out, np.float32)


def masked_slic(rgb_tensor, mask, n_segments=256):
    """Mask-constrained assignment/updates; same Lab/xy + gamma=2 DDA path family as A3.

    Boundary-only nearest-interior extension prevents an artificial hard black contour
    in the filter. DenseNet still receives the *original*, unchanged RGB input.
    """
    import boundary_map_v1 as bm
    import superpixels_v1 as sp
    import torch
    m = np.asarray(mask, bool)
    h, w = m.shape
    rgb = rgb_tensor.detach().cpu().contiguous()
    if tuple(rgb.shape) != (3, h, w) or not m.any(): raise ValueError("unaligned or empty mask")
    nearest = ndi.distance_transform_edt(~m, return_distances=False, return_indices=True)
    yy_ext, xx_ext = [torch.from_numpy(v.astype(np.int64)) for v in nearest]
    details = bm.boundary_map(rgb[:, yy_ext, xx_ext], return_details=True)
    lab = details["lab"].permute(1, 2, 0).contiguous().numpy()
    strength = details["raw_strength"][0].numpy()
    boundary = np.zeros((h, w), np.float32)
    peak = float(strength[m].max())
    if peak > bm.FLAT_EPS: boundary[m] = strength[m] / peak
    centres = _seeds(lab, m, n_segments)
    s = math.sqrt(int(m.sum()) / n_segments)
    rad = max(2, math.ceil(s))
    spatial = np.float32((10.0 / s)**2)
    dy, dx = np.mgrid[-rad:rad+1, -rad:rad+1]
    oy, ox = sp.raster_offsets(dy.ravel(), dx.ravel())
    lut = ((oy + rad) * (2*rad+1) + ox + rad).astype(np.int32)
    dy, dx = dy.ravel(), dx.ravel()
    yy, xx = np.nonzero(m)
    fields = (lab[..., 0][m], lab[..., 1][m], lab[..., 2][m], yy, xx)
    holes = 0
    for _ in range(10):
        labels = np.full((h, w), -1, np.int32)
        best = np.full((h, w), np.inf, np.float32)
        ay, ax = np.floor(centres[:, 3:].T + .5).astype(int)
        # A centroid of a concave region may round outside the FOV.
        ay, ax = nearest[0, ay, ax], nearest[1, ay, ax]
        for i, (cy, cx) in enumerate(zip(ay, ax)):
            py, px = cy+dy, cx+dx
            wy, wx = py.clip(0, h-1), px.clip(0, w-1)
            keep = (py==wy) & (px==wx) & m[wy, wx]
            if not keep.any(): continue
            y, x = wy[keep], wx[keep]
            path = boundary[wy, wx][lut[:, keep]].max(axis=0)
            dlab = lab[y, x] - centres[i, :3]
            dxy = (y.astype(np.float32)-centres[i, 3])**2 + (x.astype(np.float32)-centres[i, 4])**2
            cost = ((dlab*dlab).sum(axis=1) + spatial*dxy) * (1+2*path)
            take = cost < best[y, x]   # ascending seed order breaks ties
            labels[y[take], x[take]] = i
            best[y[take], x[take]] = cost[take]
        hy, hx = np.nonzero(m & (labels < 0))
        holes += len(hy)
        for j in range(0, len(hy), 8):
            y, x = hy[j:j+8], hx[j:j+8]
            sy, sx = sp.raster_offsets(y[None,:]-ay[:,None], x[None,:]-ax[:,None])
            path = boundary[ay[None,:,None]+sy, ax[None,:,None]+sx].max(axis=0)
            dc = ((lab[y,x][None,:,:] - centres[:,None,:3])**2).sum(axis=2)
            ds = (y[None,:]-centres[:,3,None])**2 + (x[None,:]-centres[:,4,None])**2
            labels[y, x] = ((dc + spatial*ds)*(1+2*path)).argmin(axis=0)
        ids = labels[m]
        counts = np.bincount(ids, minlength=n_segments)
        live = counts > 0
        for ch, field in enumerate(fields):
            centres[live, ch] = np.bincount(ids, weights=field, minlength=n_segments)[live]/counts[live]
    result = connected_labels(labels, max(1, int(.25*m.sum()/n_segments)))
    if not np.array_equal(result >= 0, m): raise RuntimeError("background entered a region")
    return result, holes
