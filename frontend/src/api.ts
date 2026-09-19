import type { BrainCard, ProbeMenu, ProbeResult, ReadoutStatus, Replay, Snapshot, Stimulus } from './types'

// ---------------------------------------------------------------- base64 / bytes

export function b64bytes(value: string | undefined | null): Uint8Array {
  if (!value) return new Uint8Array(0)
  const raw = atob(value)
  const out = new Uint8Array(raw.length)
  for (let i = 0; i < raw.length; i += 1) out[i] = raw.charCodeAt(i)
  return out
}

// --------------------------------------------------------------------- fetchers

async function json<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init)
  if (!response.ok) {
    const detail = await response.text().catch(() => '')
    throw new Error(`${response.status} ${response.statusText}${detail ? ` - ${detail.slice(0, 200)}` : ''}`)
  }
  return response.json() as Promise<T>
}

export const api = {
  brain: () => fetch('/api/brain').then((r) => (r.ok ? r.json() : Promise.reject(new Error(`brain ${r.status}`)))),
  games: () => json<{ games: { id: string; title: string; short_name: string }[] }>('/api/games'),
  state: () => json<Snapshot>('/api/state?viz=true'),
  command: (action: string, options: Record<string, unknown> = {}) =>
    json<Snapshot>('/api/command', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ action, ...options }),
    }),
  readouts: () => json<Record<string, ReadoutStatus>>('/api/readouts'),
  episodes: () => json<{ episodes: Snapshot['episodes'] }>('/api/episodes'),
  replay: (id: string) => json<Replay>(`/api/episodes/${id}/replay`),
  probeMenu: () => json<ProbeMenu>('/api/probe'),
  probeSearch: (q: string) => json<{ types: { name: string; neurons: number; L: number; R: number }[] }>(`/api/brain/types?q=${encodeURIComponent(q)}`),
  probeRun: (events: Stimulus[], games: string[], steps?: number) =>
    json<ProbeResult>('/api/probe', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ events, games, steps, field: true }),
    }),
}

// ------------------------------------------------------------- the point cloud

export interface PointCloud {
  count: number
  /** normalised 0..1 x,y,z with the body's real proportions */
  xyz: Float32Array
  /** index into the superclass palette */
  kind: Uint8Array
  /** 0..11 for the twelve descending command groups, -1 otherwise */
  command: Int16Array
  /** index into the probe array, -1 if this neuron is not sampled */
  probe: Int32Array
  /** cloud index of each probe id, so spike counts can light points */
  probePoint: Uint32Array
  nProbes: number
  /** 2-D projected positions, centred on the body and aspect-preserved */
  flat: Float32Array
  /** width and height of `flat`'s bounding box, in the same units */
  flatSpan: [number, number]
  superclasses: string[]
}

const MAGIC = 'FLYPT1'

/** Decode /api/brain/points.bin. One fetch, ~1.7 MB, cached by the browser forever. */
export async function loadPointCloud(card: BrainCard): Promise<PointCloud> {
  const buffer = await fetch(card.points_url).then((r) =>
    r.ok ? r.arrayBuffer() : Promise.reject(new Error(`points ${r.status}`)),
  )
  const view = new DataView(buffer)
  const magic = String.fromCharCode(...new Uint8Array(buffer, 0, 6))
  if (magic !== MAGIC) throw new Error(`unexpected point cloud header ${JSON.stringify(magic)}`)
  const version = view.getUint16(6, true)
  if (version !== 1) throw new Error(`unsupported point cloud version ${version}`)
  const pointBytes = view.getUint16(8, true)
  const count = view.getUint32(10, true)
  const min = [view.getFloat32(14, true), view.getFloat32(18, true), view.getFloat32(22, true)]
  const span = [view.getFloat32(26, true), view.getFloat32(30, true), view.getFloat32(34, true)]
  const longest = view.getFloat32(38, true) || 1
  const nProbes = view.getUint32(42, true)
  void min
  void span

  const xyz = new Float32Array(count * 3)
  const kind = new Uint8Array(count)
  const command = new Int16Array(count)
  const probe = new Int32Array(count)
  const flat = new Float32Array(count * 2)
  const base = 64
  for (let i = 0; i < count; i += 1) {
    const at = base + i * pointBytes
    const x = view.getUint16(at, true) / 65535
    const y = view.getUint16(at + 2, true) / 65535
    const z = view.getUint16(at + 4, true) / 65535
    xyz[i * 3] = x
    xyz[i * 3 + 1] = y
    xyz[i * 3 + 2] = z
    // The flat view looks down the fly's back: x along the body, y across it.
    flat[i * 2] = x
    flat[i * 2 + 1] = y
    const c = view.getUint16(at + 6, true)
    command[i] = c === 0xffff ? -1 : c
    kind[i] = view.getUint16(at + 8, true)
    const p = view.getUint16(at + 10, true)
    probe[i] = p === 0xffff ? -1 : p
  }
  const probePoint = new Uint32Array(buffer, base + count * pointBytes, nProbes)

  // The u16 positions are normalised by the longest of all three axes, so the
  // two displayed ones usually only fill part of 0..1. Re-fit the flat view to
  // its own bounding box (one shared scale, so the body keeps its proportions)
  // otherwise the flat brain floats in the top-left of a mostly empty panel.
  let loX = Infinity
  let loY = Infinity
  let hiX = -Infinity
  let hiY = -Infinity
  for (let i = 0; i < count; i += 1) {
    const x = flat[i * 2]
    const y = flat[i * 2 + 1]
    if (x < loX) loX = x
    if (y < loY) loY = y
    if (x > hiX) hiX = x
    if (y > hiY) hiY = y
  }
  // Keep the body's proportions: one shared scale for both axes, centred on
  // zero. The canvas picks the scale that fits, so a wide panel never squashes
  // the fly and a narrow one never clips it.
  const fit = Math.max(hiX - loX, hiY - loY) || 1
  const midX = (hiX + loX) / 2
  const midY = (hiY + loY) / 2
  for (let i = 0; i < count; i += 1) {
    flat[i * 2] = (flat[i * 2] - midX) / fit
    flat[i * 2 + 1] = (flat[i * 2 + 1] - midY) / fit
  }
  const flatSpan: [number, number] = [(hiX - loX) / fit, (hiY - loY) / fit]
  void longest
  return {
    count,
    xyz,
    kind,
    command,
    probe,
    probePoint,
    nProbes,
    flat,
    flatSpan,
    superclasses: card.superclasses.map((s) => s.name),
  }
}

/** Pack spike counts into a square R8 texture the shader can sample. */
export function textureSize(n: number): [number, number] {
  const side = Math.ceil(Math.sqrt(n))
  const pow2 = 2 ** Math.ceil(Math.log2(Math.max(side, 8)))
  return [pow2, Math.max(1, Math.ceil(n / pow2))]
}