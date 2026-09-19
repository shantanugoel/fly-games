import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api, b64bytes, loadPointCloud, type PointCloud } from './api'
import type { BrainCard, Snapshot } from './types'

/** One row of the decision ribbon: what the fly decided, and what it cost. */
export interface TraceRow {
  decisions: number
  frames: number
  action: string | null
  label: string | null
  coarse: string | null
  confidence: number
  model: string | null
  latencyMs: number
  command: Record<string, number>
  input: Record<string, number>
  buttons: string[]
}

export interface LabConnection {
  snapshot: Snapshot | null
  brain: BrainCard | null
  cloud: PointCloud | null
  trace: TraceRow[]
  connected: boolean
  error: string | null
  /** true while the fly is computing, including the frame we are rendering */
  thinking: boolean
  send: (action: string, options?: Record<string, unknown>) => void
}

function commandRates(snapshot: Snapshot): Record<string, number> {
  const fly = snapshot.fly
  if (!fly) return {}
  return Object.fromEntries(Object.entries(fly.command).map(([k, v]) => [k, v.rate]))
}

export function useLab(): LabConnection {
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null)
  const [brain, setBrain] = useState<BrainCard | null>(null)
  const [cloud, setCloud] = useState<PointCloud | null>(null)
  const [connected, setConnected] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [trace, setTrace] = useState<TraceRow[]>([])
  const socket = useRef<WebSocket | null>(null)
  const lastDecision = useRef(-1)

  useEffect(() => {
    let cancelled = false
    api
      .brain()
      .then(async (card) => {
        if (cancelled) return
        setBrain(card)
        const points = await loadPointCloud(card)
        if (!cancelled) setCloud(points)
      })
      .catch((exc: Error) => setError(`connectome failed to load: ${exc.message}`))
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    let alive = true
    let retry = 0
    let timer: number | undefined

    const connect = () => {
      const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
      const ws = new WebSocket(`${proto}://${window.location.host}/ws`)
      socket.current = ws
      ws.onopen = () => {
        retry = 0
        setConnected(true)
        setError(null)
      }
      ws.onmessage = (event) => {
        const payload = JSON.parse(event.data as string)
        if (payload.error) {
          setError(payload.error as string)
          return
        }
        const next = payload as Snapshot
        setSnapshot(next)
        if (next.error) setError(next.error as string)
        if (next.progress.decisions !== lastDecision.current && next.thought.action) {
          lastDecision.current = next.progress.decisions
          setTrace((rows) => {
            const row: TraceRow = {
              decisions: next.progress.decisions,
              frames: next.progress.frames,
              action: next.thought.action,
              label: next.thought.label,
              coarse: next.thought.coarse,
              confidence: next.thought.confidence ?? 0,
              model: next.thought.model,
              latencyMs: next.thought.latency_ms ?? 0,
              command: commandRates(next),
              input: next.fly?.input ?? {},
              buttons: next.thought.buttons ?? [],
            }
            const merged = [...rows, row]
            return merged.length > 90 ? merged.slice(merged.length - 90) : merged
          })
        }
      }
      ws.onclose = () => {
        if (!alive) return
        setConnected(false)
        retry = Math.min(retry + 1, 6)
        timer = window.setTimeout(connect, 400 * 2 ** retry)
      }
      ws.onerror = () => ws.close()
    }

    connect()
    return () => {
      alive = false
      if (timer) window.clearTimeout(timer)
      socket.current?.close()
    }
  }, [])

  const send = useCallback((action: string, options: Record<string, unknown> = {}) => {
    const ws = socket.current
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action, ...options }))
      setError(null)
      return
    }
    void api.command(action, options).then(setSnapshot).catch((exc: Error) => setError(exc.message))
  }, [])

  const thinking = useMemo(() => !!snapshot?.busy, [snapshot])

  return { snapshot, brain, cloud, trace, connected, error, thinking, send }
}

/** Decode a base64 uint8 heat map back to a Uint8Array of w*h. */
export function heat(data: string | undefined, size: [number, number]): Uint8Array {
  const bytes = b64bytes(data)
  const want = size[0] * size[1]
  if (bytes.length === want) return bytes
  return new Uint8Array(want)
}