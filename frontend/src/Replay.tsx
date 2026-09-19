import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api, b64bytes } from './api'
import { CommandGauges, Decision, Panel, Ribbon, Senses } from './Panels'
import type { BrainCard, EpisodeSummary, Replay as ReplayData, Snapshot } from './types'

interface Props {
  episodes: EpisodeSummary[]
  card: BrainCard | null
  snapshot: Snapshot | null
}

const SPEEDS = [0.25, 0.5, 1, 2, 4]

/** Draw a base64 uint8 heat map into a canvas with a simple ramp. */
function Heat({ data, size }: { data?: string; size?: [number, number] }) {
  const ref = useRef<HTMLCanvasElement | null>(null)
  useEffect(() => {
    const canvas = ref.current
    if (!canvas || !size) return
    const [w, h] = size
    canvas.width = w
    canvas.height = h
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const bytes = b64bytes(data)
    const image = ctx.createImageData(w, h)
    for (let i = 0; i < w * h; i += 1) {
      const v = bytes[i] ?? 0
      const t = v / 255
      image.data[i * 4] = Math.round(30 + 225 * t)
      image.data[i * 4 + 1] = Math.round(20 + 150 * t * t)
      image.data[i * 4 + 2] = Math.round(60 + 120 * (1 - t))
      image.data[i * 4 + 3] = 255
    }
    ctx.putImageData(image, 0, 0)
  }, [data, size])
  if (!size) return <p className="empty">no heat map recorded</p>
  return <canvas ref={ref} className="heat" />
}

export function Replay({ episodes, card, snapshot }: Props) {
  const [selected, setSelected] = useState<string | null>(null)
  const [data, setData] = useState<ReplayData | null>(null)
  const [index, setIndex] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [gapless, setGapless] = useState(true)
  const [speed, setSpeed] = useState(1)
  const [error, setError] = useState<string | null>(null)
  const clock = useRef(0)
  const frame = useRef<number | undefined>(undefined)
  const last = useRef(0)

  useEffect(() => {
    if (!episodes.length) return
    if (!selected || !episodes.some((e) => e.id === selected)) setSelected(episodes[episodes.length - 1].id)
  }, [episodes, selected])

  useEffect(() => {
    if (!selected) return
    let cancelled = false
    api
      .replay(selected)
      .then((replay) => {
        if (cancelled) return
        setData(replay)
        setIndex(0)
        setPlaying(false)
        setError(null)
      })
      .catch((exc: Error) => setError(exc.message))
    return () => {
      cancelled = true
    }
  }, [selected])

  const step = data?.steps[index] ?? null

  const advance = useCallback(() => {
    if (!data?.steps.length) return
    setIndex((current) => {
      if (current >= data.steps.length - 1) {
        setPlaying(false)
        return current
      }
      return current + 1
    })
  }, [data])

  useEffect(() => {
    if (!playing || !data) {
      if (frame.current) cancelAnimationFrame(frame.current)
      frame.current = undefined
      return
    }
    last.current = performance.now()
    clock.current = 0
    const tick = (now: number) => {
      const row = data.steps[index]
      const elapsed = now - last.current
      last.current = now
      const hold = (gapless ? (row?.game_ms ?? 60) : (row?.wall_ms ?? 900)) / speed
      clock.current += elapsed
      if (clock.current >= hold) {
        clock.current = 0
        advance()
      }
      frame.current = requestAnimationFrame(tick)
    }
    frame.current = requestAnimationFrame(tick)
    return () => {
      if (frame.current) cancelAnimationFrame(frame.current)
    }
  }, [playing, data, index, gapless, speed, advance])

  const totals = useMemo(() => {
    if (!data) return null
    const thinking = data.thinking_ms
    const game = data.game_ms
    const wall = thinking + game
    return { thinking, game, wall, gapShare: wall > 0 ? thinking / wall : 0 }
  }, [data])

  const liveThought = snapshot?.thought

  return (
    <div className="tab-grid tab-grid-replay">
      <div className="column column-wide">
        <Panel
          title="Replay"
          hint="the recording is one still per decision — collapse the thinking to watch it move"
          aside={
            <div className="replay-controls">
              <button type="button" className="ghost" onClick={() => setIndex((i) => Math.max(0, i - 1))}>
                ◀
              </button>
              <button type="button" className="primary" onClick={() => setPlaying((p) => !p)}>
                {playing ? 'Pause' : 'Play'}
              </button>
              <button type="button" className="ghost" onClick={() => setIndex((i) => Math.min((data?.steps.length ?? 1) - 1, i + 1))}>
                ▶
              </button>
              <button type="button" className={gapless ? 'toggle on' : 'toggle'} onClick={() => setGapless((g) => !g)}
                title="Skip the ~0.1 s the fly spent thinking between decisions">
                {gapless ? 'gaps collapsed' : 'as it happened'}
              </button>
              <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))} aria-label="playback speed">
                {SPEEDS.map((value) => (
                  <option key={value} value={value}>
                    {value}×
                  </option>
                ))}
              </select>
            </div>
          }
        >
          {error && <p className="error">{error}</p>}
          {!data || !data.steps.length ? (
            <p className="empty">Nothing recorded yet. Run an episode in the Lab tab first.</p>
          ) : (
            <div className="replay-body">
              <div>
              <div className="replay-stage">
                <img
                  className="replay-frame"
                  src={step?.image ?? data.start_image}
                  alt={`decision ${index + 1}`}
                />
                <div className="replay-hud">
                  <span className="phase-badge phase-acting">replay</span>
                  <span className="replay-clock">
                    #{index + 1}/{data.steps.length} · {step?.action ?? '—'}
                  </span>
                </div>
              </div>
              <input
                className="scrub"
                type="range"
                min={0}
                max={Math.max(0, data.steps.length - 1)}
                value={index}
                onChange={(e) => {
                  setPlaying(false)
                  setIndex(Number(e.target.value))
                }}
                aria-label="decision scrubber"
              />
              {totals && (
                <p className="footnote gap-meter">
                  <span className="gap-bar">
                    <i style={{ width: `${(totals.game / Math.max(1, totals.wall)) * 100}%` }} />
                  </span>
                  this episode spent <b>{(totals.gapShare * 100).toFixed(0)}%</b> of its wall clock waiting for
                  the fly to think ({(totals.thinking / 1000).toFixed(1)}s) and{' '}
                  {(100 - totals.gapShare * 100).toFixed(0)}% moving the emulator ({(totals.game / 1000).toFixed(1)}s).
                  {gapless ? ' Gaps are collapsed, so it plays at emulator speed.' : ''}
                </p>
              )}
              </div>
              <div>
              <Ribbon
                trace={data.steps.map((row) => ({
                  decisions: row.index,
                  frames: row.input_frame,
                  action: row.action,
                  label: row.label,
                  coarse: row.brain?.coarse ?? row.model ?? 'scripted',
                  confidence: Math.max(0, ...(Object.values(row.brain?.probabilities ?? { x: 0 }) as number[])),
                  model: row.model,
                  latencyMs: row.thinking_ms,
                  command: row.brain?.command ?? {},
                  input: row.brain?.input ?? {},
                  buttons: row.buttons,
                }))}
                selected={index}
                onSelect={setIndex}
              />
              </div>
            </div>
          )}
        </Panel>
        {step && (
          <Panel title={`Decision #${step.index + 1}`} hint="what the fly knew, and what it did about it">
            <div className="replay-detail">
              <div>
                <h3>senses</h3>
                <Senses snapshot={{ fly: { input: step.brain?.input ?? {} } } as unknown as Snapshot} />
                <h3>descending output</h3>
                <CommandGauges command={step.brain?.command ?? {}} card={card} />
              </div>
              <div>
                <h3>soma firing</h3>
                <Heat data={step.brain?.grid} size={step.brain?.grid_size} />
                <h3>readout</h3>
                <Decision
                  snapshot={
                    {
                      thought: {
                        probabilities: step.brain?.probabilities ?? {},
                        coarse: step.brain?.coarse,
                        action: step.action,
                        label: step.label,
                        buttons: step.buttons,
                        model: step.model,
                        latency_ms: step.thinking_ms,
                      },
                      progress: { decisions: step.index + 1 },
                      game: { coarse_actions: Object.keys(step.brain?.probabilities ?? {}) },
                    } as unknown as Snapshot
                  }
                />
              </div>
            </div>
          </Panel>
        )}
      </div>

      <div className="column">
        <Panel title="Recorded episodes" hint="newest first, kept in memory">
          <ul className="episode-list">
            {episodes.map((episode) => {
              const share = episode.thinking_ms + episode.game_ms > 0
                ? episode.thinking_ms / (episode.thinking_ms + episode.game_ms)
                : 0
              return (
                <li key={episode.id}>
                  <button type="button" className={episode.id === selected ? 'on' : ''} onClick={() => setSelected(episode.id)}>
                    <span className="episode-head">
                      <b>{episode.game_title}</b>
                      <em>{episode.policy}</em>
                    </span>
                    <span className="episode-meta">
                      {episode.decisions} decisions · {episode.frames} frames · seed {episode.seed}
                      {episode.completed ? ' · cleared' : episode.ended ? ' · died' : ' · open'}
                    </span>
                    <span className="episode-gap">
                      <i style={{ width: `${share * 100}%` }} />
                    </span>
                    <span className="episode-times">
                      {(episode.thinking_ms / 1000).toFixed(1)}s thinking / {(episode.game_ms / 1000).toFixed(1)}s acting
                    </span>
                  </button>
                </li>
              )
            })}
            {!episodes.length && <li className="empty">no episodes recorded this session</li>}
          </ul>
        </Panel>
        {liveThought && (
          <Panel title="Live lab" hint="the running episode, for comparison">
            <p className="footnote">
              current policy <b>{snapshot?.policy}</b>, {snapshot?.progress.decisions ?? 0} decisions so far.
              Switch to the Lab tab to keep playing.
            </p>
          </Panel>
        )}
      </div>
    </div>
  )
}