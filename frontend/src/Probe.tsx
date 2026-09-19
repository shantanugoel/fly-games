import { useEffect, useMemo, useState } from 'react'
import { api } from './api'
import { BrainView, type BrainMode } from './BrainView'
import { CommandGauges, Panel } from './Panels'
import type { PointCloud } from './api'
import type { BrainCard, ProbeMenu, ProbeResult, Stimulus } from './types'

interface Props {
  card: BrainCard | null
  cloud: PointCloud | null
  games: string[]
  mode: BrainMode
  onMode: (mode: BrainMode) => void
}

const emptyResult = null

function rowLabel(event: Stimulus): string {
  return `${event.types.join('+') || '—'}${event.side ? ` ${event.side}` : ' both'}`
}

export function Probe({ card, cloud, games, mode, onMode }: Props) {
  const [menu, setMenu] = useState<ProbeMenu | null>(null)
  const [events, setEvents] = useState<Stimulus[]>([{ types: ['LC4'], side: null, volts: 0.8 }])
  const [result, setResult] = useState<ProbeResult | null>(emptyResult)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [askGames, setAskGames] = useState<string[]>(games.slice(0, 1))
  const [steps, setSteps] = useState(24)
  const [query, setQuery] = useState('')
  const [matches, setMatches] = useState<{ name: string; neurons: number; L: number; R: number }[]>([])

  useEffect(() => {
    api.probeMenu().then(setMenu).catch((exc: Error) => setError(exc.message))
  }, [])

  useEffect(() => {
    if (query.trim().length < 2) return setMatches([])
    const timer = window.setTimeout(() => {
      api.probeSearch(query.trim()).then((r) => setMatches(r.types)).catch(() => setMatches([]))
    }, 220)
    return () => window.clearTimeout(timer)
  }, [query])

  const update = (index: number, patch: Partial<Stimulus>) =>
    setEvents((rows) => rows.map((row, i) => (i === index ? { ...row, ...patch } : row)))

  const run = async () => {
    setBusy(true)
    setError(null)
    try {
      setResult(await api.probeRun(events.filter((e) => e.types.length), askGames, steps))
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc))
    } finally {
      setBusy(false)
    }
  }

  const field = useMemo(() => {
    if (!result?.field) return null
    const raw = atob(result.field)
    const bytes = new Uint8Array(raw.length)
    for (let i = 0; i < raw.length; i += 1) bytes[i] = raw.charCodeAt(i)
    return bytes
  }, [result])

  const hot = useMemo(() => {
    if (!result) return null
    const entries = Object.entries(result.delta).sort((a, b) => b[1] - a[1])
    return entries.length && entries[0][1] > 0.5 ? entries[0][0] : null
  }, [result])

  return (
    <div className="tab-grid tab-grid-probe">
      <div className="column column-wide">
        <Panel title="Stimulate" hint="drive identified cell types — no ROM, no emulator, nothing trained">
          <div className="stimulus-list">
            {events.map((event, index) => (
              <div className="stimulus" key={index}>
                <input
                  value={event.types.join(', ')}
                  placeholder="cell types, e.g. LC4 or LPLC2,Tm3"
                  onChange={(e) => update(index, { types: e.target.value.split(',').map((t) => t.trim()).filter(Boolean) })}
                  onFocus={() => setQuery(event.types[0] ?? '')}
                  aria-label={`stimulus ${index + 1} cell types`}
                />
                <select
                  value={event.side ?? ''}
                  onChange={(e) => update(index, { side: (e.target.value || null) as Stimulus['side'] })}
                  aria-label="side"
                >
                  <option value="">both</option>
                  <option value="L">left</option>
                  <option value="R">right</option>
                </select>
                <label className="volts">
                  <input
                    type="range"
                    min={0}
                    max={menu?.max_volts ?? 0.8}
                    step={0.05}
                    value={event.volts}
                    onChange={(e) => update(index, { volts: Number(e.target.value) })}
                  />
                  <span>{event.volts.toFixed(2)}V</span>
                </label>
                <button type="button" className="ghost" onClick={() => setEvents((rows) => rows.filter((_, i) => i !== index))}>
                  ✕
                </button>
              </div>
            ))}
          </div>
          <div className="row">
            <button type="button" className="ghost" onClick={() => setEvents((rows) => [...rows, { types: [], side: null, volts: 0.6 }])}>
              + stimulus
            </button>
            <label className="inline">
              window
              <input type="number" min={4} max={120} value={steps} onChange={(e) => setSteps(Number(e.target.value))} />
              <span>steps ({(steps * 0.02).toFixed(2)}s)</span>
            </label>
            <button type="button" className="primary" onClick={run} disabled={busy}>
              {busy ? 'the fly is thinking…' : 'Stimulate'}
            </button>
          </div>
          {menu && (
            <div className="presets">
              {menu.presets.map((preset) => (
                <button
                  type="button"
                  key={preset.name}
                  title={preset.note}
                  onClick={() => {
                    setEvents(preset.events.map((e) => ({ types: e.types, side: e.side ?? null, volts: e.volts })))
                  }}
                >
                  {preset.name}
                </button>
              ))}
            </div>
          )}
          <div className="search">
            <input value={query} placeholder="search 11,863 annotated cell types…" onChange={(e) => setQuery(e.target.value)} aria-label="search cell types" />
            <ul className="matches">
              {matches.slice(0, 8).map((match) => (
                <li key={match.name}>
                  <code>{match.name}</code>
                  <span>
                    {match.neurons.toLocaleString()} neurons · L{match.L} / R{match.R}
                  </span>
                  <button type="button" onClick={() => setEvents((rows) => [...rows.filter((r) => r.types.length), { types: [match.name], side: null, volts: 0.7 }])}>
                    poke
                  </button>
                </li>
              ))}
            </ul>
          </div>
          {error && <p className="error">{error}</p>}
        </Panel>

        <Panel title="Descending output" hint="resting rate vs driven rate, in Hz">
          {result ? (
            <>
              <div className="delta-grid">
                {result.groups.map((group) => {
                  const delta = result.delta[group] ?? 0
                  const scale = Math.max(1, ...Object.values(result.delta).map(Math.abs))
                  return (
                    <div className="delta" key={group}>
                      <span className="delta-name">{group}</span>
                      <span className="delta-track">
                        <i className="zero" />
                        <i
                          className={delta >= 0 ? 'up' : 'down'}
                          style={{ width: `${(Math.abs(delta) / scale) * 50}%`, left: delta >= 0 ? '50%' : undefined }}
                        />
                      </span>
                      <span className={`delta-value ${delta > 0.05 ? 'pos' : delta < -0.05 ? 'neg' : ''}`}>
                        {delta >= 0 ? '+' : ''}
                        {delta.toFixed(2)}
                      </span>
                    </div>
                  )
                })}
              </div>
              <p className="footnote">
                {result.spikes.toLocaleString()} spikes in {(result.steps * result.dt).toFixed(2)}s of fly time,{' '}
                {result.latency_ms.toFixed(0)} ms of CPU.
              </p>
            </>
          ) : (
            <p className="empty">Pick a stimulus and press Stimulate. Try LC4 at 0.8 V — the fly&apos;s
              giant-fibre escape drive.</p>
          )}
        </Panel>
      </div>

      <div className="column">
        <Panel title="Connectome" hint="lit by the spike counts from this stimulus">
          <BrainView cloud={cloud} card={card} field={field} hot={hot} mode={mode} onMode={onMode} />
        </Panel>
        {result && (
          <Panel title="What it would do" hint="the same command rates, decoded by each game">
            <ul className="would">
              {Object.entries(result.would_do).map(([game, verdict]) => (
                <li key={game}>
                  <b>{game}</b>
                  <span className="coarse">{verdict.coarse}</span>
                  <span className="probs">
                    {Object.entries(verdict.probabilities)
                      .sort((a, b) => b[1] - a[1])
                      .map(([label, value]) => `${label} ${(value * 100).toFixed(0)}%`)
                      .join(' · ')}
                  </span>
                  <em>{verdict.source}</em>
                </li>
              ))}
              {!Object.keys(result.would_do).length && <li className="empty">Tick a game to decode the result.</li>}
            </ul>
          </Panel>
        )}
        <Panel title="Decode for" hint="which games interpret this stimulus">
          <div className="chips">
            {games.map((game) => (
              <button
                type="button"
                key={game}
                className={askGames.includes(game) ? 'on' : ''}
                onClick={() => setAskGames((list) => (list.includes(game) ? list.filter((g) => g !== game) : [...list, game]))}
              >
                {game}
              </button>
            ))}
          </div>
        </Panel>
        {result && <Panel title="Command rates" hint="driven rates"><CommandGauges command={Object.fromEntries(Object.entries(result.command).map(([k, v]) => [k, v.rate]))} card={card} hot={hot} /></Panel>}
        {result && result.kinds.length > 0 && (
          <Panel title="Most active classes" hint="spikes per superclass in the window">
            <ul className="bars">
              {result.kinds.slice(0, 8).map(([name, count]) => (
                <li key={name}>
                  <span className="bar-label">{name.replace(/_/g, ' ')}</span>
                  <span className="bar-track">
                    <span className="bar-fill" style={{ width: `${(count / result.kinds[0][1]) * 100}%` }} />
                  </span>
                  <span className="bar-value">{count.toLocaleString()}</span>
                </li>
              ))}
            </ul>
          </Panel>
        )}
        <p className="footnote stimulus-readout">
          {result ? `applied: ${events.filter((e) => e.types.length).map(rowLabel).join(', ')}` : ''}
        </p>
      </div>
    </div>
  )
}