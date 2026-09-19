import { useMemo } from 'react'
import type { BrainCard, Snapshot } from './types'
import type { TraceRow } from './useLab'

const GROUP_ORDER = ['escape', 'steer', 'forward', 'backward', 'punch', 'kick'] as const

const CHANNEL_LABEL: Record<string, string> = {
  loom: 'looming · LPLC2',
  threat: 'threat · LC4',
  shot: 'approaching · LPLC1',
  chase: 'chase · LC10a',
}

export function Panel({ title, hint, children, aside }: {
  title: string
  hint?: string
  children: React.ReactNode
  aside?: React.ReactNode
}) {
  return (
    <section className="panel">
      <header>
        <h2>{title}</h2>
        {hint && <span className="hint">{hint}</span>}
        {aside}
      </header>
      {children}
    </section>
  )
}

/** Split encoder drive labels ("looms_L") into channel + side. */
function splitDrive(key: string): { channel: string; side: string } {
  const index = key.lastIndexOf('_')
  const side = index > 0 ? key.slice(index + 1) : ''
  const stem = index > 0 ? key.slice(0, index) : key
  const channel = stem.replace(/s$/, '').replace('looms', 'loom').replace('threats', 'threat')
  return { channel: channel in CHANNEL_LABEL ? channel : stem, side }
}

export function Senses({ snapshot }: { snapshot: Snapshot }) {
  const drives = snapshot.fly?.input ?? {}
  const entries = Object.entries(drives).sort((a, b) => b[1] - a[1])
  const card = useMemo(() => new Map(), [])
  void card
  if (!entries.length) {
    return <p className="empty">Nothing is driving the fly&apos;s eyes right now — no hazard within
      range, so the connectome is at rest.</p>
  }
  return (
    <ul className="bars">
      {entries.map(([key, volts]) => {
        const { channel, side } = splitDrive(key)
        return (
          <li key={key}>
            <span className="bar-label" title={CHANNEL_LABEL[channel] ?? channel}>
              {channel}
              <b className={`side side-${side.toLowerCase()}`}>{side}</b>
            </span>
            <span className="bar-track">
              <span className="bar-fill sense" style={{ width: `${Math.min(100, (volts / 0.8) * 100)}%` }} />
            </span>
            <span className="bar-value">{volts.toFixed(2)}V</span>
          </li>
        )
      })}
    </ul>
  )
}

export function CommandGauges({
  command,
  card,
  hot,
}: {
  command: Record<string, number>
  card: BrainCard | null
  hot?: string | null
}) {
  const peak = Math.max(4, ...Object.values(command))
  return (
    <div className="command-grid">
      {GROUP_ORDER.map((group) => {
        const left = command[`${group}_L`] ?? 0
        const right = command[`${group}_R`] ?? 0
        const dominant = Math.max(left, right)
        const firing = dominant > peak * 0.35
        const neurons = card?.command_groups.find((g) => g.name === `${group}_L`)?.neurons ?? 1
        return (
          <div key={group} className={`command-row ${hot?.startsWith(group) ? 'hot' : ''} ${firing ? 'firing' : ''}`}>
            <span className="command-name">
              {group}
              <em>{neurons === 1 ? '1 neuron' : `${neurons} neurons`}</em>
            </span>
            <span className="command-pair">
              <span className="half">
                <i className="side-tag L">L</i>
                <span className="bar-track">
                  <span className="bar-fill cmd" style={{ width: `${(left / peak) * 100}%` }} />
                </span>
                <span className="bar-value">{left.toFixed(1)}</span>
              </span>
              <span className="half">
                <i className="side-tag R">R</i>
                <span className="bar-track">
                  <span className="bar-fill cmd" style={{ width: `${(right / peak) * 100}%` }} />
                </span>
                <span className="bar-value">{right.toFixed(1)}</span>
              </span>
            </span>
          </div>
        )
      })}
      <p className="footnote">Hz of the fly&apos;s descending command neurons — its output to its body.</p>
    </div>
  )
}

export function Decision({ snapshot }: { snapshot: Snapshot }) {
  const { thought, game } = snapshot
  const probabilities = Object.entries(thought.probabilities ?? {}).sort((a, b) => b[1] - a[1])
  return (
    <div className="decision">
      <div className="coarse-bar">
        {probabilities.map(([label, value]) => (
          <span
            key={label}
            className={`coarse-slice ${label === thought.coarse ? 'win' : ''}`}
            style={{ flexGrow: Math.max(0.02, value) }}
            title={`${label} ${(value * 100).toFixed(0)}%`}
          >
            {value > 0.16 ? `${label} ${(value * 100).toFixed(0)}` : ''}
          </span>
        ))}
      </div>
      <div className="decision-line">
        <span className="arrow">→</span>
        <strong className="action">{thought.label ?? thought.action ?? '—'}</strong>
        <span className="buttons">
          {(thought.buttons ?? []).length === 0
            ? 'no buttons'
            : thought.buttons.map((button) => <kbd key={button}>{button}</kbd>)}
        </span>
      </div>
      <dl className="decision-meta">
        <div>
          <dt>decoder</dt>
          <dd>{thought.model ?? '—'}</dd>
        </div>
        <div>
          <dt>thinking</dt>
          <dd>{thought.latency_ms ? `${thought.latency_ms.toFixed(0)} ms` : '—'}</dd>
        </div>
        <div>
          <dt>decisions</dt>
          <dd>{snapshot.progress.decisions}</dd>
        </div>
        <div>
          <dt>options</dt>
          <dd>{game.coarse_actions.join(' · ')}</dd>
        </div>
      </dl>
    </div>
  )
}

export function Screen({ snapshot }: { snapshot: Snapshot }) {
  const { screen, game, stats } = snapshot
  const ratio = game.aspect_width / game.aspect_height
  return (
    <div className="screen">
      <div className="screen-frame" style={{ aspectRatio: `${ratio}` }}>

        {screen.image ? <img src={screen.image} alt={`${game.title} frame`} /> : <div className="loading">starting…</div>}
        {(screen.overlays ?? []).map((overlay, index) => (
          <span
            key={`${overlay.kind}-${index}`}
            className={`overlay overlay-${overlay.kind.toLowerCase().replace(/\W+/g, '-')}`}
            style={{
              left: `${(overlay.x / game.aspect_width) * 100}%`,
              top: `${(overlay.y / game.aspect_height) * 100}%`,
              width: `${(overlay.w / game.aspect_width) * 100}%`,
              height: `${(overlay.h / game.aspect_height) * 100}%`,
            }}
            title={overlay.kind}
          />
        ))}
        <span className={`phase-badge phase-${snapshot.phase}`}>{snapshot.phase}</span>
      </div>
      <div className="screen-side">
      <ul className="stats">
        {stats.map((stat) => (
          <li key={stat.label}>
            <span>{stat.label}</span>
            <b>{stat.value}</b>
          </li>
        ))}
      </ul>
      <dl className="facts">
        {(screen.facts ?? []).map((fact) => (
          <div key={fact.label}>
            <dt>{fact.label}</dt>
            <dd>{fact.value}</dd>
          </div>
        ))}
      </dl>
      </div>
    </div>
  )
}

const COARSE_COLORS: Record<string, string> = {
  proceed: 'var(--proceed)',
  escape: 'var(--escape)',
  turn_left: 'var(--turn)',
  turn_right: 'var(--turn)',
  scripted: 'var(--scripted)',
}

/** A ribbon is a histogram of decisions, not a list of them. A 700-decision
 *  episode cannot show 700 sub-pixel bars — they collapse into a smear — so
 *  long traces are binned, each bin keeping its peak rate and its dominant
 *  coarse call. Clicking a bin jumps to its first decision. */
const MAX_RIBBON_CELLS = 140

export function Ribbon({ trace, selected, onSelect }: {
  trace: TraceRow[]
  selected?: number | null
  onSelect?: (index: number) => void
}) {
  if (!trace.length) return <p className="empty">No decisions yet. Press Run and the fly starts thinking.</p>
  const stride = Math.max(1, Math.ceil(trace.length / MAX_RIBBON_CELLS))
  const cells: { row: TraceRow; index: number; rate: number; coarse: string; span: number }[] = []
  for (let start = 0; start < trace.length; start += stride) {
    const group = trace.slice(start, start + stride)
    const counts = new Map<string, number>()
    for (const row of group) {
      const key = row.coarse ?? ''
      counts.set(key, (counts.get(key) ?? 0) + 1)
    }
    cells.push({
      row: group[0],
      index: start,
      rate: Math.max(0, ...group.flatMap((row) => Object.values(row.command))),
      coarse: [...counts.entries()].sort((a, b) => b[1] - a[1])[0][0],
      span: group.length,
    })
  }
  const peak = Math.max(1, ...cells.map((cell) => cell.rate))
  const bin = (index: number) => Math.floor(index / stride) * stride
  return (
    <div className="ribbon" style={{ gap: stride > 1 ? '1px' : '2px' }}>
      {cells.map((cell) => {
        const isSelected = selected !== undefined && selected !== null && bin(selected) === cell.index
        return (
          <button
            type="button"
            key={`${cell.row.decisions}-${cell.index}`}
            className={`ribbon-cell ${isSelected ? 'selected' : ''}`}
            onClick={() => onSelect?.(cell.index)}
            title={cell.span > 1
              ? `${cell.span} decisions · ${cell.coarse} · peak ${cell.rate.toFixed(1)} Hz`
              : `#${cell.row.decisions} ${cell.coarse} → ${cell.row.action ?? ''} · `
                + `${cell.rate.toFixed(1)} Hz · ${cell.row.latencyMs.toFixed(0)} ms`}
          >
            <span className="ribbon-activity" style={{ height: `${Math.min(100, (cell.rate / peak) * 100)}%` }} />
            <span className="ribbon-bar" style={{ background: COARSE_COLORS[cell.coarse] ?? 'var(--scripted)' }} />
          </button>
        )
      })}
    </div>
  )
}

export function Legend({ card }: { card: BrainCard | null }) {
  if (!card) return null
  const top = card.superclasses.slice(0, 8)
  return (
    <div className="legend">
      {top.map((entry) => (
        <span key={entry.name} title={`${entry.count.toLocaleString()} neurons`}>
          <i style={{ background: entry.color }} />
          {entry.name.replace(/_/g, ' ')}
        </span>
      ))}
      <span className="legend-more">+{card.superclasses.length - top.length} more classes</span>
    </div>
  )
}

/** The whole translation, in one line, so the chain is never a mystery.
 *
 * The fly has no concept of a controller: it emits graded urges from twelve
 * descending neuron groups, and a per-game decoder turns the winning urge into
 * a button combination. Showing only the urges leaves the user guessing what
 * any of it caused, and showing only the buttons hides that the fly, not a
 * script, chose it. So show every stage, left to right, with the buttons as
 * actual keys.
 */
export function Translation({ snapshot }: { snapshot: Snapshot }) {
  const { thought, fly } = snapshot
  const rates = Object.entries(fly?.command ?? {})
    .map(([name, value]) => [name, value.rate] as const)
    .sort((a, b) => b[1] - a[1])
  const loudest = rates.filter(([, rate]) => rate > 0).slice(0, 2)
  const buttons = thought.buttons ?? []
  return (
    <div className="translation">
      <div className="translation-step">
        <span className="translation-tag">the fly's loudest urge</span>
        {loudest.length === 0
          ? <span className="translation-value idle">all twelve groups quiet</span>
          : loudest.map(([name, rate]) => (
            <span className="translation-value" key={name}>
              {name.replace('_', ' ')} <b>{rate.toFixed(1)} Hz</b>
            </span>
          ))}
      </div>
      <span className="translation-arrow">→</span>
      <div className="translation-step">
        <span className="translation-tag">read it as</span>
        <span className="translation-value decision">{thought.coarse ?? '—'}</span>
      </div>
      <span className="translation-arrow">→</span>
      <div className="translation-step">
        <span className="translation-tag">{snapshot.game.short_name ?? 'the game'} does</span>
        <span className="translation-value action">{thought.label ?? thought.action ?? '—'}</span>
      </div>
      <span className="translation-arrow">→</span>
      <div className="translation-step">
        <span className="translation-tag">controller receives</span>
        {buttons.length === 0
          ? <kbd className="translation-key">no buttons</kbd>
          : buttons.map((button) => <kbd className="translation-key" key={button}>{button}</kbd>)}
      </div>
    </div>
  )
}
