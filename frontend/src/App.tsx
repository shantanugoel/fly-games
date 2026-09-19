import { useEffect, useMemo, useState } from 'react'
import type { BrainMode } from './BrainView'
import { BrainView } from './BrainView'
import { Decision, Legend, Panel, Ribbon, Screen, Senses, CommandGauges, Translation } from './Panels'
import { Probe } from './Probe'
import { Replay } from './Replay'
import { api } from './api'
import { b64bytes } from './api'
import { useLab } from './useLab'
import type { ReadoutStatus } from './types'

type Tab = 'lab' | 'probe' | 'replay'

const GAMES = [
  { id: 'mario', label: 'Mario' },
  { id: 'kungfu', label: 'Kung Fu' },
  { id: 'doom', label: 'Doom' },
]

const POLICIES = [
  { id: 'fly', label: 'fly', hint: 'trained readout if there is one, else the zero-shot rule' },
  { id: 'fly-hand', label: 'fly-hand', hint: 'force the zero-shot rule, ignore any trained readout' },
  { id: 'scripted', label: 'scripted', hint: 'the hand-written baseline; no brain involved' },
]

export function App() {
  const { snapshot, brain, cloud, trace, connected, error, send } = useLab()
  const [tab, setTab] = useState<Tab>('lab')
  const [mode, setMode] = useState<BrainMode>('2d')
  const [channel, setChannel] = useState<string | null>(null)
  const [readouts, setReadouts] = useState<Record<string, ReadoutStatus> | null>(null)
  const [dismissed, setDismissed] = useState<string | null>(null)

  useEffect(() => {
    api.readouts().then(setReadouts).catch(() => setReadouts(null))
  }, [snapshot?.readout.trained, snapshot?.game.id])

  const field = useMemo(() => {
    const raw = snapshot?.fly?.field
    return raw ? b64bytes(raw) : null
  }, [snapshot?.fly?.field])

  const hot = useMemo(() => {
    const command = snapshot?.fly?.command
    if (!command) return null
    const entries = Object.entries(command).sort((a, b) => b[1].rate - a[1].rate)
    return entries.length && entries[0][1].rate > 1 ? entries[0][0] : null
  }, [snapshot?.fly?.command])

  if (!snapshot) {
    return (
      <div className="boot">
        <h1>fly-games</h1>
        <p>{error ? `cannot reach the lab: ${error}` : 'waking the connectome…'}</p>
        {!error && <p className="boot-note">166,700 neurons have to be paged in before anything happens.</p>}
      </div>
    )
  }

  const readout = readouts?.[snapshot.game.id]
  const show = error && error !== dismissed

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand">
          <span className="mark">✈</span>
          <div>
            <h1>fly-games</h1>
            <p>
              MaleCNS v1.0 · {brain ? brain.n_neurons.toLocaleString() : '…'} neurons ·{' '}
              {brain?.n_connections ? `${(brain.n_connections / 1e6).toFixed(1)}M synapses` : 'connectome'} ·{' '}
              {brain?.device ?? '…'}
            </p>
          </div>
        </div>

        <nav className="tabs">
          {(['lab', 'probe', 'replay'] as Tab[]).map((entry) => (
            <button type="button" key={entry} className={tab === entry ? 'on' : ''} onClick={() => setTab(entry)}>
              {entry}
            </button>
          ))}
        </nav>

        <div className="controls">
          <div className="seg games">
            {GAMES.map((game) => (
              <button
                type="button"
                key={game.id}
                className={snapshot.game.id === game.id ? 'on' : ''}
                onClick={() => send('select', { game: game.id })}
              >
                {game.label}
              </button>
            ))}
          </div>
          <select
            value={snapshot.policy}
            onChange={(e) => send('configure', { policy: e.target.value })}
            aria-label="policy"
            title={POLICIES.find((p) => p.id === snapshot.policy)?.hint}
          >
            {POLICIES.map((policy) => (
              <option key={policy.id} value={policy.id}>
                {policy.label}
              </option>
            ))}
          </select>
          <select
            value={snapshot.progress.hold_frames}
            onChange={(e) => send('configure', { frames: Number(e.target.value) })}
            aria-label="hold frames"
            title="how many emulator frames each decision is held for"
          >
            {[2, 4, 8, 12, 16].map((frames) => (
              <option key={frames} value={frames}>
                hold {frames}f
              </option>
            ))}
          </select>
          <button type="button" className="primary" onClick={() => send(snapshot.running ? 'pause' : 'run')}>
            {snapshot.running ? 'Pause' : 'Run'}
          </button>
          <button type="button" className="ghost" disabled={snapshot.busy} onClick={() => send('step')} title="one decision">
            Step
          </button>
          <button type="button" className="ghost" onClick={() => send('reset')}>
            Reset
          </button>
          <span className={`phase phase-${snapshot.phase}`}>{snapshot.phase}</span>
          <span className={`link ${connected ? 'ok' : 'bad'}`} title={connected ? 'websocket connected' : 'reconnecting'}>
            ●
          </span>
        </div>
      </header>

      {show && (
        <div className="banner">
          <span>{error}</span>
          <button type="button" onClick={() => setDismissed(error)}>
            dismiss
          </button>
        </div>
      )}

      {tab === 'lab' && (
        <main className="tab-grid tab-grid-lab">
          <div className="column column-brain">
            <Panel
              title="Connectome"
              hint="every soma with a position; brightness is spikes in the last decision window"
              aside={
                <div className="chips chips-inline">
                  <button type="button" className={channel === null ? 'on' : ''} onClick={() => setChannel(null)}>
                    all
                  </button>
                  {(brain?.channels ?? []).map((entry) => (
                    <button
                      type="button"
                      key={entry.name}
                      className={channel === entry.name ? 'on' : ''}
                      title={entry.responds_to}
                      onClick={() => setChannel(channel === entry.name ? null : entry.name)}
                    >
                      {entry.name}
                    </button>
                  ))}
                </div>
              }
            >
              <BrainView cloud={cloud} card={brain} field={field} hot={hot} channel={channel} mode={mode} onMode={setMode} />
              <Legend card={brain} />
            </Panel>
          </div>

          <div className="column">
            <Panel title={snapshot.game.title} hint={`${snapshot.game.cartridge} · ${snapshot.game.platform_label}`}>
              <Screen snapshot={snapshot} />
            </Panel>
            <Panel title="Senses" hint="encoder drive on the fly's own visual-projection neurons">
              <Senses snapshot={snapshot} />
            </Panel>
            <Panel title="Decision ribbon" hint="bar colour is the fly's coarse call; height is peak descending rate">
              <Ribbon trace={trace} />
            </Panel>
          </div>

          <div className="column">
            <Panel
              title="What the fly is pressing"
              hint="every stage of the translation - the fly emits urges, never buttons"
            >
              <Translation snapshot={snapshot} />
            </Panel>
            <Panel title="Descending output" hint="the fly's commands to its body">
              <CommandGauges command={Object.fromEntries(Object.entries(snapshot.fly?.command ?? {}).map(([k, v]) => [k, v.rate]))} card={brain} hot={hot} />
            </Panel>
            <Panel title="Decision" hint={snapshot.readout.trained ? 'trained readout' : 'zero-shot rule'}>
              <Decision snapshot={snapshot} />
            </Panel>
            <Panel title="Readout" hint="the only thing that is ever fitted">
              {readout?.trained && readout.info ? (
                <dl className="readout">
                  <div>
                    <dt>model</dt>
                    <dd>
                      {readout.info.kind} · {readout.info.components} PCs · λ{readout.info.lam}
                    </dd>
                  </div>
                  <div>
                    <dt>held-out calls</dt>
                    <dd>
                      {readout.info.accuracy === null
                        ? `no episode held out (cv ${readout.info.cv_score.toFixed(3)})`
                        : `${(readout.info.accuracy * 100).toFixed(0)}% right vs ${((readout.info.baseline ?? 0) * 100).toFixed(0)}% always-same-answer`}
                    </dd>
                  </div>
                  <div>
                    <dt>cv (not accuracy)</dt>
                    <dd>
                      {readout.info.cv_score.toFixed(3)} R² on 0/1 targets, {readout.info.n_samples} decisions
                    </dd>
                  </div>
                  <div>
                    <dt>listens to</dt>
                    <dd className="weights">
                      {Object.entries(readout.weights[snapshot.thought.coarse ?? ''] ?? {})
                        .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
                        .slice(0, 4)
                        .map(([group, weight]) => (
                          <span key={group} className={weight >= 0 ? 'pos' : 'neg'}>
                            {group} {weight >= 0 ? '+' : ''}
                            {(weight * 100).toFixed(0)}
                          </span>
                        ))}
                    </dd>
                  </div>
                </dl>
              ) : (
                <p className="empty">
                  No readout for {snapshot.game.id}. The fly still plays — on a hand-written rule over its
                  command rates. Fit one with <code>fly-games train --game {snapshot.game.id}</code>.
                </p>
              )}
            </Panel>
            <Panel title="Log">
              <ul className="events">
                {snapshot.events.map((event, index) => (
                  <li key={index}>
                    <time>{event.time}</time>
                    {event.message}
                  </li>
                ))}
              </ul>
            </Panel>
          </div>
        </main>
      )}

      {tab === 'probe' && (
        <Probe card={brain} cloud={cloud} games={GAMES.map((g) => g.id)} mode={mode} onMode={setMode} />
      )}

      {tab === 'replay' && <Replay episodes={snapshot.episodes} card={brain} snapshot={snapshot} />}

      <footer className="statusline">
        <span>{snapshot.game.subtitle}</span>
        <span>
          frame {snapshot.progress.frames} · {(snapshot.progress.seconds).toFixed(1)}s of game · seed{' '}
          {snapshot.progress.seed}
        </span>
        <span className={connected ? 'ok' : 'bad'}>{connected ? 'live' : 'offline'}</span>
      </footer>
    </div>
  )
}