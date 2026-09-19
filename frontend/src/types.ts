// Shapes returned by the fly-games server. Kept in one file so the three
// surfaces (live lab, probe bench, replay) stay honest about the same payload.

export type Phase = 'thinking' | 'acting' | 'paused' | 'complete' | 'dead' | 'error'
export type Policy = 'fly' | 'fly-hand' | 'scripted'

export interface CommandRate {
  count: number
  rate: number
}

export interface FlyState {
  command: Record<string, CommandRate>
  input: Record<string, number>
  kinds: [string, number][]
  steps: number
  dt: number
  spikes: number
  groups: string[]
  /** base64 uint8, one spike count per probe neuron */
  field?: string
  /** base64 uint8 heat map of soma firing, row-major */
  grid: string
  grid_size: [number, number]
}

export interface ReadoutInfo {
  kind: string
  components: number | null
  lam: number
  cv_score: number
  n_samples: number
  trained_at: string | null
}

export interface Snapshot {
  revision: number
  phase: Phase
  policy: Policy
  running: boolean
  busy: boolean
  done: boolean
  completed: boolean
  error: string | null
  game: {
    id: string
    title: string
    short_name: string
    platform_id: string
    platform_label: string
    cartridge: string
    subtitle: string
    aspect_width: number
    aspect_height: number
    button_keys: string[]
    coarse_actions: string[]
  }
  screen: {
    image: string | null
    facts: { label: string; value: string }[]
    overlays: { kind: string; x: number; y: number; w: number; h: number }[]
    text: string
  }
  scene: string
  stats: { label: string; value: string }[]
  progress: {
    frames: number
    seconds: number
    decisions: number
    input_frame: number
    hold_frames: number
    seed: number
  }
  thought: {
    model: string | null
    coarse: string | null
    probabilities: Record<string, number>
    confidence: number | null
    action: string | null
    label: string | null
    buttons: string[]
    executed: string | null
    latency_ms: number | null
    reason: string | null
  }
  fly: FlyState | null
  readout: {
    trained: boolean
    coarse_actions: string[]
    path: string
    info: ReadoutInfo | null
  }
  actions: { key: string; label: string; buttons: string[] }[]
  events: { time: string; message: string }[]
  episodes: EpisodeSummary[]
}

export interface EpisodeSummary {
  id: string
  started_at: string
  game: string
  game_title: string
  short_name: string
  world: string
  seed: number
  policy: string
  hold_frames: number
  status: string
  decisions: number
  frames: number
  last_action: string | null
  ended: boolean
  completed: boolean
  thinking_ms: number
  game_ms: number
}

export interface ReplayStep {
  index: number
  image?: string
  input_frame: number
  action: string
  label: string
  buttons: string[]
  frames_executed: number
  reward: number
  latency_ms: number | null
  done: boolean
  completed: boolean
  model: string | null
  thinking_ms: number
  game_ms: number
  wall_ms: number
  brain: {
    command: Record<string, number>
    input: Record<string, number>
    kinds: [string, number][]
    spikes: number
    grid: string
    grid_size: [number, number]
    coarse: string | null
    probabilities: Record<string, number> | null
  } | null
}

export interface Replay {
  episode: EpisodeSummary
  start_image: string
  thinking_ms: number
  game_ms: number
  steps: ReplayStep[]
}

export interface Superclass {
  name: string
  count: number
  color: string
}

export interface BrainCard {
  device: string
  batch: number
  dt: number
  n_neurons: number
  n_connections: number | null
  grid_h: number
  grid_w: number
  command_groups: { name: string; neurons: number; probe_ids?: number[] }[]
  command_neurons: Record<string, number>
  seed: number
  positions: number
  unpositioned: number
  probes: number
  point_bytes: number
  points_url: string
  superclasses: Superclass[]
  channels: {
    name: string
    types: string[]
    responds_to: string
    sides: { L: { neurons: number; probe_ids: number[] }; R: { neurons: number; probe_ids: number[] } }
  }[]
}

export interface ProbeMenu {
  channels: { name: string; types: string[]; responds_to: string }[]
  cell_types: { name: string; role: string; responds_to: string; L: number; R: number }[]
  command_groups: { name: string; neurons: number }[]
  presets: { name: string; note: string; events: Stimulus[] }[]
  max_volts: number
  steps: number
}

export interface Stimulus {
  types: string[]
  side: 'L' | 'R' | null
  volts: number
}

export interface ProbeResult {
  input: Record<string, number>
  command: Record<string, CommandRate>
  baseline: Record<string, number>
  delta: Record<string, number>
  kinds: [string, number][]
  spikes: number
  steps: number
  dt: number
  latency_ms: number
  groups: string[]
  would_do: Record<string, { coarse: string; probabilities: Record<string, number>; source: string }>
  field?: string
  grid?: string
  grid_size?: [number, number]
}

export interface ReadoutStatus {
  trained: boolean
  coarse_actions: string[]
  features: string[]
  info: ReadoutInfo | null
  weights: Record<string, Record<string, number>>
  error?: string
}