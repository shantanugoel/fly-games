import { useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import type { PointCloud } from './api'
import type { BrainCard } from './types'

export type BrainMode = '2d' | '3d'

interface Props {
  cloud: PointCloud | null
  card: BrainCard | null
  /** per-probe spike counts for the current decision, or null for resting */
  field: Uint8Array | null
  /** which command group is currently winning, if any */
  hot?: string | null
  /** isolate one sensory channel ('loom' | 'threat' | 'shot' | 'chase') or null */
  channel?: string | null
  mode: BrainMode
  onMode: (mode: BrainMode) => void
}

const COMMAND_TINT: Record<string, [number, number, number]> = {
  escape_L: [1.0, 0.36, 0.3],
  escape_R: [1.0, 0.6, 0.25],
  steer_L: [0.42, 0.78, 1.0],
  steer_R: [0.5, 0.6, 1.0],
  forward_L: [0.5, 1.0, 0.62],
  forward_R: [0.72, 1.0, 0.45],
  backward_L: [1.0, 0.86, 0.4],
  backward_R: [1.0, 0.96, 0.58],
  punch_L: [0.96, 0.56, 0.96],
  punch_R: [0.8, 0.52, 1.0],
  kick_L: [0.6, 0.95, 0.9],
  kick_R: [0.55, 0.85, 0.82],
}

/** The resting cloud is nearly monochrome — a pale steel neuropil — because that
 *  is how every figure of this brain is drawn, and it leaves colour free to
 *  mean something: command neurons, and whatever is actually spiking. */
const TISSUE: [number, number, number] = [0.42, 0.55, 0.82]
const CLASS_MIX = 0.18

/** Landmark regions, labelled the way a neuroanatomy figure would. */
const LANDMARKS: { name: string; kind: string; split?: boolean }[] = [
  // The two lobes straddle the midline, so their shared centroid sits in the
  // middle of the central brain. Split them by side and each label lands on
  // the lobe it names.
  { name: 'optic lobe', kind: 'ol_intrinsic', split: true },
  { name: 'central brain', kind: 'cb_intrinsic' },
  { name: 'ventral nerve cord', kind: 'vnc_intrinsic' },
]

function hexToRgb(hex: string): [number, number, number] {
  const value = parseInt(hex.replace('#', ''), 16)
  return [((value >> 16) & 255) / 255, ((value >> 8) & 255) / 255, (value & 255) / 255]
}

function probeTexture(n: number): [number, number] {
  const width = 2 ** Math.ceil(Math.log2(Math.max(Math.ceil(Math.sqrt(n)), 8)))
  return [width, Math.max(1, Math.ceil(n / width))]
}

/** Blend a superclass colour into the tissue tone. */
function tissueColor(base: [number, number, number]): [number, number, number] {
  return [
    TISSUE[0] * (1 - CLASS_MIX) + base[0] * CLASS_MIX,
    TISSUE[1] * (1 - CLASS_MIX) + base[1] * CLASS_MIX,
    TISSUE[2] * (1 - CLASS_MIX) + base[2] * CLASS_MIX,
  ]
}

/** Mean projected position of each landmark class, in `flat` units. */
function landmarks(cloud: PointCloud | null): { name: string; x: number; y: number }[] {
  if (!cloud) return []
  const wanted = new Map<string, number>()
  cloud.superclasses.forEach((name, index) => {
    if (LANDMARKS.some((l) => l.kind === name)) wanted.set(name, index)
  })
  type Acc = { x: number; y: number; n: number }
  const sums = new Map<string, Acc>()
  const bucket = (key: string): Acc => {
    let acc = sums.get(key)
    if (!acc) {
      acc = { x: 0, y: 0, n: 0 }
      sums.set(key, acc)
    }
    return acc
  }
  for (let i = 0; i < cloud.count; i += 1) {
    const x = cloud.flat[i * 2]
    const y = cloud.flat[i * 2 + 1]
    const kindIndex = cloud.kind[i]
    for (const landmark of LANDMARKS) {
      if (wanted.get(landmark.kind) !== kindIndex) continue
      if (landmark.split) {
        // `flat` is centred on zero, so the sign of x is the side of the brain.
        bucket(`${landmark.name} ${x < 0 ? 'L' : 'R'}`).x += x
        bucket(`${landmark.name} ${x < 0 ? 'L' : 'R'}`).y += y
        bucket(`${landmark.name} ${x < 0 ? 'L' : 'R'}`).n += 1
      } else {
        const acc = bucket(landmark.name)
        acc.x += x
        acc.y += y
        acc.n += 1
      }
      break
    }
  }
  return [...sums.entries()]
    .filter(([, acc]) => acc.n > 50)
    .map(([name, acc]) => ({ name, x: acc.x / acc.n, y: acc.y / acc.n }))
}

// ------------------------------------------------------------------- flat view

function FlatBrain({ cloud, field, card, channel }: Props) {
  const canvas = useRef<HTMLCanvasElement | null>(null)
  const [hover, setHover] = useState<string | null>(null)

  const palette = useMemo(() => {
    const map = new Map<string, [number, number, number]>()
    card?.superclasses.forEach((s) => map.set(s.name, tissueColor(hexToRgb(s.color))))
    return map
  }, [card])

  const marks = useMemo(() => landmarks(cloud), [cloud])

  const channelProbes = useMemo(() => {
    if (!channel || !card) return null
    const found = card.channels.find((c) => c.name === channel)
    if (!found) return null
    const set = new Set<number>()
    for (const ids of Object.values(found.sides)) ids.probe_ids.forEach((id) => set.add(id))
    return set
  }, [channel, card])

  useEffect(() => {
    const canvasEl = canvas.current
    if (!canvasEl || !cloud) return
    const ratio = Math.min(2, window.devicePixelRatio || 1)
    const width = Math.max(320, Math.floor(canvasEl.clientWidth * ratio))
    const height = Math.max(220, Math.floor(canvasEl.clientHeight * ratio))
    canvasEl.width = width
    canvasEl.height = height
    const ctx = canvasEl.getContext('2d')
    if (!ctx) return

    // Accumulate into float buffers and bilinear-splat each soma across the
    // four nearest pixels. One pixel per neuron reads as TV static; a soft
    // splat reads as tissue.
    const accR = new Float32Array(width * height)
    const accG = new Float32Array(width * height)
    const accB = new Float32Array(width * height)
    const hot = new Float32Array(width * height)
    const cover = new Float32Array(width * height)

    const [spanX, spanY] = cloud.flatSpan
    const scale = 0.9 * Math.min(width / spanX, height / spanY)
    const originX = width / 2
    const originY = height / 2
    const toPixel = (ux: number, uy: number): [number, number] => [
      originX + ux * scale,
      originY + uy * scale,
    ]
    const maxSpike = field ? Math.max(1, ...field) : 0
    const tints = Object.values(COMMAND_TINT)

    for (let i = 0; i < cloud.count; i += 1) {
      const probe = cloud.probe[i]
      const spikes = field && probe >= 0 ? field[probe] : 0
      const drive = maxSpike ? spikes / maxSpike : 0
      const command = cloud.command[i]
      const base = palette.get(cloud.superclasses[cloud.kind[i]]) ?? TISSUE
      const rgb = command >= 0 ? tints[command] : base
      const dimmed = channelProbes !== null && (probe < 0 || !channelProbes.has(probe))
      const weight = dimmed ? 0.1 : 1

      const [fx, fy] = toPixel(cloud.flat[i * 2], cloud.flat[i * 2 + 1])
      const x0 = Math.floor(fx)
      const y0 = Math.floor(fy)
      if (x0 < 0 || y0 < 0 || x0 >= width - 1 || y0 >= height - 1) continue
      const tx = fx - x0
      const ty = fy - y0
      const w00 = (1 - tx) * (1 - ty)
      const w10 = tx * (1 - ty)
      const w01 = (1 - tx) * ty
      const w11 = tx * ty
      const glow = drive * drive * 2.6
      for (const [ox, oy, w] of [[0, 0, w00], [1, 0, w10], [0, 1, w01], [1, 1, w11]] as const) {
        if (w <= 0.001) continue
        const idx = (y0 + oy) * width + x0 + ox
        const amount = w * weight
        accR[idx] += rgb[0] * amount
        accG[idx] += rgb[1] * amount
        accB[idx] += rgb[2] * amount
        hot[idx] += glow * w
        cover[idx] += amount
      }
    }

    // Tone-map density, not raw channel sums. Keeping the *mean* colour of each
    // pixel and driving its brightness from coverage is what stops a dense
    // neuropil from clipping to white: it gets brighter, it never loses its
    // tint. Spiking neurons add a separate white-hot term on top.
    const image = ctx.createImageData(width, height)
    const data = image.data
    const exposure = 0.5
    for (let i = 0; i < cover.length; i += 1) {
      const c = cover[i]
      if (c <= 0) continue
      const at = i * 4
      const lum = Math.pow(1 - Math.exp(-c * exposure), 0.78)
      const heat = Math.min(1, hot[i] * 1.5)
      const r = accR[i] / c
      const g = accG[i] / c
      const b = accB[i] / c
      data[at] = Math.min(255, 255 * lum * (r * (1 - heat) + heat))
      data[at + 1] = Math.min(255, 255 * lum * (g * (1 - heat) + heat * 0.96))
      data[at + 2] = Math.min(255, 255 * lum * (b * (1 - heat) + heat * 0.9))
      data[at + 3] = Math.min(255, 40 + 215 * Math.min(1, c * 1.6))
    }
    ctx.putImageData(image, 0, 0)

    // Landmarks, then the twelve command neurons: few enough to name, and the
    // whole point of the picture.
    ctx.textBaseline = 'middle'
    ctx.font = `${9.5 * ratio}px "IBM Plex Mono", monospace`
    // Dark halo behind every label: the cloud is bright where the neuropil is
    // dense, and unhaloed text disappears into it.
    const stamp = (text: string, x: number, y: number, fill: string) => {
      const textWidth = ctx.measureText(text).width
      // Flip to the left of the anchor when the right edge would clip.
      const at = x + textWidth > width - 4 * ratio ? x - textWidth - 14 * ratio : x
      ctx.lineWidth = 3 * ratio
      ctx.strokeStyle = 'rgba(6,9,16,0.85)'
      ctx.strokeText(text, at, y)
      ctx.fillStyle = fill
      ctx.fillText(text, at, y)
      return { x: at, y, w: textWidth }
    }
    const placed: { x: number; y: number; w: number }[] = []
    card?.command_groups.forEach((group, index) => {
      const probeId = group.probe_ids?.[0]
      if (probeId === undefined) return
      const point = cloud.probePoint[probeId]
      if (point === undefined) return
      const [x, y] = toPixel(cloud.flat[point * 2], cloud.flat[point * 2 + 1])
      const tint = tints[index] ?? [1, 1, 1]
      const rgb = tint.map((v) => Math.round(v * 255)).join(',')
      const spikes = field ? field[probeId] ?? 0 : 0
      const pulse = spikes > 0 ? 5 + Math.min(10, spikes * 0.8) : 4.5
      ctx.beginPath()
      ctx.arc(x, y, pulse * ratio, 0, Math.PI * 2)
      ctx.strokeStyle = `rgba(${rgb},${spikes > 0 ? 1 : 0.55})`
      ctx.lineWidth = (spikes > 0 ? 1.8 : 1.1) * ratio
      ctx.stroke()
      if (spikes === 0 && placed.length >= 5) return
      for (const offset of [-1.6, 1.6, -3.2, 3.2]) {
        const labelY = y + offset * 8 * ratio
        const metrics = ctx.measureText(group.name).width
        const clash = placed.some((other) => Math.abs(other.y - labelY) < 9 * ratio
          && x + 9 * ratio < other.x + other.w && other.x < x + 9 * ratio + metrics)
        if (clash) continue
        placed.push({ x: x + 9 * ratio, y: labelY, w: metrics })
        stamp(group.name, x + 9 * ratio, labelY, `rgba(${rgb},0.95)`)
        break
      }
    })

    for (const mark of marks) {
      const [x, y] = toPixel(mark.x, mark.y)
      stamp(mark.name.toUpperCase(), x + 7 * ratio, y, 'rgba(168,190,226,0.8)')
    }
  }, [cloud, field, palette, card, channelProbes, marks])

  return (
    <canvas
      ref={canvas}
      className="brain-canvas"
      onMouseLeave={() => setHover(null)}
      onMouseMove={(event) => {
        if (!cloud) return
        const rect = event.currentTarget.getBoundingClientRect()
        const ratio = rect.width / canvas.current!.clientWidth || 1
        void ratio
        const scale = 0.9 * Math.min(rect.width / cloud.flatSpan[0], rect.height / cloud.flatSpan[1])
        const nx = (event.clientX - rect.left - rect.width / 2) / scale
        const ny = (event.clientY - rect.top - rect.height / 2) / scale
        let best = -1
        let bestDistance = (11 / scale) ** 2
        for (let i = 0; i < cloud.count; i += 1) {
          const dx = cloud.flat[i * 2] - nx
          const dy = cloud.flat[i * 2 + 1] - ny
          const distance = dx * dx + dy * dy
          if (distance < bestDistance) {
            bestDistance = distance
            best = i
          }
        }
        if (best < 0) return setHover(null)
        const kind = cloud.superclasses[cloud.kind[best]] ?? 'unknown'
        const command = cloud.command[best]
        const label = command >= 0 ? `${kind} · ${card?.command_groups[command]?.name ?? 'command'}` : kind
        const spikes = cloud.probe[best] >= 0 && field ? field[cloud.probe[best]] : 0
        return setHover(spikes ? `${label} · ${spikes} spikes` : label)
      }}
    >
      {hover}
    </canvas>
  )
}

// ------------------------------------------------------------------ 3-D view

const VERTEX = /* glsl */ `
attribute vec3 aColor;
attribute float aProbe;
attribute float aCommand;
attribute float aJitter;
uniform sampler2D uSpikes;
uniform vec2 uTexSize;
uniform float uMaxSpike;
uniform float uSize;
uniform float uHeight;
varying vec3 vColor;
varying float vGlow;
varying float vFade;

void main() {
  float spikes = 0.0;
  if (aProbe >= 0.0) {
    vec2 uv = vec2(mod(aProbe, uTexSize.x) + 0.5, floor(aProbe / uTexSize.x) + 0.5) / uTexSize;
    spikes = texture2D(uSpikes, uv).r * 255.0;
  }
  float drive = uMaxSpike > 0.0 ? clamp(spikes / uMaxSpike, 0.0, 1.0) : 0.0;
  bool isCommand = aCommand >= 0.0;
  vGlow = drive;
  vColor = mix(aColor, aColor + vec3(0.85), drive * drive);
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  float depth = max(0.001, -mv.z);
  // Slightly different size per soma so the cloud never looks like a grid.
  float base = uSize * (isCommand ? 7.0 : 1.0) * (0.65 + aJitter * 0.7) * (1.0 + drive * 1.8);
  gl_PointSize = clamp(base * uHeight / depth, 1.0, 30.0);
  vFade = clamp(1.35 - depth * 0.22, 0.25, 1.0);
  gl_Position = projectionMatrix * mv;
}`

const FRAGMENT = /* glsl */ `
precision mediump float;
varying vec3 vColor;
varying float vGlow;
varying float vFade;
void main() {
  vec2 d = gl_PointCoord - vec2(0.5);
  float r2 = dot(d, d);
  if (r2 > 0.25) discard;
  // Gaussian-ish falloff: this is what makes a point cloud read as glowing
  // neuropil instead of square pixels.
  float halo = exp(-r2 * 11.0);
  float alpha = halo * (0.16 + 0.26 * vFade + 0.85 * vGlow);
  gl_FragColor = vec4(vColor * (0.55 + 0.9 * vGlow) * vFade, alpha);
}`

function SolidBrain({ cloud, field, card }: Props) {
  const host = useRef<HTMLDivElement | null>(null)
  const state = useRef<{
    renderer?: THREE.WebGLRenderer
    scene?: THREE.Scene
    camera?: THREE.PerspectiveCamera
    points?: THREE.Points
    texture?: THREE.DataTexture
    frame?: number
    drag?: boolean
    last?: [number, number]
    spin?: number
    yaw?: number
    pitch?: number
    zoom?: number
  }>({})

  useEffect(() => {
    const element = host.current
    if (!element || !cloud) return
    const store = state.current
    store.renderer = new THREE.WebGLRenderer({ antialias: false, alpha: true })
    store.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1))
    element.appendChild(store.renderer.domElement)
    store.scene = new THREE.Scene()
    store.camera = new THREE.PerspectiveCamera(36, 1, 0.05, 20)
    store.yaw = 0.0
    store.pitch = 0.0
    store.zoom = 2.1
    store.spin = 0.1

    const palette = new Map((card?.superclasses ?? []).map((s) => [s.name, tissueColor(hexToRgb(s.color))]))
    const tints = Object.values(COMMAND_TINT)
    const positions = new Float32Array(cloud.count * 3)
    const colors = new Float32Array(cloud.count * 3)
    const probes = new Float32Array(cloud.count)
    const commands = new Float32Array(cloud.count)
    const jitter = new Float32Array(cloud.count)
    let maxX = 0
    let maxY = 0
    let maxZ = 0
    for (let i = 0; i < cloud.count; i += 1) {
      maxX = Math.max(maxX, cloud.xyz[i * 3])
      maxY = Math.max(maxY, cloud.xyz[i * 3 + 1])
      maxZ = Math.max(maxZ, cloud.xyz[i * 3 + 2])
    }
    for (let i = 0; i < cloud.count; i += 1) {
      // x across (left-right), y up (head at the top, so the A-P axis is
      // negated), z into the screen (dorsal-ventral).
      positions[i * 3] = cloud.xyz[i * 3] - maxX / 2
      positions[i * 3 + 1] = maxY / 2 - cloud.xyz[i * 3 + 1]
      positions[i * 3 + 2] = cloud.xyz[i * 3 + 2] - maxZ / 2
      probes[i] = cloud.probe[i]
      commands[i] = cloud.command[i]
      // Deterministic per-neuron size variation.
      jitter[i] = ((i * 2654435761) % 1000) / 1000
      const command = cloud.command[i]
      const rgb = command >= 0 ? tints[command] : palette.get(cloud.superclasses[cloud.kind[i]]) ?? TISSUE
      colors[i * 3] = rgb[0]
      colors[i * 3 + 1] = rgb[1]
      colors[i * 3 + 2] = rgb[2]
    }

    const [texW, texH] = probeTexture(cloud.nProbes)
    const data = new Uint8Array(texW * texH)
    const texture = new THREE.DataTexture(data, texW, texH, THREE.RedFormat, THREE.UnsignedByteType)
    texture.needsUpdate = true
    store.texture = texture

    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    geometry.setAttribute('aColor', new THREE.BufferAttribute(colors, 3))
    geometry.setAttribute('aProbe', new THREE.BufferAttribute(probes, 1))
    geometry.setAttribute('aCommand', new THREE.BufferAttribute(commands, 1))
    geometry.setAttribute('aJitter', new THREE.BufferAttribute(jitter, 1))
    const material = new THREE.ShaderMaterial({
      uniforms: {
        uSpikes: { value: texture },
        uTexSize: { value: new THREE.Vector2(texW, texH) },
        uMaxSpike: { value: 1 },
        uSize: { value: 0.0095 },
        uHeight: { value: 800 },
      },
      vertexShader: VERTEX,
      fragmentShader: FRAGMENT,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    })
    store.points = new THREE.Points(geometry, material)
    store.scene.add(store.points)

    const resize = () => {
      const w = element.clientWidth
      const h = element.clientHeight
      if (!w || !h || !store.renderer || !store.camera) return
      store.renderer.setSize(w, h, false)
      store.camera.aspect = w / h
      store.camera.updateProjectionMatrix()
      const material = store.points?.material as THREE.ShaderMaterial | undefined
      if (material) material.uniforms.uHeight.value = h * (store.renderer.getPixelRatio() || 1)
    }
    resize()
    const observer = new ResizeObserver(resize)
    observer.observe(element)

    const canvas = store.renderer.domElement
    const onDown = (event: PointerEvent) => {
      store.drag = true
      store.last = [event.clientX, event.clientY]
    }
    const onUp = () => {
      store.drag = false
    }
    const onMove = (event: PointerEvent) => {
      if (!store.drag || !store.last) return
      store.yaw = (store.yaw ?? 0) + (event.clientX - store.last[0]) * 0.008
      store.pitch = Math.max(-1.4, Math.min(1.4, (store.pitch ?? 0) + (event.clientY - store.last[1]) * 0.008))
      store.last = [event.clientX, event.clientY]
    }
    const onWheel = (event: WheelEvent) => {
      event.preventDefault()
      store.zoom = Math.max(0.5, Math.min(6, (store.zoom ?? 2.1) * (1 + Math.sign(event.deltaY) * 0.08)))
    }
    canvas.addEventListener('pointerdown', onDown)
    window.addEventListener('pointerup', onUp)
    canvas.addEventListener('pointermove', onMove)
    canvas.addEventListener('wheel', onWheel, { passive: false })

    let clock = performance.now()
    const loop = () => {
      const now = performance.now()
      const dt = Math.min(0.1, (now - clock) / 1000)
      clock = now
      if (!store.drag) store.yaw = (store.yaw ?? 0) + (store.spin ?? 0) * dt
      const zoom = store.zoom ?? 2.1
      const pitch = store.pitch ?? 0
      const yaw = store.yaw ?? 0
      if (store.camera) {
        store.camera.position.set(
          zoom * Math.cos(pitch) * Math.sin(yaw),
          zoom * Math.sin(pitch),
          zoom * Math.cos(pitch) * Math.cos(yaw),
        )
        store.camera.lookAt(0, 0, 0)
      }
      if (store.renderer && store.scene && store.camera) store.renderer.render(store.scene, store.camera)
      store.frame = requestAnimationFrame(loop)
    }
    loop()

    return () => {
      if (store.frame) cancelAnimationFrame(store.frame)
      observer.disconnect()
      canvas.removeEventListener('pointerdown', onDown)
      window.removeEventListener('pointerup', onUp)
      canvas.removeEventListener('pointermove', onMove)
      canvas.removeEventListener('wheel', onWheel)
      store.points?.geometry.dispose()
      ;(store.points?.material as THREE.ShaderMaterial | undefined)?.dispose()
      store.texture?.dispose()
      store.renderer?.dispose()
      canvas.remove()
      state.current = {}
    }
  }, [cloud, card])

  useEffect(() => {
    const { texture, points } = state.current
    if (!texture || !cloud) return
    const source = field ?? new Uint8Array(cloud.nProbes)
    const target = texture.image.data as Uint8Array
    target.set(source.subarray(0, Math.min(target.length, source.length)))
    texture.needsUpdate = true
    const material = points?.material as THREE.ShaderMaterial | undefined
    if (material) material.uniforms.uMaxSpike.value = Math.max(1, field ? Math.max(...field) : 1)
  }, [field, cloud])

  return <div ref={host} className="brain-canvas brain-canvas-3d" />
}

// --------------------------------------------------------------------- export

export function BrainView(props: Props) {
  const { mode, onMode, cloud, card } = props
  return (
    <div className={`brain-view brain-view-${mode}`}>
      {mode === '3d' ? <SolidBrain {...props} /> : <FlatBrain {...props} />}
      <div className="brain-view-hud">
        <div className="mode-toggle">
          <button type="button" className={mode === '2d' ? 'on' : ''} onClick={() => onMode('2d')} title="Projected soma map — cheap, always available">
            flat
          </button>
          <button type="button" className={mode === '3d' ? 'on' : ''} onClick={() => onMode('3d')} title="Real 3-D soma coordinates — drag to rotate, wheel to zoom">
            3-D
          </button>
        </div>
        {cloud && (
          <span className="brain-count">
            {cloud.count.toLocaleString()} somata · {cloud.nProbes.toLocaleString()} probed
            {card?.n_connections ? ` · ${(card.n_connections / 1e6).toFixed(1)}M synapses` : ''}
          </span>
        )}
      </div>
    </div>
  )
}