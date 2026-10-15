import Foundation
import AppKit
import SwiftUI
import MacVideoUpscalerCore

/// Hosts the player view the same way the GUI does, so harness runs exercise the
/// SwiftUI layer-hosting path rather than a bare NSWindow content view.
struct HostedSurface: NSViewRepresentable {
    let view: NSView
    func makeNSView(context: Context) -> NSView { view }
    func updateNSView(_ nsView: NSView, context: Context) {}
}

// Headless verification harness for the embedded player. It opens a file with
// PlayerController, optionally runs a small control script, and prints a status
// JSON line every 0.5 s so playback progress, buffer occupancy and the render
// clock can be checked without the GUI.
//
//   playercheck --file <video> [--seconds 30] [--mode cnn|bypass|metalfx|realesrgan] [--scale 2|4]
//               [--no-audio] [--seek 120] [--window] [--quiet] [--compare]
//               [--buffer-seconds 5] [--buffer-mb 1500]
//               [--script "5:pause,8:resume,12:seek:60,20:click"]
//
// Script verbs: pause, resume, click (a synthetic mouse click on the picture),
// seek:<s>, seekrel:<s>, mode:<cnn|bypass|denoise|metalfx>, scale:<2|4>,
// audio:<ordinal>, compare:<on|off>, compareleft:<cnn|denoise|bypass|metalfx>,
// subfile:<path>, sub:<off|ordinal>, open:<path>,
// split:<0..1>, dragdivider:<viewFraction> (a synthetic press-and-drag on the
// divider line), readback (write the frame the renderer is *showing* as a PNG —
// the paused redraw changes what is presented, which a dump of the frames rendered
// does not prove on its own).

let args = ProcessInfo.processInfo.arguments
func arg(_ name: String) -> String? {
    guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
    return args[i + 1]
}
func flag(_ name: String) -> Bool { args.contains(name) }

guard let filePath = arg("--file") else {
    FileHandle.standardError.write(Data("""
    usage: playercheck --file <video> [--seconds 30] [--mode cnn|bypass|metalfx|realesrgan|off] [--scale 2|4]
                       [--no-audio] [--seek 120] [--window] [--quiet] [--compare]
                       [--buffer-seconds 5] [--buffer-mb 1500]
                       [--script "5:pause,8:resume,12:seek:60,20:click"]
    """.utf8))
    exit(2)
}

var cfg = PlayerController.Config()
if flag("--no-audio") { cfg.audioEnabled = false }
// --mute keeps the audio path alive (decoder, timestamps, renderer) but sends it
// to a silent output, so an audio format can be checked without playing it out.
if flag("--mute") { cfg.volume = 0 }
if let v = arg("--scale"), let n = Int(v), let f = UpscaleFactor(rawValue: n) { cfg.scaleFactor = f }
if flag("--subtitles") { cfg.burnSubtitles = true }
if let v = arg("--sub-index"), let i = Int(v) { cfg.subtitleStreamIndex = i; cfg.burnSubtitles = true }
if let v = arg("--audio-index"), let i = Int(v) { cfg.audioStreamIndex = i }
// Read-ahead, in rendered seconds and capped in bytes. The point of the pair is
// that a mode slower than real time can be measured with and without a buffer:
// with a large one it plays in bursts and waits between them, with a small one it
// stutters every frame.
if let v = arg("--buffer-seconds"), let s = Double(v) { cfg.bufferSeconds = s }
if let v = arg("--buffer-mb"), let mb = Int(v) { cfg.bufferBudgetBytes = mb * 1_000_000 }

let kind = ScalerKind.parse(arg("--mode"))

let quiet = flag("--quiet")
let runSeconds = Double(arg("--seconds") ?? "30") ?? 30

// AppKit is needed even headless: the display layer lives in an NSView. A window
// is only ordered front when --window is passed; the layer consumes samples
// either way (that is what this harness verifies).
let app = NSApplication.shared
app.setActivationPolicy(flag("--window") ? .regular : .prohibited)

let controller: PlayerController
do {
    controller = try PlayerController(config: cfg)
} catch {
    FileHandle.standardError.write(Data("init failed: \(error)\n".utf8))
    exit(1)
}

let view = controller.makeView()
var window: NSWindow?
if flag("--swiftui") {
    // Exactly the GUI's hosting: SwiftUI owns the window and wraps our view in
    // an NSHostingView. With --window it is on screen, like the real app.
    let visible = flag("--window")
    let host = NSWindow(contentRect: NSRect(x: visible ? 80 : -3000, y: visible ? 80 : -3000,
                                            width: 960, height: 540),
                        styleMask: visible ? [.titled, .closable, .resizable] : [.borderless],
                        backing: .buffered, defer: false)
    host.title = "playercheck (swiftui)"
    host.contentView = NSHostingView(rootView:
        HostedSurface(view: view).frame(width: 960, height: 540))
    if visible { host.makeKeyAndOrderFront(nil) } else { host.orderBack(nil) }
    window = host
} else if flag("--window") {
    let w = NSWindow(contentRect: NSRect(x: 80, y: 80, width: 960, height: 540),
                     styleMask: [.titled, .closable, .resizable],
                     backing: .buffered, defer: false)
    w.title = "playercheck"
    w.contentView = view
    w.makeKeyAndOrderFront(nil)
    window = w
} else {
    // No window: put the view in an offscreen container so the layer has a
    // geometry, without putting anything on screen.
    let host = NSWindow(contentRect: NSRect(x: -3000, y: -3000, width: 960, height: 540),
                        styleMask: [.borderless], backing: .buffered, defer: false)
    host.contentView = view
    host.orderBack(nil)
    window = host
}
_ = window
if flag("--window") {
    app.finishLaunching()
    window?.makeKeyAndOrderFront(nil)
    app.activate(ignoringOtherApps: true)
}

var lastStatus: StatusSnapshot?

// Same wiring as the GUI: clicking the picture toggles playback. Scripts use the
// "click" verb, which sends a real press/release to the view so the mouse path
// (PlayerView -> onSurfaceClick) is exercised rather than just the callback.
controller.onSurfaceClick = { [weak controller] in
    guard let controller else { return }
    if controller.isPaused { controller.resume() } else { controller.pause() }
}

// --compare: start in the side-by-side source/enhanced view. The dumped frames
// then contain both halves, so the left one can be checked against a bypass run.
if flag("--compare") { controller.setCompare(true) }

func synthesizeClick(on target: NSView) {
    let windowNumber = target.window?.windowNumber ?? 0
    for type in [NSEvent.EventType.leftMouseDown, .leftMouseUp] {
        guard let event = NSEvent.mouseEvent(
            with: type, location: NSPoint(x: 100, y: 100), modifierFlags: [],
            timestamp: ProcessInfo.processInfo.systemUptime, windowNumber: windowNumber,
            context: nil, eventNumber: 0, clickCount: 1,
            pressure: type == .leftMouseDown ? 1 : 0) else { continue }
        if type == .leftMouseDown { target.mouseDown(with: event) }
        else { target.mouseUp(with: event) }
    }
}

/// Presses the divider where it currently sits and drags it to `viewFraction` of
/// the way across the surface, then releases. Sends the three events a real drag
/// would, so the view's grab-and-move path is exercised rather than the setter.
func dragDivider(to viewFraction: CGFloat, on target: NSView, from startViewX: CGFloat) {
    guard let windowNumber = target.window?.windowNumber else { return }
    let y = target.bounds.midY
    // View coordinates in, window coordinates out: that is what mouseDown does in
    // reverse when it reads `event.locationInWindow`.
    let start = target.convert(NSPoint(x: startViewX, y: y), to: nil)
    let end = target.convert(NSPoint(x: target.bounds.minX + viewFraction * target.bounds.width,
                                     y: y), to: nil)
    let steps: [(NSEvent.EventType, NSPoint, Float)] = [
        (.leftMouseDown, start, 1.0),
        (.leftMouseDragged, end, 1.0),
        (.leftMouseUp, end, 0.0),
    ]
    for (type, point, pressure) in steps {
        guard let event = NSEvent.mouseEvent(
            with: type, location: point, modifierFlags: [],
            timestamp: ProcessInfo.processInfo.systemUptime, windowNumber: windowNumber,
            context: nil, eventNumber: 0, clickCount: 1,
            pressure: pressure) else { continue }
        switch type {
        case .leftMouseDown: target.mouseDown(with: event)
        case .leftMouseDragged: target.mouseDragged(with: event)
        default: target.mouseUp(with: event)
        }
    }
}
// Optional frame dump: write the first N output frames as PNGs. Lets a run be
// inspected without a window (this environment has no Screen Recording access).
let dumpDir = URL(fileURLWithPath: arg("--dump-dir") ?? NSTemporaryDirectory())
// --dump <n>: first n frames. --dump-at "60,150": specific frame indices.
// --dump-sessions: prefix captures with their session ID across seeks/restarts.
let dumpFirst = arg("--dump").flatMap(Int.init)
let dumpAt = Set((arg("--dump-at") ?? "").split(separator: ",").compactMap { Int($0) })
if dumpFirst != nil || !dumpAt.isEmpty {
    // Create it: `writePNG` returns false on a missing directory and this only
    // logs on success, so a not-yet-existing --dump-dir looked like the dump had
    // never fired rather than like the write failing.
    try? FileManager.default.createDirectory(at: dumpDir, withIntermediateDirectories: true)
    var dumped = 0
    controller.onOutputFrame = { pb, idx in
        let wanted = dumpAt.contains(idx) || (dumpFirst.map { dumped < $0 } ?? false)
        guard wanted else { return }
        dumped += 1
            let prefix = flag("--dump-sessions")
                ? String(format: "session_%04d_", controller.status.current.sessionId) : ""
            let url = dumpDir.appendingPathComponent(prefix + String(format: "frame_%04d.png", idx))
        if FrameCapture.writePNG(pb, to: url) {
            FileHandle.standardError.write(Data("[dump] \(url.path)\n".utf8))
        }
    }
}
controller.status.on_change = { snap in
    lastStatus = snap
    guard !quiet else { return }
    let enc = JSONEncoder()
    enc.outputFormatting = [.sortedKeys]
    if let line = try? enc.encode(snap), var s = String(data: line, encoding: .utf8) {
        s.append("\n")
        FileHandle.standardOutput.write(s.data(using: .utf8)!)
    }
}

// Optional control script, same grammar as the stream CLI.
var controlTimer: DispatchSourceTimer?
/// How many `readback` frames have been written, so each gets its own file.
var readbacks = 0
if let script = arg("--script") {
    struct Action { let at: Double; let verb: String; let value: String?; var fired = false }
    var actions: [Action] = script.split(separator: ",").compactMap { item in
        let parts = item.split(separator: ":", maxSplits: 2)
        // Count checked before the subscript: an item with no colon (or only
        // colons) must be skipped, not trap on an out-of-range index.
        guard parts.count >= 2, let at = Double(parts[0]) else { return nil }
        return Action(at: at, verb: String(parts[1]),
                      value: parts.count > 2 ? String(parts[2]) : nil)
    }
    let started = Date()
    let timer = DispatchSource.makeTimerSource(queue: .main)
    timer.schedule(deadline: .now(), repeating: 0.1)
    timer.setEventHandler {
        let elapsed = Date().timeIntervalSince(started)
        for i in actions.indices where !actions[i].fired && elapsed >= actions[i].at {
            actions[i].fired = true
            let a = actions[i]
            let before = controller.playbackSeconds
            let statusBefore = controller.status.current
            var controlError = ""
            switch a.verb {
            case "pause": controller.pause()
            case "resume": controller.resume()
            case "click": synthesizeClick(on: view)
            case "seek": controller.seek(to: Double(a.value ?? "0") ?? 0)
            case "seekrel": controller.seek(by: Double(a.value ?? "0") ?? 0)
            case "mode": controller.setKind(ScalerKind.parse(a.value))
            case "audio": controller.setAudioTrack(Int(a.value ?? "0") ?? 0)
            case "subfile":
                do {
                    if let video = controller.currentFile {
                        try controller.loadSubtitleFile(URL(fileURLWithPath: a.value ?? ""), for: video)
                    }
                }
                catch { controlError = error.localizedDescription }
            case "sub":
                if a.value == "off" { controller.setSubtitles(false) }
                else { controller.setSubtitleTrack(Int(a.value ?? "0") ?? 0) }
            case "open":
                do { try controller.open(file: URL(fileURLWithPath: a.value ?? ""), kind: kind) }
                catch { controlError = error.localizedDescription }
            case "compare": controller.setCompare((a.value ?? "on") != "off")
            case "compareleft": controller.setCompareLeft(ScalerKind.parse(a.value))
            case "split": controller.setCompareSplit(Double(a.value ?? "0.5") ?? 0.5)
            // dragdivider:0.25 — a real mouse press on the divider, dragged to a
            // new position. Exercises the view's grab-and-move path, not just the
            // setter, which is the part that could steal the window drag.
            case "dragdivider":
                if let x = controller.dividerXInView {
                    dragDivider(to: CGFloat(Double(a.value ?? "0.5") ?? 0.5), on: view, from: x)
                }
            case "scale":
                if let n = Int(a.value ?? "2"), let f = UpscaleFactor(rawValue: n) {
                    controller.setScaleFactor(f)
                }
            // readback: dump the image the display layer is showing right now.
            // Distinct from --dump-at, which writes the frames the producer
            // rendered: those can differ from what is on screen while paused.
            case "readback":
                if let pb = controller.displayedFrame {
                    let url = dumpDir.appendingPathComponent(
                        String(format: "displayed_%03d.png", readbacks))
                    readbacks += 1
                    let ok = FrameCapture.writePNG(pb, to: url)
                    FileHandle.standardError.write(Data(
                        "[readback] \(url.path) \(CVPixelBufferGetWidth(pb))x\(CVPixelBufferGetHeight(pb)) ok=\(ok)\n".utf8))
                } else {
                    FileHandle.standardError.write(Data(
                        "[readback] no presented frame active=\(app.isActive) visible=\(window?.occlusionState.contains(.visible) == true) \(controller.status.current.renderStatus)\n".utf8))
                }
            default: break
            }
            let ev: [String: Any] = [
                "event": a.verb, "at": a.at, "wall": Date().timeIntervalSince(started),
                "value": a.value ?? "",
                "playback_before": before,
                "playback_after": controller.playbackSeconds,
                "state_before": statusBefore.state.rawValue,
                "error": controlError,
                "audio_frames_before": statusBefore.audioFramesEnqueued,
                "session_before": statusBefore.sessionId,
                "frames": controller.status.current.framesProcessed,
                "state": controller.status.current.state.rawValue,
                "compare_split": controller.compareSplitFraction,
            ]
            if let d = try? JSONSerialization.data(withJSONObject: ev, options: [.sortedKeys]) {
                FileHandle.standardOutput.write(d + Data("\n".utf8))
            }
        }
    }
    timer.resume()
    controlTimer = timer  // top-level var keeps the source alive for the process
}

do {
    try controller.open(file: URL(fileURLWithPath: filePath), kind: kind,
                        startAt: Double(arg("--seek") ?? "0") ?? 0)
} catch {
    FileHandle.standardError.write(Data("open failed: \(error)\n".utf8))
    exit(1)
}

signal(SIGINT, SIG_IGN)
let sigSrc = DispatchSource.makeSignalSource(signal: SIGINT, queue: .main)
sigSrc.setEventHandler { controller.stop(); exit(0) }
sigSrc.resume()

let deadline = Date().addingTimeInterval(runSeconds)
while Date() < deadline, controller.status.current.state != .failed {
    RunLoop.main.run(until: Date().addingTimeInterval(0.1))
}

let final = controller.status.current
let summary: [String: Any] = [
    "final_state": final.state.rawValue,
    "frames": final.framesProcessed,
    "position": final.positionSeconds,
    "playback": final.playbackSeconds,
    "duration": final.durationSeconds,
    "buffer_frames": final.bufferFrames,
    "buffer_ms": final.bufferMilliseconds,
                "audio_frames": final.audioFramesEnqueued,
    "comparing": final.comparing,
    "compare_left": final.compareLeftMode,
    "compare_split": final.compareSplit,
    "realtime_ratio": final.realtimeRatio,
    "memory_mb": final.memoryFootprintMB,
    "output": "\(final.outputWidth)x\(final.outputHeight)",
    "message": final.message,
]
if let d = try? JSONSerialization.data(withJSONObject: summary, options: [.sortedKeys]) {
    FileHandle.standardOutput.write(d + Data("\n".utf8))
}
controller.stop()
exit(final.state == .failed ? 1 : 0)
