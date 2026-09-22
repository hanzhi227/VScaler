import Foundation
import AppKit
import SwiftUI
import CoreVideo
import UniformTypeIdentifiers
import MacVideoUpscalerCore

@main
struct UpscalerApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var delegate
    @StateObject private var model = AppModel()

    var body: some Scene {
        WindowGroup("VScaler") {
            PlayerScreen()
                .environmentObject(model)
                .frame(minWidth: 480, minHeight: 270)
        }
        // No title bar: the picture is the interface. Controls live in the menu
        // bar, so the window (and full screen) is nothing but video.
        .windowStyle(.hiddenTitleBar)
        .defaultSize(width: 1100, height: 700)
        .commands { PlayerCommands(model: model) }
    }
}

// MARK: - Bridge into the engine

@MainActor
final class AppModel: ObservableObject {
    @Published var status = StatusSnapshot()
    @Published var lastError: String?
    /// Menu state (kept here so the menu bar can show checkmarks).
    @Published var kind: ScalerKind = ScalerKind.defaultKind
    @Published var factor: UpscaleFactor = .x2
    @Published var audioOrdinal = 0
    /// -1 = subtitles off.
    @Published var subOrdinal = -1
    @Published var showStats = false
    /// Side-by-side comparison: which model fills the left of the divider, and
    /// where the divider sits (0 = enhancement only, 1 = left model only). The
    /// right half is `kind` — the enhancement itself.
    @Published var comparing = false
    @Published var compareLeft: ScalerKind = .bypass
    @Published var compareSplit: Double = 0.5
    /// Whether the window is in full screen, so the menu item can say which way
    /// it will go and the diagnostics can report it.
    @Published var isFullScreen = false
    @Published private(set) var queue: [URL] = []
    @Published private(set) var currentIndex: Int?

    nonisolated(unsafe) private(set) var engine: PlayerController?
    nonisolated(unsafe) static var engineForShutdown: PlayerController?
    /// The live model, so the app delegate can hand it a file the Finder asked to
    /// open. Set at the end of `init`.
    nonisolated(unsafe) static var instance: AppModel?
    let playerView: NSView?

    private var diagTimer: DispatchSourceTimer?
    private var queueTimer: DispatchSourceTimer?
    private var displaySleepActivity: NSObjectProtocol?
    private var handledSessionId = 0
    private var clickTimers: [DispatchSourceTimer] = []

    init() {
        var ctrl: PlayerController?
        var view: NSView?
        var err: String?
        do {
            let c = try PlayerController()
            ctrl = c
            view = c.makeView()
        } catch {
            err = "\(error)"
        }
        engine = ctrl
        playerView = view
        lastError = err
        AppModel.engineForShutdown = ctrl
        ctrl?.status.on_change = { [weak self] snap in
            Task { @MainActor in self?.apply(snap) }
        }
        // Clicking the picture toggles playback, like any other player.
        ctrl?.onSurfaceClick = { [weak self] in
            Task { @MainActor in self?.togglePlayPause() }
        }
        // Double clicking it toggles full screen. The engine holds the single click
        // back long enough to tell the two apart, so this does not also pause.
        ctrl?.onSurfaceDoubleClick = { [weak self] in
            Task { @MainActor in self?.toggleFullScreen() }
        }
        observeFullScreen()
        applyLaunchDiagnostics()
        observeQueueEnd()
        AppModel.instance = self
        // A file opened from the Finder can arrive before this scene exists; the
        // delegate parks it, so collect it now (a real double-click wins over the
        // diagnostic env hook above).
        if let delegate = NSApp.delegate as? AppDelegate, !delegate.pendingOpen.isEmpty {
            let pending = delegate.pendingOpen
            delegate.pendingOpen = []
            DispatchQueue.main.async { self.addToQueue(pending) }
        }
    }

    private func apply(_ snap: StatusSnapshot) {
        status = snap
        updateDisplaySleepActivity()
        // Track selections follow the engine (it may fall back to the first track),
        // and so does the mode — a failed scaler swap reverts it, so the menu
        // checkmark must follow the engine rather than the last request.
        audioOrdinal = snap.audioTrackOrdinal
        subOrdinal = snap.subtitlesEnabled ? snap.subtitleTrackOrdinal : -1
        factor = UpscaleFactor(rawValue: snap.requestedScaleFactor) ?? factor
        comparing = snap.comparing
        compareSplit = snap.compareSplit
        if let k = ScalerKind.allCases.first(where: { $0.displayName == snap.compareLeftMode }) {
            compareLeft = k
        }
        if let k = ScalerKind.allCases.first(where: { $0.displayName == snap.mode }) { kind = k }
        if snap.state == .processing && snap.realtimeRatio < 0.95 && snap.framesProcessed > 60 {
            lastError = String(format: "Enhancing at %.2f× realtime — the picture will stutter. "
                               + "Switch to Bypass or 2×.", snap.realtimeRatio)
        } else {
            lastError = nil
        }
    }

    // MARK: file

    func openFilePanel() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        // Anything the system types as a movie. The list used to name four
        // containers, which greyed out everything else the engine can read
        // (webm, ts, flv, wmv, mpg, ...). Every one of those conforms to
        // public.movie, so this admits them all without also admitting audio-only
        // files.
        panel.allowedContentTypes = [.movie]
        panel.message = "Choose videos to add to the queue"
        guard panel.runModal() == .OK else { return }
        addToQueue(panel.urls)
    }

    func openFolderPanel() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.message = "Choose a folder of videos to add to the queue"
        guard panel.runModal() == .OK, let folder = panel.url else { return }
        lastError = nil
        let found = videos(in: folder)
        if found.isEmpty {
            if lastError == nil { lastError = "No videos found in the selected folder." }
        } else {
            addToQueue(found)
        }
    }

    private func videos(in folder: URL) -> [URL] {
        let keys: Set<URLResourceKey> = [.isRegularFileKey, .isDirectoryKey]
        guard let files = FileManager.default.enumerator(at: folder, includingPropertiesForKeys: Array(keys),
                                                         options: [.skipsHiddenFiles, .skipsPackageDescendants]) else {
            lastError = "Could not read the selected folder."
            return []
        }
        let extensions: Set<String> = ["mp4", "m4v", "mov", "mkv", "webm", "ts", "m2ts", "mts",
                                       "flv", "avi", "divx", "mpg", "mpeg", "vob", "mxf", "wmv",
                                       "asf", "3gp", "ogv", "rm", "rmvb"]
        return files.compactMap { $0 as? URL }
            .filter { url in
                let ext = url.pathExtension.lowercased()
                guard extensions.contains(ext)
                    || UTType(filenameExtension: ext)?.conforms(to: .movie) == true else { return false }
                return (try? url.resourceValues(forKeys: keys).isRegularFile) == true
            }
            .sorted { $0.path.localizedStandardCompare($1.path) == .orderedAscending }
    }

    func addToQueue(_ urls: [URL]) {
        let expanded = urls.flatMap { url -> [URL] in
            if (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true {
                return videos(in: url)
            }
            return [url]
        }
        let existing = Set(queue.map(\.standardizedFileURL))
        var seen = existing
        let added = expanded.filter { url in
            let file = url.standardizedFileURL
            guard !seen.contains(file) else { return false }
            seen.insert(file)
            return true
        }
        guard !added.isEmpty else {
            if queue.isEmpty { lastError = "No videos found to add." }
            return
        }
        queue.append(contentsOf: added)
        if currentIndex == nil { playQueued(at: 0) }
    }

    func open(_ url: URL) { addToQueue([url]) }

    func playQueued(at index: Int) {
        guard queue.indices.contains(index) else { return }
        guard let c = engine else { return }
        currentIndex = index
        handledSessionId = status.sessionId
        do {
            c.config.scaleFactor = factor
            c.config.audioStreamIndex = audioOrdinal
            c.config.burnSubtitles = subOrdinal >= 0
            if subOrdinal >= 0 { c.config.subtitleStreamIndex = subOrdinal }
            try c.open(file: queue[index], kind: kind)
            lastError = nil
        } catch {
            lastError = "\(error)"
        }
    }

    func nextInQueue() {
        guard let currentIndex, queue.indices.contains(currentIndex + 1) else { return }
        playQueued(at: currentIndex + 1)
    }

    func previousInQueue() {
        guard let currentIndex else { return }
        playQueued(at: max(0, currentIndex - 1))
    }

    func removeQueued(at index: Int) {
        guard queue.indices.contains(index) else { return }
        queue.remove(at: index)
        guard let currentIndex else { return }
        if index < currentIndex {
            self.currentIndex = currentIndex - 1
        } else if index == currentIndex {
            if queue.isEmpty {
                self.currentIndex = nil
                engine?.stop()
            } else {
                playQueued(at: min(index, queue.count - 1))
            }
        }
    }

    func clearQueue() {
        queue = []
        currentIndex = nil
        engine?.stop()
        lastError = nil
    }

    private func observeQueueEnd() {
        let timer = DispatchSource.makeTimerSource(queue: .main)
        timer.schedule(deadline: .now() + 0.25, repeating: 0.25)
        timer.setEventHandler { [weak self] in
            MainActor.assumeIsolated {
                self?.updateDisplaySleepActivity()
                guard let self, let currentIndex = self.currentIndex,
                      self.queue.indices.contains(currentIndex + 1),
                      self.status.state == .finished,
                      self.status.sessionId > self.handledSessionId,
                      let engine = self.engine else { return }
                let end = self.status.presentationEndSeconds
                guard end > 0, engine.playbackSeconds >= end - 0.02 else { return }
                self.handledSessionId = self.status.sessionId
                self.nextInQueue()
            }
        }
        timer.resume()
        queueTimer = timer
    }

    /// AVSampleBufferRenderSynchronizer does not keep the display awake like a
    /// system video player. Hold an activity only while the viewer intends to
    /// watch; a finished feeder still has frames on the display layer until its
    /// presentation clock reaches the last frame.
    private func updateDisplaySleepActivity() {
        let playing: Bool
        if let engine, !engine.isPaused {
            switch status.state {
            case .preparing, .processing:
                playing = true
            case .finished:
                playing = status.presentationEndSeconds > 0
                    && engine.playbackSeconds < status.presentationEndSeconds
            case .idle, .paused, .failed:
                playing = false
            }
        } else {
            playing = false
        }
        if playing && displaySleepActivity == nil {
            displaySleepActivity = ProcessInfo.processInfo.beginActivity(
                options: [.userInitiated, .idleDisplaySleepDisabled],
                reason: "Playing video in VScaler")
        } else if !playing, let activity = displaySleepActivity {
            ProcessInfo.processInfo.endActivity(activity)
            displaySleepActivity = nil
        }
    }

    // MARK: controls

    func togglePlayPause() {
        guard let c = engine, status.state != .idle, status.state != .failed else { return }
        if c.isPaused { c.resume() } else { c.pause() }
    }

    func seek(by delta: Double) {
        if ProcessInfo.processInfo.environment["MACVIDEOUPSCALER_DIAG"] == "1" {
            NSLog("[seek] request delta=%.1f from=%.2f", delta, engine?.playbackSeconds ?? -1)
        }
        engine?.seek(by: delta)
    }
    func seek(to seconds: Double) { engine?.seek(to: seconds) }

    func setKind(_ k: ScalerKind) { kind = k; engine?.setKind(k) }

    func setFactor(_ f: UpscaleFactor) { factor = f; engine?.setScaleFactor(f) }

    /// Side-by-side comparison: the chosen model on the left, the enhancement on
    /// the right, and a divider you can move anywhere between them.
    func toggleCompare() { setCompare(!comparing) }

    func setCompare(_ on: Bool) {
        comparing = on
        // Turning it on re-applies the chosen model and split, so a comparison
        // that was switched off and on again comes back as it was left.
        engine?.setCompareLeft(compareLeft)
        engine?.setCompareSplit(compareSplit)
        engine?.setCompare(on)
    }

    func setCompareLeft(_ k: ScalerKind) {
        compareLeft = k
        engine?.setCompareLeft(k)
    }

    func setCompareSplit(_ fraction: Double) {
        let clamped = min(max(fraction, 0), 1)
        compareSplit = clamped
        engine?.setCompareSplit(clamped)
    }

    /// The name to show for a model in the comparison, where the bypass path is
    /// what the original video looks like: the source with no enhancement at all,
    /// upscaled by a plain bilinear filter.
    static func compareName(_ k: ScalerKind) -> String {
        k == .bypass ? "Original (no enhancement)" : k.displayName
    }

    func setAudioTrack(_ ordinal: Int) { audioOrdinal = ordinal; engine?.setAudioTrack(ordinal) }

    func setSubtitleTrack(_ ordinal: Int) {
        subOrdinal = ordinal
        if ordinal < 0 { engine?.setSubtitles(false) } else { engine?.setSubtitleTrack(ordinal) }
    }

    func toggleFullScreen() { NSApp.keyWindow?.toggleFullScreen(nil) }

    /// Track full screen from the window's own notifications rather than from the
    /// calls that request it: entering and leaving are asynchronous and can also be
    /// driven by the green button, ⌃⌘F or a swipe, so the notification is the only
    /// thing that knows the real state.
    private func observeFullScreen() {
        let center = NotificationCenter.default
        let pairs: [(NSNotification.Name, Bool)] = [
            (NSWindow.didEnterFullScreenNotification, true),
            (NSWindow.didExitFullScreenNotification, false),
        ]
        for (name, entered) in pairs {
            center.addObserver(forName: name, object: nil, queue: .main) { [weak self] _ in
                MainActor.assumeIsolated { self?.isFullScreen = entered }
            }
        }
    }

    func stop() {
        engine?.stop()
        if let activity = displaySleepActivity {
            ProcessInfo.processInfo.endActivity(activity)
            displaySleepActivity = nil
        }
    }

    // MARK: diagnostics (opt-in; see applyLaunchDiagnostics)

    private func applyLaunchDiagnostics() {
        let env = ProcessInfo.processInfo.environment
        // Each hook stands alone, so for example `_DIAG=1` alone still logs a run
        // whose file was opened by double-clicking it in the Finder.
        guard let c = engine else { return }
        if let s = env["MACVIDEOUPSCALER_SCALE"], let n = Int(s), let f = UpscaleFactor(rawValue: n) {
            factor = f
        }
        if let s = env["MACVIDEOUPSCALER_MODE"] { kind = ScalerKind.parse(s) }
        if let s = env["MACVIDEOUPSCALER_AUDIO"], let n = Int(s) { audioOrdinal = n }
        if let s = env["MACVIDEOUPSCALER_SUB"], let n = Int(s) { subOrdinal = n }
        if env["MACVIDEOUPSCALER_STATS"] == "1" { showStats = true }
        // Silence for a diagnostic run: the film is still decoded and fed to the
        // renderer, it just has nowhere to be heard. Without this a harness-driven
        // app plays the film out loud, which is not something a test should do to
        // whoever is using the machine.
        if env["MACVIDEOUPSCALER_MUTE"] == "1" { c.config.volume = 0 }
        if let s = env["MACVIDEOUPSCALER_SPLIT"], let f = Double(s) { compareSplit = f }
        if let s = env["MACVIDEOUPSCALER_COMPARE_LEFT"] { compareLeft = ScalerKind.parse(s) }
        if env["MACVIDEOUPSCALER_COMPARE"] == "1" { setCompare(true) }

        if let list = env["MACVIDEOUPSCALER_DUMP_AT"] {
            let indices = Set(list.split(separator: ",").compactMap { Int($0) })
            if !indices.isEmpty {
                let dir = URL(fileURLWithPath: env["MACVIDEOUPSCALER_DUMP_DIR"] ?? NSTemporaryDirectory())
                c.onOutputFrame = { pb, idx in
                    guard indices.contains(idx) else { return }
                    let url = dir.appendingPathComponent(String(format: "app_frame_%04d.png", idx))
                    if FrameCapture.writePNG(pb, to: url) {
                        NSLog("[diag] dumped %@", url.path)
                    }
                }
            }
        }
        if env["MACVIDEOUPSCALER_DIAG"] == "1" {
            let timer = DispatchSource.makeTimerSource(queue: .main)
            timer.schedule(deadline: .now() + 1, repeating: 1)
            timer.setEventHandler { [weak self, weak c] in
                MainActor.assumeIsolated {
                    guard let c else { return }
                    let s = c.status.current
                    let line = "[diag] file=\(s.file) session=\(s.sessionId) "
                        + "state=\(s.state.rawValue) frames=\(s.framesProcessed) "
                        + "playback=\(String(format: "%.2f", s.playbackSeconds)) "
                        + "out=\(s.sourceWidth)x\(s.sourceHeight)->\(s.outputWidth)x\(s.outputHeight) "
                        + "audio=\(String(format: "%.2f", Double(s.audioFramesEnqueued) / 48000)) "
                        + "full=\(self?.isFullScreen == true ? 1 : 0) "
                        + "cmp=\(s.comparing ? 1 : 0) "
                        + "skip=\(String(format: "%.0f", s.pendingSkipSeconds)) "
                        + "raw=\(s.renderStatus)"
                    NSLog("%@", line)
                }
            }
            timer.resume()
            diagTimer = timer
        }
        // _CLICK_AT="6,10" / _DBLCLICK_AT="12": click the centre of the picture at
        // those seconds, once or twice. The events go through the app's own
        // dispatch (`NSApp.sendEvent`), so they exercise the real window
        // hit-testing — the only way to check that a click reaches the video
        // surface and not the SwiftUI overlays above it, since neither
        // Accessibility nor Screen Recording is granted here and so a click
        // cannot be posted from outside the process.
        if let list = env["MACVIDEOUPSCALER_CLICK_AT"] { scheduleClicks(list, count: 1) }
        if let list = env["MACVIDEOUPSCALER_DBLCLICK_AT"] { scheduleClicks(list, count: 2) }
        // _SPLIT_AT="3.5:0.25,6:0.75": move the comparison divider at those
        // seconds, through the same `setCompareSplit` the slider, the menu
        // presets and a drag on the line call. Exists because the paused
        // comparison is only half-proven by the harness: that the app's own
        // control reaches the engine — and redraws the frozen frame — can only be
        // shown in the app.
        if let list = env["MACVIDEOUPSCALER_SPLIT_AT"] { scheduleSplits(list) }
        // _KEY_AT="8:right,12:left,16:space": press those keys at those seconds.
        // Sent through the app's own dispatch, so menu key equivalents are
        // exercised exactly the way a real keyboard exercises them.
        if let list = env["MACVIDEOUPSCALER_KEY_AT"] { scheduleKeys(list, throughMenu: false) }
        if let list = env["MACVIDEOUPSCALER_KEYEQ_AT"] { scheduleKeys(list, throughMenu: true) }
        if env["MACVIDEOUPSCALER_MENU"] == "1" {
            let timer = DispatchSource.makeTimerSource(queue: .main)
            timer.schedule(deadline: .now() + 2, repeating: 4)
            timer.setEventHandler { [weak self] in
                MainActor.assumeIsolated {
                    NSApp.mainMenu?.update()
                    self?.dumpMenu()
                }
            }
            timer.resume()
            clickTimers.append(timer)
        }
        if let openPath = env["MACVIDEOUPSCALER_OPEN"] {
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.8) { [weak self] in
                guard let self else { return }
                self.addToQueue([URL(fileURLWithPath: openPath)])
            }
        }
        if let paths = env["MACVIDEOUPSCALER_QUEUE"] {
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.8) { [weak self] in
                let urls = paths.split(separator: "|").map { URL(fileURLWithPath: String($0)) }
                self?.addToQueue(urls)
            }
        }
    }

    private func scheduleClicks(_ list: String, count: Int) {        let times = list.split(separator: ",").compactMap { Double($0) }
        let started = Date()
        var fired = Set<Double>()
        let timer = DispatchSource.makeTimerSource(queue: .main)
        timer.schedule(deadline: .now() + 0.3, repeating: 0.3)
        timer.setEventHandler { [weak self] in
            MainActor.assumeIsolated {
                guard let self else { return }
                let elapsed = Date().timeIntervalSince(started)
                for t in times where !fired.contains(t) && elapsed >= t {
                    fired.insert(t)
                    self.sendClick(count)
                }
            }
        }
        timer.resume()
        clickTimers.append(timer)
    }

    /// Moves the comparison divider at the given seconds. The value goes through
    /// `setCompareSplit`, so what this exercises is the slider's own path — and,
    /// with the picture paused, the redraw that has to follow it.
    private func scheduleSplits(_ list: String) {
        let actions: [(Double, Double)] = list.split(separator: ",").compactMap { item in
            let parts = item.split(separator: ":")
            guard parts.count == 2, let at = Double(parts[0]), let value = Double(parts[1]) else {
                return nil
            }
            return (at, value)
        }
        let started = Date()
        var fired = Set<Double>()
        let timer = DispatchSource.makeTimerSource(queue: .main)
        timer.schedule(deadline: .now() + 0.2, repeating: 0.2)
        timer.setEventHandler { [weak self] in
            MainActor.assumeIsolated {
                guard let self else { return }
                let elapsed = Date().timeIntervalSince(started)
                for (at, value) in actions where !fired.contains(at) && elapsed >= at {
                    fired.insert(at)
                    self.setCompareSplit(value)
                    NSLog("[diag] split=%.2f state=%@ comparing=%d", value,
                          self.status.state.rawValue, self.comparing ? 1 : 0)
                }
            }
        }
        timer.resume()
        clickTimers.append(timer)
    }

    /// A double click is two press/release pairs with the second stamped
    /// `clickCount: 2`, which is how AppKit reports one.
    private func sendClick(_ count: Int) {
        guard let win = NSApp.windows.first(where: { $0.isVisible }),
              let content = win.contentView else { return }
        let point = NSPoint(x: content.bounds.midX, y: content.bounds.midY)
        let hit = content.hitTest(point)
        let location = content.convert(point, to: nil)
        for pass in 1...max(1, count) {
            for type in [NSEvent.EventType.leftMouseDown, .leftMouseUp] {
                guard let event = NSEvent.mouseEvent(
                    with: type, location: location, modifierFlags: [],
                    timestamp: ProcessInfo.processInfo.systemUptime, windowNumber: win.windowNumber,
                    context: nil, eventNumber: 0, clickCount: pass,
                    pressure: type == .leftMouseDown ? 1 : 0) else { continue }
                NSApp.sendEvent(event)
            }
        }
        let line = "[click] count=\(count) "
            + "hit=\(hit.map { String(describing: type(of: $0)) } ?? "nothing") "
            + "state=\(status.state.rawValue) full=\(isFullScreen ? 1 : 0)"
        NSLog("%@", line)
    }

    private func scheduleKeys(_ list: String, throughMenu: Bool) {
        let actions: [(Double, String)] = list.split(separator: ",").compactMap { item in
            let parts = item.split(separator: ":")
            guard parts.count == 2, let at = Double(parts[0]) else { return nil }
            return (at, String(parts[1]))
        }
        let started = Date()
        var fired = Set<Double>()
        let timer = DispatchSource.makeTimerSource(queue: .main)
        timer.schedule(deadline: .now() + 0.2, repeating: 0.2)
        timer.setEventHandler { [weak self] in
            MainActor.assumeIsolated {
                guard let self else { return }
                let elapsed = Date().timeIntervalSince(started)
                for (at, key) in actions where !fired.contains(at) && elapsed >= at {
                    fired.insert(at)
                    self.deliverKey(key, throughMenu: throughMenu)
                }
            }
        }
        timer.resume()
        clickTimers.append(timer)
    }

    /// Presses a key the way the keyboard does. `NSApp.sendEvent` runs the menu
    /// key equivalents before the responder chain, which is the path a shortcut
    /// takes — so this reports whether the item's shortcut is really reachable,
    /// not just declared.
    private func keyEvent(_ name: String, type: NSEvent.EventType) -> NSEvent? {
        // "cmd+ctrl+f", "right", "space": modifiers first, key last.
        var parts = name.split(separator: "+").map(String.init)
        let key = parts.popLast() ?? ""
        var flags: NSEvent.ModifierFlags = []
        for part in parts {
            switch part {
            case "cmd", "command": flags.insert(.command)
            case "ctrl", "control": flags.insert(.control)
            case "shift": flags.insert(.shift)
            case "opt", "option": flags.insert(.option)
            default: break
            }
        }
        let keys: [String: (UInt16, String)] = [
            "left": (123, String(UnicodeScalar(NSLeftArrowFunctionKey)!)),
            "right": (124, String(UnicodeScalar(NSRightArrowFunctionKey)!)),
            "down": (125, String(UnicodeScalar(NSDownArrowFunctionKey)!)),
            "up": (126, String(UnicodeScalar(NSUpArrowFunctionKey)!)),
            "space": (49, " "),
            "s": (1, "s"),
            "c": (8, "c"),
            "f": (3, "f"),
        ]
        guard let (code, characters) = keys[key],
              let win = NSApp.windows.first(where: { $0.isVisible }) else { return nil }
        // A real press carries the *shifted* character in both fields — shift is
        // the one modifier that changes the character itself, so it is not ignored.
        // Sending lowercase here made AppKit miss items whose shortcut needs shift
        // and report the event as handled by something else, so the shortcut looked
        // broken when it was the synthetic event.
        let shifted = flags.contains(.shift) ? characters.uppercased() : characters
        return NSEvent.keyEvent(with: type, location: .zero, modifierFlags: flags,
                                timestamp: ProcessInfo.processInfo.systemUptime,
                                windowNumber: win.windowNumber, context: nil,
                                characters: shifted, charactersIgnoringModifiers: shifted,
                                isARepeat: false, keyCode: code)
    }

    /// Two ways to deliver a shortcut, to tell "the menu item is wired up" apart
    /// from "AppKit routes a real key press to it": `sendEvent` is the path an
    /// event from the keyboard takes, `performKeyEquivalent` asks the menu itself
    /// and reports whether it claimed the event.
    private func deliverKey(_ name: String, throughMenu: Bool) {
        let before = status.playbackSeconds
        var handled: Bool?
        if throughMenu {
            if let down = keyEvent(name, type: .keyDown) {
                handled = NSApp.mainMenu?.performKeyEquivalent(with: down) ?? false
            }
        } else {
            // Faithful to a keyboard: the app has to be active for a key equivalent
            // to be routed at all, and the events go through the event *queue*
            // rather than straight to `sendEvent`. Calling `sendEvent` directly
            // skips the queue, which made AppKit's menu-highlighting machinery
            // re-dispatch one press as several actions.
            NSApp.activate(ignoringOtherApps: true)
            for type in [NSEvent.EventType.keyDown, .keyUp] {
                if let event = keyEvent(name, type: type) { NSApp.postEvent(event, atStart: false) }
            }
        }
        // A seek rebuilds the session, so read the position back once it has landed.
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.8) { [weak self] in
            guard let self else { return }
            let after = self.status.playbackSeconds
            NSLog("%@", "[key] \(name) via=\(throughMenu ? "menu" : "sendEvent") "
                  + "handled=\(handled.map(String.init) ?? "-") "
                  + "before=\(String(format: "%.2f", before)) "
                  + "after=\(String(format: "%.2f", after)) "
                  + "moved=\(String(format: "%+.2f", after - before)) "
                  + "state=\(self.status.state.rawValue) full=\(self.isFullScreen ? 1 : 0)")
        }
    }

    /// Dumps the menu SwiftUI actually built, with each item's key equivalent —
    /// the ground truth for "is this shortcut installed, or only written down?".
    private func dumpMenu() {
        func describe(_ modifiers: NSEvent.ModifierFlags) -> String {
            var s = ""
            if modifiers.contains(.control) { s += "ctrl+" }
            if modifiers.contains(.option) { s += "opt+" }
            if modifiers.contains(.shift) { s += "shift+" }
            if modifiers.contains(.command) { s += "cmd+" }
            return s.isEmpty ? "" : s
        }
        func walk(_ menu: NSMenu, depth: Int) {
            for item in menu.items {
                let key = item.keyEquivalent
                let name: String
                if key.isEmpty {
                    name = "(no key)"
                } else {
                    name = key.unicodeScalars.map { $0.value >= 0xF700 ? "U+\(String($0.value, radix: 16, uppercase: true))" : String($0) }
                        .joined()
                }
                NSLog("%@", "[menu] \(String(repeating: "  ", count: depth))\(item.title) "
                      + "key=\(describe(item.keyEquivalentModifierMask))\(name) "
                      + "enabled=\(item.isEnabled) check=\(item.state.rawValue) "
                      + "action=\(item.action.map(String.init(describing:)) ?? "nil")")
                if let sub = item.submenu { walk(sub, depth: depth + 1) }
            }
        }
        guard let main = NSApp.mainMenu else { return }
        NSLog("%@", "[menu] ===== main menu =====")
        walk(main, depth: 0)
    }
}

/// Hosts the player's display layer inside SwiftUI.
struct PlayerSurface: NSViewRepresentable {
    let view: NSView
    func makeNSView(context: Context) -> NSView { view }
    func updateNSView(_ nsView: NSView, context: Context) {}
}

// MARK: - Menu bar

struct PlayerCommands: Commands {
    @ObservedObject var model: AppModel

    private var idle: Bool { model.status.state == .idle || model.status.state == .failed }

    var body: some Commands {
        CommandGroup(replacing: .newItem) {
            Button("Add Videos…") { model.openFilePanel() }
                .keyboardShortcut("o", modifiers: .command)
            Button("Add Folder…") { model.openFolderPanel() }
                .keyboardShortcut("o", modifiers: [.command, .shift])
        }

        CommandMenu("Playback") {
            Button(model.status.state == .paused ? "Play" : "Pause") { model.togglePlayPause() }
                .keyboardShortcut(.space, modifiers: [])
                .disabled(idle)
            Divider()
            Button("Back 10 Seconds") { model.seek(by: -10) }
                .keyboardShortcut(.leftArrow, modifiers: [])
                .disabled(idle)
            Button("Forward 10 Seconds") { model.seek(by: 10) }
                .keyboardShortcut(.rightArrow, modifiers: [])
                .disabled(idle)
            Button("Back 1 Minute") { model.seek(by: -60) }
                .keyboardShortcut(.leftArrow, modifiers: .command)
                .disabled(idle)
            Button("Forward 1 Minute") { model.seek(by: 60) }
                .keyboardShortcut(.rightArrow, modifiers: .command)
                .disabled(idle)
            Divider()
            Button("Go to Start") { model.seek(to: 0) }.disabled(idle)
            Divider()
            Button("Previous Video") { model.previousInQueue() }
                .keyboardShortcut(.leftArrow, modifiers: [.command, .option])
                .disabled(model.currentIndex == nil)
            Button("Next Video") { model.nextInQueue() }
                .keyboardShortcut(.rightArrow, modifiers: [.command, .option])
                .disabled(model.currentIndex.map { $0 + 1 >= model.queue.count } ?? true)
            if !model.queue.isEmpty {
                Menu("Queue") {
                    ForEach(model.queue.indices, id: \.self) { index in
                        Button("\(index == model.currentIndex ? "▶ " : "")\(model.queue[index].lastPathComponent)") {
                            model.playQueued(at: index)
                        }
                    }
                    Divider()
                    Button("Clear Queue") { model.clearQueue() }
                }
            }
        }

        CommandMenu("Enhancement") {
            ForEach(ScalerKind.allCases, id: \.self) { k in
                Toggle(k.displayName, isOn: Binding(
                    get: { model.kind == k },
                    set: { _ in model.setKind(k) }))
            }
            // What the selected model is for, where it is selected: the names
            // alone do not say that one of them is the live-action option.
            Text(model.kind.hint)
            Divider()
            // A mode whose network exists at one scale (Real-ESRGAN is x4 only)
            // makes this control inert rather than authoritative, so it is
            // disabled and said out loud instead of silently ignoring a click.
            let pinned = model.kind.pinnedFactor
            Toggle("2×", isOn: Binding(
                get: { model.factor == .x2 },
                set: { _ in model.setFactor(.x2) }))
                .disabled(!model.kind.magnifies || pinned != nil)
            Toggle("4×", isOn: Binding(
                get: { model.factor == .x4 },
                set: { _ in model.setFactor(.x4) }))
                .disabled(!model.kind.magnifies || pinned != nil)
            if !model.kind.magnifies {
                // Off is 1:1, so the magnification above is inert. Saying so beats
                // leaving two controls that look like they should do something.
                Text("Off plays at the source's own size — magnification does not apply")
            } else if let pinned {
                Text("\(model.kind.displayName) is \(pinned.displayName) only — the network "
                     + "has no 2× checkpoint, so this mode always runs at \(pinned.displayName)")
            } else if model.status.scaleFactor > 0,
               model.status.scaleFactor != model.status.requestedScaleFactor {
                Text("Effective: \(model.status.scaleFactor)× (clamped for this source)")
            }
            if pinned == nil {
                Text("4× is SD-only: bigger sources clamp to 2×")
            } else {
                Text("Past 1024×576 the 4× output exceeds the encoder's limit, so this "
                     + "mode is SD-only rather than clamped")
            }
            Divider()
            Toggle("Comparison View", isOn: Binding(
                get: { model.comparing },
                set: { model.setCompare($0) }))
                // Not ⇧⌘C, which is the natural sibling of ⇧⌘S: a shifted
                // shortcut could not be verified here at all — a synthetic ⇧⌘
                // press matches no SwiftUI item in this app, for any letter, while
                // the ⌥⌘ form is delivered and fires (measured). ⌥⌘C is the shape
                // this environment can prove, so it is the one shipped rather than
                // a keystroke that might do nothing.
                .keyboardShortcut("c", modifiers: [.command, .option])
                // A model's output is 2x/4x and Off's is 1:1, so the two have no
                // shared pixel grid to be split on. The engine refuses it; disabling
                // the item shows that before it is triggered rather than after.
                .disabled(!model.kind.magnifies)
            // Deliberately not gated on a file being open. It was, and the item
            // then held a stale disabled state that swallowed its key equivalent
            // (measured: the gated item did nothing while the ungated ⌃⌘F worked
            // through the same path). With nothing open the toggle is a no-op that
            // simply applies when a file is opened.
            if model.comparing {
                // Only the magnifying kinds: a half is composited into the output
                // at its geometry, and Off has none to offer. "The original" is
                // already here as bypass, which is the same picture scaled.
                Menu("Left Half") {
                    ForEach(ScalerKind.allCases.filter(\.magnifies), id: \.self) { k in
                        Toggle(AppModel.compareName(k), isOn: Binding(
                            get: { model.compareLeft == k },
                            set: { _ in model.setCompareLeft(k) }))
                    }
                }
                Menu("Right Half") {
                    ForEach(ScalerKind.allCases.filter(\.magnifies), id: \.self) { k in
                        Toggle(AppModel.compareName(k), isOn: Binding(
                            get: { model.kind == k },
                            set: { _ in model.setKind(k) }))
                    }
                }
                Button("Show Only Right Half") { model.setCompareSplit(0) }
                Button("Split Evenly") { model.setCompareSplit(0.5) }
                Button("Show Only Left Half") { model.setCompareSplit(1) }
                Text("Drag the line on the picture to move the split")
            } else {
                Text("Splits the picture so two models can be compared")
            }
        }

        CommandMenu("Audio") {
            if model.status.audioTracks.isEmpty {
                Text("No audio tracks")
            } else {
                ForEach(model.status.audioTracks) { t in
                    Toggle(t.label, isOn: Binding(
                        get: { model.audioOrdinal == t.ordinal },
                        set: { _ in model.setAudioTrack(t.ordinal) }))
                }
            }
        }

        CommandMenu("Subtitles") {
            Toggle("Off", isOn: Binding(
                get: { model.subOrdinal < 0 },
                set: { _ in model.setSubtitleTrack(-1) }))
            if model.status.subtitleTracks.isEmpty {
                Text("No subtitle tracks in file")
            } else {
                Divider()
                ForEach(model.status.subtitleTracks) { t in
                    Toggle(t.label, isOn: Binding(
                        get: { model.subOrdinal == t.ordinal },
                        set: { _ in model.setSubtitleTrack(t.ordinal) }))
                }
                Text("Subtitles are burned into the picture")
            }
        }

        CommandGroup(after: .toolbar) {
            Toggle("Show Stats", isOn: $model.showStats)
                .keyboardShortcut("s", modifiers: [.command, .shift])
            Button(model.isFullScreen ? "Exit Full Screen" : "Enter Full Screen") { model.toggleFullScreen() }
                .keyboardShortcut("f", modifiers: [.command, .control])
            Text("Double-click the picture to toggle full screen")
        }
    }
}

// MARK: - Screen

struct PlayerScreen: View {
    @EnvironmentObject var model: AppModel

    @State private var chromeVisible = true
    @State private var hideWorkItem: DispatchWorkItem?
    @State private var scrub: Double?
    /// Last non-zero skip total, and whether the badge is showing: the badge holds
    /// its number through the fade-out instead of blanking to "0s".
    @State private var shownSkip: Double = 0
    @State private var skipBadgeVisible = false
    /// Slider position while it is being dragged; nil when not dragging.
    @State private var splitDrag: Double?
    @State private var showQueue = false

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            if let v = model.playerView {
                PlayerSurface(view: v)
                    .ignoresSafeArea()
            }

            if model.status.state == .idle {
                idleOverlay
            } else if isPreparing {
                preparingOverlay
            }

            skipBadge
            compareLabels

            VStack(spacing: 0) {
                if model.showStats { statsHUD }
                Spacer(minLength: 0)
                if chromeVisible && model.comparing && model.status.state != .idle { compareBar }
                if chromeVisible && model.status.state != .idle { transport }
            }

            if let err = model.lastError {
                VStack { Spacer(); note(err, .orange); Spacer().frame(height: 90) }
            } else if model.status.state == .failed, !model.status.message.isEmpty {
                VStack { Spacer(); note(model.status.message, .red); Spacer().frame(height: 90) }
            }
        }
        .onContinuousHover { phase in
            if case .active = phase { pokeChrome() }
        }
        .onAppear { pokeChrome() }
        .onChange(of: showQueue) { _, _ in pokeChrome() }
    }

    /// Nothing has been shown yet: the decoders, scaler and first frame are still
    /// coming up. Saying so avoids the "is it broken?" black screen.
    private var isPreparing: Bool {
        model.status.framesProcessed == 0 &&
        (model.status.state == .preparing || model.status.state == .processing)
    }

    /// The running total of skips still waiting on the debounce: "+30s" for three
    /// quick "forward 10" presses. It appears on the first press so the presses are
    /// acknowledged immediately, and clears when the skip actually happens.
    /// The running total of skips waiting on the debounce.
    ///
    /// The view is always in the hierarchy and only its opacity is animated. It
    /// used to be inserted and removed with a transition while the *number* changed,
    /// which cross-faded the digits, so "+20s" and "+30s" smeared into each other —
    /// and because that animation sat on the enclosing container, a skip starting
    /// also smeared the unrelated "Preparing…" text underneath it. With a stable
    /// view, the digits now swap instantly and only the fade in/out animates.
    private var skipBadge: some View {
        let forward = shownSkip > 0
        return HStack(spacing: 10) {
            Image(systemName: forward ? "goforward" : "gobackward")
                .font(.system(size: 30, weight: .semibold))
            Text("\(forward ? "+" : "\u{2212}")\(Int(abs(shownSkip).rounded()))s")
                .font(.system(size: 34, weight: .semibold, design: .rounded))
                .monospacedDigit()
        }
        .foregroundStyle(.white)
        .padding(.horizontal, 24).padding(.vertical, 12)
        .background(.black.opacity(0.55), in: Capsule())
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .padding(.top, 54)
        .opacity(skipBadgeVisible ? 1 : 0)
        .animation(.easeOut(duration: 0.16), value: skipBadgeVisible)
        .allowsHitTesting(false)
        .onChange(of: model.status.pendingSkipSeconds) { _, total in
            if abs(total) >= 0.5 {
                shownSkip = total
                skipBadgeVisible = true
            } else {
                skipBadgeVisible = false     // keeps the last total while it fades
            }
        }
    }

    /// The comparison controls: which model is on each side, and the slider that
    /// moves the divider. The slider goes all the way to either end, so either
    /// model can be made to fill the whole picture — the divider's position *is*
    /// the split, and 0% and 100% are the two single-model views.
    private var compareBar: some View {
        HStack(spacing: 12) {
            Picker("", selection: Binding(
                get: { model.compareLeft },
                set: { model.setCompareLeft($0) })) {
                ForEach(ScalerKind.allCases, id: \.self) { k in
                    Text(AppModel.compareName(k)).tag(k)
                }
            }
            .labelsHidden().frame(width: 190)

            Slider(value: Binding(get: { splitValue }, set: { splitDrag = $0; model.setCompareSplit($0) }),
                   in: 0...1,
                   onEditingChanged: { editing in if !editing { splitDrag = nil } })
                .frame(width: 200)

            Text("\(Int((splitValue * 100).rounded()))%")
                .font(.callout.monospacedDigit())
                .frame(width: 46, alignment: .trailing)

            Picker("", selection: Binding(
                get: { model.kind },
                set: { model.setKind($0) })) {
                ForEach(ScalerKind.allCases, id: \.self) { k in
                    Text(AppModel.compareName(k)).tag(k)
                }
            }
            .labelsHidden().frame(width: 190)
        }
        .foregroundStyle(.white)
        .padding(.horizontal, 14).padding(.vertical, 8)
        .background(.black.opacity(0.55), in: Capsule())
        .padding(.bottom, 8)
    }

    /// Names the two halves of the comparison view. The labels straddle the
    /// picture's centre — which is where the picture starts out split, because the
    /// view centres the picture — so they read as captions for the divider rather
    /// than for the window, at any window size and however the picture is
    /// letterboxed. They sit above the comparison bar so a skip's "+10s" at the
    /// top of the screen cannot collide with them.
    private var compareLabels: some View {
        let visible = model.comparing && !isPreparing && model.status.state != .idle
        let left = AppModel.compareName(model.compareLeft)
        let right = model.status.mode.isEmpty ? "Enhanced" : model.status.mode
        let rightFull = model.status.scaleFactor > 0 ? "\(right) \(model.status.scaleFactor)×" : right
        // At either extreme only one model is on screen, so a two-sided caption
        // would be pointing at a half that is not there.
        let single = splitValue <= 0.001 ? rightFull : (splitValue >= 0.999 ? left : nil)
        return HStack(spacing: 26) {
            if let single {
                sideLabel(single)
            } else {
                sideLabel(left)
                sideLabel(rightFull)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottom)
        .padding(.bottom, 118)
        // Stable view, opacity-only: the same construction as the skip badge, so
        // showing and hiding cannot smear the text underneath it.
        .opacity(visible ? 1 : 0)
        .animation(.easeOut(duration: 0.16), value: visible)
        .allowsHitTesting(false)
    }

    /// The slider's value while it is being dragged, so the engine's round-trip
    /// through the status store cannot pull the knob back under the pointer.
    private var splitValue: Double { splitDrag ?? model.compareSplit }

    private func sideLabel(_ text: String) -> some View {
        Text(text)
            .font(.system(size: 13, weight: .medium))
            .foregroundStyle(.white)
            .padding(.horizontal, 10).padding(.vertical, 5)
            .background(.black.opacity(0.55), in: Capsule())
    }

    private var idleOverlay: some View {
        VStack(spacing: 14) {
            Image(systemName: "film.stack").font(.system(size: 46))
                .foregroundStyle(.white.opacity(0.35))
            Text("VScaler").font(.title2).foregroundStyle(.white.opacity(0.85))
            Text("Add videos or a folder to start playing. Controls are in the menu bar.")
                .font(.callout).foregroundStyle(.white.opacity(0.55))
            HStack(spacing: 10) {
                Button("Add Videos…") { model.openFilePanel() }
                    .buttonStyle(.borderedProminent)
                Button("Add Folder…") { model.openFolderPanel() }
                    .buttonStyle(.bordered)
            }
        }
    }

    private var preparingOverlay: some View {
        VStack(spacing: 12) {
            ProgressView().controlSize(.large)
            Text("Preparing \((model.status.file.isEmpty ? "video" : model.status.file))…")
                .font(.callout).foregroundStyle(.white.opacity(0.7))
                .lineLimit(1).truncationMode(.middle)
            if !model.status.message.isEmpty {
                Text(model.status.message).font(.caption).foregroundStyle(.orange)
            }
        }
        .padding(24)
        .background(.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 10))
        // It appears in the middle of the picture on every seek, so it must not
        // eat clicks aimed at the video underneath — it did: a click during a
        // seek landed on the spinner instead of toggling playback.
        .allowsHitTesting(false)
    }

    // MARK: transport

    private var transport: some View {
        HStack(spacing: 12) {
            Button(action: { model.togglePlayPause() }) {
                Image(systemName: model.status.state == .paused ? "play.fill" : "pause.fill")
                    .font(.system(size: 15))
            }
            .buttonStyle(.plain)

            Text(timeString(shownPosition))
                .font(.callout.monospacedDigit()).frame(width: 58, alignment: .leading)

            Slider(value: Binding(get: { sliderValue }, set: { scrub = $0 }),
                   in: 0...max(model.status.durationSeconds, 1),
                   onEditingChanged: { editing in
                       if !editing, let t = scrub { scrub = nil; model.seek(to: t) }
                   })
                .disabled(model.status.durationSeconds <= 0)

            Text(timeString(model.status.durationSeconds))
                .font(.callout.monospacedDigit()).foregroundStyle(.secondary)
                .frame(width: 58, alignment: .trailing)

            Button("−10s") { model.seek(by: -10) }.buttonStyle(.plain)
            Button("+10s") { model.seek(by: 10) }.buttonStyle(.plain)

            Button(action: { model.previousInQueue() }) {
                Image(systemName: "backward.end.fill")
            }
            .buttonStyle(.plain)
            .disabled(model.currentIndex == nil)
            .help("Previous video")

            Button(action: { model.nextInQueue() }) {
                Image(systemName: "forward.end.fill")
            }
            .buttonStyle(.plain)
            .disabled(model.currentIndex.map { $0 + 1 >= model.queue.count } ?? true)
            .help("Next video")

            Button(action: { showQueue.toggle() }) {
                Image(systemName: "list.bullet")
            }
            .buttonStyle(.plain)
            .help("Queue (\(model.queue.count) videos)")
            .popover(isPresented: $showQueue, arrowEdge: .top) { queuePopover }

            Button(action: { model.toggleCompare() }) {
                Image(systemName: model.comparing ? "rectangle.split.2x1.fill" : "rectangle.split.2x1")
            }
            .buttonStyle(.plain)
            .disabled(model.status.state == .idle)
            .help("Compare the source against the enhancement (⌥⌘C)")

            Button(action: { model.showStats.toggle() }) {
                Image(systemName: "chart.bar.xaxis")
            }
            .buttonStyle(.plain)
            .help("Show stats (⇧⌘S)")
        }
        .foregroundStyle(.white)
        .padding(.horizontal, 14).padding(.vertical, 9)
        .background(.black.opacity(0.55), in: Capsule())
        .padding(.bottom, 16)
        .transition(.opacity)
    }

    private var queuePopover: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Queue").font(.headline)
                Spacer()
                Text("\(model.queue.count) videos").font(.caption).foregroundStyle(.secondary)
            }
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 2) {
                    ForEach(model.queue.indices, id: \.self) { index in
                        HStack(alignment: .top, spacing: 8) {
                            Button {
                                model.playQueued(at: index)
                            } label: {
                                HStack(alignment: .top, spacing: 8) {
                                    Image(systemName: index == model.currentIndex ? "waveform" : "film")
                                        .frame(width: 18)
                                        .padding(.top, 2)
                                    Text(model.queue[index].lastPathComponent)
                                        .fixedSize(horizontal: false, vertical: true)
                                        .frame(maxWidth: .infinity, alignment: .leading)
                                }
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .contentShape(Rectangle())
                            }
                            .buttonStyle(.plain)
                            .help(model.queue[index].path)
                            Button { model.removeQueued(at: index) } label: {
                                Image(systemName: "xmark").font(.caption)
                            }
                            .buttonStyle(.plain)
                            .help("Remove from queue")
                        }
                        .padding(.horizontal, 8).padding(.vertical, 7)
                        .background(index == model.currentIndex ? Color.accentColor.opacity(0.18) : .clear,
                                    in: RoundedRectangle(cornerRadius: 6))
                    }
                }
            }
            .frame(maxHeight: 320)
            Divider()
            HStack {
                Button("Add Videos…") { model.openFilePanel() }
                Button("Add Folder…") { model.openFolderPanel() }
                Spacer()
                Button("Clear") { model.clearQueue() }
                    .disabled(model.queue.isEmpty)
            }
            .font(.caption)
        }
        .padding(14)
        .frame(width: 440)
    }

    // MARK: stats

    private var statsHUD: some View {
        let s = model.status
        return VStack(alignment: .leading, spacing: 3) {
            Text(s.file.isEmpty ? "—" : s.file)
                .font(.caption.bold()).lineLimit(1).truncationMode(.middle)
            Group {
                row("source", "\(s.sourceWidth)×\(s.sourceHeight)")
                row("output", "\(s.outputWidth)×\(s.outputHeight)  \(s.scaleFactor)×")
                row("mode", s.mode.isEmpty ? "—" : s.mode)
                row("view", s.comparing
                    ? "\(Int((s.compareSplit * 100).rounded()))% left=\(s.compareLeftMode)"
                    : "enhanced")
                row("state", s.state.rawValue)
                row("frames", "\(s.framesProcessed)")
                row("process", s.processFps <= 0 ? "—" : String(format: "%.1f fps · %.2f× RT", s.processFps, s.realtimeRatio))
                row("position", String(format: "%.1f / %.0f s", s.playbackSeconds, s.durationSeconds))
                row("audio fed", s.audioFramesEnqueued <= 0 ? "—" : String(format: "%.1f s", Double(s.audioFramesEnqueued) / 48000))
                row("buffer", "\(s.bufferFrames) frames · \(Int(s.bufferMilliseconds.rounded())) ms")
                row("buffer mem", ByteCountFormatter.string(fromByteCount: Int64(s.bufferBytes), countStyle: .file))
                row("memory", String(format: "%.0f MB", s.memoryFootprintMB))
                row("thermal", s.thermalState)
                row("renderers", s.renderStatus.isEmpty ? "—" : s.renderStatus)
            }
        }
        .font(.caption.monospaced())
        .foregroundStyle(.white.opacity(0.9))
        .padding(10)
        .background(.black.opacity(0.6), in: RoundedRectangle(cornerRadius: 8))
        .padding(14)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .allowsHitTesting(false)
    }

    private func row(_ label: String, _ value: String) -> some View {
        HStack(spacing: 8) {
            Text(label).foregroundStyle(.white.opacity(0.5))
                .frame(width: 78, alignment: .leading)
            Text(value)
        }
    }

    private func note(_ text: String, _ tint: Color) -> some View {
        HStack(alignment: .top, spacing: 6) {
            Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(tint)
            Text(text).font(.callout).fixedSize(horizontal: false, vertical: true)
        }
        .padding(8)
        .background(tint.opacity(0.18), in: RoundedRectangle(cornerRadius: 6))
        .padding(.horizontal, 24)
    }

    // MARK: helpers

    /// Show the transport, and hide it again after a couple of seconds of no
    /// mouse movement — but never while paused or the queue is open.
    private func pokeChrome() {
        chromeVisible = true
        hideWorkItem?.cancel()
        guard model.status.state != .paused, !showQueue else { return }
        let item = DispatchWorkItem {
            if scrub == nil && model.status.state != .paused && !showQueue {
                chromeVisible = false
            }
        }
        hideWorkItem = item
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.5, execute: item)
    }

    private var shownPosition: Double {
        scrub ?? model.status.playbackSeconds
    }

    private var sliderValue: Double {
        min(max(shownPosition, 0), max(model.status.durationSeconds, 1))
    }

    private func timeString(_ seconds: Double) -> String {
        guard seconds.isFinite, seconds >= 0 else { return "0:00" }
        let t = Int(seconds.rounded())
        if t >= 3600 { return String(format: "%d:%02d:%02d", t / 3600, (t % 3600) / 60, t % 60) }
        return String(format: "%d:%02d", t / 60, t % 60)
    }
}

// In a bundle this is what makes the app a real document-handling GUI app; when
// run straight from .build it also promotes the process from a background agent
// to a regular app, so its window appears either way.
final class AppDelegate: NSObject, NSApplicationDelegate {
    /// A file the Finder asked us to open (double-click, "Open With", or a drop on
    /// the Dock icon). Parked here because that Apple Event can arrive before the
    /// SwiftUI scene — and so its `AppModel` — exists.
    var pendingOpen: [URL] = []

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }

    func application(_ application: NSApplication, open urls: [URL]) {
        guard !urls.isEmpty else { return }
        MainActor.assumeIsolated {
            if let model = AppModel.instance {
                model.addToQueue(urls)
            } else {
                pendingOpen.append(contentsOf: urls)
            }
        }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }

    /// Shut the engine down on quit so the ffmpeg children do not outlive the app.
    func applicationWillTerminate(_ notification: Notification) {
        MainActor.assumeIsolated { AppModel.instance?.stop() }
    }
}
