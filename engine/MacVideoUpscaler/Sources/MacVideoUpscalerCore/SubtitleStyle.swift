import AppKit

public struct SubtitleStyle: Codable, Equatable {
    public enum Typeface: String, Codable, CaseIterable {
        case system, avenir, helvetica, georgia
        public var label: String {
            switch self {
            case .system: return "System"
            case .avenir: return "Avenir Next"
            case .helvetica: return "Helvetica Neue"
            case .georgia: return "Georgia"
            }
        }
        func font(size: CGFloat) -> NSFont {
            let name: String
            switch self {
            case .system: return NSFont.systemFont(ofSize: size, weight: .medium)
            case .avenir: name = "AvenirNext-DemiBold"
            case .helvetica: name = "HelveticaNeue-Medium"
            case .georgia: name = "Georgia"
            }
            return NSFont(name: name, size: size) ?? NSFont.systemFont(ofSize: size, weight: .medium)
        }
    }
    public enum Tint: String, Codable, CaseIterable {
        case white, warm, yellow
        public var label: String {
            switch self {
            case .white: return "White"
            case .warm: return "Warm white"
            case .yellow: return "Yellow"
            }
        }
        var cgColor: CGColor {
            switch self {
            case .white: return CGColor(red: 1, green: 1, blue: 1, alpha: 1)
            case .warm: return CGColor(red: 1, green: 0.96, blue: 0.86, alpha: 1)
            case .yellow: return CGColor(red: 1, green: 0.88, blue: 0.25, alpha: 1)
            }
        }
    }
    public enum Outline: String, Codable, CaseIterable {
        case subtle, standard, strong
        public var label: String { rawValue.capitalized }
        var width: Double {
            switch self {
            case .subtle: return 2
            case .standard: return 3.5
            case .strong: return 6
            }
        }
    }
    public var typeface: Typeface = .system
    public var size: Double = 1
    public var tint: Tint = .white
    public var outline: Outline = .standard
    /// Fractions of the video dimensions; positive values move right and up.
    public var horizontalOffset: Double = 0
    public var verticalOffset: Double = 0
    public init() {}

    private enum CodingKeys: String, CodingKey {
        case typeface, size, tint, outline, horizontalOffset, verticalOffset
    }
    public init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        typeface = try values.decodeIfPresent(Typeface.self, forKey: .typeface) ?? .system
        size = try values.decodeIfPresent(Double.self, forKey: .size) ?? 1
        tint = try values.decodeIfPresent(Tint.self, forKey: .tint) ?? .white
        outline = try values.decodeIfPresent(Outline.self, forKey: .outline) ?? .standard
        horizontalOffset = try values.decodeIfPresent(Double.self, forKey: .horizontalOffset) ?? 0
        verticalOffset = try values.decodeIfPresent(Double.self, forKey: .verticalOffset) ?? 0
    }

    public var normalized: Self {
        var result = self
        result.size = size.isFinite ? min(1.5, max(0.75, size)) : 1
        result.horizontalOffset = horizontalOffset.isFinite ? min(0.4, max(-0.4, horizontalOffset)) : 0
        result.verticalOffset = verticalOffset.isFinite ? min(0.8, max(-0.1, verticalOffset)) : 0
        return result
    }

    public func previewImage() -> CGImage? {
        TextSubtitleRenderer(cues: [.init(start: 0, end: 1,
            lines: ["A clearer view of every word."])],
            sourceWidth: 960, sourceHeight: 540, style: normalized).previewImage()
    }
}
