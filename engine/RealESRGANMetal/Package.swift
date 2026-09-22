// swift-tools-version:5.10
import PackageDescription

let package = Package(
    name: "RealESRGANMetal",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .library(name: "RealESRGANMetal", targets: ["RealESRGANMetal"]),
        .executable(name: "resrbench", targets: ["resrbench"]),
    ],
    targets: [
        .target(
            name: "RealESRGANMetal",
            path: "Sources/RealESRGANMetal"
        ),
        .executableTarget(
            name: "resrbench",
            dependencies: ["RealESRGANMetal"],
            path: "Sources/resrbench"
        ),
    ]
)
