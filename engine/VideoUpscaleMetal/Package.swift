// swift-tools-version:5.10
import PackageDescription

let package = Package(
    name: "VideoUpscaleMetal",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .library(name: "VideoUpscaleMetal", targets: ["VideoUpscaleMetal"]),
        .executable(name: "vupbench", targets: ["vupbench"]),
    ],
    targets: [
        .target(
            name: "VideoUpscaleMetal",
            path: "Sources/VideoUpscaleMetal"
        ),
        .executableTarget(
            name: "vupbench",
            dependencies: ["VideoUpscaleMetal"],
            path: "Sources/vupbench"
        ),
    ]
)
