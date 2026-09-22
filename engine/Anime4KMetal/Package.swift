// swift-tools-version:5.10
import PackageDescription

let package = Package(
    name: "Anime4KMetal",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .library(name: "Anime4KMetal", targets: ["Anime4KMetal"]),
        .executable(name: "an4kbench", targets: ["an4kbench"]),
    ],
    targets: [
        .target(
            name: "Anime4KMetal",
            path: "Sources/Anime4KMetal"
        ),
        .executableTarget(
            name: "an4kbench",
            dependencies: ["Anime4KMetal"],
            path: "Sources/an4kbench"
        ),
    ]
)
