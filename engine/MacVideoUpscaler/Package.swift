// swift-tools-version:5.10
import PackageDescription

let package = Package(
    name: "MacVideoUpscaler",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "MacVideoUpscalerCore", targets: ["MacVideoUpscalerCore"]),
        .executable(name: "macvideoupscale", targets: ["macvideoupscale"]),
        .executable(name: "MacVideoUpscalerApp", targets: ["MacVideoUpscalerApp"]),
        .executable(name: "playercheck", targets: ["playercheck"]),
    ],
    dependencies: [
        .package(path: "../Anime4KMetal"),
        .package(path: "../RealESRGANMetal"),
    ],
    targets: [
        .target(
            name: "MacVideoUpscalerCore",
            dependencies: [
                .product(name: "Anime4KMetal", package: "Anime4KMetal"),
                .product(name: "RealESRGANMetal", package: "RealESRGANMetal"),
            ],
            swiftSettings: [.define("HAVE_ANIME4K_METAL")]
        ),
        .executableTarget(
            name: "macvideoupscale",
            dependencies: ["MacVideoUpscalerCore"]
        ),
        .executableTarget(
            name: "playercheck",
            dependencies: ["MacVideoUpscalerCore"]
        ),
        .executableTarget(
            name: "MacVideoUpscalerApp",
            dependencies: ["MacVideoUpscalerCore"]
        ),
    ]
)
