import * as esbuild from "npm:esbuild"
import { denoPlugins } from "jsr:@luca/esbuild-deno-loader"
import type { BuildOptions } from "npm:esbuild"

const args = Deno.args
const watchMode = args.includes("--watch")

const buildOptions: BuildOptions = {
    plugins: [...denoPlugins()],
    conditions: ["browser", "deno", "node"],
    entryPoints: [
        "./clientside/init.ts",
    ],
    outfile: "./static/js/clientside.js",
    bundle: true,
    format: "esm",
    target: ["es2020"],
    define: {
        "import.meta.url": '""',
        "import.meta": "false",
    },
}

async function build() {
    try {
        const timestamp = new Date().toLocaleTimeString()
        await esbuild.build(buildOptions)
        console.log(`[${timestamp}] Build completed successfully`)
    } catch (error) {
        console.error(`Build failed:`, error)
    }
}

if (watchMode) {
    // Use Deno's built-in watch functionality
    const watcher = Deno.watchFs(["./clientside"], { recursive: true })

    // Initial build
    await build()
    console.log("Watching for changes...")

    for await (const event of watcher) {
        if (["create", "modify"].includes(event.kind)) {
            console.log(`Changes detected in ${event.paths}`)
            await build()
        }
    }
} else {
    await build()
    esbuild.stop()
}
