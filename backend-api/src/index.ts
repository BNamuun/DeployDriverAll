import express, { Request, Response } from "express"
import cors from "cors"
import dotenv from "dotenv"
import axios from "axios"

dotenv.config()

const app = express()

app.use(cors())
// base64 frames can be >100kb; bump the limit.
app.use(express.json({ limit: "10mb" }))

const PORT = process.env.PORT ? Number(process.env.PORT) : 3000
// Python ml-service (FastAPI) runs on 8000.
// Use 127.0.0.1 to avoid any proxy/DNS edge cases with "localhost".
// const ML_SERVICE_URL = process.env.ML_SERVICE_URL ?? "http://127.0.0.1:8000"
const ML_SERVICE_URL = process.env.ML_SERVICE_URL
// Health check
app.get("/health", (_req: Request, res: Response) => {
  console.log("xaxaxa")
  res.json({ status: "ok", time: new Date().toISOString() })
})

// Proxy health to ml-service
app.get("/api/drive/health", async (_req: Request, res: Response) => {
  try {
    const { data } = await axios.get(`${ML_SERVICE_URL}/health`, {
      timeout: 3000,
      // Never proxy localhost calls.
      proxy: false,
    })
    res.json(data)
  } catch (err) {
    res.status(502).json({
      ok: false,
      error: "Failed to reach ml-service",
      details: String(err),
    })
  }
})

// Forward base64 inference payload to ml-service
app.post("/api/drive/infer", async (req: Request, res: Response) => {
  console.log("xaxa")
  try {
    const { data } = await axios.post(
      `${ML_SERVICE_URL}/infer-base64`,
      req.body,
      {
        timeout: 8000,
        headers: { "Content-Type": "application/json" },
        // Never proxy localhost calls.
        proxy: false,
      }
    )
    console.log("infer-base64 response data:", data)
    res.json(data)
  } catch (err) {
    res.status(502).json({ error: "Inference failed", details: String(err) })
  }
})

// Example API route
app.get("/api/hello", (req: Request, res: Response) => {
  const name = (req.query.name as string) ?? "world"
  res.json({ message: `Hello, ${name}!` })
})

// Example POST route
app.post("/api/users", (req: Request, res: Response) => {
  const { email } = req.body as { email?: string }

  if (!email) {
    return res.status(400).json({ error: "email is required" })
  }

  // pretend we saved a user
  res.status(201).json({ id: "u_123", email })
})

app.listen(PORT, "0.0.0.0", () => {
  console.log(`API running on http://localhost:${PORT}`)
  console.log(`Proxying ml-service at ${ML_SERVICE_URL}`)
})
