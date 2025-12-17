"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
var _a;
Object.defineProperty(exports, "__esModule", { value: true });
const express_1 = __importDefault(require("express"));
const cors_1 = __importDefault(require("cors"));
const dotenv_1 = __importDefault(require("dotenv"));
const axios_1 = __importDefault(require("axios"));
dotenv_1.default.config();
const app = (0, express_1.default)();
app.use((0, cors_1.default)());
// base64 frames can be >100kb; bump the limit.
app.use(express_1.default.json({ limit: "10mb" }));
const PORT = process.env.PORT ? Number(process.env.PORT) : 3000;
// Python ml-service (FastAPI) runs on 8000.
// Use 127.0.0.1 to avoid any proxy/DNS edge cases with "localhost".
const ML_SERVICE_URL = (_a = process.env.ML_SERVICE_URL) !== null && _a !== void 0 ? _a : "http://127.0.0.1:8000";
// Health check
app.get("/health", (_req, res) => {
    console.log("xaxaxa");
    res.json({ status: "ok", time: new Date().toISOString() });
});
// Proxy health to ml-service
app.get("/api/drive/health", (_req, res) => __awaiter(void 0, void 0, void 0, function* () {
    try {
        const { data } = yield axios_1.default.get(`${ML_SERVICE_URL}/health`, {
            timeout: 3000,
            // Never proxy localhost calls.
            proxy: false,
        });
        res.json(data);
    }
    catch (err) {
        res.status(502).json({
            ok: false,
            error: "Failed to reach ml-service",
            details: String(err),
        });
    }
}));
// Forward base64 inference payload to ml-service
app.post("/api/drive/infer", (req, res) => __awaiter(void 0, void 0, void 0, function* () {
    console.log("xaxa");
    try {
        const { data } = yield axios_1.default.post(`${ML_SERVICE_URL}/infer-base64`, req.body, {
            timeout: 8000,
            headers: { "Content-Type": "application/json" },
            // Never proxy localhost calls.
            proxy: false,
        });
        console.log("infer-base64 response data:", data);
        res.json(data);
    }
    catch (err) {
        res.status(502).json({ error: "Inference failed", details: String(err) });
    }
}));
// Example API route
app.get("/api/hello", (req, res) => {
    var _a;
    const name = (_a = req.query.name) !== null && _a !== void 0 ? _a : "world";
    res.json({ message: `Hello, ${name}!` });
});
// Example POST route
app.post("/api/users", (req, res) => {
    const { email } = req.body;
    if (!email) {
        return res.status(400).json({ error: "email is required" });
    }
    // pretend we saved a user
    res.status(201).json({ id: "u_123", email });
});
app.listen(PORT, () => {
    console.log(`API running on http://localhost:${PORT}`);
    console.log(`Proxying ml-service at ${ML_SERVICE_URL}`);
});
