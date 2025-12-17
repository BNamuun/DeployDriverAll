"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const express_1 = __importDefault(require("express"));
const cors_1 = __importDefault(require("cors"));
const dotenv_1 = __importDefault(require("dotenv"));
dotenv_1.default.config();
const app = (0, express_1.default)();
app.use((0, cors_1.default)());
app.use(express_1.default.json());
// Health check
app.get("/health", (_req, res) => {
    res.json({ status: "ok", time: new Date().toISOString() });
});
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
const PORT = process.env.PORT ? Number(process.env.PORT) : 3000;
app.listen(PORT, () => {
    console.log(`API running on http://localhost:${PORT}`);
});
