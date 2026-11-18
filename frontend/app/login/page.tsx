"use client";

import { FormEvent, useState } from "react";
import { ArrowRight, Command, Eye, EyeOff, LockKeyhole, ShieldCheck } from "lucide-react";
import { apiFetch } from "@/lib/api";

export default function LoginPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [visible, setVisible] = useState(false);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setPending(true);
    setError("");
    try {
      await apiFetch<{ user: { id: number; email: string } }>("/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });
      const requested = new URLSearchParams(window.location.search).get("next");
      const destination = requested?.startsWith("/") && !requested.startsWith("//") ? requested : "/dashboard";
      window.location.replace(destination);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Sign-in failed. Please try again.");
    } finally {
      setPending(false);
    }
  }

  return (
    <main className="login-page">
      <div className="login-glow" />
      <section className="login-card" aria-labelledby="login-title">
        <div className="login-brand">
          <span className="login-mark"><Command size={20} strokeWidth={2.5} /></span>
          <span>ORBIT <small>RESEARCH WORKSPACE</small></span>
        </div>
        <div className="login-intro">
          <span className="eyebrow">SECURE RESEARCH ACCESS</span>
          <h1 id="login-title">Welcome back</h1>
          <p>Sign in to continue to your satellite imagery workspace.</p>
        </div>
        <form className="login-form" onSubmit={submit}>
          <label className="login-field">
            <span>Work email</span>
            <input autoComplete="username" autoFocus maxLength={254} onChange={(event) => setEmail(event.target.value)} placeholder="name@organization.org" required type="email" value={email} />
          </label>
          <label className="login-field">
            <span>Password</span>
            <span className="password-wrap">
              <input autoComplete="current-password" maxLength={128} onChange={(event) => setPassword(event.target.value)} placeholder="Enter your password" required type={visible ? "text" : "password"} value={password} />
              <button aria-label={visible ? "Hide password" : "Show password"} onClick={() => setVisible((current) => !current)} type="button">{visible ? <EyeOff size={16} /> : <Eye size={16} />}</button>
            </span>
          </label>
          {error && <div className="login-error" role="alert">{error}</div>}
          <button className="login-submit" disabled={pending} type="submit">
            {pending ? "Signing in…" : "Sign in"} {!pending && <ArrowRight size={16} />}
          </button>
        </form>
        <div className="login-security">
          <ShieldCheck size={15} />
          <span>Protected with scrypt password hashing and a secure, HttpOnly session.</span>
        </div>
        <div className="login-access"><LockKeyhole size={13} /> Accounts are provisioned by your workspace administrator.</div>
      </section>
      <footer className="login-footer">MULTIMODAL SATELLITE CLOUD REMOVAL · RESEARCH ACCESS</footer>
    </main>
  );
}
