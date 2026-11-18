"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import {
  Activity,
  ArrowUpRight,
  Bell,
  ChevronDown,
  CircleHelp,
  Clock3,
  Command,
  Database,
  FolderKanban,
  LayoutDashboard,
  LogOut,
  Search,
  SlidersHorizontal,
  Sparkles,
  Waypoints,
} from "lucide-react";
import { type FormEvent, useEffect, useState } from "react";
import { API_BASE, apiFetch } from "@/lib/api";

const navigation = [
  { href: "/dashboard", label: "Overview", icon: LayoutDashboard },
  { href: "/studio", label: "Removal studio", icon: Sparkles },
  { href: "/projects", label: "Projects & datasets", icon: FolderKanban },
  { href: "/history", label: "Processing history", icon: Clock3 },
  { href: "/analytics", label: "Model analytics", icon: Activity },
  { href: "/comparison", label: "Model comparison", icon: Waypoints },
];

const pageNames: Record<string, string> = {
  "/dashboard": "Overview",
  "/studio": "Cloud removal studio",
  "/projects": "Projects & datasets",
  "/history": "Processing history",
  "/analytics": "Model analytics",
  "/comparison": "Model comparison",
};

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const [query, setQuery] = useState("");
  const [user, setUser] = useState<{ id: number; email: string } | null>(null);
  const [authStatus, setAuthStatus] = useState<"checking" | "authenticated" | "unavailable">("checking");
  const [authError, setAuthError] = useState("");
  const [logoutError, setLogoutError] = useState("");

  useEffect(() => {
    if (pathname === "/login") return;
    let active = true;
    fetch(`${API_BASE}/auth/me`, { credentials: "include", cache: "no-store" })
      .then(async (response) => {
        if (!active) return;
        if (response.status === 401) {
          router.replace(`/login?next=${encodeURIComponent(pathname)}`);
          return;
        }
        if (!response.ok) throw new Error(`Authentication service returned ${response.status}`);
        const result = await response.json() as { user: { id: number; email: string } };
        if (active) {
          setUser(result.user);
          setAuthStatus("authenticated");
          setAuthError("");
        }
      })
      .catch((error: unknown) => {
        if (active) {
          setAuthStatus("unavailable");
          setAuthError(error instanceof Error ? error.message : "Could not verify your sign-in.");
        }
      });
    return () => { active = false; };
  }, [pathname, router]);

  useEffect(() => {
    function handleUnauthorized() {
      if (pathname !== "/login") router.replace(`/login?next=${encodeURIComponent(pathname)}`);
    }
    window.addEventListener("orbit:unauthorized", handleUnauthorized);
    return () => window.removeEventListener("orbit:unauthorized", handleUnauthorized);
  }, [pathname, router]);

  if (pathname === "/login") return children;

  async function logout() {
    setLogoutError("");
    try {
      await apiFetch<{ ok: boolean }>("/auth/logout", { method: "POST" });
      setUser(null);
      router.replace("/login");
    } catch (error) {
      setLogoutError(error instanceof Error ? error.message : "Sign-out failed.");
    }
  }

  if (authStatus !== "authenticated") {
    return (
      <div className="auth-gate">
        <div className="auth-gate-mark"><Command size={20} /></div>
        <strong>{authStatus === "checking" ? "Checking your session" : "Unable to verify your session"}</strong>
        <p>{authStatus === "checking" ? "Connecting securely to the research workspace…" : authError}</p>
        {authStatus === "unavailable" && <button className="auth-retry" onClick={() => window.location.reload()}>Retry connection</button>}
      </div>
    );
  }

  function search(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const value = query.trim();
    router.push(value ? `/history?search=${encodeURIComponent(value)}` : "/history");
  }

  return (
    <div className="outer-shell">
      <aside className="sidebar" aria-label="Primary navigation">
        <Link className="brand-mark" href="/dashboard" aria-label="Orbit home">
          <Command size={19} strokeWidth={2.7} />
        </Link>
        <div className="side-separator" />
        <nav className="side-nav">
          {navigation.map(({ href, label, icon: Icon }) => (
            <Link
              aria-label={label}
              aria-current={pathname === href ? "page" : undefined}
              className={`nav-icon ${pathname === href ? "active" : ""}`}
              href={href}
              key={href}
              title={label}
            >
              <Icon size={19} strokeWidth={1.8} />
            </Link>
          ))}
        </nav>
        <div className="side-spacer" />
        <Link className="nav-icon" href="/analytics" aria-label="Research settings" title="Research settings">
          <SlidersHorizontal size={18} />
        </Link>
        <Link className="nav-icon help-icon" href="/comparison" aria-label="Evaluation details" title="Evaluation details">
          <CircleHelp size={18} />
        </Link>
        <div className="avatar" aria-label="Researcher profile">R</div>
      </aside>

      <div className="workspace">
        <header className="topbar">
          <div className="page-crumb">
            <span className="crumb-overline">ORBIT / RESEARCH WORKSPACE</span>
            <span className="crumb-title">{pageNames[pathname] ?? "Research workspace"}</span>
          </div>
          <form className="search-box" onSubmit={search} role="search">
            <Search size={16} />
            <input
              aria-label="Search processing history"
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search scenes, jobs…"
              value={query}
            />
            <kbd>⌘ K</kbd>
          </form>
          <div className="top-actions">
            <span className="account-email" title={user?.email}>{user?.email}</span>
            <button className="icon-button sign-out" onClick={() => void logout()} aria-label="Sign out" title="Sign out">
              <LogOut size={16} />
            </button>
            <Link className="icon-button" href="/history" aria-label="Processing history" title="Processing history">
              <Bell size={17} />
            </Link>
            <Link className="workspace-switch" href="/projects" title="Projects and datasets">
              <Database size={15} />
              <span>Workspace</span>
              <ChevronDown size={14} />
            </Link>
            <Link className="new-run" href="/studio">
              <span className="new-run-icon"><Sparkles size={14} /></span>
              <span>New reconstruction</span>
              <ArrowUpRight size={15} />
            </Link>
          </div>
        </header>
        {logoutError && <div className="auth-inline-error" role="alert">{logoutError}</div>}
        <main className="main-content">{children}</main>
      </div>
      <nav className="mobile-nav" aria-label="Mobile navigation">
        {navigation.map(({ href, label, icon: Icon }) => (
          <Link aria-label={label} aria-current={pathname === href ? "page" : undefined} className={pathname === href ? "active" : ""} href={href} key={href}>
            <Icon size={19} />
          </Link>
        ))}
      </nav>
    </div>
  );
}
