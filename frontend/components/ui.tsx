import { ArrowUpRight, LoaderCircle, TriangleAlert } from "lucide-react";
import type { ReactNode } from "react";

export function Panel({
  children,
  className = "",
}: {
  children: ReactNode;
  className?: string;
}) {
  return <section className={`panel ${className}`}>{children}</section>;
}

export function SectionTitle({
  eyebrow,
  title,
  detail,
  action,
}: {
  eyebrow?: string;
  title: string;
  detail?: string;
  action?: ReactNode;
}) {
  return (
    <div className="section-title">
      <div>
        {eyebrow && <div className="eyebrow">{eyebrow}</div>}
        <h1>{title}</h1>
        {detail && <p>{detail}</p>}
      </div>
      {action && <div className="section-action">{action}</div>}
    </div>
  );
}

export function EmptyState({
  title,
  detail,
  action,
}: {
  title: string;
  detail: string;
  action?: ReactNode;
}) {
  return (
    <div className="empty-state">
      <span className="empty-orb" />
      <h3>{title}</h3>
      <p>{detail}</p>
      {action}
    </div>
  );
}

export function ErrorAlert({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="alert alert-error" role="alert">
      <TriangleAlert size={17} />
      <span>{message}</span>
      {onRetry && (
        <button className="text-button" onClick={onRetry} type="button">
          Retry
        </button>
      )}
    </div>
  );
}

export function Notice({ children, tone = "info" }: { children: ReactNode; tone?: "info" | "warning" | "success" }) {
  return <div className={`alert alert-${tone}`}>{children}</div>;
}

export function LoadingState({ label = "Loading verified project data…" }: { label?: string }) {
  return (
    <div className="loading-state">
      <LoaderCircle className="spin" size={18} />
      <span>{label}</span>
    </div>
  );
}

export function StatusBadge({ children, tone = "neutral" }: { children: ReactNode; tone?: string }) {
  return <span className={`badge badge-${tone}`}>{children}</span>;
}

export function Button({
  children,
  href,
  variant = "primary",
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & {
  href?: string;
  variant?: "primary" | "secondary" | "quiet" | "danger";
}) {
  const className = `button button-${variant} ${props.className ?? ""}`;
  if (href) {
    return (
      <a className={className} href={href}>
        {children}
        {variant === "primary" && <ArrowUpRight size={16} />}
      </a>
    );
  }
  return (
    <button {...props} className={className}>
      {children}
    </button>
  );
}

export function SelectField({
  label,
  value,
  onChange,
  children,
  disabled = false,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  children: ReactNode;
  disabled?: boolean;
}) {
  return (
    <label className="field">
      <span>{label}</span>
      <select disabled={disabled} value={value} onChange={(event) => onChange(event.target.value)}>
        {children}
      </select>
    </label>
  );
}

export function Modal({
  title,
  detail,
  onClose,
  children,
}: {
  title: string;
  detail?: string;
  onClose: () => void;
  children: ReactNode;
}) {
  return (
    <div className="modal-backdrop" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <section aria-labelledby="modal-title" aria-modal="true" className="modal" role="dialog">
        <div className="modal-header">
          <div>
            <h2 id="modal-title">{title}</h2>
            {detail && <p>{detail}</p>}
          </div>
          <button aria-label="Close dialog" className="modal-close" onClick={onClose} type="button">×</button>
        </div>
        {children}
      </section>
    </div>
  );
}
