import { Component } from "react";
import type { ReactNode } from "react";

type Props = { children: ReactNode; fallback?: ReactNode };
type State = { err: Error | null };

// Keeps a single panel's render crash from unmounting the whole app
// (the "EE page sometimes blanks" class of bug). Shows an inline error
// with a retry instead of a white screen.
export default class ErrorBoundary extends Component<Props, State> {
  state: State = { err: null };
  static getDerivedStateFromError(err: Error): State {
    return { err };
  }
  componentDidCatch(err: Error, info: unknown) {
    console.error("panel crashed:", err, info);
  }
  render() {
    const { err } = this.state;
    if (err) {
      return (
        this.props.fallback ?? (
          <div className="rounded-md border border-red-500/40 bg-red-500/5 p-4 text-xs text-red-400">
            <div className="font-semibold mb-1">This panel crashed - the rest of the page is unaffected.</div>
            <div className="font-mono opacity-80 break-all">{String(err.message || err)}</div>
            <button className="mt-2 underline hover:text-red-300" onClick={() => this.setState({ err: null })}>
              retry
            </button>
          </div>
        )
      );
    }
    return this.props.children;
  }
}
