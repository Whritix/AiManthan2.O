import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { api } from "../lib/api";
import type { ControlState, FrameResult, Health } from "../lib/types";

interface LiveContextValue {
  frame: FrameResult | null;
  fps: number;
  health: Health | null;
  control: ControlState | null;
  connected: boolean;
  lastFrameAt: number;
  error: string | null;
  refreshControl: () => Promise<void>;
  refreshHealth: () => Promise<void>;
}

const LiveContext = createContext<LiveContextValue | null>(null);

export function LiveProvider({ children }: { children: ReactNode }) {
  const [frame, setFrame] = useState<FrameResult | null>(null);
  const [fps, setFps] = useState(0);
  const [health, setHealth] = useState<Health | null>(null);
  const [control, setControl] = useState<ControlState | null>(null);
  const [connected, setConnected] = useState(false);
  const [lastFrameAt, setLastFrameAt] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const frameRef = useRef<FrameResult | null>(null);

  const applyFrame = useCallback((incoming: FrameResult) => {
    if (frameRef.current?.frame_id === incoming.frame_id) return;
    frameRef.current = incoming;
    setFrame(incoming);
    setFps(incoming.fps || 0);
    setLastFrameAt(Date.now());
    setError(null);
  }, []);

  const refreshHealth = useCallback(async () => {
    try {
      const next = await api.health();
      setHealth(next);
      setFps(next.fps);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }, []);

  const refreshControl = useCallback(async () => {
    try {
      const next = await api.controlState();
      setControl(next);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }, []);

  // WebSocket is the primary transport; reconnects forever with a short backoff.
  useEffect(() => {
    let socket: WebSocket | null = null;
    let retry: number | undefined;
    let disposed = false;

    const connect = () => {
      if (disposed) return;
      const protocol = window.location.protocol === "https:" ? "wss" : "ws";
      socket = new WebSocket(`${protocol}://${window.location.host}/ws/live`);
      socket.onopen = () => {
        if (!disposed) {
          setConnected(true);
          setError(null);
        }
      };
      socket.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data) as { type?: string; frame?: FrameResult };
          if (message.type === "frame" && message.frame) applyFrame(message.frame);
        } catch {
          /* Ignore malformed frames; polling remains as fallback. */
        }
      };
      socket.onerror = () => {
        try {
          socket?.close();
        } catch {
          /* noop */
        }
      };
      socket.onclose = () => {
        if (disposed) return;
        setConnected(false);
        retry = window.setTimeout(connect, 2000);
      };
    };

    connect();
    return () => {
      disposed = true;
      if (retry) window.clearTimeout(retry);
      try {
        socket?.close();
      } catch {
        /* noop */
      }
    };
  }, [applyFrame]);

  // Snapshot polling covers a blocked WebSocket and hydrates the first paint.
  useEffect(() => {
    let disposed = false;
    const tick = async () => {
      try {
        const latest = await api.latest();
        if (disposed) return;
        if (latest.frame && latest.frame.frame_id !== frameRef.current?.frame_id) {
          applyFrame(latest.frame);
        }
        if (!latest.frame) setFps(latest.fps);
      } catch (err) {
        if (!disposed) setError(err instanceof Error ? err.message : String(err));
      }
    };
    void tick();
    const timer = window.setInterval(() => void tick(), 3000);
    return () => {
      disposed = true;
      window.clearInterval(timer);
    };
  }, [applyFrame]);

  useEffect(() => {
    void refreshHealth();
    const timer = window.setInterval(() => void refreshHealth(), 5000);
    return () => window.clearInterval(timer);
  }, [refreshHealth]);

  useEffect(() => {
    void refreshControl();
    const timer = window.setInterval(() => void refreshControl(), 4000);
    return () => window.clearInterval(timer);
  }, [refreshControl]);

  const value = useMemo(
    () => ({
      frame,
      fps,
      health,
      control,
      connected,
      lastFrameAt,
      error,
      refreshControl,
      refreshHealth,
    }),
    [frame, fps, health, control, connected, lastFrameAt, error, refreshControl, refreshHealth],
  );

  return <LiveContext.Provider value={value}>{children}</LiveContext.Provider>;
}

export function useLive(): LiveContextValue {
  const value = useContext(LiveContext);
  if (!value) throw new Error("useLive must be used inside LiveProvider");
  return value;
}
