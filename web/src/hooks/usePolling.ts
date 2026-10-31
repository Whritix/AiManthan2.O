import { useCallback, useEffect, useRef, useState } from "react";

interface PollingState<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
  refresh: () => Promise<void>;
  setData: (next: T | null) => void;
}

/**
 * Poll an async source while mounted. The latest fetcher is kept in a ref so
 * inline arrow functions do not restart the interval on every render.
 */
export function usePolling<T>(fetcher: () => Promise<T>, intervalMs = 0): PollingState<T> {
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;
  const mountedRef = useRef(true);
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      const next = await fetcherRef.current();
      if (mountedRef.current) {
        setData(next);
        setError(null);
      }
    } catch (err) {
      if (mountedRef.current) {
        setError(err instanceof Error ? err.message : String(err));
      }
    } finally {
      if (mountedRef.current) setLoading(false);
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    void refresh();
    if (intervalMs <= 0) {
      return () => {
        mountedRef.current = false;
      };
    }
    const timer = window.setInterval(() => void refresh(), intervalMs);
    return () => {
      mountedRef.current = false;
      window.clearInterval(timer);
    };
  }, [intervalMs, refresh]);

  return { data, error, loading, refresh, setData };
}

/** Re-render at a fixed cadence; used for clocks and relative timestamps. */
export function useNow(intervalMs = 1000): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), intervalMs);
    return () => window.clearInterval(timer);
  }, [intervalMs]);
  return now;
}
