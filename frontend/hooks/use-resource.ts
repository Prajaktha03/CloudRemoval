"use client";

import { useCallback, useEffect, useState } from "react";
import { apiFetch, errorMessage } from "@/lib/api";

export function useResource<T>(path: string) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setData(await apiFetch<T>(path));
    } catch (caught) {
      setError(errorMessage(caught));
    } finally {
      setLoading(false);
    }
  }, [path]);

  useEffect(() => {
    let mounted = true;
    apiFetch<T>(path).then((result) => {
      if (mounted) setData(result);
    }).catch((caught: unknown) => {
      if (mounted) setError(errorMessage(caught));
    }).finally(() => {
      if (mounted) setLoading(false);
    });
    return () => { mounted = false; };
  }, [path]);

  return { data, error, loading, refresh };
}
