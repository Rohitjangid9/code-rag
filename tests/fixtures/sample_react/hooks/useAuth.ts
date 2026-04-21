import { useState, useEffect } from "react";

interface User {
  id: string;
  name: string;
  email: string;
  avatarUrl?: string;
}

export function useAuth(userId: string) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    fetch(`/api/v1/users/${userId}`)
      .then((r) => r.json())
      .then((data) => setUser(data))
      .finally(() => setLoading(false));
  }, [userId]);

  return { user, loading };
}
