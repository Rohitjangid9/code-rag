import React from "react";
import { useAuth } from "./hooks/useAuth";
import Avatar from "./Avatar";

interface UserCardProps {
  userId: string;
  showEmail?: boolean;
}

export function UserCard({ userId, showEmail = false }: UserCardProps) {
  /**
   * Displays a single user card with avatar and optional email.
   */
  const { user, loading } = useAuth(userId);

  if (loading) return <div>Loading…</div>;

  return (
    <div className="user-card">
      <Avatar src={user?.avatarUrl} />
      <h2>{user?.name}</h2>
      {showEmail && <p>{user?.email}</p>}
    </div>
  );
}

export default UserCard;
