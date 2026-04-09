"""Role-based access control for memory operations.

Provides a lightweight access control layer that governs who can
read, write, and manage memories. Designed for multi-agent and
multi-user deployments where different actors have different
permission levels.

Access levels:
    - READ: Can recall/list/get memories
    - WRITE: Can remember/update/smart_remember + all READ permissions
    - ADMIN: Can forget/decay/consolidate + all WRITE permissions

Usage::

    policy = AccessPolicy()
    policy.grant("agent_reader", AccessLevel.READ, namespace="shared")
    policy.grant("agent_writer", AccessLevel.WRITE, namespace="shared")
    policy.grant("admin_bot", AccessLevel.ADMIN)

    policy.check("agent_reader", AccessLevel.READ, namespace="shared")  # True
    policy.check("agent_reader", AccessLevel.WRITE, namespace="shared")  # False
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from neuragram.core.exceptions import EngramError


class AccessDeniedError(EngramError):
    """Raised when an operation is denied by access control."""

    def __init__(self, actor: str, operation: str, reason: str = "") -> None:
        self.actor = actor
        self.operation = operation
        msg = f"Access denied: {actor} cannot perform '{operation}'"
        if reason:
            msg += f" ({reason})"
        super().__init__(msg)


class AccessLevel(IntEnum):
    """Permission levels, ordered by increasing privilege.

    Higher levels include all permissions of lower levels.
    """

    NONE = 0
    READ = 1
    WRITE = 2
    ADMIN = 3


@dataclass
class AccessGrant:
    """A single permission grant for an actor."""

    actor_id: str
    level: AccessLevel
    namespace: str | None = None  # None means all namespaces
    user_id: str | None = None  # None means all users


class AccessPolicy:
    """Manages access control grants and permission checks.

    Thread-safe for read operations. Grants are stored in memory
    and can be persisted externally if needed.

    Args:
        enabled: Whether access control is enforced. When False,
            all operations are allowed (default for backward compatibility).
        default_level: Default access level for unregistered actors.
    """

    def __init__(
        self,
        enabled: bool = False,
        default_level: AccessLevel = AccessLevel.ADMIN,
    ) -> None:
        self._enabled = enabled
        self._default_level = default_level
        self._grants: list[AccessGrant] = []

    @property
    def enabled(self) -> bool:
        """Whether access control is active."""
        return self._enabled

    def enable(self) -> None:
        """Activate access control enforcement."""
        self._enabled = True

    def disable(self) -> None:
        """Deactivate access control (all operations allowed)."""
        self._enabled = False

    def grant(
        self,
        actor_id: str,
        level: AccessLevel,
        namespace: str | None = None,
        user_id: str | None = None,
    ) -> None:
        """Grant an access level to an actor.

        Args:
            actor_id: The actor (agent, user, service) receiving the grant.
            level: Permission level to grant.
            namespace: Restrict grant to a specific namespace (None = all).
            user_id: Restrict grant to a specific user's memories (None = all).
        """
        # Remove existing grants for the same scope
        self._grants = [
            g for g in self._grants
            if not (
                g.actor_id == actor_id
                and g.namespace == namespace
                and g.user_id == user_id
            )
        ]
        self._grants.append(
            AccessGrant(
                actor_id=actor_id,
                level=level,
                namespace=namespace,
                user_id=user_id,
            )
        )

    def revoke(
        self,
        actor_id: str,
        namespace: str | None = None,
        user_id: str | None = None,
    ) -> None:
        """Revoke all grants for an actor in a given scope.

        Args:
            actor_id: The actor to revoke.
            namespace: Scope to revoke (None = all namespaces).
            user_id: Scope to revoke (None = all users).
        """
        self._grants = [
            g for g in self._grants
            if not (
                g.actor_id == actor_id
                and (namespace is None or g.namespace == namespace)
                and (user_id is None or g.user_id == user_id)
            )
        ]

    def get_level(
        self,
        actor_id: str,
        namespace: str | None = None,
        user_id: str | None = None,
    ) -> AccessLevel:
        """Get the effective access level for an actor in a given scope.

        Checks grants from most specific to least specific:
        1. Exact match (actor + namespace + user_id)
        2. Namespace-only match (actor + namespace)
        3. User-only match (actor + user_id)
        4. Global match (actor only)
        5. Default level

        Args:
            actor_id: The actor to check.
            namespace: The namespace context.
            user_id: The user context.

        Returns:
            The highest applicable AccessLevel.
        """
        if not self._enabled:
            return AccessLevel.ADMIN

        best_level = AccessLevel.NONE
        has_any_grant = False

        for grant in self._grants:
            if grant.actor_id != actor_id:
                continue

            has_any_grant = True

            # Check scope match
            namespace_match = grant.namespace is None or grant.namespace == namespace
            user_match = grant.user_id is None or grant.user_id == user_id

            if namespace_match and user_match:
                best_level = max(best_level, grant.level)

        if not has_any_grant:
            return self._default_level

        return best_level

    def check(
        self,
        actor_id: str,
        required_level: AccessLevel,
        namespace: str | None = None,
        user_id: str | None = None,
    ) -> bool:
        """Check if an actor has the required access level.

        Args:
            actor_id: The actor to check.
            required_level: Minimum required permission level.
            namespace: The namespace context.
            user_id: The user context.

        Returns:
            True if the actor has sufficient permissions.
        """
        effective = self.get_level(actor_id, namespace=namespace, user_id=user_id)
        return effective >= required_level

    def enforce(
        self,
        actor_id: str,
        required_level: AccessLevel,
        operation: str,
        namespace: str | None = None,
        user_id: str | None = None,
    ) -> None:
        """Enforce access control, raising AccessDeniedError if denied.

        Args:
            actor_id: The actor attempting the operation.
            required_level: Minimum required permission level.
            operation: Description of the operation (for error messages).
            namespace: The namespace context.
            user_id: The user context.

        Raises:
            AccessDeniedError: If the actor lacks sufficient permissions.
        """
        if not self.check(actor_id, required_level, namespace=namespace, user_id=user_id):
            effective = self.get_level(actor_id, namespace=namespace, user_id=user_id)
            raise AccessDeniedError(
                actor=actor_id,
                operation=operation,
                reason=f"requires {required_level.name}, has {effective.name}",
            )

    def list_grants(self, actor_id: str | None = None) -> list[dict[str, Any]]:
        """List all grants, optionally filtered by actor.

        Args:
            actor_id: Filter by actor (None = all actors).

        Returns:
            List of grant dicts with actor_id, level, namespace, user_id.
        """
        grants = self._grants
        if actor_id is not None:
            grants = [g for g in grants if g.actor_id == actor_id]

        return [
            {
                "actor_id": g.actor_id,
                "level": g.level.name,
                "namespace": g.namespace,
                "user_id": g.user_id,
            }
            for g in grants
        ]
