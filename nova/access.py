"""Sign-in policy: who may use the app, decided from secrets and ID-token claims.

Imports no streamlit. The UI hands `decide_access` a snapshot of `st.secrets`, the
`st.user` claims, and the environment facts it reads, then renders the returned
`Decision`. Everything here FAILS CLOSED: a missing, partial, or unreadable
configuration blocks the app rather than downgrading it to anonymous.

Operator-facing `problem` strings name configuration keys, never their values, and
never user data; visitors are shown only a generic message (see streamlit_app).
"""

import logging
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from nova.config import (
    ALLOW_ANONYMOUS_ENV,
    MAX_CLOCK_SKEW_SECONDS,
    MAX_SESSION_AGE_SECONDS,
    MIN_COOKIE_SECRET_LENGTH,
)

SecretsState = Literal["ok", "missing", "malformed"]
DecisionKind = Literal["allow", "anonymous", "login", "deny", "blocked"]
DenyReason = Literal[
    "auth_not_configured",
    "auth_misconfigured",
    "email_missing",
    "email_unverified",
    "domain_not_allowed",
]

# The three keys every OIDC provider needs (Streamlit's validate_auth_credentials).
PROVIDER_KEYS = ("client_id", "client_secret", "server_metadata_url")
# Name of the flat [auth] provider: Streamlit's `provider` claim for st.login(None),
# and how [access] unverified_email_providers refers to it.
DEFAULT_PROVIDER = "default"
# Issuers whose tokens must also carry a matching `hd` (hosted domain) claim: Google
# says to check `hd`, not the email's domain, when restricting to an organization —
# a personal Google account can be registered on a work address.
GOOGLE_ISSUERS = frozenset({"https://accounts.google.com", "accounts.google.com"})
MAX_EMAIL_LENGTH = 254
# cookie_secret values that are obviously a template placeholder, or too uniform.
_PLACEHOLDER_MARKERS = ("change-me", "changeme", "replace")
_MIN_SECRET_DISTINCT_CHARS = 10
# An allowlisted domain: lowercase labels joined by dots, at least two labels. Rejects
# wildcards, "@", whitespace, and bare hosts.
_DOMAIN = re.compile(r"[a-z0-9-]+(\.[a-z0-9-]+)+")

# Operator-facing problems (written to stderr by the UI; never shown to visitors).
PROBLEM_SECRETS_UNPARSEABLE = (
    "the secrets file could not be read — check .streamlit/secrets.toml for TOML "
    "syntax errors"
)
PROBLEM_OPT_OUT_IN_SECRETS = (
    f"{ALLOW_ANONYMOUS_ENV} is set in secrets.toml — remove it; anonymous mode can "
    "only be enabled in the process environment"
)
PROBLEM_OPT_OUT_IN_DOTENV = (
    f"{ALLOW_ANONYMOUS_ENV} is set in .env — remove it; anonymous mode can only be "
    "enabled in the process environment"
)
PROBLEM_NOT_CONFIGURED = (
    "sign-in is not configured — add [auth] and [access] to .streamlit/secrets.toml "
    "(see .streamlit/secrets.toml.example), or, for local development only, launch "
    f"with {ALLOW_ANONYMOUS_ENV}=1 set in the process environment"
)
PROBLEM_ACCESS_WITHOUT_AUTH = (
    "[access] is set but [auth] is missing — configure [auth], or remove [access]"
)
PROBLEM_AUTH_TABLE = "[auth] must be a table"
PROBLEM_REDIRECT_URI = "[auth] redirect_uri must end in /oauth2callback"
PROBLEM_COOKIE_SECRET = (
    f"[auth] cookie_secret must be a random string of at least "
    f"{MIN_COOKIE_SECRET_LENGTH} characters, not the template placeholder — generate "
    'one with: python -c "import secrets; print(secrets.token_hex(32))"'
)
PROBLEM_PROVIDER = (
    "each sign-in provider needs non-empty client_id, client_secret and "
    "server_metadata_url — flat in [auth] for the default provider, or in an "
    '[auth.<name>] table whose name has no "_" and is not "default"'
)
PROBLEM_POLICY = (
    "[access] allowed_email_domains must list at least one domain (like "
    '"hospital.org"), and unverified_email_providers may list only configured '
    'provider names ("default" for the flat [auth] provider)'
)
PROBLEM_AUTHLIB = "Authlib is not installed — install the streamlit[auth] extra"
PROBLEM_TRUSTED_HEADERS = (
    "server.trustedUserHeaders is set — header claims override sign-in claims, so "
    "it is refused while [auth] is configured"
)

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SecretsSnapshot:
    """What the UI read from `st.secrets` (its raw values; AttrDicts are Mappings)."""

    state: SecretsState
    auth: object = None
    access: object = None
    anonymous_opt_out_present: bool = False  # ALLOW_ANONYMOUS_ENV found in secrets


@dataclass(frozen=True)
class AuthConfig:
    """The parsed [auth] section: providers in button order, or the first problem."""

    providers: tuple[str | None, ...]  # None = the flat default provider
    problem: str | None = None


@dataclass(frozen=True)
class AccessPolicy:
    """The parsed [access] section."""

    allowed_domains: frozenset[str]
    # Providers whose tokens may omit `email_verified` (DEFAULT_PROVIDER = flat one).
    unverified_email_providers: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Decision:
    """What the gate decided. Build deny/blocked ones via `_deny` / `_blocked`, which
    require a reason, so every refusal carries one."""

    kind: DecisionKind
    reason: DenyReason | None = None
    email: str | None = None
    providers: tuple[str | None, ...] = ()
    problem: str | None = None  # operator-facing; names keys, never values
    reauth: bool = False  # login because the previous sign-in expired or is stale


def _blocked(reason: DenyReason, problem: str) -> Decision:
    return Decision("blocked", reason=reason, problem=problem)


def _deny(reason: DenyReason, email: str | None) -> Decision:
    return Decision("deny", reason=reason, email=email)


def anonymous_opt_out(environ: Mapping[str, str]) -> bool:
    """True only for the exact value "1" ("true", "yes", "0", "" are all False)."""
    return environ.get(ALLOW_ANONYMOUS_ENV) == "1"


def normalize_email(value: object) -> str | None:
    """Strip and lowercase an email claim; None unless it is one plausible address."""
    if not isinstance(value, str):
        return None
    email = value.strip().lower()
    if (
        len(email) > MAX_EMAIL_LENGTH
        or email.count("@") != 1
        or any(c.isspace() for c in email)
    ):
        return None
    local, _, domain = email.partition("@")
    return email if local and domain else None


def _non_empty(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _strong_secret(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) >= MIN_COOKIE_SECRET_LENGTH
        and not any(marker in value.lower() for marker in _PLACEHOLDER_MARKERS)
        and len(set(value)) >= _MIN_SECRET_DISTINCT_CHARS
    )


def provider_name(provider: str | None) -> str:
    """The `provider` claim / policy name for a configured provider (None = default)."""
    return DEFAULT_PROVIDER if provider is None else provider


def parse_auth(auth: object) -> AuthConfig | None:
    """Validate [auth] the way Streamlit's `validate_auth_credentials` would, and more.

    Returns None when [auth] is absent. Otherwise the providers in button order — the
    flat default provider first, then each `[auth.<name>]` table in file order (the
    `client_kwargs` table is options, not a provider) — or the first problem found.
    Stricter than Streamlit: every provider is checked up front (not on click), the
    cookie secret must look random, and "default" is reserved for the flat provider.
    """
    if auth is None:
        return None
    if not isinstance(auth, Mapping):
        return AuthConfig((), PROBLEM_AUTH_TABLE)
    redirect_uri = auth.get("redirect_uri")
    if not isinstance(redirect_uri, str) or not redirect_uri.endswith(
        "/oauth2callback"
    ):
        return AuthConfig((), PROBLEM_REDIRECT_URI)
    if not _strong_secret(auth.get("cookie_secret")):
        return AuthConfig((), PROBLEM_COOKIE_SECRET)

    providers: list[str | None] = []
    if any(key in auth for key in PROVIDER_KEYS):
        if not all(_non_empty(auth.get(key)) for key in PROVIDER_KEYS):
            return AuthConfig((), PROBLEM_PROVIDER)  # a partial flat provider
        providers.append(None)
    for name, section in auth.items():
        if name == "client_kwargs" or not isinstance(section, Mapping):
            continue
        if (
            "_" in name
            or name == DEFAULT_PROVIDER
            or not all(_non_empty(section.get(key)) for key in PROVIDER_KEYS)
        ):
            return AuthConfig((), PROBLEM_PROVIDER)
        providers.append(name)
    if not providers:
        return AuthConfig((), PROBLEM_PROVIDER)
    return AuthConfig(tuple(providers))


def _string_list(value: object) -> list[str] | None:
    """A TOML list of strings, or a comma-separated string (k8s directory secrets
    only yield strings); None for anything else."""
    if isinstance(value, str):
        return [part for part in value.split(",") if part.strip()]
    if isinstance(value, list | tuple) and all(isinstance(v, str) for v in value):
        return list(value)
    return None


def parse_policy(
    access: object, providers: tuple[str | None, ...]
) -> AccessPolicy | None:
    """Validate [access]; None (fail closed) when it is missing or invalid.

    `allowed_email_domains` must name at least one exact domain (normalized:
    stripped, lowercased, a leading "@" dropped). `unverified_email_providers`
    (optional, default none) may list only the configured `providers`' names.
    """
    if not isinstance(access, Mapping):
        return None
    domains = _string_list(access.get("allowed_email_domains"))
    if not domains:
        return None
    allowed = frozenset(d.strip().lower().removeprefix("@") for d in domains)
    if not all(_DOMAIN.fullmatch(d) for d in allowed):
        return None
    unverified = _string_list(access.get("unverified_email_providers", []))
    if unverified is None:
        return None
    trusted = frozenset(name.strip() for name in unverified)
    if not trusted <= {provider_name(p) for p in providers}:
        return None
    return AccessPolicy(allowed, trusted)


def check_email(
    claims: Mapping[str, object], policy: AccessPolicy
) -> DenyReason | None:
    """None when the signed-in email passes the policy, else why it does not.

    The email must be verified: `email_verified` must be True (or the string
    "true"); a token that omits the claim passes only from a provider listed in
    `unverified_email_providers`. Its domain must be allowlisted EXACTLY — no
    suffix or subdomain match — and a Google token's `hd` must be allowlisted too.
    """
    email = normalize_email(claims.get("email"))
    if email is None:
        return "email_missing"
    if "email_verified" in claims:
        verified = claims["email_verified"]
        if not (
            verified is True
            or (isinstance(verified, str) and verified.lower() == "true")
        ):
            return "email_unverified"
    else:
        provider = claims.get("provider")
        if (
            not isinstance(provider, str)
            or provider not in policy.unverified_email_providers
        ):
            return "email_unverified"
    if email.rpartition("@")[2] not in policy.allowed_domains:
        return "domain_not_allowed"
    issuer = claims.get("iss")
    if isinstance(issuer, str) and issuer in GOOGLE_ISSUERS:
        hosted_domain = claims.get("hd")
        if (
            not isinstance(hosted_domain, str)
            or hosted_domain.strip().lower() not in policy.allowed_domains
        ):
            return "domain_not_allowed"
    return None


def session_is_fresh(
    claims: Mapping[str, object], providers: tuple[str | None, ...], now: float
) -> bool:
    """True when the sign-in is recent enough and from a provider still configured.

    Stale when `iat` is not a finite number (bools excluded), is older than
    MAX_SESSION_AGE_SECONDS, or is more than MAX_CLOCK_SKEW_SECONDS in the future;
    also when the `provider` claim names a provider no longer in [auth].
    """
    issued_at = claims.get("iat")
    if (
        isinstance(issued_at, bool)
        or not isinstance(issued_at, int | float)
        or not math.isfinite(issued_at)
    ):
        return False
    if now - issued_at > MAX_SESSION_AGE_SECONDS:
        return False
    if issued_at > now + MAX_CLOCK_SKEW_SECONDS:
        return False
    provider = claims.get("provider")
    return isinstance(provider, str) and provider in {
        provider_name(p) for p in providers
    }


def decide_access(
    secrets: SecretsSnapshot,
    *,
    claims: Mapping[str, object],
    allow_anonymous: bool,
    opt_out_in_dotenv: bool,
    trusted_headers: bool,
    authlib_installed: bool,
    now: float,
) -> Decision:
    """Decide who may use the app. The first matching rule wins:

    1. Unreadable secrets, or the anonymous opt-out found in secrets.toml or .env ->
       blocked, even with the opt-out set, so a broken or copied deploy never
       downgrades to anonymous.
    2. No [auth]: [access] alone -> blocked; the opt-out -> anonymous; else blocked.
    3. [auth] invalid, Authlib missing, trusted user headers on, or [access]
       invalid -> blocked. (The opt-out is ignored whenever [auth] exists.)
    4. Not signed in (`is_logged_in` is not exactly True — an injected `email` alone
       never counts) -> login; signed in but stale -> login with `reauth`.
    5. The email policy -> deny with a reason, or allow.
    """
    if secrets.state == "malformed":
        return _blocked("auth_misconfigured", PROBLEM_SECRETS_UNPARSEABLE)
    if secrets.anonymous_opt_out_present:
        return _blocked("auth_misconfigured", PROBLEM_OPT_OUT_IN_SECRETS)
    if opt_out_in_dotenv:
        return _blocked("auth_misconfigured", PROBLEM_OPT_OUT_IN_DOTENV)

    auth = parse_auth(secrets.auth)
    if auth is None:
        if secrets.access is not None:
            return _blocked("auth_misconfigured", PROBLEM_ACCESS_WITHOUT_AUTH)
        if allow_anonymous:
            return Decision("anonymous")
        return _blocked("auth_not_configured", PROBLEM_NOT_CONFIGURED)
    if auth.problem is not None:
        return _blocked("auth_misconfigured", auth.problem)
    if not authlib_installed:
        return _blocked("auth_misconfigured", PROBLEM_AUTHLIB)
    if trusted_headers:
        return _blocked("auth_misconfigured", PROBLEM_TRUSTED_HEADERS)
    policy = parse_policy(secrets.access, auth.providers)
    if policy is None:
        return _blocked("auth_misconfigured", PROBLEM_POLICY)

    if claims.get("is_logged_in") is not True:
        return Decision("login", providers=auth.providers)
    if not session_is_fresh(claims, auth.providers, now):
        return Decision("login", providers=auth.providers, reauth=True)
    email = normalize_email(claims.get("email"))
    reason = check_email(claims, policy)
    if reason is not None:
        return _deny(reason, email)
    return Decision("allow", email=email, providers=auth.providers)


def report_problem(problem: str) -> None:
    """Log an operator-facing configuration problem to stderr.

    A plain stdlib logger ("nova.access"): with no handlers configured it reaches
    stderr through logging's last-resort handler. `problem` names keys, never values.
    """
    _LOGGER.error("Sign-in is unavailable: %s", problem)
