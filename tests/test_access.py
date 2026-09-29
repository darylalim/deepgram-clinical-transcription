"""The sign-in policy (`nova.access`), tested directly with plain dicts — no streamlit.

The UI's rendering of each decision is tested in test_streamlit_app.py
(TestAccessGate); the tracked secrets template is parsed here through Streamlit's own
secrets loader, so the example an operator copies is known to fail closed unedited.
"""

import math
import os
import secrets as pysecrets
from pathlib import Path
from unittest.mock import patch

import pytest

from nova import access
from nova.access import (
    PROBLEM_ACCESS_WITHOUT_AUTH,
    PROBLEM_AUTH_TABLE,
    PROBLEM_AUTHLIB,
    PROBLEM_COOKIE_SECRET,
    PROBLEM_NOT_CONFIGURED,
    PROBLEM_OPT_OUT_IN_DOTENV,
    PROBLEM_OPT_OUT_IN_SECRETS,
    PROBLEM_POLICY,
    PROBLEM_PROVIDER,
    PROBLEM_REDIRECT_URI,
    PROBLEM_SECRETS_UNPARSEABLE,
    PROBLEM_TRUSTED_HEADERS,
    AccessPolicy,
    SecretsSnapshot,
    anonymous_opt_out,
    check_email,
    decide_access,
    normalize_email,
    parse_auth,
    parse_policy,
    session_is_fresh,
)
from nova.config import (
    ALLOW_ANONYMOUS_ENV,
    MAX_CLOCK_SKEW_SECONDS,
    MAX_SESSION_AGE_SECONDS,
)

TEMPLATE = Path(__file__).resolve().parent.parent / ".streamlit/secrets.toml.example"
NOW = 1_800_000_000.0
SECRET = "k3y-0123456789abcdefghijklmnopqrstuv"  # 36 chars, many distinct

FLAT = {
    "client_id": "id",
    "client_secret": "secret",
    "server_metadata_url": "https://idp.example/.well-known/openid-configuration",
}
AUTH = {
    "redirect_uri": "http://localhost:8501/oauth2callback",
    "cookie_secret": SECRET,
    **FLAT,
}
ACCESS = {"allowed_email_domains": ["hospital.org"]}
POLICY = AccessPolicy(frozenset({"hospital.org"}))
SIGNED_IN = {
    "is_logged_in": True,
    "email": "Dr@Hospital.org",
    "email_verified": True,
    "provider": "default",
    "iat": NOW - 60,
}


def _decide(
    snapshot=None,
    *,
    claims=None,
    allow_anonymous=False,
    opt_out_in_dotenv=False,
    trusted_headers=False,
    authlib_installed=True,
    now=NOW,
):
    return decide_access(
        SecretsSnapshot("ok", AUTH, ACCESS) if snapshot is None else snapshot,
        claims=SIGNED_IN if claims is None else claims,
        allow_anonymous=allow_anonymous,
        opt_out_in_dotenv=opt_out_in_dotenv,
        trusted_headers=trusted_headers,
        authlib_installed=authlib_installed,
        now=now,
    )


class TestAnonymousOptOut:
    def test_only_the_exact_value_1_opts_out(self):
        assert anonymous_opt_out({ALLOW_ANONYMOUS_ENV: "1"}) is True

    @pytest.mark.parametrize("value", ["true", "yes", "0", "", " 1", "1 "])
    def test_anything_else_does_not(self, value):
        assert anonymous_opt_out({ALLOW_ANONYMOUS_ENV: value}) is False

    def test_absent_does_not(self):
        assert anonymous_opt_out({}) is False


class TestNormalizeEmail:
    def test_strips_and_lowercases(self):
        assert normalize_email("  Dr.Smith@Hospital.ORG ") == "dr.smith@hospital.org"

    @pytest.mark.parametrize(
        "value",
        [
            None,
            42,
            ["a@b.org"],
            "no-at-sign",
            "a@b@c.org",
            "dr smith@hospital.org",
            "dr@hospital.org\nx",
            "@hospital.org",
            "dr@",
            "a" * 250 + "@b.org",  # 256 chars > 254
        ],
    )
    def test_rejects_anything_but_one_plausible_address(self, value):
        assert normalize_email(value) is None


class TestParseAuth:
    def test_absent_is_none(self):
        assert parse_auth(None) is None

    def test_non_table_is_a_problem(self):
        assert parse_auth("oops") == access.AuthConfig((), PROBLEM_AUTH_TABLE)

    def test_flat_provider_is_the_default(self):
        assert parse_auth(AUTH) == access.AuthConfig((None,))

    def test_named_providers_follow_the_default_in_file_order(self):
        auth = {
            **AUTH,
            "client_kwargs": {"prompt": "login"},  # options, not a provider
            "microsoft": dict(FLAT),
            "okta": dict(FLAT),
        }

        assert parse_auth(auth) == access.AuthConfig((None, "microsoft", "okta"))

    def test_named_providers_without_a_default(self):
        auth: dict[str, object] = {k: v for k, v in AUTH.items() if k not in FLAT}
        auth["google"] = dict(FLAT)

        assert parse_auth(auth) == access.AuthConfig(("google",))

    @pytest.mark.parametrize(
        "auth",
        [
            {**AUTH, "my_idp": dict(FLAT)},  # "_" is not allowed in a provider name
            {**AUTH, "default": dict(FLAT)},  # reserved for the flat provider
            {**AUTH, "okta": {"client_id": "id", "client_secret": "s"}},  # incomplete
            {**AUTH, "okta": {**FLAT, "client_secret": "  "}},  # blank value
            {k: v for k, v in AUTH.items() if k != "client_secret"},  # partial flat
            {**AUTH, "client_id": 123},  # non-string flat value
            {"redirect_uri": AUTH["redirect_uri"], "cookie_secret": SECRET},  # none
        ],
    )
    def test_provider_problems(self, auth):
        assert parse_auth(auth) == access.AuthConfig((), PROBLEM_PROVIDER)

    @pytest.mark.parametrize(
        "redirect_uri", [None, 1, "http://localhost:8501/", "http://x/oauth2callback/"]
    )
    def test_redirect_uri_must_end_in_oauth2callback(self, redirect_uri):
        auth = {**AUTH, "redirect_uri": redirect_uri}

        assert parse_auth(auth) == access.AuthConfig((), PROBLEM_REDIRECT_URI)

    @pytest.mark.parametrize(
        "cookie_secret",
        [
            None,
            12345678901234567890123456789012345,
            "abcdefghij" + "k" * 21,  # 31 chars
            "CHANGE-ME",  # the template placeholder
            "change-me-0123456789abcdefghijklmnopqrstuvwxyz",
            "CHANGEME-0123456789abcdefghijklmnopqrstuvwxyz",
            "replace-with-output-of-python-secrets-token-hex-32",
            "ab" * 20,  # long enough, but only 2 distinct characters
            "012345678" * 4,  # 9 distinct characters
        ],
    )
    def test_weak_or_placeholder_cookie_secret_is_a_problem(self, cookie_secret):
        auth = {**AUTH, "cookie_secret": cookie_secret}

        assert parse_auth(auth) == access.AuthConfig((), PROBLEM_COOKIE_SECRET)

    def test_random_cookie_secret_passes(self):
        auth = {**AUTH, "cookie_secret": pysecrets.token_hex(32)}

        assert parse_auth(auth) == access.AuthConfig((None,))

    def test_ten_distinct_characters_is_enough(self):
        auth = {**AUTH, "cookie_secret": "0123456789" * 4}

        assert parse_auth(auth) == access.AuthConfig((None,))

    def test_exactly_the_minimum_length_is_enough(self):
        # 32 characters, all distinct, no placeholder marker: the boundary itself.
        secret = "0123456789abcdefghijklmnopqrstuv"
        assert len(secret) == access.MIN_COOKIE_SECRET_LENGTH

        auth = {**AUTH, "cookie_secret": secret}

        assert parse_auth(auth) == access.AuthConfig((None,))


class TestParsePolicy:
    def test_list_of_domains(self):
        policy = parse_policy(
            {"allowed_email_domains": ["hospital.org", "clinic.org"]}, (None,)
        )

        assert policy == AccessPolicy(frozenset({"hospital.org", "clinic.org"}))

    def test_comma_separated_string(self):
        # Kubernetes directory secrets only yield strings.
        policy = parse_policy(
            {"allowed_email_domains": "hospital.org, clinic.org,"}, (None,)
        )

        assert policy == AccessPolicy(frozenset({"hospital.org", "clinic.org"}))

    def test_entries_are_normalized(self):
        policy = parse_policy({"allowed_email_domains": ["@Hospital.ORG "]}, (None,))

        assert policy == AccessPolicy(frozenset({"hospital.org"}))

    @pytest.mark.parametrize(
        "access_section",
        [
            None,
            "hospital.org",  # not a table
            {},
            {"allowed_email_domains": []},
            {"allowed_email_domains": ""},
            {"allowed_email_domains": " , "},
            {"allowed_email_domains": ["*.hospital.org"]},
            {"allowed_email_domains": ["a@b.org"]},
            {"allowed_email_domains": ["localhost"]},  # a bare host
            {"allowed_email_domains": ["hospital .org"]},
            {"allowed_email_domains": ["hospital.org", ""]},
            {"allowed_email_domains": ["hospital.org", 7]},
            {"allowed_email_domains": 7},
        ],
    )
    def test_missing_or_invalid_domains_fail_closed(self, access_section):
        assert parse_policy(access_section, (None,)) is None

    def test_no_unverified_providers_by_default(self):
        assert parse_policy(ACCESS, (None,)) == POLICY

    def test_unverified_providers_must_be_configured(self):
        section = {**ACCESS, "unverified_email_providers": ["microsoft", "default"]}

        policy = parse_policy(section, (None, "microsoft"))

        assert policy == AccessPolicy(
            frozenset({"hospital.org"}), frozenset({"microsoft", "default"})
        )

    def test_unverified_providers_as_a_comma_string(self):
        section = {**ACCESS, "unverified_email_providers": "microsoft"}

        policy = parse_policy(section, (None, "microsoft"))

        assert policy == AccessPolicy(
            frozenset({"hospital.org"}), frozenset({"microsoft"})
        )

    @pytest.mark.parametrize(
        "names", [["microsoft"], ["default"], [""], "okta", 1, [True]]
    )
    def test_unknown_unverified_provider_fails_closed(self, names):
        # "default" names the flat provider, which this config does not have.
        section = {**ACCESS, "unverified_email_providers": names}

        assert parse_policy(section, ("google",)) is None


class TestCheckEmail:
    def test_allowlisted_verified_email_passes(self):
        assert check_email(SIGNED_IN, POLICY) is None

    @pytest.mark.parametrize(
        "email",
        ["dr@evil-hospital.org", "dr@sub.hospital.org", "dr@hospital.org.evil.com"],
    )
    def test_domain_match_is_exact(self, email):
        assert (
            check_email({**SIGNED_IN, "email": email}, POLICY) == "domain_not_allowed"
        )

    def test_listed_subdomain_passes(self):
        policy = AccessPolicy(frozenset({"hospital.org", "sub.hospital.org"}))

        assert (
            check_email({**SIGNED_IN, "email": "dr@sub.hospital.org"}, policy) is None
        )

    @pytest.mark.parametrize("email", [None, 7, "not-an-email"])
    def test_unusable_email_is_missing(self, email):
        assert check_email({**SIGNED_IN, "email": email}, POLICY) == "email_missing"

    def test_email_absent_is_missing(self):
        claims = {k: v for k, v in SIGNED_IN.items() if k != "email"}

        assert check_email(claims, POLICY) == "email_missing"

    @pytest.mark.parametrize("verified", [False, "false", "no", 1, None, "True "])
    def test_email_verified_must_be_true(self, verified):
        claims = {**SIGNED_IN, "email_verified": verified}

        assert check_email(claims, POLICY) == "email_unverified"

    @pytest.mark.parametrize("verified", [True, "true", "TRUE"])
    def test_stringified_true_is_accepted(self, verified):
        assert check_email({**SIGNED_IN, "email_verified": verified}, POLICY) is None

    def test_absent_email_verified_is_unverified_by_default(self):
        claims = {k: v for k, v in SIGNED_IN.items() if k != "email_verified"}

        assert check_email(claims, POLICY) == "email_unverified"

    def test_absent_email_verified_passes_for_a_listed_provider(self):
        # Entra sends no email_verified; the operator lists that provider by name.
        policy = AccessPolicy(frozenset({"hospital.org"}), frozenset({"microsoft"}))
        claims = {k: v for k, v in SIGNED_IN.items() if k != "email_verified"}

        assert check_email({**claims, "provider": "microsoft"}, policy) is None
        # ...but only for that provider: the others must still assert it.
        assert (
            check_email({**claims, "provider": "default"}, policy) == "email_unverified"
        )
        assert (
            check_email({**claims, "provider": ["microsoft"]}, policy)
            == "email_unverified"
        )

    def test_listed_provider_still_rejects_an_explicit_false(self):
        policy = AccessPolicy(frozenset({"hospital.org"}), frozenset({"default"}))

        claims = {**SIGNED_IN, "email_verified": False}

        assert check_email(claims, policy) == "email_unverified"

    @pytest.mark.parametrize(
        "iss", ["https://accounts.google.com", "accounts.google.com"]
    )
    def test_google_requires_an_allowlisted_hosted_domain(self, iss):
        # A personal Google account on a work address has a verified email but no hd.
        google = {**SIGNED_IN, "iss": iss}

        assert check_email(google, POLICY) == "domain_not_allowed"
        assert (
            check_email({**google, "hd": "other.org"}, POLICY) == "domain_not_allowed"
        )
        assert check_email({**google, "hd": 7}, POLICY) == "domain_not_allowed"
        assert check_email({**google, "hd": "Hospital.org"}, POLICY) is None

    def test_other_issuers_need_no_hosted_domain(self):
        claims = {**SIGNED_IN, "iss": "https://login.microsoftonline.com/tid/v2.0"}

        assert check_email(claims, POLICY) is None
        # An unhashable issuer claim is just "not Google", never a crash.
        assert check_email({**SIGNED_IN, "iss": ["x"]}, POLICY) is None


class TestSessionIsFresh:
    PROVIDERS = (None, "microsoft")

    def _fresh(self, **overrides):
        return session_is_fresh({**SIGNED_IN, **overrides}, self.PROVIDERS, NOW)

    def test_recent_sign_in_is_fresh(self):
        assert self._fresh() is True

    def test_boundaries(self):
        assert self._fresh(iat=NOW - MAX_SESSION_AGE_SECONDS) is True
        assert self._fresh(iat=NOW - MAX_SESSION_AGE_SECONDS - 1) is False
        assert self._fresh(iat=NOW + MAX_CLOCK_SKEW_SECONDS) is True
        assert self._fresh(iat=NOW + MAX_CLOCK_SKEW_SECONDS + 1) is False

    @pytest.mark.parametrize(
        "iat", [None, "1800000000", True, math.nan, math.inf, -math.inf, [NOW]]
    )
    def test_iat_must_be_a_finite_number(self, iat):
        assert self._fresh(iat=iat) is False

    def test_iat_missing_is_stale(self):
        claims = {k: v for k, v in SIGNED_IN.items() if k != "iat"}

        assert session_is_fresh(claims, self.PROVIDERS, NOW) is False

    def test_integer_iat(self):
        assert self._fresh(iat=int(NOW) - 5) is True

    @pytest.mark.parametrize("provider", ["microsoft", "default"])
    def test_configured_provider(self, provider):
        assert self._fresh(provider=provider) is True

    @pytest.mark.parametrize("provider", ["okta", None, "", ["default"]])
    def test_provider_no_longer_configured_is_stale(self, provider):
        assert self._fresh(provider=provider) is False


class TestDecideAccess:
    """One test per ordered rule in `decide_access`."""

    def test_malformed_secrets_block_even_with_the_opt_out(self):
        decision = _decide(SecretsSnapshot("malformed"), allow_anonymous=True)

        assert decision.kind == "blocked"
        assert decision.reason == "auth_misconfigured"
        assert decision.problem == PROBLEM_SECRETS_UNPARSEABLE

    def test_opt_out_in_secrets_blocks(self):
        snapshot = SecretsSnapshot("ok", anonymous_opt_out_present=True)

        decision = _decide(snapshot, allow_anonymous=True)

        assert (decision.kind, decision.problem) == (
            "blocked",
            PROBLEM_OPT_OUT_IN_SECRETS,
        )

    @pytest.mark.parametrize(
        "snapshot", [SecretsSnapshot("missing"), SecretsSnapshot("ok", AUTH, ACCESS)]
    )
    def test_opt_out_in_dotenv_blocks_whatever_the_config(self, snapshot):
        decision = _decide(snapshot, allow_anonymous=True, opt_out_in_dotenv=True)

        assert (decision.kind, decision.problem) == (
            "blocked",
            PROBLEM_OPT_OUT_IN_DOTENV,
        )

    def test_access_without_auth_blocks_even_with_the_opt_out(self):
        decision = _decide(SecretsSnapshot("ok", None, ACCESS), allow_anonymous=True)

        assert (decision.kind, decision.problem) == (
            "blocked",
            PROBLEM_ACCESS_WITHOUT_AUTH,
        )

    @pytest.mark.parametrize("state", ["missing", "ok"])
    def test_no_auth_with_the_opt_out_is_anonymous(self, state):
        decision = _decide(SecretsSnapshot(state), allow_anonymous=True)

        assert decision == access.Decision("anonymous")

    @pytest.mark.parametrize("state", ["missing", "ok"])
    def test_no_auth_without_the_opt_out_is_not_configured(self, state):
        decision = _decide(SecretsSnapshot(state))

        assert decision.kind == "blocked"
        assert decision.reason == "auth_not_configured"
        assert decision.problem == PROBLEM_NOT_CONFIGURED

    def test_invalid_auth_blocks_with_its_problem(self):
        snapshot = SecretsSnapshot("ok", {**AUTH, "cookie_secret": "CHANGE-ME"}, ACCESS)

        decision = _decide(snapshot, allow_anonymous=True)

        assert (decision.kind, decision.problem) == ("blocked", PROBLEM_COOKIE_SECRET)

    def test_missing_authlib_blocks(self):
        decision = _decide(authlib_installed=False)

        assert (decision.kind, decision.problem) == ("blocked", PROBLEM_AUTHLIB)

    def test_trusted_user_headers_block(self):
        decision = _decide(trusted_headers=True)

        assert (decision.kind, decision.problem) == ("blocked", PROBLEM_TRUSTED_HEADERS)

    @pytest.mark.parametrize("access_section", [None, {"allowed_email_domains": []}])
    def test_missing_or_empty_allowlist_blocks(self, access_section):
        decision = _decide(SecretsSnapshot("ok", AUTH, access_section))

        assert (decision.kind, decision.problem) == ("blocked", PROBLEM_POLICY)

    @pytest.mark.parametrize(
        "claims",
        [
            {},
            {"is_logged_in": False},  # production's logged-out shape with [auth] set
            {"is_logged_in": "true"},  # identity, not truthiness
            {"is_logged_in": 1},
            # AppTest (and a host or header) can inject an email with no sign-in.
            {"email": "test@example.com"},
            {"email": "dr@hospital.org", "email_verified": True},
        ],
    )
    def test_not_signed_in_means_login(self, claims):
        decision = _decide(claims=claims)

        assert decision == access.Decision("login", providers=(None,))

    def test_opt_out_is_ignored_once_auth_is_configured(self):
        decision = _decide(claims={}, allow_anonymous=True)

        assert decision.kind == "login"

    def test_login_offers_every_provider(self):
        auth = {**AUTH, "microsoft": dict(FLAT)}

        decision = _decide(SecretsSnapshot("ok", auth, ACCESS), claims={})

        assert decision.providers == (None, "microsoft")

    @pytest.mark.parametrize(
        "claims",
        [
            {**SIGNED_IN, "iat": NOW - MAX_SESSION_AGE_SECONDS - 1},
            {k: v for k, v in SIGNED_IN.items() if k != "iat"},
            {**SIGNED_IN, "provider": "okta"},  # removed from [auth] since sign-in
            # Freshness is checked before the email: a stale sign-in by an account
            # that would be refused is asked to sign in again, never denied (and
            # never logged as access_denied).
            {
                **SIGNED_IN,
                "email": "dr@evil.org",
                "iat": NOW - MAX_SESSION_AGE_SECONDS - 1,
            },
            {**SIGNED_IN, "email": "dr@evil.org", "provider": "okta"},
        ],
    )
    def test_stale_sign_in_means_login_again(self, claims):
        decision = _decide(claims=claims)

        assert decision == access.Decision("login", providers=(None,), reauth=True)

    def test_allowed(self):
        decision = _decide()

        assert decision == access.Decision(
            "allow",
            email="dr@hospital.org",
            providers=(None,),
            expires_at=NOW - 60 + MAX_SESSION_AGE_SECONDS,  # SIGNED_IN's iat
        )

    def test_allowed_sign_in_expires_when_it_turns_stale(self):
        # The UI re-checks `expires_at` where the gate does not run; it must agree
        # with the gate's own boundary: still allowed at it, stale one second past.
        claims = {**SIGNED_IN, "iat": NOW - 60}
        expires_at = _decide(claims=claims).expires_at
        assert expires_at == NOW - 60 + MAX_SESSION_AGE_SECONDS

        assert _decide(claims=claims, now=expires_at).kind == "allow"
        assert _decide(claims=claims, now=expires_at + 1).reauth is True

    def test_only_an_allowed_sign_in_carries_an_expiry(self):
        refused = _decide(claims={**SIGNED_IN, "email": "dr@other.org"})
        anonymous = _decide(SecretsSnapshot("missing"), allow_anonymous=True)

        assert (refused.kind, refused.expires_at) == ("deny", None)
        assert (anonymous.kind, anonymous.expires_at) == ("anonymous", None)

    def test_denied_carries_the_reason_and_normalized_email(self):
        decision = _decide(claims={**SIGNED_IN, "email": " Dr@Evil-Hospital.org"})

        assert decision == access.Decision(
            "deny", reason="domain_not_allowed", email="dr@evil-hospital.org"
        )

    def test_denied_without_an_email(self):
        claims = {k: v for k, v in SIGNED_IN.items() if k != "email"}

        decision = _decide(claims=claims)

        assert (decision.kind, decision.reason, decision.email) == (
            "deny",
            "email_missing",
            None,
        )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"snapshot": SecretsSnapshot("malformed")},
            {"snapshot": SecretsSnapshot("ok", anonymous_opt_out_present=True)},
            {"opt_out_in_dotenv": True},
            {"snapshot": SecretsSnapshot("ok", None, ACCESS)},
            {"snapshot": SecretsSnapshot("missing")},
            {"snapshot": SecretsSnapshot("ok", "oops", ACCESS)},
            {"authlib_installed": False},
            {"trusted_headers": True},
            {"snapshot": SecretsSnapshot("ok", AUTH, None)},
            {"claims": {**SIGNED_IN, "email": "dr@other.org"}},
            {"claims": {**SIGNED_IN, "email_verified": False}},
            {"claims": {**SIGNED_IN, "email": None}},
        ],
    )
    def test_every_refusal_carries_a_reason(self, kwargs):
        # The audit trail logs `reason` for each refusal, so none may lack one.
        decision = _decide(**kwargs)

        assert decision.kind in ("blocked", "deny")
        assert decision.reason is not None
        if decision.kind == "blocked":
            assert decision.problem


class TestReportProblem:
    def test_logs_to_the_access_logger(self, caplog):
        with caplog.at_level("ERROR", logger="nova.access"):
            access.report_problem(PROBLEM_POLICY)

        (record,) = caplog.records
        assert record.name == "nova.access"
        assert record.getMessage() == f"Sign-in is unavailable: {PROBLEM_POLICY}"


class TestSecretsTemplate:
    """`.streamlit/secrets.toml.example`, loaded through Streamlit's own secrets
    parser (TOML flavor and AttrDict wrapping included), must fail closed as
    shipped — its cookie_secret is a placeholder — and pass once that is replaced."""

    @staticmethod
    def _load(tmp_path, text):
        from streamlit import config
        from streamlit.runtime.secrets import Secrets

        path = tmp_path / "secrets.toml"
        path.write_text(text)
        saved = config.get_option("secrets.files")
        config.set_option("secrets.files", [str(path)])
        # patch.dict restores os.environ, which a parse mirrors top-level scalars into.
        try:
            with patch.dict(os.environ):
                loaded = Secrets()
                # No poll-watcher thread on a temp file in the test process.
                loaded._file_watchers_installed = True
                return {key: loaded[key] for key in loaded}
        finally:
            config.set_option("secrets.files", saved)

    def test_template_fails_closed_until_the_secret_is_replaced(self, tmp_path):
        text = TEMPLATE.read_text()
        loaded = self._load(tmp_path, text)

        auth = parse_auth(loaded["auth"])
        assert auth is not None and auth.problem == PROBLEM_COOKIE_SECRET

        placeholder = 'cookie_secret = "CHANGE-ME"'
        assert text.count(placeholder) == 1
        real = text.replace(placeholder, f'cookie_secret = "{pysecrets.token_hex(32)}"')
        loaded = self._load(tmp_path, real)
        auth = parse_auth(loaded["auth"])
        assert auth is not None and auth.problem is None
        assert auth.providers == (None,)
        assert parse_policy(loaded["access"], auth.providers) is not None

    def test_template_holds_only_tables(self, tmp_path):
        # No top-level values: Streamlit mirrors those into os.environ, and the
        # anonymous opt-out must never be set from a secrets file.
        loaded = self._load(tmp_path, TEMPLATE.read_text())

        assert ALLOW_ANONYMOUS_ENV not in loaded
        assert set(loaded) == {"auth", "access"}
