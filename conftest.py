# Root conftest.py — pytest adds this file's directory to sys.path,
# allowing `import streamlit_app` in tests without packaging or sys.path hacks.
#
# It also pins what `import streamlit_app` sees, before any test module is collected:
# the import runs the module-level access gate (in bare mode), and neither a
# developer's .env nor their secrets.toml may steer it.
import os

from streamlit import config

from nova.config import ALLOW_ANONYMOUS_ENV

# Any value but "1", so a developer's shell cannot put this process on the anonymous
# branch. (.env never can: the app's _load_dotenv never copies the opt-out.)
os.environ[ALLOW_ANONYMOUS_ENV] = "0"
# No secrets files, so neither ./.streamlit/secrets.toml nor ~/.streamlit/secrets.toml
# reaches the gate (or mirrors its top-level values into os.environ).
config.set_option("secrets.files", [])
