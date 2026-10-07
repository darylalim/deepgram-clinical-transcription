"""Transcription core for the Deepgram clinical transcription app.

Imports no streamlit; the UI consumes it in-process and the walkers/option builders
are unit-tested directly. It also holds the sign-in access policy (`nova.access`)
and the PHI-free audit trail (`nova.audit`).
"""
