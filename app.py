"""
app.py — TradeVision Streamlit entry point.

Multi-page setup using st.navigation.
The model is loaded once via @st.cache_resource and passed into pages.
"""

# ── Ensure project root is on sys.path so `config` and package imports work ──
# _bootstrap patches sys.path at import time, before any project modules load.
import _bootstrap  # noqa: F401

import streamlit as st

from tradevision.inference.predictor import ModelNotFoundError, get_predictor
from tradevision.utils.logger import get_logger
from ui.pages.home import render_home
from ui.pages.about import render_about

logger = get_logger(__name__)


# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TradeVision",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/youruser/tradevision",
        "Report a bug": "https://github.com/youruser/tradevision/issues",
        "About": "TradeVision — AI chart pattern classifier",
    },
)

# ─── Global CSS ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* Dark sidebar */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        }

        section[data-testid="stSidebar"] * {
            color: #cbd5e1 !important;
        }

        /* Hide default Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* Main content area */
        .main .block-container {
            padding-top: 1.5rem;
            max-width: 1200px;
        }

        /* Spinner colour */
        .stSpinner > div {
            border-top-color: #6366f1 !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─── Cached model loader ──────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading TradeVision model …")
def _load_predictor():
    """Load the Keras model once for the lifetime of the Streamlit process."""
    predictor = get_predictor()
    try:
        # Eagerly trigger load so the first request is fast
        predictor._ensure_loaded()  # noqa: SLF001
        logger.info("Model pre-loaded via cache_resource")
    except ModelNotFoundError:
        logger.warning("Model file not found; will show info banner on Home page")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error during model pre-load: %s", exc)
    return predictor


predictor = _load_predictor()


# ─── Navigation ──────────────────────────────────────────────────────────────

def _home_page():
    render_home(predictor)


def _about_page():
    render_about()


home_page = st.Page(_home_page, title="Home", icon="📈", default=True)
about_page = st.Page(_about_page, title="About", icon="ℹ️")

pg = st.navigation(
    {
        "TradeVision": [home_page, about_page],
    }
)
pg.run()
