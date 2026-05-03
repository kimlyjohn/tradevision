"""
ui/theme.py -- Shared presentation helpers for the Streamlit UI layer.

Defines the global design system, reusable HTML helpers, and small
rendering utilities so pages/components stay visually consistent.
"""

from __future__ import annotations

from html import escape

import streamlit as st


STATUS_STYLES: dict[str, dict[str, str]] = {
    "neutral": {
        "label": "System ready",
        "background": "#EFF6FF",
        "foreground": "#1D4ED8",
        "border": "#BFDBFE",
    },
    "success": {
        "label": "Analysis complete",
        "background": "#ECFDF3",
        "foreground": "#047857",
        "border": "#A7F3D0",
    },
    "warning": {
        "label": "Review advised",
        "background": "#FFF7ED",
        "foreground": "#B45309",
        "border": "#FED7AA",
    },
    "critical": {
        "label": "Model unavailable",
        "background": "#FEF2F2",
        "foreground": "#B91C1C",
        "border": "#FECACA",
    },
    "processing": {
        "label": "Analyzing chart",
        "background": "#EEF2FF",
        "foreground": "#4338CA",
        "border": "#C7D2FE",
    },
}


def apply_theme() -> None:
    """Inject the shared design system CSS."""
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

            :root {
                --tv-bg: #f3f7fb;
                --tv-surface: rgba(255, 255, 255, 0.88);
                --tv-surface-strong: #ffffff;
                --tv-surface-muted: #f8fbff;
                --tv-sidebar-top: #0f172a;
                --tv-sidebar-bottom: #111c33;
                --tv-text: #0f172a;
                --tv-text-soft: #475569;
                --tv-text-faint: #64748b;
                --tv-border: #dbe7f3;
                --tv-border-strong: #c6d7ea;
                --tv-primary: #1d4ed8;
                --tv-primary-strong: #1e40af;
                --tv-primary-soft: #e8f0ff;
                --tv-accent: #f59e0b;
                --tv-danger: #dc2626;
                --tv-success: #059669;
                --tv-shadow: 0 18px 48px rgba(15, 23, 42, 0.08);
                --tv-shadow-soft: 0 10px 26px rgba(15, 23, 42, 0.06);
                --tv-radius-lg: 20px;
                --tv-radius-md: 16px;
                --tv-radius-sm: 12px;
                --tv-transition: all 180ms ease;
            }

            html, body, [class*="css"], [data-testid="stAppViewContainer"], [data-testid="stApp"] {
                font-family: 'DM Sans', sans-serif;
                color: var(--tv-text);
            }

            [data-testid="stAppViewContainer"] {
                background:
                    radial-gradient(circle at top left, rgba(59, 130, 246, 0.12), transparent 22%),
                    radial-gradient(circle at top right, rgba(14, 165, 233, 0.10), transparent 18%),
                    linear-gradient(180deg, #f8fbff 0%, #eef4f9 100%);
            }

            [data-testid="stHeader"] {
                background: transparent;
            }

            #MainMenu, footer {
                visibility: hidden;
            }

            .main .block-container {
                padding-top: 2rem;
                padding-bottom: 3rem;
                max-width: 1240px;
            }

            section[data-testid="stSidebar"] {
                background: linear-gradient(180deg, var(--tv-sidebar-top) 0%, var(--tv-sidebar-bottom) 100%);
                border-right: 1px solid rgba(148, 163, 184, 0.14);
            }

            section[data-testid="stSidebar"] * {
                color: #d7e1f0;
            }

            section[data-testid="stSidebar"] [data-testid="stSidebarNav"] {
                padding-top: 0.75rem;
            }

            section[data-testid="stSidebar"] [data-testid="stSidebarNav"] ul {
                gap: 0.35rem;
            }

            section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a {
                border-radius: 14px;
                min-height: 44px;
                padding: 0.5rem 0.8rem;
                transition: var(--tv-transition);
            }

            section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover {
                background: rgba(148, 163, 184, 0.12);
            }

            section[data-testid="stSidebar"] [aria-current="page"] {
                background: linear-gradient(135deg, rgba(59, 130, 246, 0.20), rgba(37, 99, 235, 0.32));
                box-shadow: inset 0 0 0 1px rgba(147, 197, 253, 0.25);
            }

            .tv-sidebar-brand {
                background: linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.04));
                border: 1px solid rgba(148, 163, 184, 0.18);
                border-radius: 22px;
                padding: 1.15rem 1rem;
                margin-bottom: 1rem;
                box-shadow: 0 12px 32px rgba(2, 8, 23, 0.18);
            }

            .tv-sidebar-brand h2,
            .tv-sidebar-brand p,
            .tv-sidebar-brand span {
                margin: 0;
            }

            .tv-sidebar-kicker {
                color: #93c5fd;
                font-size: 0.72rem;
                font-weight: 700;
                letter-spacing: 0.14em;
                text-transform: uppercase;
            }

            .tv-sidebar-title {
                margin-top: 0.55rem;
                font-size: 1.35rem;
                font-weight: 800;
                color: #f8fafc;
            }

            .tv-sidebar-copy {
                margin-top: 0.45rem;
                color: #cbd5e1;
                font-size: 0.92rem;
                line-height: 1.45;
            }

            .tv-sidebar-meta {
                margin-top: 0.9rem;
                display: grid;
                gap: 0.55rem;
            }

            .tv-sidebar-pill {
                display: inline-flex;
                align-items: center;
                gap: 0.45rem;
                width: fit-content;
                border-radius: 999px;
                padding: 0.35rem 0.65rem;
                background: rgba(147, 197, 253, 0.12);
                color: #dbeafe;
                font-size: 0.78rem;
                font-weight: 600;
                border: 1px solid rgba(147, 197, 253, 0.16);
            }

            .tv-page-header {
                position: relative;
                overflow: hidden;
                background:
                    linear-gradient(135deg, rgba(255,255,255,0.92), rgba(248,251,255,0.86)),
                    linear-gradient(135deg, rgba(59,130,246,0.05), transparent 60%);
                border: 1px solid rgba(198, 215, 234, 0.9);
                border-radius: 26px;
                padding: 1.65rem 1.75rem;
                box-shadow: var(--tv-shadow);
                margin-bottom: 1.25rem;
            }

            .tv-page-header::after {
                content: "";
                position: absolute;
                inset: auto -8% -35% auto;
                width: 220px;
                height: 220px;
                background: radial-gradient(circle, rgba(37, 99, 235, 0.12), transparent 65%);
                pointer-events: none;
            }

            .tv-kicker {
                display: inline-flex;
                align-items: center;
                gap: 0.5rem;
                color: var(--tv-primary);
                font-size: 0.76rem;
                font-weight: 800;
                letter-spacing: 0.14em;
                text-transform: uppercase;
            }

            .tv-header-grid {
                display: flex;
                justify-content: space-between;
                gap: 1rem;
                align-items: flex-start;
                flex-wrap: wrap;
            }

            .tv-page-header h1 {
                margin: 0.7rem 0 0.45rem 0;
                font-size: clamp(2rem, 3vw, 3.1rem);
                line-height: 1.02;
                font-weight: 800;
                color: var(--tv-text);
                letter-spacing: -0.03em;
                max-width: 720px;
            }

            .tv-page-header p {
                margin: 0;
                max-width: 690px;
                color: var(--tv-text-soft);
                font-size: 1rem;
                line-height: 1.6;
            }

            .tv-status-pill {
                display: inline-flex;
                align-items: center;
                gap: 0.55rem;
                border-radius: 999px;
                padding: 0.65rem 0.9rem;
                border: 1px solid;
                font-weight: 700;
                font-size: 0.88rem;
                white-space: nowrap;
            }

            .tv-status-dot {
                width: 0.55rem;
                height: 0.55rem;
                border-radius: 999px;
                background: currentColor;
                display: inline-block;
            }

            .tv-section-label {
                margin: 0 0 0.9rem 0;
                color: var(--tv-text);
                font-size: 1.02rem;
                font-weight: 700;
                letter-spacing: -0.01em;
            }

            .tv-surface {
                background: linear-gradient(180deg, rgba(255,255,255,0.9), rgba(249,252,255,0.88));
                border: 1px solid rgba(198, 215, 234, 0.92);
                border-radius: var(--tv-radius-lg);
                box-shadow: var(--tv-shadow-soft);
                padding: 1.2rem 1.2rem 1.1rem 1.2rem;
                margin-bottom: 1.5rem;
            }

            .tv-surface + .tv-surface {
                margin-top: 1.5rem;
            }

            .tv-surface-muted {
                background: linear-gradient(180deg, rgba(248,251,255,0.95), rgba(243,247,252,0.95));
            }

            .tv-meta-row {
                display: flex;
                flex-wrap: wrap;
                gap: 0.6rem;
                margin-top: 1rem;
            }

            .tv-meta-chip {
                border-radius: 999px;
                background: var(--tv-primary-soft);
                color: var(--tv-primary-strong);
                border: 1px solid #bfdbfe;
                padding: 0.45rem 0.75rem;
                font-size: 0.82rem;
                font-weight: 700;
            }

            .tv-helper {
                color: var(--tv-text-soft);
                font-size: 0.95rem;
                line-height: 1.55;
                margin: 0;
            }

            .tv-empty-preview,
            .tv-image-frame,
            .tv-panel {
                border-radius: var(--tv-radius-lg);
                border: 1px solid rgba(198, 215, 234, 0.9);
                background: linear-gradient(180deg, rgba(255,255,255,0.82), rgba(246,250,255,0.95));
                box-shadow: var(--tv-shadow-soft);
            }

            .tv-empty-preview {
                min-height: 360px;
                padding: 1.4rem;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                text-align: center;
                margin-top: 1.25rem;
            }

            .tv-empty-preview h3 {
                margin: 0.9rem 0 0.4rem 0;
                color: var(--tv-text);
                font-size: 1.2rem;
                font-weight: 700;
            }

            .tv-empty-preview p {
                margin: 0;
                max-width: 430px;
                color: var(--tv-text-soft);
                line-height: 1.6;
            }

            .tv-preview-caption {
                margin-top: 0.85rem;
                color: var(--tv-text-soft);
                font-size: 0.9rem;
            }

            .tv-icon-badge {
                width: 3.35rem;
                height: 3.35rem;
                border-radius: 18px;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                background: linear-gradient(135deg, rgba(59,130,246,0.16), rgba(37,99,235,0.06));
                border: 1px solid rgba(147, 197, 253, 0.5);
                color: var(--tv-primary);
            }

            .tv-result-grid {
                display: grid;
                grid-template-columns: 1.4fr 0.9fr;
                gap: 1rem;
            }

            .tv-kpi-card {
                background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(245,249,255,0.96));
                border: 1px solid rgba(198, 215, 234, 0.95);
                border-radius: 24px;
                box-shadow: var(--tv-shadow);
                padding: 1.4rem;
            }

            .tv-kpi-label {
                color: var(--tv-text-faint);
                text-transform: uppercase;
                letter-spacing: 0.12em;
                font-size: 0.74rem;
                font-weight: 800;
                margin: 0 0 0.5rem 0;
            }

            .tv-kpi-title {
                color: var(--tv-text);
                font-size: 2rem;
                font-weight: 800;
                line-height: 1.1;
                margin: 0;
                letter-spacing: -0.03em;
            }

            .tv-kpi-note {
                margin: 0.75rem 0 0 0;
                color: var(--tv-text-soft);
                font-size: 0.97rem;
                line-height: 1.6;
            }

            .tv-metric-row {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 0.85rem;
                margin-top: 1rem;
            }

            .tv-metric {
                background: rgba(248, 251, 255, 0.95);
                border: 1px solid rgba(203, 213, 225, 0.7);
                border-radius: 18px;
                padding: 0.9rem 1rem;
            }

            .tv-metric span {
                display: block;
            }

            .tv-metric-label {
                color: var(--tv-text-faint);
                font-size: 0.78rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.1em;
            }

            .tv-metric-value {
                color: var(--tv-text);
                font-family: 'IBM Plex Mono', monospace;
                font-size: 1.1rem;
                font-weight: 600;
                margin-top: 0.32rem;
            }

            .tv-probability-card {
                padding: 1.15rem 1.15rem 1.05rem 1.15rem;
            }

            .tv-probability-list {
                margin-top: 0.85rem;
                display: grid;
                gap: 0.65rem;
            }

            .tv-probability-row {
                border-radius: 16px;
                background: rgba(247, 250, 255, 0.95);
                border: 1px solid rgba(212, 224, 238, 0.95);
                padding: 0.72rem 0.82rem 0.68rem 0.82rem;
            }

            .tv-probability-row-top {
                background: linear-gradient(180deg, rgba(239, 246, 255, 0.98), rgba(232, 240, 255, 0.92));
                border-color: rgba(147, 197, 253, 0.95);
            }

            .tv-probability-meta {
                display: grid;
                grid-template-columns: auto 1fr auto;
                gap: 0.65rem;
                align-items: center;
                margin-bottom: 0.45rem;
            }

            .tv-probability-rank {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                width: 1.7rem;
                height: 1.7rem;
                border-radius: 999px;
                background: rgba(29, 78, 216, 0.08);
                color: var(--tv-primary-strong);
                font-family: 'IBM Plex Mono', monospace;
                font-size: 0.72rem;
                font-weight: 700;
            }

            .tv-probability-label {
                color: var(--tv-text);
                font-size: 0.92rem;
                font-weight: 700;
            }

            .tv-probability-value {
                color: var(--tv-text);
                font-family: 'IBM Plex Mono', monospace;
                font-size: 0.84rem;
                font-weight: 700;
            }

            .tv-probability-track {
                width: 100%;
                height: 8px;
                border-radius: 999px;
                background: #dde6f1;
                overflow: hidden;
            }

            .tv-probability-fill {
                height: 100%;
                border-radius: 999px;
            }

            .tv-note-callout {
                margin-top: 1rem;
                border-radius: 18px;
                border: 1px solid #fed7aa;
                background: #fff7ed;
                padding: 0.95rem 1rem;
                color: #9a3412;
                font-size: 0.95rem;
                line-height: 1.55;
            }

            .tv-trust-strip {
                display: flex;
                justify-content: space-between;
                gap: 1rem;
                flex-wrap: wrap;
                border-radius: 20px;
                border: 1px solid rgba(198, 215, 234, 0.95);
                background: rgba(255,255,255,0.82);
                box-shadow: var(--tv-shadow-soft);
                padding: 1rem 1.15rem;
                margin-top: 1rem;
            }

            .tv-error-gap {
                height: 1rem;
            }

            .tv-trust-strip strong {
                display: block;
                color: var(--tv-text);
                font-size: 0.92rem;
                margin-bottom: 0.22rem;
            }

            .tv-trust-strip span {
                color: var(--tv-text-soft);
                font-size: 0.9rem;
                line-height: 1.5;
            }

            .tv-about-grid {
                display: grid;
                gap: 1rem;
            }

            .tv-about-card h3 {
                margin: 0 0 0.6rem 0;
                color: var(--tv-text);
                font-size: 1.08rem;
                font-weight: 700;
            }

            .tv-about-card p,
            .tv-about-card li,
            .tv-about-card td,
            .tv-about-card th {
                color: var(--tv-text-soft);
            }

            .tv-about-card ul {
                margin: 0;
                padding-left: 1.15rem;
            }

            .tv-about-card {
                margin-bottom: 1.75rem !important;
            }

            .tv-about-column {
                display: flex;
                flex-direction: column;
                gap: 0.2rem;
            }

            div[data-testid="stFileUploader"] {
                border: 1px dashed #93c5fd;
                border-radius: 22px;
                background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
                padding: 0.4rem;
            }

            div[data-testid="stFileUploader"] section {
                padding: 1rem 1rem 1.15rem 1rem;
            }

            div[data-testid="stFileUploader"] button {
                background: linear-gradient(135deg, var(--tv-primary), var(--tv-primary-strong));
                color: white;
                border: none;
                border-radius: 14px;
                font-weight: 700;
                box-shadow: 0 10px 20px rgba(37, 99, 235, 0.18);
                transition: var(--tv-transition);
            }

            div[data-testid="stFileUploader"] button:hover {
                background: linear-gradient(135deg, #1e40af, #1d4ed8);
            }

            .stButton > button {
                border-radius: 14px;
                border: 1px solid #bfdbfe;
                color: var(--tv-primary-strong);
                background: #eff6ff;
                font-weight: 700;
                transition: var(--tv-transition);
            }

            .stButton > button:hover {
                border-color: #93c5fd;
                background: #dbeafe;
                color: #1d4ed8;
            }

            [data-testid="stAlert"] {
                border-radius: 18px;
                border-width: 1px;
                box-shadow: none;
                margin-top: 1.5rem !important;
                margin-bottom: 1.5rem !important;
            }

            .stExpander {
                border: 1px solid rgba(198, 215, 234, 0.95);
                border-radius: 18px;
                background: rgba(255, 255, 255, 0.76);
            }

            .stExpander summary {
                font-weight: 700;
                color: var(--tv-text);
            }

            [data-testid="stMarkdownContainer"] code {
                font-family: 'IBM Plex Mono', monospace;
                background: rgba(15, 23, 42, 0.04);
                border-radius: 8px;
                padding: 0.15rem 0.35rem;
            }

            @media (max-width: 900px) {
                .tv-result-grid {
                    grid-template-columns: 1fr;
                }

                .tv-page-header {
                    padding: 1.35rem;
                }
            }

            @media (prefers-reduced-motion: reduce) {
                *, *::before, *::after {
                    transition: none !important;
                    animation: none !important;
                    scroll-behavior: auto !important;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def icon_svg(name: str, stroke: str = "currentColor") -> str:
    """Return a simple inline SVG icon by name."""
    icons = {
        "spark": """
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                <path d="M12 3L13.9 8.1L19 10L13.9 11.9L12 17L10.1 11.9L5 10L10.1 8.1L12 3Z"
                    stroke="{stroke}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
        """,
        "upload": """
            <svg width="26" height="26" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                <path d="M12 16V5" stroke="{stroke}" stroke-width="1.8" stroke-linecap="round"/>
                <path d="M8.5 8.5L12 5L15.5 8.5" stroke="{stroke}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M5 18.5H19" stroke="{stroke}" stroke-width="1.8" stroke-linecap="round"/>
            </svg>
        """,
        "chart": """
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                <path d="M4 19.5H20" stroke="{stroke}" stroke-width="1.8" stroke-linecap="round"/>
                <path d="M7 16L10 12L13 14L17 8" stroke="{stroke}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
                <circle cx="7" cy="16" r="1.2" fill="{stroke}"/>
                <circle cx="10" cy="12" r="1.2" fill="{stroke}"/>
                <circle cx="13" cy="14" r="1.2" fill="{stroke}"/>
                <circle cx="17" cy="8" r="1.2" fill="{stroke}"/>
            </svg>
        """,
        "shield": """
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                <path d="M12 3L18 5.5V10.8C18 14.6 15.4 18.1 12 19.5C8.6 18.1 6 14.6 6 10.8V5.5L12 3Z"
                    stroke="{stroke}" stroke-width="1.8" stroke-linejoin="round"/>
                <path d="M9.5 11.5L11.2 13.2L14.8 9.6" stroke="{stroke}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
        """,
    }
    return icons.get(name, icons["spark"]).format(stroke=stroke)


def render_sidebar_brand() -> None:
    """Render the branded sidebar card."""
    st.sidebar.markdown(
        """
        <div class="tv-sidebar-brand">
            <span class="tv-sidebar-kicker">Workspace</span>
            <h2 class="tv-sidebar-title">TradeVision</h2>
            <p class="tv-sidebar-copy">
                Technical chart classification for uploaded candlestick and OHLC images.
            </p>
            <div class="tv-sidebar-meta">
                <span class="tv-sidebar-pill">Single-image inference</span>
                <span class="tv-sidebar-pill">Research use only</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(
    *,
    kicker: str,
    title: str,
    subtitle: str,
    status_key: str = "neutral",
    status_label: str | None = None,
) -> None:
    """Render the standard page header."""
    status = STATUS_STYLES.get(status_key, STATUS_STYLES["neutral"])
    label = escape(status_label or status["label"])
    st.markdown(
        f"""
        <section class="tv-page-header">
            <div class="tv-header-grid">
                <div>
                    <span class="tv-kicker">{icon_svg("spark")} {escape(kicker)}</span>
                    <h1>{escape(title)}</h1>
                    <p>{escape(subtitle)}</p>
                </div>
                <span class="tv-status-pill"
                      style="background:{status['background']}; color:{status['foreground']}; border-color:{status['border']};">
                    <span class="tv-status-dot"></span>
                    {label}
                </span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_section_intro(title: str, body: str, chips: list[str] | None = None) -> None:
    """Render a surface with section title and helper copy."""
    chips_markup = ""
    if chips:
        chips_markup = "".join(
            f'<span class="tv-meta-chip">{escape(chip)}</span>' for chip in chips
        )
        chips_markup = f'<div class="tv-meta-row">{chips_markup}</div>'
    st.markdown(
        f"""
        <div class="tv-surface tv-surface-muted">
            <p class="tv-section-label">{escape(title)}</p>
            <p class="tv-helper">{escape(body)}</p>
            {chips_markup}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_preview() -> None:
    """Render the preview placeholder."""
    st.markdown(
        f"""
        <div class="tv-empty-preview">
            <div class="tv-icon-badge">{icon_svg("chart")}</div>
            <h3>Preview workspace</h3>
            <p>
                Upload a candlestick or OHLC chart to inspect the image before inference.
                The preview pane stays focused on clarity, not decoration.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_trust_strip(items: list[tuple[str, str]]) -> None:
    """Render a compact trust/disclaimer strip."""
    blocks = "".join(
        f"<div><strong>{escape(title)}</strong><span>{escape(body)}</span></div>"
        for title, body in items
    )
    st.markdown(
        f'<section class="tv-trust-strip">{blocks}</section>',
        unsafe_allow_html=True,
    )
