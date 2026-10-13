"""Chat bubble widgets (split from `ui_qt.py`)."""
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ui_base import PALETTE, get_palette

MONO = QFont("Monospace", 9)
MONO.setStyleHint(QFont.StyleHint.TypeWriter)


# ---------------------------------------------------------------------------
# Chat bubbles
# ---------------------------------------------------------------------------

class MessageBubble(QWidget):
    """Base chat message bubble (user / assistant / system / error)."""

    def __init__(self, role: str, text: str, palette: dict = None):
        super().__init__()
        self.role = role
        self.text_label = QLabel(text)
        self.text_label.setWordWrap(True)
        self.text_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.text_label.setTextFormat(Qt.TextFormat.PlainText)
        self.text_label.setMaximumWidth(680)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 2, 0, 2)
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)

        p = palette or PALETTE
        if role == "user":
            self.text_label.setStyleSheet(
                f"background: {p['accent']}; color: white; padding: 8px 12px;"
                "border-radius: 10px; font-weight: bold;"
            )
            body.addStretch(1)
            body.addWidget(self.text_label)
        elif role == "assistant":
            self.text_label.setStyleSheet(
                f"background: {p['surface']}; color: {p['text']};"
                f"border: 1px solid {p['border']}; border-left: 4px solid {p['success']};"
                "padding: 8px 12px; border-radius: 10px;"
            )
            body.addWidget(self.text_label)
            body.addStretch(1)
        else:  # system / error
            color = p["error"] if role == "error" else p["muted"]
            style = "font-style: italic;" if role == "system" else "font-weight: bold;"
            self.text_label.setStyleSheet(f"color: {color}; padding: 2px 6px; {style}")
            body.addStretch(1)
            body.addWidget(self.text_label)
            body.addStretch(1)

        outer.addLayout(body)


class AssistantBubble(MessageBubble):
    """Assistant bubble with thumbs up/down feedback (rating 5 / 1)."""

    def __init__(self, text: str = "", palette: dict = None):
        super().__init__("assistant", text, palette)
        p = palette or PALETTE

        self.feedback_row = QWidget()
        fb = QHBoxLayout(self.feedback_row)
        fb.setContentsMargins(14, 0, 0, 0)
        fb.setSpacing(6)

        rate_label = QLabel("Rate:")
        rate_label.setStyleSheet(f"color: {p['muted']}; font-size: 11px;")
        fb.addWidget(rate_label)

        self.thumbs_up = QPushButton("👍")
        self.thumbs_up.setToolTip("Good answer — feeds auto-training (rating 5/5)")
        self.thumbs_down = QPushButton("👎")
        self.thumbs_down.setToolTip("Bad answer — flags for review (rating 1/5)")
        for b in (self.thumbs_up, self.thumbs_down):
            b.setFixedWidth(34)
            fb.addWidget(b)

        self.rated_label = QLabel("")
        self.rated_label.setStyleSheet(f"color: {p['success']}; font-size: 11px;")
        fb.addWidget(self.rated_label)
        fb.addStretch(1)

        # Insert the feedback row under the bubble text
        self.layout().addWidget(self.feedback_row)
        self.feedback_row.hide()

    def show_feedback(self):
        self.feedback_row.show()

    def set_rated(self, rating: int):
        self.thumbs_up.setEnabled(False)
        self.thumbs_down.setEnabled(False)
        mark = "👍" if rating >= 4 else "👎"
        self.rated_label.setText(f"Rated {rating}/5 {mark}")


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

