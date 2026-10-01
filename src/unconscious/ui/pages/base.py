"""A scrolling page with a centred reading column and a few shared builders."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QScrollArea, QVBoxLayout, QWidget

from unconscious.ui.cards import SparkCard
from unconscious.ui.widgets import CardGrid, GridSurface, clear, label

if TYPE_CHECKING:
    from unconscious.ui.window import MainWindow


class Page(QScrollArea):
    max_width = 1100

    def __init__(self, window: MainWindow, **params):
        super().__init__()
        self.window = window
        self.app = window.app
        self.params = params
        self.setObjectName("Page")
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        container = GridSurface()
        container.setObjectName("Content")
        outer = QHBoxLayout(container)
        outer.setContentsMargins(52, 40, 52, 72)
        self.column = QWidget()
        self.column.setMaximumWidth(self.max_width)
        self.body = QVBoxLayout(self.column)
        self.body.setContentsMargins(0, 0, 0, 0)
        self.body.setSpacing(0)
        outer.addStretch(1)
        outer.addWidget(self.column, 1000)
        outer.addStretch(1)
        self.setWidget(container)

    # -- lifecycle -----------------------------------------------------------

    def build(self) -> None:  # override
        raise NotImplementedError

    def refresh(self) -> None:
        position = self.verticalScrollBar().value()
        clear(self.body)
        try:
            self.build()
        except Exception as exc:  # a broken page should say so, not crash the app
            import logging

            logging.getLogger(__name__).exception("page build failed")
            self.body.addWidget(label(f"{type(exc).__name__}: {exc}", "body", "error"))
        self.body.addStretch(1)
        self.verticalScrollBar().setValue(position)

    def on_state(self, state: dict) -> None:  # override for live updates
        pass

    # -- helpers -------------------------------------------------------------

    def add(self, widget_or_layout, space_before: int = 0) -> None:
        if space_before:
            self.body.addSpacing(space_before)
        if isinstance(widget_or_layout, QWidget):
            self.body.addWidget(widget_or_layout)
        else:
            self.body.addLayout(widget_or_layout)

    def cards(self, sparks: list[dict], compact: bool = False, min_width: int = 310) -> CardGrid:
        widgets = []
        for spark in sparks:
            card = SparkCard(spark, compact=compact)
            card.open_spark.connect(lambda sid: self.window.go("spark", id=sid))
            card.open_thread.connect(lambda tid: self.window.go("thread", id=tid))
            card.feedback.connect(self.window.spark_feedback)
            widgets.append(card)
        return CardGrid(widgets, min_width=min_width + 24, spacing=0)

    def page_head(self, title: str, sub: str = "") -> QVBoxLayout:
        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.addWidget(label(title, "display-l"))
        if sub:
            text = label(sub, "body", "ink2")
            text.setMaximumWidth(640)
            layout.addWidget(text)
        layout.addSpacing(34)
        return layout
