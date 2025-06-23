import typing
"""Pydantic models for browser automation action validation."""

from typing import Any
from typing import Literal

from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator


class BaseAction(BaseModel):
    """Base class for all browser actions."""

    type: str = Field(..., description="Type of action to perform")

    class Config:
        extra = "forbid"


class ClickAction(BaseAction):
    """Click on an element."""

    type: Literal["click"] = "click"
    selector: str = Field(..., description="CSS selector for element to click")


class FillAction(BaseAction):
    """Fill an input field."""

    type: Literal["fill"] = "fill"
    selector: str = Field(..., description="CSS selector for input field")
    text: str = Field(..., description="Text to fill into the field")


class TypeAction(BaseAction):
    """Type text into an element (character by character)."""

    type: Literal["type"] = "type"
    selector: str = Field(..., description="CSS selector for element")
    text: str = Field(..., description="Text to type")


class WaitAction(BaseAction):
    """Wait for a specified duration."""

    type: Literal["wait"] = "wait"
    timeout: int = Field(
        ..., gt=0, le=30000, description="Wait duration in milliseconds"
    )


class WaitForSelectorAction(BaseAction):
    """Wait for an element to appear."""

    type: Literal["wait_for_selector"] = "wait_for_selector"
    selector: str = Field(..., description="CSS selector to wait for")
    timeout: int = Field(
        default=5000, gt=0, le=30000, description="Timeout in milliseconds"
    )


class WaitForLoadStateAction(BaseAction):
    """Wait for page load state."""

    type: Literal["wait_for_load_state"] = "wait_for_load_state"
    state: Literal["load", "domcontentloaded", "networkidle"] = Field(
        default="networkidle", description="Load state to wait for"
    )


class ScrollAction(BaseAction):
    """Scroll the page."""

    type: Literal["scroll"] = "scroll"
    direction: Literal["top", "bottom", "position"] = Field(
        default="bottom", description="Scroll direction"
    )
    y: int = Field(default=0, description="Y position for position-based scrolling")

    @model_validator(mode="after")
    def validate_position_scrolling(self):
        """Validate Y position is required for position scrolling."""
        if self.direction == "position" and (self.y is None or self.y == 0):
            raise ValueError(
                "Y position must be specified for position-based scrolling"
            )
        return self


class ScreenshotAction(BaseAction):
    """Take a screenshot."""

    type: Literal["screenshot"] = "screenshot"
    path: str = Field(default="", description="Screenshot file path")
    full_page: bool = Field(default=False, description="Take full page screenshot")


class EvaluateAction(BaseAction):
    """Execute JavaScript code."""

    type: Literal["evaluate"] = "evaluate"
    script: str = Field(..., description="JavaScript code to execute")


class HoverAction(BaseAction):
    """Hover over an element."""

    type: Literal["hover"] = "hover"
    selector: str = Field(..., description="CSS selector for element to hover")


class SelectAction(BaseAction):
    """Select an option from a select element."""

    type: Literal["select"] = "select"
    selector: str = Field(..., description="CSS selector for select element")
    value: str = Field(..., description="Value to select")


class PressAction(BaseAction):
    """Press a keyboard key."""

    type: Literal["press"] = "press"
    key: str = Field(..., description="Key to press (e.g., 'Enter', 'ArrowDown')")
    selector: str = Field(
        default="", description="Optional selector to focus before pressing key"
    )


class DragAndDropAction(BaseAction):
    """Drag and drop between elements."""

    type: Literal["drag_and_drop"] = "drag_and_drop"
    source: str = Field(..., description="CSS selector for source element")
    target: str = Field(..., description="CSS selector for target element")


# Union type for all valid actions
BrowserAction = (
    ClickAction
    | FillAction
    | TypeAction
    | WaitAction
    | WaitForSelectorAction
    | WaitForLoadStateAction
    | ScrollAction
    | ScreenshotAction
    | EvaluateAction
    | HoverAction
    | SelectAction
    | PressAction
    | DragAndDropAction
)


def validate_action(action: dict[str, Any]) -> BrowserAction:
    """Validate and parse a browser action.

    Args:
        action: Raw action dictionary

    Returns:
        Validated action model

    Raises:
        ValidationError: If action is invalid
    """
    action_type = action.get("type")

    # Map action types to their corresponding models
    action_models = {
        "click": ClickAction,
        "fill": FillAction,
        "type": TypeAction,
        "wait": WaitAction,
        "wait_for_selector": WaitForSelectorAction,
        "wait_for_load_state": WaitForLoadStateAction,
        "scroll": ScrollAction,
        "screenshot": ScreenshotAction,
        "evaluate": EvaluateAction,
        "hover": HoverAction,
        "select": SelectAction,
        "press": PressAction,
        "drag_and_drop": DragAndDropAction,
    }

    if action_type not in action_models:
        raise ValueError(f"Unsupported action type: {action_type}")

    model_class = action_models[action_type]
    return model_class(**action)


def validate_actions(actions: list[dict[str, Any]]) -> list[BrowserAction]:
    """Validate a list of browser actions.

    Args:
        actions: List of raw action dictionaries

    Returns:
        List of validated action models

    Raises:
        ValidationError: If any action is invalid
    """
    return [validate_action(action) for action in actions]
