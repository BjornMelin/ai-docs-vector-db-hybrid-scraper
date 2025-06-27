"""Unit tests for browser automation action schemas with Pydantic v2.

This test module demonstrates:
- Comprehensive validation testing for Pydantic v2 models
- Error case handling and validation error testing
- Union type validation
- Custom validator testing
- Field constraint validation
"""

import pytest
from pydantic import ValidationError

from src.services.browser.action_schemas import (
    BaseAction,
    ClickAction,
    DragAndDropAction,
    EvaluateAction,
    FillAction,
    HoverAction,
    PressAction,
    ScreenshotAction,
    ScrollAction,
    SelectAction,
    TypeAction,
    WaitAction,
    WaitForLoadStateAction,
    WaitForSelectorAction,
    validate_action,
    validate_actions,
)


class TestBaseAction:
    """Test BaseAction base class."""

    def test_base_action_requires_type(self):
        """Test that BaseAction requires a type field."""
        with pytest.raises(ValidationError) as excinfo:
            BaseAction()

        errors = excinfo.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("type",)
        assert errors[0]["type"] == "missing"

    def test_base_action_forbids_extra_fields(self):
        """Test that BaseAction forbids extra fields."""
        with pytest.raises(ValidationError) as excinfo:
            BaseAction(type="test", extra_field="not_allowed")

        errors = excinfo.value.errors()
        assert any(error["type"] == "extra_forbidden" for error in errors)


class TestClickAction:
    """Test ClickAction model."""

    def test_valid_click_action(self):
        """Test creating valid click action."""
        action = ClickAction(selector="#button")
        assert action.type == "click"
        assert action.selector == "#button"

    def test_click_action_requires_selector(self):
        """Test that click action requires selector."""
        with pytest.raises(ValidationError) as excinfo:
            ClickAction()

        errors = excinfo.value.errors()
        assert any(error["loc"] == ("selector",) for error in errors)

    def test_click_action_type_is_literal(self):
        """Test that click action type must be 'click'."""
        with pytest.raises(ValidationError) as excinfo:
            ClickAction(type="wrong", selector="#button")

        errors = excinfo.value.errors()
        assert any("literal_error" in error["type"] for error in errors)


class TestFillAction:
    """Test FillAction model."""

    def test_valid_fill_action(self):
        """Test creating valid fill action."""
        action = FillAction(selector="input[name='email']", text="test@example.com")
        assert action.type == "fill"
        assert action.selector == "input[name='email']"
        assert action.text == "test@example.com"

    def test_fill_action_requires_all_fields(self):
        """Test that fill action requires selector and text."""
        with pytest.raises(ValidationError) as excinfo:
            FillAction()

        errors = excinfo.value.errors()
        field_errors = {error["loc"][0] for error in errors}
        assert "selector" in field_errors
        assert "text" in field_errors

    def test_fill_action_with_empty_text(self):
        """Test fill action accepts empty text."""
        action = FillAction(selector="input", text="")
        assert action.text == ""


class TestTypeAction:
    """Test TypeAction model."""

    def test_valid_type_action(self):
        """Test creating valid type action."""
        action = TypeAction(selector="textarea", text="Hello, World!")
        assert action.type == "type"
        assert action.selector == "textarea"
        assert action.text == "Hello, World!"

    def test_type_action_differs_from_fill(self):
        """Test that type action is distinct from fill action."""
        type_action = TypeAction(selector="input", text="test")
        fill_action = FillAction(selector="input", text="test")

        assert type_action.type != fill_action.type
        assert isinstance(type_action, TypeAction)
        assert isinstance(fill_action, FillAction)


class TestWaitAction:
    """Test WaitAction model."""

    def test_valid_wait_action(self):
        """Test creating valid wait action."""
        action = WaitAction(timeout=1000)
        assert action.type == "wait"
        assert action.timeout == 1000

    def test_wait_action_timeout_constraints(self):
        """Test wait action timeout constraints."""
        # Test minimum constraint
        with pytest.raises(ValidationError) as excinfo:
            WaitAction(timeout=0)
        assert any("greater_than" in str(error) for error in excinfo.value.errors())

        # Test maximum constraint
        with pytest.raises(ValidationError) as excinfo:
            WaitAction(timeout=30001)
        assert any("less_than_equal" in str(error) for error in excinfo.value.errors())

    def test_wait_action_requires_timeout(self):
        """Test that wait action requires timeout."""
        with pytest.raises(ValidationError) as excinfo:
            WaitAction()

        errors = excinfo.value.errors()
        assert any(error["loc"] == ("timeout",) for error in errors)


class TestWaitForSelectorAction:
    """Test WaitForSelectorAction model."""

    def test_valid_wait_for_selector_action(self):
        """Test creating valid wait for selector action."""
        action = WaitForSelectorAction(selector=".loading-complete")
        assert action.type == "wait_for_selector"
        assert action.selector == ".loading-complete"
        assert action.timeout == 5000  # Default

    def test_wait_for_selector_custom_timeout(self):
        """Test wait for selector with custom timeout."""
        action = WaitForSelectorAction(selector="#element", timeout=10000)
        assert action.timeout == 10000

    def test_wait_for_selector_timeout_constraints(self):
        """Test wait for selector timeout constraints."""
        with pytest.raises(ValidationError):
            WaitForSelectorAction(selector="#element", timeout=0)

        with pytest.raises(ValidationError):
            WaitForSelectorAction(selector="#element", timeout=30001)


class TestWaitForLoadStateAction:
    """Test WaitForLoadStateAction model."""

    def test_valid_wait_for_load_state_default(self):
        """Test creating wait for load state with default."""
        action = WaitForLoadStateAction()
        assert action.type == "wait_for_load_state"
        assert action.state == "networkidle"

    def test_wait_for_load_state_options(self):
        """Test all valid load state options."""
        states = ["load", "domcontentloaded", "networkidle"]

        for state in states:
            action = WaitForLoadStateAction(state=state)
            assert action.state == state

    def test_wait_for_load_state_invalid_state(self):
        """Test invalid load state."""
        with pytest.raises(ValidationError) as excinfo:
            WaitForLoadStateAction(state="invalid")

        errors = excinfo.value.errors()
        assert any("literal_error" in error["type"] for error in errors)


class TestScrollAction:
    """Test ScrollAction model with custom validator."""

    def test_valid_scroll_bottom(self):
        """Test scroll to bottom (default)."""
        action = ScrollAction()
        assert action.type == "scroll"
        assert action.direction == "bottom"
        assert action.y == 0

    def test_valid_scroll_top(self):
        """Test scroll to top."""
        action = ScrollAction(direction="top")
        assert action.direction == "top"

    def test_valid_scroll_position(self):
        """Test scroll to specific position."""
        action = ScrollAction(direction="position", y=500)
        assert action.direction == "position"
        assert action.y == 500

    def test_scroll_position_requires_y(self):
        """Test that position scrolling requires Y coordinate."""
        # Test with y=0 (default)
        with pytest.raises(ValidationError) as excinfo:
            ScrollAction(direction="position")

        error_msg = str(excinfo.value)
        assert "Y position must be specified" in error_msg

    def test_scroll_position_with_y_zero(self):
        """Test that position scrolling rejects y=0."""
        with pytest.raises(ValidationError) as excinfo:
            ScrollAction(direction="position", y=0)

        error_msg = str(excinfo.value)
        assert "Y position must be specified" in error_msg

    def test_scroll_invalid_direction(self):
        """Test invalid scroll direction."""
        with pytest.raises(ValidationError):
            ScrollAction(direction="invalid")


class TestScreenshotAction:
    """Test ScreenshotAction model."""

    def test_valid_screenshot_default(self):
        """Test screenshot with defaults."""
        action = ScreenshotAction()
        assert action.type == "screenshot"
        assert action.path == ""
        assert action.full_page is False

    def test_screenshot_with_options(self):
        """Test screenshot with all options."""
        action = ScreenshotAction(path="/tmp/screenshot.png", full_page=True)  # noqa: S108 # test temp path
        assert action.path == "/tmp/screenshot.png"  # noqa: S108 # test temp path
        assert action.full_page is True


class TestEvaluateAction:
    """Test EvaluateAction model."""

    def test_valid_evaluate_action(self):
        """Test creating valid evaluate action."""
        script = "document.querySelector('#element').click()"
        action = EvaluateAction(script=script)
        assert action.type == "evaluate"
        assert action.script == script

    def test_evaluate_requires_script(self):
        """Test that evaluate action requires script."""
        with pytest.raises(ValidationError) as excinfo:
            EvaluateAction()

        errors = excinfo.value.errors()
        assert any(error["loc"] == ("script",) for error in errors)


class TestHoverAction:
    """Test HoverAction model."""

    def test_valid_hover_action(self):
        """Test creating valid hover action."""
        action = HoverAction(selector="a.link")
        assert action.type == "hover"
        assert action.selector == "a.link"


class TestSelectAction:
    """Test SelectAction model."""

    def test_valid_select_action(self):
        """Test creating valid select action."""
        action = SelectAction(selector="select#country", value="US")
        assert action.type == "select"
        assert action.selector == "select#country"
        assert action.value == "US"

    def test_select_requires_value(self):
        """Test that select action requires value."""
        with pytest.raises(ValidationError) as excinfo:
            SelectAction(selector="select")

        errors = excinfo.value.errors()
        assert any(error["loc"] == ("value",) for error in errors)


class TestPressAction:
    """Test PressAction model."""

    def test_valid_press_action_simple(self):
        """Test press action without selector."""
        action = PressAction(key="Enter")
        assert action.type == "press"
        assert action.key == "Enter"
        assert action.selector == ""

    def test_press_action_with_selector(self):
        """Test press action with selector."""
        action = PressAction(key="Tab", selector="input#first")
        assert action.key == "Tab"
        assert action.selector == "input#first"

    def test_press_special_keys(self):
        """Test various special key names."""
        special_keys = ["Enter", "Tab", "ArrowDown", "ArrowUp", "Escape", "Space"]

        for key in special_keys:
            action = PressAction(key=key)
            assert action.key == key


class TestDragAndDropAction:
    """Test DragAndDropAction model."""

    def test_valid_drag_and_drop(self):
        """Test creating valid drag and drop action."""
        action = DragAndDropAction(source="#draggable", target="#droppable")
        assert action.type == "drag_and_drop"
        assert action.source == "#draggable"
        assert action.target == "#droppable"

    def test_drag_and_drop_requires_both_selectors(self):
        """Test that drag and drop requires both selectors."""
        with pytest.raises(ValidationError) as excinfo:
            DragAndDropAction()

        errors = excinfo.value.errors()
        field_errors = {error["loc"][0] for error in errors}
        assert "source" in field_errors
        assert "target" in field_errors


class TestValidateAction:
    """Test validate_action function."""

    def test_validate_click_action(self):
        """Test validating click action dictionary."""
        action_dict = {"type": "click", "selector": "#button"}
        action = validate_action(action_dict)

        assert isinstance(action, ClickAction)
        assert action.selector == "#button"

    def test_validate_scroll_action_with_validator(self):
        """Test validating scroll action triggers custom validator."""
        # Valid position scroll
        action_dict = {"type": "scroll", "direction": "position", "y": 100}
        action = validate_action(action_dict)
        assert isinstance(action, ScrollAction)

        # Invalid position scroll
        action_dict = {"type": "scroll", "direction": "position", "y": 0}
        with pytest.raises(ValidationError):
            validate_action(action_dict)

    def test_validate_unknown_action_type(self):
        """Test validating unknown action type."""
        action_dict = {"type": "unknown_action"}

        with pytest.raises(ValueError) as excinfo:
            validate_action(action_dict)

        assert "Unsupported action type: unknown_action" in str(excinfo.value)

    def test_validate_missing_required_fields(self):
        """Test validation catches missing required fields."""
        action_dict = {"type": "fill", "selector": "input"}
        # Missing 'text' field

        with pytest.raises(ValidationError):
            validate_action(action_dict)

    def test_validate_extra_fields_forbidden(self):
        """Test validation forbids extra fields."""
        action_dict = {
            "type": "click",
            "selector": "#button",
            "extra_field": "not_allowed",
        }

        with pytest.raises(ValidationError):
            validate_action(action_dict)


class TestValidateActions:
    """Test validate_actions function."""

    def test_validate_empty_list(self):
        """Test validating empty action list."""
        actions = validate_actions([])
        assert actions == []

    def test_validate_multiple_actions(self):
        """Test validating multiple actions."""
        action_dicts = [
            {"type": "click", "selector": "#button1"},
            {"type": "wait", "timeout": 1000},
            {"type": "fill", "selector": "input", "text": "test"},
            {"type": "screenshot", "full_page": True},
        ]

        actions = validate_actions(action_dicts)

        assert len(actions) == 4
        assert isinstance(actions[0], ClickAction)
        assert isinstance(actions[1], WaitAction)
        assert isinstance(actions[2], FillAction)
        assert isinstance(actions[3], ScreenshotAction)

    def test_validate_actions_stops_on_first_error(self):
        """Test that validation error includes all errors."""
        action_dicts = [
            {"type": "click", "selector": "#button"},
            {"type": "invalid"},  # Invalid
            {"type": "wait", "timeout": 1000},
        ]

        with pytest.raises(ValueError):
            validate_actions(action_dicts)


class TestBrowserActionUnion:
    """Test BrowserAction union type."""

    def test_all_action_types_in_union(self):
        """Test that all action types are part of BrowserAction union."""
        # Create instances of each action type
        actions = [
            ClickAction(selector="#btn"),
            FillAction(selector="input", text="test"),
            TypeAction(selector="textarea", text="test"),
            WaitAction(timeout=1000),
            WaitForSelectorAction(selector=".elem"),
            WaitForLoadStateAction(),
            ScrollAction(),
            ScreenshotAction(),
            EvaluateAction(script="console.log('test')"),
            HoverAction(selector="a"),
            SelectAction(selector="select", value="opt1"),
            PressAction(key="Enter"),
            DragAndDropAction(source="#src", target="#tgt"),
        ]

        # All should be valid BrowserAction instances
        for action in actions:
            # Python's union type doesn't have isinstance support,
            # but we can verify the action has the expected attributes
            assert hasattr(action, "type")
            assert isinstance(action, BaseAction)


class TestPydanticV2Features:
    """Test Pydantic v2 specific features."""

    def test_model_validator_mode_after(self):
        """Test that model_validator with mode='after' works correctly."""
        # This tests the ScrollAction validator specifically
        action = ScrollAction(direction="bottom", y=100)
        assert action.y == 100  # Y is ignored for non-position scrolling

        # Test that validator runs after field validation
        with pytest.raises(ValidationError) as excinfo:
            ScrollAction(direction="position")  # y defaults to 0

        # The error should come from our custom validator, not field validation
        assert "Y position must be specified" in str(excinfo.value)

    def test_field_constraints(self):
        """Test Pydantic v2 field constraints."""
        # Test Field with gt constraint
        with pytest.raises(ValidationError) as excinfo:
            WaitAction(timeout=-1)

        errors = excinfo.value.errors()
        assert any("greater_than" in str(error) for error in errors)

        # Test Field with le constraint
        with pytest.raises(ValidationError) as excinfo:
            WaitAction(timeout=40000)

        errors = excinfo.value.errors()
        assert any("less_than_equal" in str(error) for error in errors)

    def test_literal_types(self):
        """Test Pydantic v2 Literal type validation."""
        # Valid literal
        action = ClickAction(selector="#btn")
        assert action.type == "click"

        # Invalid literal
        with pytest.raises(ValidationError) as excinfo:
            # Manually construct to bypass type checking
            ClickAction.model_validate({"type": "wrong", "selector": "#btn"})

        errors = excinfo.value.errors()
        assert any("literal_error" in error["type"] for error in errors)

    def test_extra_forbid_config(self):
        """Test that extra='forbid' config works in Pydantic v2."""
        with pytest.raises(ValidationError) as excinfo:
            ClickAction(selector="#btn", unexpected_field="value")

        errors = excinfo.value.errors()
        assert any(error["type"] == "extra_forbidden" for error in errors)


class TestEdgeCases:
    """Test edge cases and error scenarios."""

    def test_empty_selector(self):
        """Test actions with empty selectors."""
        # Some actions might accept empty selectors
        action = PressAction(key="Enter", selector="")
        assert action.selector == ""

        # But required selectors should not be empty
        # (Pydantic doesn't validate string content by default)
        action = ClickAction(selector="")
        assert action.selector == ""  # Empty but valid

    def test_unicode_in_fields(self):
        """Test unicode characters in text fields."""
        action = FillAction(selector="input", text="Hello 世界 🌍")
        assert action.text == "Hello 世界 🌍"

    def test_special_characters_in_selectors(self):
        """Test CSS selectors with special characters."""
        selectors = [
            "input[name='email']",
            "#id-with-dash",
            ".class\\.with\\.dots",
            "[data-test-id='complex']",
            "div > span + p",
        ]

        for selector in selectors:
            action = ClickAction(selector=selector)
            assert action.selector == selector

    def test_large_timeout_values(self):
        """Test maximum allowed timeout values."""
        action = WaitAction(timeout=30000)  # Maximum
        assert action.timeout == 30000

        action = WaitForSelectorAction(selector="#elem", timeout=30000)
        assert action.timeout == 30000

    def test_javascript_code_in_evaluate(self):
        """Test various JavaScript code snippets."""
        scripts = [
            "return document.title",
            "window.scrollTo(0, 0)",
            "const elem = document.querySelector('#id'); elem.click();",
            """
            // Multi-line script
            const items = document.querySelectorAll('.item');
            return items.length;
            """,
        ]

        for script in scripts:
            action = EvaluateAction(script=script)
            assert action.script == script
