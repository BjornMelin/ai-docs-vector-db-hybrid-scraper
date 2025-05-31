"""Comprehensive tests for browser action schemas."""

import pytest
from pydantic import ValidationError
from src.services.browser.action_schemas import BaseAction
from src.services.browser.action_schemas import ClickAction
from src.services.browser.action_schemas import DragAndDropAction
from src.services.browser.action_schemas import EvaluateAction
from src.services.browser.action_schemas import FillAction
from src.services.browser.action_schemas import HoverAction
from src.services.browser.action_schemas import PressAction
from src.services.browser.action_schemas import ScreenshotAction
from src.services.browser.action_schemas import ScrollAction
from src.services.browser.action_schemas import SelectAction
from src.services.browser.action_schemas import TypeAction
from src.services.browser.action_schemas import WaitAction
from src.services.browser.action_schemas import WaitForLoadStateAction
from src.services.browser.action_schemas import WaitForSelectorAction
from src.services.browser.action_schemas import validate_action
from src.services.browser.action_schemas import validate_actions


class TestBaseAction:
    """Test BaseAction functionality."""

    def test_base_action_validation(self):
        """Test base action validation."""
        # Should not be instantiated directly
        with pytest.raises(ValidationError):
            BaseAction()

        # Should require type field
        with pytest.raises(ValidationError):
            BaseAction(extra_field="value")


class TestClickAction:
    """Test ClickAction validation and creation."""

    def test_valid_click_action(self):
        """Test creating valid click action."""
        action = ClickAction(selector="#submit-button")
        assert action.type == "click"
        assert action.selector == "#submit-button"

    def test_click_action_missing_selector(self):
        """Test click action without selector."""
        with pytest.raises(ValidationError) as exc_info:
            ClickAction()
        assert "selector" in str(exc_info.value)

    def test_click_action_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError) as exc_info:
            ClickAction(selector="#button", extra="field")
        assert "extra" in str(exc_info.value).lower()


class TestFillAction:
    """Test FillAction validation and creation."""

    def test_valid_fill_action(self):
        """Test creating valid fill action."""
        action = FillAction(selector="input[name='email']", text="test@example.com")
        assert action.type == "fill"
        assert action.selector == "input[name='email']"
        assert action.text == "test@example.com"

    def test_fill_action_missing_fields(self):
        """Test fill action with missing fields."""
        # Missing selector
        with pytest.raises(ValidationError) as exc_info:
            FillAction(text="test")
        assert "selector" in str(exc_info.value)

        # Missing text
        with pytest.raises(ValidationError) as exc_info:
            FillAction(selector="input")
        assert "text" in str(exc_info.value)

    def test_fill_action_empty_text(self):
        """Test fill action with empty text is valid."""
        action = FillAction(selector="input", text="")
        assert action.text == ""


class TestTypeAction:
    """Test TypeAction validation and creation."""

    def test_valid_type_action(self):
        """Test creating valid type action."""
        action = TypeAction(selector=".search-box", text="search query")
        assert action.type == "type"
        assert action.selector == ".search-box"
        assert action.text == "search query"

    def test_type_action_differs_from_fill(self):
        """Test that type action is distinct from fill action."""
        type_action = TypeAction(selector="input", text="test")
        fill_action = FillAction(selector="input", text="test")
        assert type_action.type != fill_action.type


class TestWaitAction:
    """Test WaitAction validation and creation."""

    def test_valid_wait_action(self):
        """Test creating valid wait action."""
        action = WaitAction(timeout=1000)
        assert action.type == "wait"
        assert action.timeout == 1000

    def test_wait_action_timeout_validation(self):
        """Test wait action timeout validation."""
        # Too short (0 or negative)
        with pytest.raises(ValidationError):
            WaitAction(timeout=0)

        with pytest.raises(ValidationError):
            WaitAction(timeout=-100)

        # Too long (> 30000)
        with pytest.raises(ValidationError):
            WaitAction(timeout=30001)

        # Valid edge cases
        action1 = WaitAction(timeout=1)
        assert action1.timeout == 1

        action2 = WaitAction(timeout=30000)
        assert action2.timeout == 30000


class TestWaitForSelectorAction:
    """Test WaitForSelectorAction validation and creation."""

    def test_valid_wait_for_selector(self):
        """Test creating valid wait for selector action."""
        action = WaitForSelectorAction(selector=".loading-complete")
        assert action.type == "wait_for_selector"
        assert action.selector == ".loading-complete"
        assert action.timeout == 5000  # Default

    def test_wait_for_selector_custom_timeout(self):
        """Test wait for selector with custom timeout."""
        action = WaitForSelectorAction(selector="#modal", timeout=10000)
        assert action.timeout == 10000

    def test_wait_for_selector_timeout_validation(self):
        """Test timeout validation for wait for selector."""
        with pytest.raises(ValidationError):
            WaitForSelectorAction(selector="#test", timeout=0)

        with pytest.raises(ValidationError):
            WaitForSelectorAction(selector="#test", timeout=30001)


class TestWaitForLoadStateAction:
    """Test WaitForLoadStateAction validation and creation."""

    def test_valid_wait_for_load_state_default(self):
        """Test creating wait for load state with default."""
        action = WaitForLoadStateAction()
        assert action.type == "wait_for_load_state"
        assert action.state == "networkidle"  # Default

    def test_wait_for_load_state_options(self):
        """Test all valid load state options."""
        states = ["load", "domcontentloaded", "networkidle"]
        for state in states:
            action = WaitForLoadStateAction(state=state)
            assert action.state == state

    def test_wait_for_load_state_invalid(self):
        """Test invalid load state."""
        with pytest.raises(ValidationError):
            WaitForLoadStateAction(state="invalid")


class TestScrollAction:
    """Test ScrollAction validation and creation."""

    def test_valid_scroll_default(self):
        """Test creating scroll action with defaults."""
        action = ScrollAction()
        assert action.type == "scroll"
        assert action.direction == "bottom"  # Default
        assert action.y == 0

    def test_scroll_directions(self):
        """Test all valid scroll directions."""
        directions = ["top", "bottom", "position"]
        for direction in directions:
            if direction == "position":
                action = ScrollAction(direction=direction, y=500)
            else:
                action = ScrollAction(direction=direction)
            assert action.direction == direction

    def test_scroll_position_requires_y(self):
        """Test position scrolling requires y value."""
        # Valid position scroll
        action = ScrollAction(direction="position", y=1000)
        assert action.y == 1000

        # Invalid: position without y > 0
        with pytest.raises(ValidationError) as exc_info:
            ScrollAction(direction="position", y=0)
        assert "Y position must be specified" in str(exc_info.value)

    def test_scroll_y_ignored_for_other_directions(self):
        """Test y value is allowed but not required for top/bottom."""
        action = ScrollAction(direction="top", y=500)
        assert action.y == 500  # Allowed but not used


class TestScreenshotAction:
    """Test ScreenshotAction validation and creation."""

    def test_valid_screenshot_default(self):
        """Test creating screenshot action with defaults."""
        action = ScreenshotAction()
        assert action.type == "screenshot"
        assert action.path == ""
        assert action.full_page is False

    def test_screenshot_with_options(self):
        """Test screenshot with all options."""
        action = ScreenshotAction(path="/tmp/screenshot.png", full_page=True)
        assert action.path == "/tmp/screenshot.png"
        assert action.full_page is True


class TestEvaluateAction:
    """Test EvaluateAction validation and creation."""

    def test_valid_evaluate_action(self):
        """Test creating valid evaluate action."""
        script = "document.querySelector('#result').textContent"
        action = EvaluateAction(script=script)
        assert action.type == "evaluate"
        assert action.script == script

    def test_evaluate_missing_script(self):
        """Test evaluate action requires script."""
        with pytest.raises(ValidationError) as exc_info:
            EvaluateAction()
        assert "script" in str(exc_info.value)

    def test_evaluate_complex_script(self):
        """Test evaluate with complex JavaScript."""
        script = """
        const elements = document.querySelectorAll('.item');
        return Array.from(elements).map(el => el.textContent);
        """
        action = EvaluateAction(script=script)
        assert action.script == script


class TestHoverAction:
    """Test HoverAction validation and creation."""

    def test_valid_hover_action(self):
        """Test creating valid hover action."""
        action = HoverAction(selector=".tooltip-trigger")
        assert action.type == "hover"
        assert action.selector == ".tooltip-trigger"

    def test_hover_missing_selector(self):
        """Test hover action requires selector."""
        with pytest.raises(ValidationError):
            HoverAction()


class TestSelectAction:
    """Test SelectAction validation and creation."""

    def test_valid_select_action(self):
        """Test creating valid select action."""
        action = SelectAction(selector="select[name='country']", value="US")
        assert action.type == "select"
        assert action.selector == "select[name='country']"
        assert action.value == "US"

    def test_select_missing_fields(self):
        """Test select action field validation."""
        # Missing selector
        with pytest.raises(ValidationError):
            SelectAction(value="option1")

        # Missing value
        with pytest.raises(ValidationError):
            SelectAction(selector="select")


class TestPressAction:
    """Test PressAction validation and creation."""

    def test_valid_press_action(self):
        """Test creating valid press action."""
        action = PressAction(key="Enter")
        assert action.type == "press"
        assert action.key == "Enter"
        assert action.selector == ""  # Default

    def test_press_with_selector(self):
        """Test press action with optional selector."""
        action = PressAction(key="Tab", selector="input#username")
        assert action.selector == "input#username"

    def test_press_missing_key(self):
        """Test press action requires key."""
        with pytest.raises(ValidationError):
            PressAction()

    def test_press_special_keys(self):
        """Test various special key values."""
        keys = ["Enter", "Tab", "Escape", "ArrowDown", "ArrowUp", "Space", "Control+A"]
        for key in keys:
            action = PressAction(key=key)
            assert action.key == key


class TestDragAndDropAction:
    """Test DragAndDropAction validation and creation."""

    def test_valid_drag_and_drop(self):
        """Test creating valid drag and drop action."""
        action = DragAndDropAction(source="#draggable", target="#droppable")
        assert action.type == "drag_and_drop"
        assert action.source == "#draggable"
        assert action.target == "#droppable"

    def test_drag_and_drop_missing_fields(self):
        """Test drag and drop field validation."""
        # Missing source
        with pytest.raises(ValidationError):
            DragAndDropAction(target="#droppable")

        # Missing target
        with pytest.raises(ValidationError):
            DragAndDropAction(source="#draggable")


class TestValidateAction:
    """Test validate_action function."""

    def test_validate_click_action(self):
        """Test validating click action dictionary."""
        action_dict = {"type": "click", "selector": "#button"}
        action = validate_action(action_dict)
        assert isinstance(action, ClickAction)
        assert action.selector == "#button"

    def test_validate_fill_action(self):
        """Test validating fill action dictionary."""
        action_dict = {"type": "fill", "selector": "input", "text": "test"}
        action = validate_action(action_dict)
        assert isinstance(action, FillAction)
        assert action.text == "test"

    def test_validate_wait_action(self):
        """Test validating wait action dictionary."""
        action_dict = {"type": "wait", "timeout": 2000}
        action = validate_action(action_dict)
        assert isinstance(action, WaitAction)
        assert action.timeout == 2000

    def test_validate_scroll_position(self):
        """Test validating scroll position action."""
        action_dict = {"type": "scroll", "direction": "position", "y": 1500}
        action = validate_action(action_dict)
        assert isinstance(action, ScrollAction)
        assert action.y == 1500

    def test_validate_invalid_action_type(self):
        """Test validation with invalid action type."""
        with pytest.raises(ValueError) as exc_info:
            validate_action({"type": "invalid_action"})
        assert "Unsupported action type: invalid_action" in str(exc_info.value)

    def test_validate_missing_type(self):
        """Test validation with missing type."""
        with pytest.raises(ValueError):
            validate_action({"selector": "#button"})

    def test_validate_invalid_field_values(self):
        """Test validation with invalid field values."""
        # Invalid timeout
        with pytest.raises(ValidationError):
            validate_action({"type": "wait", "timeout": -100})

        # Missing required field
        with pytest.raises(ValidationError):
            validate_action({"type": "click"})


class TestValidateActions:
    """Test validate_actions function."""

    def test_validate_empty_list(self):
        """Test validating empty action list."""
        actions = validate_actions([])
        assert actions == []

    def test_validate_single_action(self):
        """Test validating single action in list."""
        action_dicts = [{"type": "click", "selector": "#button"}]
        actions = validate_actions(action_dicts)
        assert len(actions) == 1
        assert isinstance(actions[0], ClickAction)

    def test_validate_multiple_actions(self):
        """Test validating multiple actions."""
        action_dicts = [
            {"type": "click", "selector": "#button"},
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

    def test_validate_complex_workflow(self):
        """Test validating a complex action workflow."""
        workflow = [
            {"type": "wait_for_load_state", "state": "networkidle"},
            {"type": "click", "selector": "#login-button"},
            {"type": "wait_for_selector", "selector": "#login-form"},
            {"type": "fill", "selector": "#username", "text": "user@example.com"},
            {"type": "fill", "selector": "#password", "text": "password123"},
            {"type": "press", "key": "Enter"},
            {"type": "wait_for_selector", "selector": ".dashboard", "timeout": 10000},
            {"type": "screenshot", "path": "dashboard.png", "full_page": True},
        ]
        actions = validate_actions(workflow)
        assert len(actions) == 8

        # Verify action types in order
        expected_types = [
            WaitForLoadStateAction,
            ClickAction,
            WaitForSelectorAction,
            FillAction,
            FillAction,
            PressAction,
            WaitForSelectorAction,
            ScreenshotAction,
        ]
        for action, expected_type in zip(actions, expected_types, strict=False):
            assert isinstance(action, expected_type)

    def test_validate_actions_with_one_invalid(self):
        """Test validation fails if any action is invalid."""
        action_dicts = [
            {"type": "click", "selector": "#button"},
            {"type": "invalid_type"},  # Invalid
            {"type": "wait", "timeout": 1000},
        ]
        with pytest.raises(ValueError):
            validate_actions(action_dicts)

    def test_validate_all_action_types(self):
        """Test validating all supported action types."""
        all_actions = [
            {"type": "click", "selector": "#btn"},
            {"type": "fill", "selector": "input", "text": "test"},
            {"type": "type", "selector": "input", "text": "test"},
            {"type": "wait", "timeout": 1000},
            {"type": "wait_for_selector", "selector": ".loaded"},
            {"type": "wait_for_load_state", "state": "load"},
            {"type": "scroll", "direction": "bottom"},
            {"type": "screenshot"},
            {"type": "evaluate", "script": "return true;"},
            {"type": "hover", "selector": ".menu"},
            {"type": "select", "selector": "select", "value": "opt1"},
            {"type": "press", "key": "Enter"},
            {"type": "drag_and_drop", "source": "#src", "target": "#tgt"},
        ]

        actions = validate_actions(all_actions)
        assert len(actions) == 13

        # Verify each action type
        assert isinstance(actions[0], ClickAction)
        assert isinstance(actions[1], FillAction)
        assert isinstance(actions[2], TypeAction)
        assert isinstance(actions[3], WaitAction)
        assert isinstance(actions[4], WaitForSelectorAction)
        assert isinstance(actions[5], WaitForLoadStateAction)
        assert isinstance(actions[6], ScrollAction)
        assert isinstance(actions[7], ScreenshotAction)
        assert isinstance(actions[8], EvaluateAction)
        assert isinstance(actions[9], HoverAction)
        assert isinstance(actions[10], SelectAction)
        assert isinstance(actions[11], PressAction)
        assert isinstance(actions[12], DragAndDropAction)
