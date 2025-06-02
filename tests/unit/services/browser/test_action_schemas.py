"""Comprehensive tests for browser action schema validation."""

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
    """Test the base action class."""

    def test_valid_base_action(self):
        """Test valid base action creation."""
        action = BaseAction(type="test")
        assert action.type == "test"

    def test_base_action_forbids_extra_fields(self):
        """Test that BaseAction forbids extra fields."""
        with pytest.raises(ValidationError) as exc_info:
            BaseAction(type="test", invalid_field="value")
        assert "Extra inputs are not permitted" in str(exc_info.value)

    def test_base_action_requires_type(self):
        """Test that type field is required."""
        with pytest.raises(ValidationError) as exc_info:
            BaseAction()
        assert "Field required" in str(exc_info.value)


class TestClickAction:
    """Test click action validation."""

    def test_valid_click_action(self):
        """Test valid click action."""
        action = ClickAction(selector="button#submit")
        assert action.type == "click"
        assert action.selector == "button#submit"

    def test_click_action_requires_selector(self):
        """Test that selector is required."""
        with pytest.raises(ValidationError) as exc_info:
            ClickAction()
        assert "Field required" in str(exc_info.value)

    def test_click_action_type_literal(self):
        """Test that type is automatically set to 'click'."""
        action = ClickAction(selector=".test")
        assert action.type == "click"

    def test_click_action_forbids_extra_fields(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError) as exc_info:
            ClickAction(selector=".test", extra_field="value")
        assert "Extra inputs are not permitted" in str(exc_info.value)


class TestFillAction:
    """Test fill action validation."""

    def test_valid_fill_action(self):
        """Test valid fill action."""
        action = FillAction(selector="input[name='email']", text="test@example.com")
        assert action.type == "fill"
        assert action.selector == "input[name='email']"
        assert action.text == "test@example.com"

    def test_fill_action_requires_selector_and_text(self):
        """Test that both selector and text are required."""
        with pytest.raises(ValidationError):
            FillAction(selector="input")

        with pytest.raises(ValidationError):
            FillAction(text="test")

    def test_fill_action_empty_text(self):
        """Test fill action with empty text."""
        action = FillAction(selector="input", text="")
        assert action.text == ""


class TestTypeAction:
    """Test type action validation."""

    def test_valid_type_action(self):
        """Test valid type action."""
        action = TypeAction(selector="input", text="Hello World")
        assert action.type == "type"
        assert action.selector == "input"
        assert action.text == "Hello World"

    def test_type_action_special_characters(self):
        """Test type action with special characters."""
        action = TypeAction(selector="input", text="test@#$%^&*()")
        assert action.text == "test@#$%^&*()"


class TestWaitAction:
    """Test wait action validation."""

    def test_valid_wait_action(self):
        """Test valid wait action."""
        action = WaitAction(timeout=5000)
        assert action.type == "wait"
        assert action.timeout == 5000

    def test_wait_action_timeout_validation(self):
        """Test timeout validation rules."""
        # Valid timeout
        action = WaitAction(timeout=1000)
        assert action.timeout == 1000

        # Invalid: zero timeout
        with pytest.raises(ValidationError) as exc_info:
            WaitAction(timeout=0)
        assert "Input should be greater than 0" in str(exc_info.value)

        # Invalid: negative timeout
        with pytest.raises(ValidationError) as exc_info:
            WaitAction(timeout=-1000)
        assert "Input should be greater than 0" in str(exc_info.value)

        # Invalid: timeout too long
        with pytest.raises(ValidationError) as exc_info:
            WaitAction(timeout=50000)
        assert "Input should be less than or equal to 30000" in str(exc_info.value)

    def test_wait_action_maximum_timeout(self):
        """Test maximum allowed timeout."""
        action = WaitAction(timeout=30000)
        assert action.timeout == 30000


class TestWaitForSelectorAction:
    """Test wait for selector action validation."""

    def test_valid_wait_for_selector_action(self):
        """Test valid wait for selector action."""
        action = WaitForSelectorAction(selector=".loading")
        assert action.type == "wait_for_selector"
        assert action.selector == ".loading"
        assert action.timeout == 5000  # default

    def test_wait_for_selector_custom_timeout(self):
        """Test wait for selector with custom timeout."""
        action = WaitForSelectorAction(selector=".element", timeout=10000)
        assert action.timeout == 10000

    def test_wait_for_selector_timeout_validation(self):
        """Test timeout validation for wait for selector."""
        with pytest.raises(ValidationError):
            WaitForSelectorAction(selector=".test", timeout=0)

        with pytest.raises(ValidationError):
            WaitForSelectorAction(selector=".test", timeout=50000)


class TestWaitForLoadStateAction:
    """Test wait for load state action validation."""

    def test_valid_wait_for_load_state_action(self):
        """Test valid wait for load state action."""
        action = WaitForLoadStateAction()
        assert action.type == "wait_for_load_state"
        assert action.state == "networkidle"  # default

    def test_wait_for_load_state_custom_state(self):
        """Test wait for load state with custom state."""
        for state in ["load", "domcontentloaded", "networkidle"]:
            action = WaitForLoadStateAction(state=state)
            assert action.state == state

    def test_wait_for_load_state_invalid_state(self):
        """Test invalid load state."""
        with pytest.raises(ValidationError) as exc_info:
            WaitForLoadStateAction(state="invalid_state")
        assert "Input should be 'load', 'domcontentloaded' or 'networkidle'" in str(
            exc_info.value
        )


class TestScrollAction:
    """Test scroll action validation."""

    def test_valid_scroll_action_bottom(self):
        """Test valid scroll to bottom."""
        action = ScrollAction()
        assert action.type == "scroll"
        assert action.direction == "bottom"  # default
        assert action.y == 0

    def test_valid_scroll_action_top(self):
        """Test valid scroll to top."""
        action = ScrollAction(direction="top")
        assert action.direction == "top"

    def test_valid_scroll_action_position(self):
        """Test valid scroll to position."""
        action = ScrollAction(direction="position", y=500)
        assert action.direction == "position"
        assert action.y == 500

    def test_scroll_action_position_requires_y(self):
        """Test that position scrolling requires Y coordinate."""
        with pytest.raises(ValidationError) as exc_info:
            ScrollAction(direction="position")
        assert "Y position must be specified for position-based scrolling" in str(
            exc_info.value
        )

    def test_scroll_action_position_zero_y_invalid(self):
        """Test that Y=0 is invalid for position scrolling."""
        with pytest.raises(ValidationError) as exc_info:
            ScrollAction(direction="position", y=0)
        assert "Y position must be specified for position-based scrolling" in str(
            exc_info.value
        )

    def test_scroll_action_invalid_direction(self):
        """Test invalid scroll direction."""
        with pytest.raises(ValidationError) as exc_info:
            ScrollAction(direction="invalid")
        assert "Input should be 'top', 'bottom' or 'position'" in str(exc_info.value)


class TestScreenshotAction:
    """Test screenshot action validation."""

    def test_valid_screenshot_action_defaults(self):
        """Test screenshot action with defaults."""
        action = ScreenshotAction()
        assert action.type == "screenshot"
        assert action.path == ""
        assert action.full_page is False

    def test_valid_screenshot_action_custom(self):
        """Test screenshot action with custom values."""
        action = ScreenshotAction(path="/tmp/screenshot.png", full_page=True)
        assert action.path == "/tmp/screenshot.png"
        assert action.full_page is True


class TestEvaluateAction:
    """Test evaluate action validation."""

    def test_valid_evaluate_action(self):
        """Test valid evaluate action."""
        script = "document.title"
        action = EvaluateAction(script=script)
        assert action.type == "evaluate"
        assert action.script == script

    def test_evaluate_action_complex_script(self):
        """Test evaluate action with complex JavaScript."""
        script = """
        const elements = document.querySelectorAll('h1, h2, h3');
        return Array.from(elements).map(el => el.textContent);
        """
        action = EvaluateAction(script=script)
        assert action.script == script

    def test_evaluate_action_requires_script(self):
        """Test that script is required."""
        with pytest.raises(ValidationError):
            EvaluateAction()


class TestHoverAction:
    """Test hover action validation."""

    def test_valid_hover_action(self):
        """Test valid hover action."""
        action = HoverAction(selector=".menu-item")
        assert action.type == "hover"
        assert action.selector == ".menu-item"

    def test_hover_action_requires_selector(self):
        """Test that selector is required."""
        with pytest.raises(ValidationError):
            HoverAction()


class TestSelectAction:
    """Test select action validation."""

    def test_valid_select_action(self):
        """Test valid select action."""
        action = SelectAction(selector="select[name='country']", value="US")
        assert action.type == "select"
        assert action.selector == "select[name='country']"
        assert action.value == "US"

    def test_select_action_requires_selector_and_value(self):
        """Test that both selector and value are required."""
        with pytest.raises(ValidationError):
            SelectAction(selector="select")

        with pytest.raises(ValidationError):
            SelectAction(value="option")


class TestPressAction:
    """Test press action validation."""

    def test_valid_press_action_with_selector(self):
        """Test valid press action with selector."""
        action = PressAction(key="Enter", selector="input")
        assert action.type == "press"
        assert action.key == "Enter"
        assert action.selector == "input"

    def test_valid_press_action_without_selector(self):
        """Test valid press action without selector."""
        action = PressAction(key="ArrowDown")
        assert action.key == "ArrowDown"
        assert action.selector == ""  # default

    def test_press_action_requires_key(self):
        """Test that key is required."""
        with pytest.raises(ValidationError):
            PressAction()

    def test_press_action_special_keys(self):
        """Test press action with special keys."""
        special_keys = [
            "Enter",
            "Escape",
            "Tab",
            "ArrowUp",
            "ArrowDown",
            "F1",
            "Control+c",
        ]
        for key in special_keys:
            action = PressAction(key=key)
            assert action.key == key


class TestDragAndDropAction:
    """Test drag and drop action validation."""

    def test_valid_drag_and_drop_action(self):
        """Test valid drag and drop action."""
        action = DragAndDropAction(source=".drag-source", target=".drop-target")
        assert action.type == "drag_and_drop"
        assert action.source == ".drag-source"
        assert action.target == ".drop-target"

    def test_drag_and_drop_requires_source_and_target(self):
        """Test that both source and target are required."""
        with pytest.raises(ValidationError):
            DragAndDropAction(source=".source")

        with pytest.raises(ValidationError):
            DragAndDropAction(target=".target")


class TestValidateAction:
    """Test the validate_action utility function."""

    def test_validate_action_click(self):
        """Test validating click action."""
        action_dict = {"type": "click", "selector": "button"}
        result = validate_action(action_dict)
        assert isinstance(result, ClickAction)
        assert result.selector == "button"

    def test_validate_action_fill(self):
        """Test validating fill action."""
        action_dict = {"type": "fill", "selector": "input", "text": "test"}
        result = validate_action(action_dict)
        assert isinstance(result, FillAction)
        assert result.text == "test"

    def test_validate_action_wait(self):
        """Test validating wait action."""
        action_dict = {"type": "wait", "timeout": 2000}
        result = validate_action(action_dict)
        assert isinstance(result, WaitAction)
        assert result.timeout == 2000

    def test_validate_action_scroll_position(self):
        """Test validating scroll action with position."""
        action_dict = {"type": "scroll", "direction": "position", "y": 1000}
        result = validate_action(action_dict)
        assert isinstance(result, ScrollAction)
        assert result.y == 1000

    def test_validate_action_screenshot(self):
        """Test validating screenshot action."""
        action_dict = {"type": "screenshot", "path": "test.png", "full_page": True}
        result = validate_action(action_dict)
        assert isinstance(result, ScreenshotAction)
        assert result.path == "test.png"
        assert result.full_page is True

    def test_validate_action_evaluate(self):
        """Test validating evaluate action."""
        action_dict = {"type": "evaluate", "script": "console.log('test')"}
        result = validate_action(action_dict)
        assert isinstance(result, EvaluateAction)
        assert result.script == "console.log('test')"

    def test_validate_action_unsupported_type(self):
        """Test validating unsupported action type."""
        action_dict = {"type": "unsupported_action"}
        with pytest.raises(ValueError) as exc_info:
            validate_action(action_dict)
        assert "Unsupported action type: unsupported_action" in str(exc_info.value)

    def test_validate_action_invalid_data(self):
        """Test validating action with invalid data."""
        action_dict = {"type": "wait", "timeout": -1000}
        with pytest.raises(ValidationError):
            validate_action(action_dict)

    def test_validate_action_missing_type(self):
        """Test validating action without type."""
        action_dict = {"selector": "button"}
        with pytest.raises(ValueError) as exc_info:
            validate_action(action_dict)
        assert "Unsupported action type: None" in str(exc_info.value)


class TestValidateActions:
    """Test the validate_actions utility function."""

    def test_validate_actions_single(self):
        """Test validating single action."""
        actions = [{"type": "click", "selector": "button"}]
        results = validate_actions(actions)
        assert len(results) == 1
        assert isinstance(results[0], ClickAction)

    def test_validate_actions_multiple(self):
        """Test validating multiple actions."""
        actions = [
            {"type": "click", "selector": "button"},
            {"type": "wait", "timeout": 1000},
            {"type": "fill", "selector": "input", "text": "test"},
        ]
        results = validate_actions(actions)
        assert len(results) == 3
        assert isinstance(results[0], ClickAction)
        assert isinstance(results[1], WaitAction)
        assert isinstance(results[2], FillAction)

    def test_validate_actions_empty_list(self):
        """Test validating empty action list."""
        actions = []
        results = validate_actions(actions)
        assert results == []

    def test_validate_actions_complex_workflow(self):
        """Test validating complex workflow."""
        actions = [
            {"type": "wait_for_load_state", "state": "networkidle"},
            {"type": "click", "selector": ".menu-toggle"},
            {"type": "wait_for_selector", "selector": ".menu", "timeout": 3000},
            {"type": "hover", "selector": ".menu-item"},
            {"type": "click", "selector": ".submenu-item"},
            {"type": "fill", "selector": "#search", "text": "test query"},
            {"type": "press", "key": "Enter", "selector": "#search"},
            {"type": "wait", "timeout": 2000},
            {"type": "scroll", "direction": "bottom"},
            {"type": "screenshot", "path": "result.png", "full_page": True},
        ]
        results = validate_actions(actions)
        assert len(results) == 10
        assert all(hasattr(action, "type") for action in results)

    def test_validate_actions_with_invalid_action(self):
        """Test validating actions with one invalid action."""
        actions = [
            {"type": "click", "selector": "button"},
            {"type": "wait", "timeout": -1000},  # Invalid
        ]
        with pytest.raises(ValidationError):
            validate_actions(actions)

    def test_validate_actions_all_action_types(self):
        """Test validating all supported action types."""
        actions = [
            {"type": "click", "selector": "button"},
            {"type": "fill", "selector": "input", "text": "test"},
            {"type": "type", "selector": "textarea", "text": "type test"},
            {"type": "wait", "timeout": 1000},
            {"type": "wait_for_selector", "selector": ".element"},
            {"type": "wait_for_load_state", "state": "load"},
            {"type": "scroll", "direction": "top"},
            {"type": "screenshot", "path": "test.png"},
            {"type": "evaluate", "script": "document.title"},
            {"type": "hover", "selector": ".item"},
            {"type": "select", "selector": "select", "value": "option1"},
            {"type": "press", "key": "Enter"},
            {"type": "drag_and_drop", "source": ".source", "target": ".target"},
        ]

        results = validate_actions(actions)
        assert len(results) == 13

        # Verify each action type
        expected_types = [
            ClickAction,
            FillAction,
            TypeAction,
            WaitAction,
            WaitForSelectorAction,
            WaitForLoadStateAction,
            ScrollAction,
            ScreenshotAction,
            EvaluateAction,
            HoverAction,
            SelectAction,
            PressAction,
            DragAndDropAction,
        ]

        for result, expected_type in zip(results, expected_types, strict=False):
            assert isinstance(result, expected_type)


class TestBrowserActionUnion:
    """Test the BrowserAction union type."""

    def test_browser_action_union_coverage(self):
        """Test that BrowserAction union includes all action types."""
        # This test ensures the union type is properly defined
        # by creating instances of each action type
        actions = [
            ClickAction(selector="button"),
            FillAction(selector="input", text="test"),
            TypeAction(selector="textarea", text="type"),
            WaitAction(timeout=1000),
            WaitForSelectorAction(selector=".element"),
            WaitForLoadStateAction(),
            ScrollAction(),
            ScreenshotAction(),
            EvaluateAction(script="console.log('test')"),
            HoverAction(selector=".item"),
            SelectAction(selector="select", value="option"),
            PressAction(key="Enter"),
            DragAndDropAction(source=".source", target=".target"),
        ]

        # All actions should be valid BrowserAction types
        for action in actions:
            assert hasattr(action, "type")
            assert isinstance(action.type, str)
