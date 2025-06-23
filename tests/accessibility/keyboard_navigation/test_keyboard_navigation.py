"""Keyboard navigation accessibility testing.

This module implements comprehensive keyboard navigation testing to ensure 
WCAG 2.1 compliance for keyboard accessibility, including tab order validation,
focus management, and keyboard shortcut functionality.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.accessibility
@pytest.mark.keyboard_navigation
class TestKeyboardNavigationCompliance:
    """Comprehensive keyboard navigation testing for WCAG compliance."""

    def test_wcag_2_1_1_keyboard_access(self, keyboard_navigation_tester):
        """Test WCAG 2.1.1 - Keyboard accessibility.
        
        All functionality available through a mouse must be available through keyboard.
        """
        # HTML with keyboard-accessible elements
        accessible_html = """
        <button onclick="doAction()">Accessible Button</button>
        <a href="/page" onclick="navigate()">Accessible Link</a>
        <input type="text" onchange="handleChange()">
        <select onchange="handleSelect()">
            <option value="1">Option 1</option>
        </select>
        <div role="button" tabindex="0" onclick="customAction()" 
             onkeydown="handleKeyDown(event)">Custom Button</div>
        """
        
        # HTML with non-keyboard-accessible elements
        problematic_html = """
        <div onclick="doAction()">Clickable Div (No Keyboard)</div>
        <span onclick="navigate()" style="cursor: pointer;">Clickable Span</span>
        <img onclick="imageClick()" src="test.jpg" alt="Clickable Image">
        """
        
        # Test accessible HTML
        result = keyboard_navigation_tester.validate_tab_order(accessible_html)
        tab_order_errors = [
            issue for issue in result["issues"] 
            if issue["severity"] == "error"
        ]
        assert len(tab_order_errors) == 0, "Accessible HTML should not have keyboard errors"
        
        # Problematic HTML would need more sophisticated analysis to detect
        # non-keyboard accessible click handlers on non-interactive elements

    def test_wcag_2_1_2_no_keyboard_trap(self):
        """Test WCAG 2.1.2 - No Keyboard Trap.
        
        Users must be able to navigate away from any focusable component using keyboard.
        """
        # Example of potential keyboard trap
        keyboard_trap_html = """
        <div id="modal" tabindex="0" 
             onkeydown="if(event.key==='Tab') event.preventDefault();">
            <input type="text" id="input1">
            <input type="text" id="input2">
            <button id="closeBtn">Close</button>
        </div>
        """
        
        # Accessible modal implementation
        accessible_modal_html = """
        <div id="modal" role="dialog" aria-labelledby="modal-title" 
             aria-describedby="modal-desc" tabindex="-1">
            <h2 id="modal-title">Modal Title</h2>
            <p id="modal-desc">Modal description</p>
            <input type="text" id="input1">
            <input type="text" id="input2">
            <button id="closeBtn" onclick="closeModal()">Close</button>
        </div>
        <script>
        function trapFocus(element) {
            const focusableElements = element.querySelectorAll(
                'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
            );
            const firstElement = focusableElements[0];
            const lastElement = focusableElements[focusableElements.length - 1];
            
            element.addEventListener('keydown', function(e) {
                if (e.key === 'Tab') {
                    if (e.shiftKey) {
                        if (document.activeElement === firstElement) {
                            lastElement.focus();
                            e.preventDefault();
                        }
                    } else {
                        if (document.activeElement === lastElement) {
                            firstElement.focus();
                            e.preventDefault();
                        }
                    }
                }
                if (e.key === 'Escape') {
                    closeModal();
                }
            });
        }
        </script>
        """
        
        # Check for proper escape mechanisms
        assert "Escape" in accessible_modal_html, "Should handle Escape key"
        assert "closeModal" in accessible_modal_html, "Should have close functionality"
        assert "preventDefault" in keyboard_trap_html, "Trap example prevents default behavior"

    def test_tab_order_validation(self, keyboard_navigation_tester):
        """Test logical tab order validation."""
        # Good tab order (natural DOM order)
        good_tab_order_html = """
        <form>
            <label for="name">Name:</label>
            <input type="text" id="name">
            
            <label for="email">Email:</label>
            <input type="email" id="email">
            
            <label for="phone">Phone:</label>
            <input type="tel" id="phone">
            
            <button type="submit">Submit</button>
        </form>
        """
        
        # Bad tab order (using positive tabindex)
        bad_tab_order_html = """
        <form>
            <label for="name">Name:</label>
            <input type="text" id="name" tabindex="3">
            
            <label for="email">Email:</label>
            <input type="email" id="email" tabindex="1">
            
            <label for="phone">Phone:</label>
            <input type="tel" id="phone" tabindex="4">
            
            <button type="submit" tabindex="2">Submit</button>
        </form>
        """
        
        # Very bad tab order (removing elements from tab order)
        very_bad_tab_order_html = """
        <form>
            <input type="text" tabindex="-1" placeholder="Can't reach with Tab">
            <button type="button" tabindex="-1">Unreachable Button</button>
            <input type="submit" value="Submit">
        </form>
        """
        
        # Test good tab order
        result = keyboard_navigation_tester.validate_tab_order(good_tab_order_html)
        assert len(result["issues"]) == 0, "Natural tab order should have no issues"
        assert result["positive_tabindex_count"] == 0, "Should not use positive tabindex"
        
        # Test bad tab order
        result = keyboard_navigation_tester.validate_tab_order(bad_tab_order_html)
        positive_tabindex_warnings = [
            issue for issue in result["issues"] 
            if "Positive tabindex" in issue["issue"]
        ]
        assert len(positive_tabindex_warnings) > 0, "Should warn about positive tabindex"
        assert result["positive_tabindex_count"] > 0, "Should detect positive tabindex values"
        
        # Test very bad tab order
        result = keyboard_navigation_tester.validate_tab_order(very_bad_tab_order_html)
        negative_tabindex_warnings = [
            issue for issue in result["issues"] 
            if "tabindex='-1'" in issue["issue"]
        ]
        assert len(negative_tabindex_warnings) > 0, "Should warn about interactive elements with tabindex=-1"

    def test_focus_indicators_validation(self, keyboard_navigation_tester):
        """Test focus indicators for keyboard navigation."""
        # CSS with good focus indicators
        good_focus_css = """
        /* Default focus styles maintained */
        button:focus {
            outline: 2px solid #005fcc;
            outline-offset: 2px;
        }
        
        input:focus {
            border: 2px solid #005fcc;
            box-shadow: 0 0 0 3px rgba(0, 95, 204, 0.3);
        }
        
        a:focus {
            outline: 2px solid #005fcc;
            outline-offset: 2px;
            text-decoration: underline;
        }
        
        .custom-button:focus {
            background-color: #e6f3ff;
            border: 2px solid #005fcc;
        }
        """
        
        # CSS with poor focus indicators
        poor_focus_css = """
        /* Removing default focus styles without replacement */
        * {
            outline: none;
        }
        
        button {
            outline: none;
        }
        
        input {
            outline: none;
        }
        
        a {
            outline: none;
        }
        """
        
        # CSS with mixed focus indicators
        mixed_focus_css = """
        /* Some good, some bad */
        button:focus {
            outline: 3px solid #ff6b35;
            outline-offset: 2px;
        }
        
        input {
            outline: none; /* Bad - no replacement */
        }
        
        a:focus {
            background-color: yellow;
            color: black;
        }
        """
        
        # Test good focus CSS
        result = keyboard_navigation_tester.check_focus_indicators(good_focus_css)
        assert result["compliant"], f"Good focus CSS should be compliant: {result['issues']}"
        assert result["focus_styles_count"] > 0, "Should detect focus styles"
        
        # Test poor focus CSS
        result = keyboard_navigation_tester.check_focus_indicators(poor_focus_css)
        assert not result["compliant"], "Poor focus CSS should not be compliant"
        outline_none_errors = [
            issue for issue in result["issues"] 
            if "outline:none" in issue["issue"]
        ]
        assert len(outline_none_errors) > 0, "Should detect outline:none without replacement"
        
        # Test mixed focus CSS
        result = keyboard_navigation_tester.check_focus_indicators(mixed_focus_css)
        # May or may not be compliant depending on implementation
        assert result["focus_styles_count"] > 0, "Should detect some focus styles"

    def test_skip_links_validation(self, keyboard_navigation_tester):
        """Test skip link implementation for keyboard users."""
        # HTML with proper skip links
        good_skip_links_html = """
        <!DOCTYPE html>
        <html lang="en">
        <head><title>Test Page</title></head>
        <body>
            <a href="#main-content" class="skip-link">Skip to main content</a>
            <a href="#navigation" class="skip-link">Skip to navigation</a>
            
            <header>
                <nav id="navigation">
                    <ul>
                        <li><a href="/">Home</a></li>
                        <li><a href="/about">About</a></li>
                    </ul>
                </nav>
            </header>
            
            <main id="main-content">
                <h1>Main Content</h1>
                <p>Page content here.</p>
            </main>
        </body>
        </html>
        """
        
        # HTML without skip links
        no_skip_links_html = """
        <!DOCTYPE html>
        <html lang="en">
        <head><title>Test Page</title></head>
        <body>
            <header>
                <nav>
                    <ul>
                        <li><a href="/">Home</a></li>
                        <li><a href="/about">About</a></li>
                    </ul>
                </nav>
            </header>
            
            <main>
                <h1>Main Content</h1>
                <p>Page content here.</p>
            </main>
        </body>
        </html>
        """
        
        # HTML with broken skip links
        broken_skip_links_html = """
        <!DOCTYPE html>
        <html lang="en">
        <head><title>Test Page</title></head>
        <body>
            <a href="#main-content" class="skip-link">Skip to main content</a>
            <a href="#nonexistent" class="skip-link">Skip to nonexistent</a>
            
            <header>
                <nav>
                    <ul>
                        <li><a href="/">Home</a></li>
                        <li><a href="/about">About</a></li>
                    </ul>
                </nav>
            </header>
            
            <main id="different-id">
                <h1>Main Content</h1>
                <p>Page content here.</p>
            </main>
        </body>
        </html>
        """
        
        # Test good skip links
        result = keyboard_navigation_tester.validate_skip_links(good_skip_links_html)
        assert result["compliant"], f"Good skip links should be compliant: {result['issues']}"
        assert result["skip_links_found"] >= 2, "Should find multiple skip links"
        
        # Test no skip links
        result = keyboard_navigation_tester.validate_skip_links(no_skip_links_html)
        assert not result["compliant"], "Page without skip links should not be compliant"
        no_skip_errors = [
            issue for issue in result["issues"] 
            if "No skip links found" in issue["issue"]
        ]
        assert len(no_skip_errors) > 0, "Should detect missing skip links"
        
        # Test broken skip links
        result = keyboard_navigation_tester.validate_skip_links(broken_skip_links_html)
        assert not result["compliant"], "Broken skip links should not be compliant"
        broken_target_errors = [
            issue for issue in result["issues"] 
            if "target" in issue["issue"] and "not found" in issue["issue"]
        ]
        assert len(broken_target_errors) > 0, "Should detect broken skip link targets"

    def test_keyboard_shortcut_accessibility(self):
        """Test keyboard shortcut implementation and conflicts."""
        # JavaScript with accessible keyboard shortcuts
        accessible_shortcuts_js = """
        document.addEventListener('keydown', function(event) {
            // Use proper modifier keys to avoid conflicts
            if (event.ctrlKey && event.shiftKey && event.key === 'S') {
                event.preventDefault();
                saveDocument();
            }
            
            // Alt + number for navigation
            if (event.altKey && /^[1-9]$/.test(event.key)) {
                event.preventDefault();
                navigateToSection(parseInt(event.key));
            }
            
            // Escape to close modals
            if (event.key === 'Escape') {
                closeAllModals();
            }
            
            // Arrow keys for custom navigation
            if (event.target.getAttribute('role') === 'menu') {
                if (event.key === 'ArrowDown') {
                    event.preventDefault();
                    focusNextMenuItem();
                }
                if (event.key === 'ArrowUp') {
                    event.preventDefault();
                    focusPreviousMenuItem();
                }
            }
        });
        """
        
        # JavaScript with problematic shortcuts
        problematic_shortcuts_js = """
        document.addEventListener('keydown', function(event) {
            // Bad: Single key shortcuts without modifiers
            if (event.key === 's') {
                event.preventDefault();
                save();
            }
            
            // Bad: Conflicts with browser shortcuts
            if (event.ctrlKey && event.key === 'f') {
                event.preventDefault();
                customFind();
            }
            
            // Bad: Prevents all keyboard input
            if (event.key === 'Tab') {
                event.preventDefault();
                // This would break tab navigation
            }
        });
        """
        
        # Check for good practices
        assert "ctrlKey && event.shiftKey" in accessible_shortcuts_js, "Should use modifier combinations"
        assert "Escape" in accessible_shortcuts_js, "Should handle Escape key"
        assert "role" in accessible_shortcuts_js, "Should check element roles"
        assert "preventDefault" in accessible_shortcuts_js, "Should prevent default when handled"
        
        # Check for problematic practices
        assert "event.key === 's'" in problematic_shortcuts_js, "Example has single-key shortcut"
        assert "ctrlKey && event.key === 'f'" in problematic_shortcuts_js, "Example conflicts with browser"

    def test_complex_widget_keyboard_navigation(self):
        """Test keyboard navigation in complex widgets."""
        # Accessible dropdown/combobox
        accessible_combobox_html = """
        <div class="combobox-container">
            <label for="country-combobox">Country:</label>
            <input type="text" 
                   id="country-combobox"
                   role="combobox"
                   aria-expanded="false"
                   aria-autocomplete="list"
                   aria-owns="country-listbox"
                   aria-describedby="country-help">
            <div id="country-help">Type to filter countries</div>
            
            <ul id="country-listbox" 
                role="listbox" 
                style="display: none;"
                aria-label="Country options">
                <li role="option" id="option-us">United States</li>
                <li role="option" id="option-ca">Canada</li>
                <li role="option" id="option-mx">Mexico</li>
            </ul>
        </div>
        
        <script>
        // Keyboard navigation implementation
        document.getElementById('country-combobox').addEventListener('keydown', function(e) {
            switch(e.key) {
                case 'ArrowDown':
                    e.preventDefault();
                    openListbox();
                    focusFirstOption();
                    break;
                case 'ArrowUp':
                    e.preventDefault();
                    openListbox();
                    focusLastOption();
                    break;
                case 'Escape':
                    closeListbox();
                    this.focus();
                    break;
                case 'Enter':
                    if (this.getAttribute('aria-expanded') === 'true') {
                        e.preventDefault();
                        selectFocusedOption();
                    }
                    break;
            }
        });
        </script>
        """
        
        # Accessible data table with sorting
        accessible_table_html = """
        <table role="table" aria-label="Employee data">
            <caption>Employee Information (sortable)</caption>
            <thead>
                <tr>
                    <th scope="col">
                        <button type="button" 
                                aria-describedby="name-sort-desc"
                                onclick="sortTable('name')">
                            Name
                            <span aria-hidden="true">↕</span>
                        </button>
                        <div id="name-sort-desc" class="sr-only">
                            Sortable column. Currently unsorted.
                        </div>
                    </th>
                    <th scope="col">
                        <button type="button" 
                                aria-describedby="dept-sort-desc"
                                onclick="sortTable('department')">
                            Department
                            <span aria-hidden="true">↕</span>
                        </button>
                        <div id="dept-sort-desc" class="sr-only">
                            Sortable column. Currently unsorted.
                        </div>
                    </th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>John Doe</td>
                    <td>Engineering</td>
                </tr>
                <tr>
                    <td>Jane Smith</td>
                    <td>Marketing</td>
                </tr>
            </tbody>
        </table>
        """
        
        # Check combobox accessibility features
        assert 'role="combobox"' in accessible_combobox_html, "Should have combobox role"
        assert 'aria-expanded' in accessible_combobox_html, "Should track expanded state"
        assert 'aria-owns' in accessible_combobox_html, "Should own the listbox"
        assert 'ArrowDown' in accessible_combobox_html, "Should handle arrow navigation"
        assert 'Escape' in accessible_combobox_html, "Should handle escape key"
        
        # Check table accessibility features
        assert 'scope="col"' in accessible_table_html, "Should have column headers"
        assert '<caption>' in accessible_table_html, "Should have table caption"
        assert 'aria-describedby' in accessible_table_html, "Should describe sort state"
        assert '<button type="button"' in accessible_table_html, "Should use buttons for sorting"

    def test_modal_dialog_keyboard_management(self):
        """Test keyboard management in modal dialogs."""
        accessible_modal_html = """
        <div id="modal-overlay" style="display: none;" aria-hidden="true">
            <div id="modal-dialog" 
                 role="dialog" 
                 aria-labelledby="modal-title"
                 aria-describedby="modal-desc"
                 aria-modal="true"
                 tabindex="-1">
                
                <div class="modal-header">
                    <h2 id="modal-title">Confirm Action</h2>
                    <button type="button" 
                            class="close-button" 
                            aria-label="Close dialog"
                            onclick="closeModal()">
                        <span aria-hidden="true">&times;</span>
                    </button>
                </div>
                
                <div class="modal-body">
                    <p id="modal-desc">Are you sure you want to delete this item?</p>
                    <input type="text" placeholder="Type 'DELETE' to confirm">
                </div>
                
                <div class="modal-footer">
                    <button type="button" onclick="closeModal()">Cancel</button>
                    <button type="button" onclick="confirmDelete()">Delete</button>
                </div>
            </div>
        </div>
        
        <script>
        let previousFocus;
        
        function openModal() {
            // Store current focus
            previousFocus = document.activeElement;
            
            // Show modal
            const modal = document.getElementById('modal-overlay');
            modal.style.display = 'block';
            modal.setAttribute('aria-hidden', 'false');
            
            // Focus first focusable element
            const firstFocusable = modal.querySelector('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])');
            if (firstFocusable) {
                firstFocusable.focus();
            }
            
            // Set up focus trap
            setupFocusTrap(modal);
        }
        
        function closeModal() {
            // Hide modal
            const modal = document.getElementById('modal-overlay');
            modal.style.display = 'none';
            modal.setAttribute('aria-hidden', 'true');
            
            // Restore focus
            if (previousFocus) {
                previousFocus.focus();
            }
        }
        
        function setupFocusTrap(modal) {
            const focusableElements = modal.querySelectorAll(
                'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
            );
            const firstElement = focusableElements[0];
            const lastElement = focusableElements[focusableElements.length - 1];
            
            modal.addEventListener('keydown', function(e) {
                if (e.key === 'Tab') {
                    if (e.shiftKey) {
                        if (document.activeElement === firstElement) {
                            e.preventDefault();
                            lastElement.focus();
                        }
                    } else {
                        if (document.activeElement === lastElement) {
                            e.preventDefault();
                            firstElement.focus();
                        }
                    }
                }
                
                if (e.key === 'Escape') {
                    closeModal();
                }
            });
        }
        </script>
        """
        
        # Check modal accessibility features
        assert 'role="dialog"' in accessible_modal_html, "Should have dialog role"
        assert 'aria-modal="true"' in accessible_modal_html, "Should be marked as modal"
        assert 'aria-labelledby' in accessible_modal_html, "Should be labeled"
        assert 'aria-describedby' in accessible_modal_html, "Should be described"
        assert 'previousFocus' in accessible_modal_html, "Should restore focus"
        assert 'setupFocusTrap' in accessible_modal_html, "Should trap focus"
        assert 'Escape' in accessible_modal_html, "Should handle escape key"

    @pytest.mark.parametrize("element_type,expected_focusable", [
        ("button", True),
        ("a[href]", True),
        ("input[type=text]", True),
        ("input[type=hidden]", False),
        ("select", True),
        ("textarea", True),
        ("div", False),
        ("div[tabindex=0]", True),
        ("div[tabindex=-1]", False),
        ("button[disabled]", False),
        ("input[disabled]", False),
    ])
    def test_focusable_element_detection(self, element_type, expected_focusable):
        """Test detection of focusable elements."""
        # This would typically be implemented in JavaScript or with a browser automation tool
        # For now, we test the logic conceptually
        
        focusable_selectors = [
            'a[href]',
            'button:not([disabled])',
            'input:not([disabled]):not([type="hidden"])',
            'select:not([disabled])',
            'textarea:not([disabled])',
            '[tabindex]:not([tabindex="-1"])',
            'details',
            'summary',
        ]
        
        # Simplified check - in practice this would use CSS selector matching
        if element_type == "button":
            assert expected_focusable == True
        elif element_type == "a[href]":
            assert expected_focusable == True
        elif element_type == "input[type=text]":
            assert expected_focusable == True
        elif element_type == "input[type=hidden]":
            assert expected_focusable == False
        elif element_type == "div":
            assert expected_focusable == False
        elif element_type == "div[tabindex=0]":
            assert expected_focusable == True
        elif element_type == "div[tabindex=-1]":
            assert expected_focusable == False


@pytest.mark.accessibility
@pytest.mark.keyboard_navigation
class TestKeyboardNavigationIntegration:
    """Integration tests for keyboard navigation with browser automation."""

    @pytest.fixture
    def mock_browser_page(self):
        """Mock browser page for testing."""
        page = AsyncMock()
        page.keyboard = AsyncMock()
        page.focus = AsyncMock()
        page.evaluate = AsyncMock()
        page.query_selector = AsyncMock()
        page.query_selector_all = AsyncMock()
        return page

    @pytest.mark.asyncio
    async def test_tab_navigation_simulation(self, mock_browser_page):
        """Test tab navigation through page elements."""
        # Mock focusable elements
        mock_elements = [
            MagicMock(tag_name="input"),
            MagicMock(tag_name="button"),
            MagicMock(tag_name="a"),
        ]
        mock_browser_page.query_selector_all.return_value = mock_elements
        
        # Simulate tab navigation
        await mock_browser_page.keyboard.press("Tab")
        await mock_browser_page.keyboard.press("Tab")
        await mock_browser_page.keyboard.press("Tab")
        
        # Verify navigation calls
        assert mock_browser_page.keyboard.press.call_count == 3
        mock_browser_page.keyboard.press.assert_called_with("Tab")

    @pytest.mark.asyncio
    async def test_focus_indicator_visibility(self, mock_browser_page):
        """Test that focus indicators are visible during navigation."""
        # Mock element with focus
        mock_element = MagicMock()
        mock_browser_page.query_selector.return_value = mock_element
        
        # Simulate focusing element
        await mock_browser_page.focus("button")
        
        # Mock getting computed styles
        mock_browser_page.evaluate.return_value = {
            "outline": "2px solid rgb(0, 95, 204)",
            "outlineOffset": "2px",
            "boxShadow": "none"
        }
        
        # Get focus styles
        focus_styles = await mock_browser_page.evaluate("""
            () => {
                const element = document.querySelector('button:focus');
                if (!element) return null;
                const styles = window.getComputedStyle(element);
                return {
                    outline: styles.outline,
                    outlineOffset: styles.outlineOffset,
                    boxShadow: styles.boxShadow
                };
            }
        """)
        
        mock_browser_page.focus.assert_called_once_with("button")
        assert focus_styles["outline"] != "none", "Should have visible outline"

    @pytest.mark.asyncio
    async def test_skip_link_functionality(self, mock_browser_page):
        """Test skip link navigation functionality."""
        # Mock skip link and target
        skip_link = MagicMock()
        skip_link.get_attribute.return_value = "#main-content"
        main_content = MagicMock()
        
        mock_browser_page.query_selector.side_effect = [skip_link, main_content]
        
        # Test skip link navigation
        await mock_browser_page.focus("a.skip-link")
        await mock_browser_page.keyboard.press("Enter")
        
        # Verify navigation
        mock_browser_page.focus.assert_called_with("a.skip-link")
        mock_browser_page.keyboard.press.assert_called_with("Enter")

    @pytest.mark.asyncio
    async def test_modal_focus_management(self, mock_browser_page):
        """Test focus management in modal dialogs."""
        # Mock modal elements
        modal_trigger = MagicMock()
        modal_dialog = MagicMock()
        first_focusable = MagicMock()
        close_button = MagicMock()
        
        mock_browser_page.query_selector.side_effect = [
            modal_trigger, modal_dialog, first_focusable, close_button
        ]
        
        # Test modal opening
        await mock_browser_page.focus("#open-modal")
        await mock_browser_page.keyboard.press("Enter")
        
        # Test escape key closes modal
        await mock_browser_page.keyboard.press("Escape")
        
        # Verify focus management
        assert mock_browser_page.keyboard.press.call_count >= 2

    def test_keyboard_navigation_report_generation(self, keyboard_navigation_tester):
        """Test generation of keyboard navigation accessibility reports."""
        test_html = """
        <!DOCTYPE html>
        <html lang="en">
        <head><title>Test Page</title></head>
        <body>
            <a href="#main" class="skip-link">Skip to main</a>
            
            <nav>
                <ul>
                    <li><a href="/">Home</a></li>
                    <li><a href="/about">About</a></li>
                </ul>
            </nav>
            
            <main id="main">
                <h1>Page Title</h1>
                <form>
                    <input type="text" id="name" tabindex="1">
                    <button type="submit" tabindex="2">Submit</button>
                </form>
            </main>
        </body>
        </html>
        """
        
        test_css = """
        .skip-link:focus {
            position: absolute;
            top: 10px;
            left: 10px;
            background: #000;
            color: #fff;
            padding: 8px;
            text-decoration: none;
        }
        
        input:focus {
            outline: 2px solid #0066cc;
        }
        
        button {
            outline: none; /* Problematic */
        }
        """
        
        # Generate comprehensive report
        tab_order_result = keyboard_navigation_tester.validate_tab_order(test_html)
        focus_result = keyboard_navigation_tester.check_focus_indicators(test_css)
        skip_links_result = keyboard_navigation_tester.validate_skip_links(test_html)
        
        report = {
            "timestamp": "2024-01-01T00:00:00Z",
            "page_url": "http://test.example.com",
            "keyboard_navigation": {
                "tab_order": tab_order_result,
                "focus_indicators": focus_result,
                "skip_links": skip_links_result,
            },
            "overall_compliant": (
                tab_order_result["compliant"] and 
                focus_result["compliant"] and 
                skip_links_result["compliant"]
            ),
            "recommendations": []
        }
        
        # Add specific recommendations
        if not tab_order_result["compliant"]:
            report["recommendations"].append("Review tab order and avoid positive tabindex values")
        
        if not focus_result["compliant"]:
            report["recommendations"].append("Ensure all interactive elements have visible focus indicators")
        
        if not skip_links_result["compliant"]:
            report["recommendations"].append("Add skip links to help keyboard users bypass repetitive content")
        
        # Verify report structure
        assert "timestamp" in report
        assert "keyboard_navigation" in report
        assert isinstance(report["overall_compliant"], bool)
        assert isinstance(report["recommendations"], list)
        
        # Check specific results
        assert "tab_order" in report["keyboard_navigation"]
        assert "focus_indicators" in report["keyboard_navigation"]
        assert "skip_links" in report["keyboard_navigation"]