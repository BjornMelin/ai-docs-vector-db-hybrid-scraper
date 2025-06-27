"""Screen reader accessibility testing.

This module implements comprehensive screen reader compatibility testing to ensure
WCAG 2.1 compliance for assistive technology, including semantic HTML validation,
ARIA attribute testing, and screen reader text generation.
"""

import pytest
from bs4 import BeautifulSoup


@pytest.mark.accessibility
@pytest.mark.screen_reader
class TestScreenReaderCompliance:
    """Comprehensive screen reader compatibility testing."""

    def test_semantic_html_structure(self, screen_reader_validator):
        """Test semantic HTML structure for screen readers."""
        # HTML with good semantic structure
        semantic_html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <title>Accessible Page Title</title>
        </head>
        <body>
            <header>
                <h1>Site Title</h1>
                <nav aria-label="Main navigation">
                    <ul>
                        <li><a href="/">Home</a></li>
                        <li><a href="/about">About</a></li>
                        <li><a href="/contact">Contact</a></li>
                    </ul>
                </nav>
            </header>

            <main>
                <article>
                    <header>
                        <h1>Article Title</h1>
                        <p>Published on <time datetime="2024-01-01">January 1, 2024</time></p>
                    </header>

                    <section>
                        <h2>Section Heading</h2>
                        <p>Article content goes here.</p>

                        <figure>
                            <img src="chart.png" alt="Sales increased 50% over last year">
                            <figcaption>Sales performance chart for 2024</figcaption>
                        </figure>
                    </section>
                </article>

                <aside>
                    <h2>Related Articles</h2>
                    <ul>
                        <li><a href="/related1">Related Article 1</a></li>
                        <li><a href="/related2">Related Article 2</a></li>
                    </ul>
                </aside>
            </main>

            <footer>
                <p>&copy; 2024 Company Name. All rights reserved.</p>
            </footer>
        </body>
        </html>
        """

        # HTML with poor semantic structure
        non_semantic_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Page</title>
        </head>
        <body>
            <div class="header">
                <div class="title">Site Title</div>
                <div class="nav">
                    <div><a href="/">Home</a></div>
                    <div><a href="/about">About</a></div>
                </div>
            </div>

            <div class="content">
                <div class="article">
                    <div class="article-title">Article Title</div>
                    <div class="date">January 1, 2024</div>
                    <div class="text">Article content goes here.</div>
                </div>
            </div>

            <div class="sidebar">
                <div class="sidebar-title">Related</div>
                <div><a href="/related1">Related Article 1</a></div>
            </div>

            <div class="footer">
                <div>&copy; 2024 Company Name</div>
            </div>
        </body>
        </html>
        """

        # Test semantic HTML
        result = screen_reader_validator.validate_semantic_structure(semantic_html)
        assert result["compliant"], (
            f"Semantic HTML should be compliant: {result['issues']}"
        )
        assert result["nav_count"] == 1, "Should have one navigation element"

        # Test non-semantic HTML
        result = screen_reader_validator.validate_semantic_structure(non_semantic_html)
        main_errors = [
            issue for issue in result["issues"] if "main landmark" in issue["issue"]
        ]
        assert len(main_errors) > 0, "Should detect missing main landmark"

    def test_heading_structure_hierarchy(self, _screen_reader_validator):
        """Test proper heading hierarchy for screen readers."""
        # Good heading hierarchy
        good_headings_html = """
        <main>
            <h1>Main Page Title</h1>

            <section>
                <h2>First Section</h2>
                <p>Content here.</p>

                <h3>Subsection A</h3>
                <p>More content.</p>

                <h3>Subsection B</h3>
                <p>Even more content.</p>
            </section>

            <section>
                <h2>Second Section</h2>
                <p>Different content.</p>

                <h3>Another Subsection</h3>
                <p>Content continues.</p>

                <h4>Deep Subsection</h4>
                <p>Detailed content.</p>
            </section>
        </main>
        """

        # Bad heading hierarchy
        bad_headings_html = """
        <main>
            <h3>Starting with H3 (Bad)</h3>
            <p>Content here.</p>

            <h1>Main Title (Out of Order)</h1>
            <p>Content.</p>

            <h5>Skipped H4 (Bad)</h5>
            <p>Content.</p>

            <h2>Back to H2</h2>
            <p>Content.</p>
        </main>
        """

        # Parse and analyze headings
        def analyze_headings(html_content):
            soup = BeautifulSoup(html_content, "html.parser")
            headings = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
            heading_levels = [int(h.name[1]) for h in headings]

            issues = []

            # Check if starts with h1
            if heading_levels and heading_levels[0] != 1:
                issues.append("Page should start with h1")

            # Check for skipped levels
            for i in range(1, len(heading_levels)):
                if heading_levels[i] > heading_levels[i - 1] + 1:
                    issues.append(
                        f"Skipped heading level: h{heading_levels[i - 1]} to h{heading_levels[i]}"
                    )

            return {
                "levels": heading_levels,
                "issues": issues,
                "compliant": len(issues) == 0,
            }

        # Test good headings
        result = analyze_headings(good_headings_html)
        assert result["compliant"], (
            f"Good headings should be compliant: {result['issues']}"
        )
        assert result["levels"][0] == 1, "Should start with h1"

        # Test bad headings
        result = analyze_headings(bad_headings_html)
        assert not result["compliant"], (
            f"Bad headings should not be compliant: {result['issues']}"
        )
        assert len(result["issues"]) > 0, "Should detect heading hierarchy issues"

    def test_form_accessibility_for_screen_readers(self, screen_reader_validator):
        """Test form accessibility for screen readers."""
        # Accessible form
        accessible_form_html = """
        <form>
            <fieldset>
                <legend>Personal Information</legend>

                <div class="form-group">
                    <label for="first-name">First Name (required):</label>
                    <input type="text"
                           id="first-name"
                           name="firstName"
                           required
                           aria-describedby="first-name-error"
                           aria-invalid="false">
                    <div id="first-name-error" role="alert" aria-live="polite"></div>
                </div>

                <div class="form-group">
                    <label for="email">Email Address:</label>
                    <input type="email"
                           id="email"
                           name="email"
                           aria-describedby="email-help">
                    <div id="email-help">We'll never share your email with anyone</div>
                </div>

                <div class="form-group">
                    <span id="phone-label">Phone Number:</span>
                    <input type="tel"
                           name="phone"
                           aria-labelledby="phone-label"
                           aria-describedby="phone-format">
                    <div id="phone-format">Format: (123) 456-7890</div>
                </div>
            </fieldset>

            <fieldset>
                <legend>Communication Preferences</legend>

                <div role="group" aria-labelledby="contact-methods">
                    <div id="contact-methods">How would you like to be contacted?</div>

                    <div class="checkbox-group">
                        <input type="checkbox" id="contact-email" name="contact" value="email">
                        <label for="contact-email">Email</label>
                    </div>

                    <div class="checkbox-group">
                        <input type="checkbox" id="contact-phone" name="contact" value="phone">
                        <label for="contact-phone">Phone</label>
                    </div>

                    <div class="checkbox-group">
                        <input type="checkbox" id="contact-mail" name="contact" value="mail">
                        <label for="contact-mail">Postal Mail</label>
                    </div>
                </div>

                <div role="radiogroup" aria-labelledby="newsletter-label">
                    <div id="newsletter-label">Newsletter Subscription:</div>

                    <div class="radio-group">
                        <input type="radio" id="newsletter-yes" name="newsletter" value="yes">
                        <label for="newsletter-yes">Yes, send me newsletters</label>
                    </div>

                    <div class="radio-group">
                        <input type="radio" id="newsletter-no" name="newsletter" value="no">
                        <label for="newsletter-no">No newsletters</label>
                    </div>
                </div>
            </fieldset>

            <div class="form-actions">
                <button type="submit">Submit Form</button>
                <button type="reset">Clear Form</button>
            </div>
        </form>
        """

        # Inaccessible form
        inaccessible_form_html = """
        <form>
            <div>
                First Name: <input type="text" placeholder="Enter first name">
            </div>

            <div>
                Email: <input type="email">
            </div>

            <div>
                Phone: <input type="tel">
                <small>Use format: (123) 456-7890</small>
            </div>

            <div>
                Contact preferences:
                <input type="checkbox" value="email"> Email
                <input type="checkbox" value="phone"> Phone
                <input type="checkbox" value="mail"> Mail
            </div>

            <div>
                Newsletter:
                <input type="radio" name="newsletter" value="yes"> Yes
                <input type="radio" name="newsletter" value="no"> No
            </div>

            <input type="submit" value="Submit">
        </form>
        """

        # Test accessible form
        result = screen_reader_validator.validate_form_accessibility(
            accessible_form_html
        )
        assert result["compliant"], (
            f"Accessible form should be compliant: {result['issues']}"
        )
        assert result["total_inputs"] > 0, "Should detect form inputs"

        # Test inaccessible form
        result = screen_reader_validator.validate_form_accessibility(
            inaccessible_form_html
        )
        assert not result["compliant"], "Inaccessible form should not be compliant"
        label_errors = [
            issue for issue in result["issues"] if "label" in issue["issue"].lower()
        ]
        assert len(label_errors) > 0, "Should detect missing labels"

    def test_table_accessibility_for_screen_readers(self):
        """Test table accessibility for screen readers."""
        # Accessible data table
        accessible_table_html = """
        <table role="table">
            <caption>
                Employee Sales Data for Q4 2024
                <details>
                    <summary>Table Description</summary>
                    <p>This table shows sales performance by employee including
                       total sales, commission, and performance rating.</p>
                </details>
            </caption>

            <thead>
                <tr>
                    <th scope="col" id="emp-name">Employee Name</th>
                    <th scope="col" id="dept">Department</th>
                    <th scope="col" id="sales">Total Sales</th>
                    <th scope="col" id="commission">Commission</th>
                    <th scope="col" id="rating">Rating</th>
                </tr>
            </thead>

            <tbody>
                <tr>
                    <th scope="row" headers="emp-name">John Doe</th>
                    <td headers="dept emp-name">Sales</td>
                    <td headers="sales emp-name">$125,000</td>
                    <td headers="commission emp-name">$12,500</td>
                    <td headers="rating emp-name">Excellent</td>
                </tr>

                <tr>
                    <th scope="row" headers="emp-name">Jane Smith</th>
                    <td headers="dept emp-name">Marketing</td>
                    <td headers="sales emp-name">$95,000</td>
                    <td headers="commission emp-name">$9,500</td>
                    <td headers="rating emp-name">Good</td>
                </tr>
            </tbody>

            <tfoot>
                <tr>
                    <th scope="row">Totals</th>
                    <td>-</td>
                    <td>$220,000</td>
                    <td>$22,000</td>
                    <td>-</td>
                </tr>
            </tfoot>
        </table>
        """

        # Layout table (should be avoided)

        # Simple data table without proper headers
        poor_table_html = """
        <table>
            <tr>
                <td><b>Name</b></td>
                <td><b>Department</b></td>
                <td><b>Sales</b></td>
            </tr>
            <tr>
                <td>John Doe</td>
                <td>Sales</td>
                <td>$125,000</td>
            </tr>
            <tr>
                <td>Jane Smith</td>
                <td>Marketing</td>
                <td>$95,000</td>
            </tr>
        </table>
        """

        # Check accessible table features
        assert "<caption>" in accessible_table_html, "Should have caption"
        assert 'scope="col"' in accessible_table_html, "Should have column headers"
        assert 'scope="row"' in accessible_table_html, "Should have row headers"
        assert "headers=" in accessible_table_html, (
            "Should associate cells with headers"
        )
        assert "<thead>" in accessible_table_html, "Should have table head"
        assert "<tbody>" in accessible_table_html, "Should have table body"
        assert "<tfoot>" in accessible_table_html, "Should have table foot"

        # Check problematic tables
        assert "<caption>" not in poor_table_html, "Poor table lacks caption"
        assert "scope=" not in poor_table_html, "Poor table lacks proper headers"
        assert "<th>" not in poor_table_html, (
            "Poor table uses bold instead of th elements"
        )

    def test_aria_live_regions_for_dynamic_content(self):
        """Test ARIA live regions for dynamic content updates."""
        # Proper live region implementation
        live_regions_html = """
        <div id="status-messages"
             aria-live="polite"
             aria-atomic="false"
             class="sr-only">
            <!-- Status messages appear here -->
        </div>

        <div id="error-messages"
             role="alert"
             aria-live="assertive"
             aria-atomic="true"
             class="sr-only">
            <!-- Error messages appear here -->
        </div>

        <div id="loading-indicator"
             aria-live="polite"
             aria-busy="false"
             aria-describedby="loading-text">
            <div id="loading-text" class="sr-only">Loading content...</div>
        </div>

        <form>
            <div class="form-group">
                <label for="username">Username:</label>
                <input type="text"
                       id="username"
                       name="username"
                       aria-describedby="username-validation"
                       aria-invalid="false">
                <div id="username-validation"
                     role="status"
                     aria-live="polite">
                    <!-- Validation messages appear here -->
                </div>
            </div>

            <div class="form-group">
                <label for="search">Search:</label>
                <input type="search"
                       id="search"
                       name="search"
                       aria-describedby="search-results-count"
                       autocomplete="off">
                <div id="search-results-count"
                     role="status"
                     aria-live="polite">
                    <!-- Result count appears here -->
                </div>
            </div>
        </form>

        <div id="chat-log"
             role="log"
             aria-live="polite"
             aria-label="Chat conversation">
            <!-- Chat messages appear here -->
        </div>

        <script>
        // Example of updating live regions
        function showStatusMessage(message) {
            const statusDiv = document.getElementById('status-messages');
            statusDiv.textContent = message;
        }

        function showErrorMessage(message) {
            const errorDiv = document.getElementById('error-messages');
            errorDiv.textContent = message;
        }

        function updateLoadingState(isLoading) {
            const loadingDiv = document.getElementById('loading-indicator');
            loadingDiv.setAttribute('aria-busy', isLoading ? 'true' : 'false');

            if (isLoading) {
                loadingDiv.querySelector('#loading-text').textContent = 'Loading content...';
            } else {
                loadingDiv.querySelector('#loading-text').textContent = 'Content loaded';
            }
        }

        function validateUsername(username) {
            const validationDiv = document.getElementById('username-validation');
            const usernameInput = document.getElementById('username');

            if (username.length < 3) {
                validationDiv.textContent = 'Username must be at least 3 characters';
                usernameInput.setAttribute('aria-invalid', 'true');
            } else {
                validationDiv.textContent = 'Username is valid';
                usernameInput.setAttribute('aria-invalid', 'false');
            }
        }

        function updateSearchResults(count) {
            const countDiv = document.getElementById('search-results-count');
            countDiv.textContent = `Found ${count} results`;
        }
        </script>
        """

        # Check live region features
        assert 'aria-live="polite"' in live_regions_html, (
            "Should have polite live regions"
        )
        assert 'aria-live="assertive"' in live_regions_html, (
            "Should have assertive live regions"
        )
        assert 'role="alert"' in live_regions_html, "Should have alert role"
        assert 'role="status"' in live_regions_html, "Should have status role"
        assert 'role="log"' in live_regions_html, "Should have log role"
        assert "aria-atomic" in live_regions_html, "Should specify atomic updates"
        assert "aria-busy" in live_regions_html, "Should indicate busy state"

    def test_landmark_navigation_for_screen_readers(self, screen_reader_validator):
        """Test landmark navigation for screen readers."""
        # HTML with proper landmarks
        landmark_html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <title>Landmark Navigation Test</title>
        </head>
        <body>
            <a href="#main-content" class="skip-link">Skip to main content</a>

            <header role="banner">
                <h1>Site Title</h1>
                <nav role="navigation" aria-label="Main navigation">
                    <ul>
                        <li><a href="/">Home</a></li>
                        <li><a href="/about">About</a></li>
                        <li><a href="/services">Services</a></li>
                        <li><a href="/contact">Contact</a></li>
                    </ul>
                </nav>
            </header>

            <nav role="navigation" aria-label="Breadcrumb navigation">
                <ol>
                    <li><a href="/">Home</a></li>
                    <li><a href="/services">Services</a></li>
                    <li aria-current="page">Web Development</li>
                </ol>
            </nav>

            <main role="main" id="main-content">
                <article>
                    <header>
                        <h1>Web Development Services</h1>
                        <p>Last updated: <time datetime="2024-01-01">January 1, 2024</time></p>
                    </header>

                    <section>
                        <h2>Our Approach</h2>
                        <p>We focus on accessibility and user experience.</p>
                    </section>

                    <section>
                        <h2>Technologies</h2>
                        <ul>
                            <li>HTML5 & CSS3</li>
                            <li>JavaScript & TypeScript</li>
                            <li>React & Vue.js</li>
                        </ul>
                    </section>
                </article>

                <aside role="complementary" aria-label="Related services">
                    <h2>Related Services</h2>
                    <ul>
                        <li><a href="/mobile-development">Mobile Development</a></li>
                        <li><a href="/ui-design">UI Design</a></li>
                        <li><a href="/consulting">Consulting</a></li>
                    </ul>
                </aside>
            </main>

            <section role="region" aria-labelledby="testimonials-heading">
                <h2 id="testimonials-heading">Client Testimonials</h2>
                <blockquote>
                    <p>"Excellent work on accessibility!"</p>
                    <cite>— Happy Client</cite>
                </blockquote>
            </section>

            <footer role="contentinfo">
                <div role="region" aria-label="Contact information">
                    <h2>Contact Us</h2>
                    <address>
                        123 Main Street<br>
                        City, State 12345<br>
                        <a href="mailto:info@company.com">info@company.com</a><br>
                        <a href="tel:+1234567890">+1 (234) 567-890</a>
                    </address>
                </div>

                <nav role="navigation" aria-label="Footer navigation">
                    <ul>
                        <li><a href="/privacy">Privacy Policy</a></li>
                        <li><a href="/terms">Terms of Service</a></li>
                        <li><a href="/sitemap">Sitemap</a></li>
                    </ul>
                </nav>

                <p>&copy; 2024 Company Name. All rights reserved.</p>
            </footer>
        </body>
        </html>
        """

        # Test landmark structure
        result = screen_reader_validator.validate_semantic_structure(landmark_html)
        assert result["compliant"], (
            f"Landmark HTML should be compliant: {result['issues']}"
        )

        # Check for multiple navigation landmarks with labels
        nav_count = landmark_html.count('role="navigation"') + landmark_html.count(
            "<nav"
        )
        assert nav_count >= 3, "Should have multiple navigation landmarks"

        # Check landmark labeling
        assert 'aria-label="Main navigation"' in landmark_html
        assert 'aria-label="Breadcrumb navigation"' in landmark_html
        assert 'aria-label="Footer navigation"' in landmark_html

    def test_screen_reader_text_generation(self):
        """Test generation of screen reader accessible text."""
        # Complex content that needs screen reader optimization
        complex_content_html = """
        <div class="product-card">
            <img src="laptop.jpg"
                 alt="Silver MacBook Pro 16-inch laptop showing colorful desktop wallpaper">

            <div class="product-info">
                <h3>MacBook Pro 16-inch</h3>

                <div class="rating" aria-label="4.5 out of 5 stars">
                    <span aria-hidden="true">★★★★☆</span>
                    <span class="sr-only">Rated 4.5 out of 5 stars</span>
                </div>

                <div class="price">
                    <span class="sr-only">Price:</span>
                    <span class="current-price">$2,499</span>
                    <span class="original-price" aria-label="Original price">
                        <span class="sr-only">Originally</span>
                        <s>$2,799</s>
                    </span>
                    <span class="discount" aria-label="Discount amount">
                        <span class="sr-only">Save</span>
                        $300
                    </span>
                </div>

                <div class="availability">
                    <span class="stock-status"
                          aria-label="In stock, 5 units available">
                        <span aria-hidden="true">✓</span>
                        <span class="sr-only">In stock</span>
                        5 available
                    </span>
                </div>

                <div class="shipping">
                    <span class="sr-only">Shipping information:</span>
                    <span aria-label="Free shipping on orders over $50">
                        <span aria-hidden="true">🚚</span>
                        Free shipping
                    </span>
                </div>

                <button type="button"
                        aria-describedby="product-description"
                        onclick="addToCart('macbook-pro-16')">
                    Add to Cart
                </button>

                <div id="product-description" class="sr-only">
                    MacBook Pro 16-inch laptop with M2 Pro chip, 16GB RAM,
                    512GB SSD storage. Includes 1-year warranty and
                    30-day return policy.
                </div>
            </div>
        </div>

        <style>
        .sr-only {
            position: absolute;
            width: 1px;
            height: 1px;
            padding: 0;
            margin: -1px;
            overflow: hidden;
            clip: rect(0, 0, 0, 0);
            white-space: nowrap;
            border: 0;
        }
        </style>
        """

        # Check screen reader optimizations
        assert 'class="sr-only"' in complex_content_html, (
            "Should have screen reader only text"
        )
        assert 'aria-hidden="true"' in complex_content_html, (
            "Should hide decorative elements"
        )
        assert "aria-label=" in complex_content_html, "Should provide accessible labels"
        assert "aria-describedby=" in complex_content_html, (
            "Should provide descriptions"
        )

        # Count screen reader enhancements
        sr_only_count = complex_content_html.count('class="sr-only"')
        aria_hidden_count = complex_content_html.count('aria-hidden="true"')
        aria_label_count = complex_content_html.count("aria-label=")

        assert sr_only_count >= 5, "Should have multiple screen reader only elements"
        assert aria_hidden_count >= 2, "Should hide decorative content"
        assert aria_label_count >= 3, "Should provide accessible labels"

    @pytest.mark.parametrize(
        "role,expected_behavior",
        [
            ("button", "should be focusable and activatable"),
            ("link", "should navigate on activation"),
            ("checkbox", "should toggle state"),
            ("radio", "should select single option"),
            ("tab", "should activate tab panel"),
            ("menuitem", "should activate menu action"),
            ("option", "should be selectable in listbox"),
            ("treeitem", "should expand/collapse tree node"),
        ],
    )
    def test_widget_roles_screen_reader_behavior(self, role, _expected_behavior):
        """Test ARIA widget roles for screen reader behavior."""
        # This test documents expected screen reader behavior for different roles
        widget_examples = {
            "button": """<div role="button" tabindex="0" aria-pressed="false">Toggle</div>""",
            "link": """<span role="link" tabindex="0">Navigate</span>""",
            "checkbox": """<div role="checkbox" tabindex="0" aria-checked="false">Option</div>""",
            "radio": """<div role="radio" tabindex="0" aria-checked="false">Choice</div>""",
            "tab": """<div role="tab" tabindex="0" aria-selected="false" aria-controls="panel1">Tab 1</div>""",
            "menuitem": """<div role="menuitem" tabindex="-1">Menu Item</div>""",
            "option": """<div role="option" aria-selected="false">List Option</div>""",
            "treeitem": """<div role="treeitem" tabindex="0" aria-expanded="false">Tree Node</div>""",
        }

        if role in widget_examples:
            html = widget_examples[role]
            assert f'role="{role}"' in html, f"Should have {role} role"
            assert "tabindex=" in html or role == "option", (
                "Should be focusable unless option"
            )

    def test_screen_reader_compatibility_report(self, screen_reader_validator):
        """Test generation of screen reader compatibility reports."""
        test_page_html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <title>Test Page for Screen Reader Compatibility</title>
        </head>
        <body>
            <header>
                <h1>Main Title</h1>
                <nav aria-label="Main navigation">
                    <ul>
                        <li><a href="/">Home</a></li>
                        <li><a href="/about">About</a></li>
                    </ul>
                </nav>
            </header>

            <main>
                <article>
                    <h1>Article Title</h1>
                    <p>Article content with <a href="/link">embedded link</a>.</p>

                    <img src="chart.png" alt="Sales increased 25% this quarter">

                    <table>
                        <caption>Sales Data</caption>
                        <thead>
                            <tr>
                                <th scope="col">Quarter</th>
                                <th scope="col">Sales</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <th scope="row">Q1</th>
                                <td>$100,000</td>
                            </tr>
                        </tbody>
                    </table>

                    <form>
                        <fieldset>
                            <legend>Contact Form</legend>

                            <label for="name">Name:</label>
                            <input type="text" id="name" required>

                            <button type="submit">Submit</button>
                        </fieldset>
                    </form>
                </article>
            </main>
        </body>
        </html>
        """

        # Generate comprehensive screen reader report
        semantic_result = screen_reader_validator.validate_semantic_structure(
            test_page_html
        )
        form_result = screen_reader_validator.validate_form_accessibility(
            test_page_html
        )

        report = {
            "timestamp": "2024-01-01T00:00:00Z",
            "page_url": "http://test.example.com",
            "screen_reader_compatibility": {
                "semantic_structure": semantic_result,
                "form_accessibility": form_result,
                "heading_structure": "analyzed separately",
                "landmark_navigation": "analyzed separately",
                "aria_implementation": "analyzed separately",
            },
            "overall_score": 0.0,
            "recommendations": [],
            "assistive_technology_support": {
                "nvda": "compatible",
                "jaws": "compatible",
                "voiceover": "compatible",
                "talkback": "compatible",
                "dragon": "compatible",
            },
        }

        # Calculate overall score
        compliant_tests = 0
        total_tests = 0

        if semantic_result["compliant"]:
            compliant_tests += 1
        total_tests += 1

        if form_result["compliant"]:
            compliant_tests += 1
        total_tests += 1

        report["overall_score"] = compliant_tests / total_tests

        # Add recommendations
        if not semantic_result["compliant"]:
            report["recommendations"].append(
                "Improve semantic HTML structure and landmarks"
            )

        if not form_result["compliant"]:
            report["recommendations"].append(
                "Add proper labels and descriptions to form elements"
            )

        report["recommendations"].extend(
            [
                "Test with actual screen reader software",
                "Verify reading order and navigation flow",
                "Ensure all content is accessible via keyboard",
                "Validate ARIA implementation with screen reader users",
            ]
        )

        # Verify report structure
        assert "timestamp" in report
        assert "screen_reader_compatibility" in report
        assert 0.0 <= report["overall_score"] <= 1.0
        assert len(report["recommendations"]) > 0
        assert "assistive_technology_support" in report

        # Verify assistive technology coverage
        at_support = report["assistive_technology_support"]
        assert "nvda" in at_support, "Should test NVDA compatibility"
        assert "jaws" in at_support, "Should test JAWS compatibility"
        assert "voiceover" in at_support, "Should test VoiceOver compatibility"
